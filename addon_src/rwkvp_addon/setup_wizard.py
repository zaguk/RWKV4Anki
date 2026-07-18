from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .addon_config import (
    ACTIVE_REVIEW_PROTOTYPE_CONFIG_KEY,
    ADAPTIVE_DESIRED_RETENTION_CONFIG_KEY,
    CALCULATE_FORGETTING_CURVES_CONFIG_KEY,
    CARD_INFO_FORGETTING_CURVE_GRAPH_CONFIG_KEY,
    CARD_INFO_INTERVALS_CONFIG_KEY,
    CARD_INFO_RETRIEVABILITY_CONFIG_KEY,
    CURVE_RESCHEDULING_CONFIG_KEY,
    ENABLE_RWKV_IMMEDIATE_CONFIG_KEY,
    EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY,
    EXPERIMENTAL_SHORT_TERM_RESCHEDULING_CONFIG_KEY,
    MODEL_CONFIG_KEY,
)
from .config_options import (
    CONFIG_OPTIONS,
    ConfigOption,
    RestartRequirementContext,
    restart_required_option_labels,
)
from .metrics import MetricResult
from .rwkv_performance_modes import PREDICT_MANY_FAST_MODE

# Accuracy-tuning decisions use exactly the precision shown to the user.  This
# prevents invisible fifth-or-later decimal differences from changing a setup
# choice that appears tied in the result table.
METRIC_DISPLAY_DECIMAL_PLACES = 4

_MODE_EQUIVALENCE_REL_TOLERANCE = 1e-9
_MODE_EQUIVALENCE_ABS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SetupConfigChange:
    key_path: tuple[str, ...]
    label: str
    old_value: object
    new_value: object
    old_display: str
    new_display: str

    @property
    def description(self) -> str:
        return f"{self.label}: {self.old_display} → {self.new_display}"


@dataclass(frozen=True)
class CheckpointRebuildReason:
    key_path: tuple[str, ...]
    label: str
    explanation: str


@dataclass(frozen=True)
class SetupConfigSummary:
    changes: tuple[SetupConfigChange, ...]
    restart_required_labels: tuple[str, ...]
    checkpoint_rebuild_reasons: tuple[CheckpointRebuildReason, ...]


@dataclass(frozen=True)
class OptimizationDurationEstimate:
    reviews_per_minute: float | None
    model_processing_reviews: int
    deleted_reviews_processing_reviews: int
    matrix_processing_reviews: int
    model_comparison_seconds: float | None
    deleted_reviews_comparison_seconds: float | None
    matrix_comparison_seconds: float | None


@dataclass(frozen=True)
class MetricWinnerSelection:
    winner_key: str
    reason: str
    comparison_key: str | None


def apply_setup_feature_choices(
    config: Mapping[str, Any],
    *,
    immediate_enabled: bool,
    curves_enabled: bool,
) -> dict[str, Any]:
    """Return a changed copy of ``config`` with coherent feature dependencies.

    The input mapping is the transaction's immutable baseline. Experimental
    curve-dependent features are disabled when curves are disabled, rather than
    merely becoming temporarily ineffective and unexpectedly reappearing later.
    Behavior Lab and unrelated settings are deliberately left alone.
    """

    updated = deepcopy(dict(config))
    immediate = bool(immediate_enabled)
    curves = bool(curves_enabled)

    updated[ENABLE_RWKV_IMMEDIATE_CONFIG_KEY] = immediate
    updated[ACTIVE_REVIEW_PROTOTYPE_CONFIG_KEY] = immediate
    updated[CALCULATE_FORGETTING_CURVES_CONFIG_KEY] = curves
    updated[CARD_INFO_RETRIEVABILITY_CONFIG_KEY] = immediate
    updated[CARD_INFO_INTERVALS_CONFIG_KEY] = curves
    updated[CARD_INFO_FORGETTING_CURVE_GRAPH_CONFIG_KEY] = curves

    if not curves:
        updated[ADAPTIVE_DESIRED_RETENTION_CONFIG_KEY] = False
        updated[CURVE_RESCHEDULING_CONFIG_KEY] = False
        updated[EXPERIMENTAL_SHORT_TERM_RESCHEDULING_CONFIG_KEY] = False

    return updated


def apply_setup_immediate_choice(
    config: Mapping[str, Any],
    *,
    immediate_enabled: bool,
) -> dict[str, Any]:
    """Apply an Immediate answer before the later curve question is answered.

    Setup is transactional, but leaving midway can keep the choices made so
    far.  Record the Immediate answer as soon as it is made while preserving
    the existing curve choice and its dependent settings.
    """

    updated = deepcopy(dict(config))
    immediate = bool(immediate_enabled)
    updated[ENABLE_RWKV_IMMEDIATE_CONFIG_KEY] = immediate
    updated[ACTIVE_REVIEW_PROTOTYPE_CONFIG_KEY] = immediate
    updated[CARD_INFO_RETRIEVABILITY_CONFIG_KEY] = immediate
    return updated


def setup_config_changes(
    current: Mapping[str, Any],
    updated: Mapping[str, Any],
) -> tuple[SetupConfigChange, ...]:
    """Describe changes to user-facing settings in their normal visual order."""

    changes: list[SetupConfigChange] = []
    for option in CONFIG_OPTIONS:
        old_value = _path_value(current, option.key_path)
        new_value = _path_value(updated, option.key_path)
        if old_value == new_value:
            continue
        changes.append(
            SetupConfigChange(
                key_path=option.key_path,
                label=option.label,
                old_value=deepcopy(old_value),
                new_value=deepcopy(new_value),
                old_display=_display_option_value(option, old_value),
                new_display=_display_option_value(option, new_value),
            )
        )
    return tuple(changes)


def checkpoint_rebuild_reasons(
    current: Mapping[str, Any],
    updated: Mapping[str, Any],
) -> tuple[CheckpointRebuildReason, ...]:
    """Return setting changes that make the existing checkpoint insufficient."""

    reasons: list[CheckpointRebuildReason] = []
    if _path_value(current, (MODEL_CONFIG_KEY,)) != _path_value(updated, (MODEL_CONFIG_KEY,)):
        reasons.append(
            CheckpointRebuildReason(
                key_path=(MODEL_CONFIG_KEY,),
                label="Underlying Model",
                explanation=("The underlying model changed; RWKV checkpoints are model-specific."),
            )
        )

    if bool(_path_value(current, (EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY,))) != bool(
        _path_value(updated, (EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY,))
    ):
        reasons.append(
            CheckpointRebuildReason(
                key_path=(EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY,),
                label="Include History from Deleted Cards",
                explanation=(
                    "The review-history inclusion policy changed; rebuild so the "
                    "checkpoint uses the selected history."
                ),
            )
        )

    curves_were_enabled = bool(_path_value(current, (CALCULATE_FORGETTING_CURVES_CONFIG_KEY,)))
    curves_are_enabled = bool(_path_value(updated, (CALCULATE_FORGETTING_CURVES_CONFIG_KEY,)))
    if not curves_were_enabled and curves_are_enabled:
        reasons.append(
            CheckpointRebuildReason(
                key_path=(CALCULATE_FORGETTING_CURVES_CONFIG_KEY,),
                label="Calculate Forgetting Curves",
                explanation=(
                    "Forgetting Curves were enabled; rebuild to calculate the missing "
                    "curve history."
                ),
            )
        )
    return tuple(reasons)


def summarize_setup_config(
    current: Mapping[str, Any],
    updated: Mapping[str, Any],
    *,
    restart_context: RestartRequirementContext | None = None,
) -> SetupConfigSummary:
    current_dict = deepcopy(dict(current))
    updated_dict = deepcopy(dict(updated))
    return SetupConfigSummary(
        changes=setup_config_changes(current_dict, updated_dict),
        restart_required_labels=restart_required_option_labels(
            current_dict,
            updated_dict,
            context=restart_context,
        ),
        checkpoint_rebuild_reasons=checkpoint_rebuild_reasons(
            current_dict,
            updated_dict,
        ),
    )


def average_duration(durations: Sequence[float]) -> float | None:
    """Return a valid arithmetic mean, or ``None`` when a run failed."""

    values: list[float] = []
    for duration in durations:
        if isinstance(duration, bool):
            return None
        try:
            value = float(duration)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value < 0:
            return None
        values.append(value)
    if not values:
        return None
    return math.fsum(values) / len(values)


def choose_faster_mode(
    durations_by_mode: Mapping[str, Sequence[float]],
    *,
    fallback_mode: str = PREDICT_MANY_FAST_MODE,
) -> str:
    """Choose the lowest average duration, preferring Fast for ties/failures."""

    fallback = str(fallback_mode).strip().lower() or PREDICT_MANY_FAST_MODE
    averages = {
        str(mode).strip().lower(): average_duration(durations)
        for mode, durations in durations_by_mode.items()
    }
    valid = {mode: value for mode, value in averages.items() if value is not None}
    if not valid:
        return fallback

    best_average = min(valid.values())
    tied = {
        mode
        for mode, value in valid.items()
        if math.isclose(
            value,
            best_average,
            rel_tol=_MODE_EQUIVALENCE_REL_TOLERANCE,
            abs_tol=_MODE_EQUIVALENCE_ABS_TOLERANCE,
        )
    }
    if PREDICT_MANY_FAST_MODE in tied:
        return PREDICT_MANY_FAST_MODE
    if fallback in tied:
        return fallback
    return next(mode for mode in valid if mode in tied)


def choose_metric_winner(
    metrics_by_key: Mapping[str, MetricResult],
    *,
    default_key: str,
) -> str:
    """Select by displayed LogLoss, displayed RMSE(bins), then the default."""

    return select_metric_winner(
        metrics_by_key,
        default_key=default_key,
    ).winner_key


def select_metric_winner(
    metrics_by_key: Mapping[str, MetricResult],
    *,
    default_key: str,
    checkpoint_sizes_by_key: Mapping[str, int | None] | None = None,
) -> MetricWinnerSelection:
    """Return the winner, actual deciding value, and best alternative.

    LogLoss and RMSE(bins) are rounded to the same four decimal places used by
    the Setup Wizard before they participate in selection.  RMSE(bins) is only
    a tiebreaker when displayed LogLoss is tied.  When checkpoint sizes are
    supplied for every metric finalist, the smaller checkpoint breaks a tie on
    both displayed metrics.  The configured default is the final stable tie
    policy.
    """

    default = str(default_key)
    normalized = {str(key): metrics for key, metrics in metrics_by_key.items()}
    if default not in normalized:
        raise ValueError("The default metric candidate is missing.")

    valid = {key: metrics for key, metrics in normalized.items() if _valid_metrics(metrics)}
    if not valid:
        return MetricWinnerSelection(default, "no-valid-metrics", None)

    if len(valid) == 1:
        winner = next(iter(valid))
        return MetricWinnerSelection(winner, "only-valid-metric", None)

    minimum_log_loss = min(
        _metric_display_value(float(metrics.log_loss)) for metrics in valid.values()
    )
    log_loss_contenders = {
        key: metrics
        for key, metrics in valid.items()
        if _metric_display_value(float(metrics.log_loss)) == minimum_log_loss
    }
    reason = "log-loss"
    if len(log_loss_contenders) > 1:
        reason = "rmse-bins"
    minimum_rmse = min(
        _metric_display_value(float(metrics.rmse_bins)) for metrics in log_loss_contenders.values()
    )
    finalists = tuple(
        key
        for key, metrics in log_loss_contenders.items()
        if _metric_display_value(float(metrics.rmse_bins)) == minimum_rmse
    )
    normalized_sizes = (
        {
            str(key): _nonnegative_checkpoint_size(value)
            for key, value in checkpoint_sizes_by_key.items()
        }
        if checkpoint_sizes_by_key is not None
        else {}
    )
    if len(finalists) > 1 and normalized_sizes and all(
        normalized_sizes.get(key) is not None for key in finalists
    ):
        minimum_size = min(int(normalized_sizes[key]) for key in finalists)
        size_finalists = tuple(
            key for key in finalists if normalized_sizes[key] == minimum_size
        )
        if len(size_finalists) < len(finalists):
            reason = "checkpoint-size"
        finalists = size_finalists
    if len(finalists) > 1:
        reason = "default-tie" if default in finalists else "stable-tie"
    winner = default if default in finalists else finalists[0]

    valid_order = {key: index for index, key in enumerate(valid)}
    alternatives = [key for key in valid if key != winner]
    comparison = min(
        alternatives,
        key=lambda key: (
            _metric_display_value(float(valid[key].log_loss)),
            _metric_display_value(float(valid[key].rmse_bins)),
            (
                normalized_sizes.get(key)
                if normalized_sizes.get(key) is not None
                else math.inf
            ),
            0 if key == default else 1,
            valid_order[key],
        ),
    )
    return MetricWinnerSelection(winner, reason, comparison)


def metric_values_tie_at_display_precision(left: float, right: float) -> bool:
    """Return whether two finite metric values display identically in Setup."""

    return _metric_display_value(left) == _metric_display_value(right)


def valid_metric_candidate_keys(
    metrics_by_key: Mapping[str, MetricResult],
) -> tuple[str, ...]:
    """Return candidates with finite, successful LogLoss and RMSE values."""

    return tuple(str(key) for key, metrics in metrics_by_key.items() if _valid_metrics(metrics))


def lower_is_better_improvement_percent(
    baseline_value: float | None,
    candidate_value: float | None,
) -> float | None:
    """Return the candidate's percentage improvement over a lower-is-better baseline."""

    if baseline_value is None or candidate_value is None:
        return None
    try:
        baseline = float(baseline_value)
        candidate = float(candidate_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(baseline) or not math.isfinite(candidate) or baseline < 0 or candidate < 0:
        return None
    if baseline == 0:
        return 0.0 if candidate == 0 else None
    return (baseline - candidate) / baseline * 100.0


def estimate_optimization_durations(
    *,
    reviews_per_minute: float | None,
    model_count: int,
    model_review_count: int,
    without_deleted_review_count: int,
    with_deleted_review_count: int,
) -> OptimizationDurationEstimate:
    """Estimate each selectable full-history accuracy-tuning scope."""

    normalized_model_count = _nonnegative_int(model_count, "model_count")
    normalized_model_reviews = _nonnegative_int(
        model_review_count,
        "model_review_count",
    )
    without_deleted = _nonnegative_int(
        without_deleted_review_count,
        "without_deleted_review_count",
    )
    with_deleted = _nonnegative_int(
        with_deleted_review_count,
        "with_deleted_review_count",
    )
    model_work = normalized_model_count * normalized_model_reviews
    deleted_work = without_deleted + with_deleted
    matrix_work = normalized_model_count * deleted_work
    rate = _positive_finite_float(reviews_per_minute)
    if rate is None:
        return OptimizationDurationEstimate(
            reviews_per_minute=None,
            model_processing_reviews=model_work,
            deleted_reviews_processing_reviews=deleted_work,
            matrix_processing_reviews=matrix_work,
            model_comparison_seconds=None,
            deleted_reviews_comparison_seconds=None,
            matrix_comparison_seconds=None,
        )

    model_seconds = model_work * 60.0 / rate
    deleted_seconds = deleted_work * 60.0 / rate
    matrix_seconds = matrix_work * 60.0 / rate
    return OptimizationDurationEstimate(
        reviews_per_minute=rate,
        model_processing_reviews=model_work,
        deleted_reviews_processing_reviews=deleted_work,
        matrix_processing_reviews=matrix_work,
        model_comparison_seconds=model_seconds,
        deleted_reviews_comparison_seconds=deleted_seconds,
        matrix_comparison_seconds=matrix_seconds,
    )


def _nonnegative_checkpoint_size(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized if normalized >= 0 else None


def _display_option_value(option: ConfigOption, value: object) -> str:
    display_value = not bool(value) if option.inverted else value
    if option.value_type == "bool":
        return "On" if bool(display_value) else "Off"
    if option.value_type == "int":
        try:
            return f"{int(display_value):,}"
        except (TypeError, ValueError):
            return str(display_value)
    if option.key_path == (MODEL_CONFIG_KEY,):
        return str(display_value or "")
    choice = str(display_value or "")
    return "GPU" if choice.casefold() == "gpu" else choice.replace("_", " ").title()


def _path_value(config: Mapping[str, Any], path: tuple[str, ...]) -> object:
    value: object = config
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _valid_metrics(metrics: MetricResult) -> bool:
    if metrics.error or metrics.log_loss is None or metrics.rmse_bins is None:
        return False
    try:
        log_loss = float(metrics.log_loss)
        rmse = float(metrics.rmse_bins)
    except (TypeError, ValueError):
        return False
    return math.isfinite(log_loss) and math.isfinite(rmse) and log_loss >= 0 and rmse >= 0


def _metric_display_value(value: float) -> float:
    return float(f"{float(value):.{METRIC_DISPLAY_DECIMAL_PLACES}f}")


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if parsed < 0 or parsed != value:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _positive_finite_float(value: float | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None
