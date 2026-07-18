from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .checkpoint_storage import rust_checkpoint_identity_counts
from .compact_review_data import PackedProcessReviewRows
from .metrics import (
    EvaluationScope,
    MetricResult,
    ScopedMetricResults,
    calculate_aligned_scoped_metrics,
)
from .prediction_cache import PredictionRecordSet
from .progress import ProgressReporter
from .review_rows import day_offset_origin_from_rows
from .rwkv_performance_modes import normalize_process_many_mode
from .rwkv_processing import new_rwkvp_runtime, process_review_rows_with_predictions
from .rwkv_runtime_resources import release_runtime_resources

DELETED_REVIEWS_COMPARISON = "deleted-reviews"
MODELS_COMPARISON = "models"
MODELS_AND_DELETED_REVIEWS_COMPARISON = "models-and-deleted-reviews"

_COLLECTION_SCOPE_KEY = "current-collection"

ComparisonReviewRows = list[dict[str, Any]] | PackedProcessReviewRows


class RuntimeFactory(Protocol):
    def __call__(self, *, model_id: str, process_many_mode: str) -> object: ...


@dataclass(frozen=True)
class ImmediateStateComparisonVariant:
    key: str
    label: str
    model_id: str
    include_deleted_reviews: bool
    rows: ComparisonReviewRows
    day_offset_adjustment: int = 0


@dataclass(frozen=True)
class ImmediateStateComparisonPlan:
    comparison: str
    baseline_key: str
    process_many_mode: str
    current_card_ids: frozenset[int]
    current_review_count: int
    variants: tuple[ImmediateStateComparisonVariant, ...]


@dataclass(frozen=True)
class ImmediateStateComparisonMeasurement:
    key: str
    label: str
    model_id: str
    include_deleted_reviews: bool
    processed_review_count: int
    evaluated_review_count: int
    metrics: MetricResult
    expected_checkpoint_bytes: int | None = None


@dataclass(frozen=True)
class ImmediateStateVariantEvaluation:
    """Scalar result from one disposable, curve-free RWKV state build."""

    processed_review_count: int
    evaluated_review_count: int
    metrics: MetricResult
    expected_checkpoint_bytes: int | None = None


@dataclass(frozen=True)
class ImmediateStateComparisonResult:
    comparison: str
    baseline_key: str
    process_many_mode: str
    current_card_count: int
    current_review_count: int
    measurements: tuple[ImmediateStateComparisonMeasurement, ...]

    @property
    def baseline(self) -> ImmediateStateComparisonMeasurement:
        return self.measurement(self.baseline_key)

    def measurement(self, key: str) -> ImmediateStateComparisonMeasurement:
        for measurement in self.measurements:
            if measurement.key == str(key):
                return measurement
        raise KeyError(key)


def deleted_review_comparison_plan(
    *,
    current_rows: list[dict[str, Any]],
    all_rows: ComparisonReviewRows,
    current_card_ids: Iterable[int],
    model_id: str,
    current_includes_deleted_reviews: bool,
    process_many_mode: str,
    without_deleted_day_offset_adjustment: int = 0,
) -> ImmediateStateComparisonPlan:
    without_deleted = ImmediateStateComparisonVariant(
        key="without-deleted-reviews",
        label="Without deleted-card history",
        model_id=str(model_id),
        include_deleted_reviews=False,
        rows=current_rows,
        day_offset_adjustment=int(without_deleted_day_offset_adjustment),
    )
    with_deleted = ImmediateStateComparisonVariant(
        key="with-deleted-reviews",
        label="With deleted-card history",
        model_id=str(model_id),
        include_deleted_reviews=True,
        rows=all_rows,
    )
    baseline_key = with_deleted.key if current_includes_deleted_reviews else without_deleted.key
    return ImmediateStateComparisonPlan(
        comparison=DELETED_REVIEWS_COMPARISON,
        baseline_key=baseline_key,
        process_many_mode=normalize_process_many_mode(process_many_mode),
        current_card_ids=frozenset(int(card_id) for card_id in current_card_ids),
        current_review_count=len(current_rows),
        variants=_baseline_first((without_deleted, with_deleted), baseline_key),
    )


def model_deleted_review_matrix_plan(
    *,
    current_rows: list[dict[str, Any]],
    all_rows: ComparisonReviewRows,
    current_card_ids: Iterable[int],
    model_ids: Sequence[str],
    current_model_id: str,
    current_includes_deleted_reviews: bool,
    process_many_mode: str,
    without_deleted_day_offset_adjustment: int = 0,
) -> ImmediateStateComparisonPlan:
    """Compare every model with both deleted-history policies.

    All without-deleted variants deliberately share ``current_rows`` and all
    with-deleted variants share ``all_rows``. Setup therefore retains only two
    potentially large histories while still constructing and releasing one
    disposable runtime per matrix cell.
    """

    normalized_models = tuple(dict.fromkeys(str(model_id) for model_id in model_ids))
    if not normalized_models:
        raise ValueError("No RWKV models are available for comparison.")
    selected_model = str(current_model_id)
    if selected_model not in normalized_models:
        raise ValueError("The selected RWKV model is not available for comparison.")

    variants: list[ImmediateStateComparisonVariant] = []
    baseline_key = ""
    for index, model_id in enumerate(normalized_models):
        for include_deleted, history_key, history_label in (
            (False, "without-deleted-reviews", "Without deleted-card history"),
            (True, "with-deleted-reviews", "With deleted-card history"),
        ):
            key = f"model-{index}-{history_key}"
            variants.append(
                ImmediateStateComparisonVariant(
                    key=key,
                    label=f"{model_id} · {history_label}",
                    model_id=model_id,
                    include_deleted_reviews=include_deleted,
                    rows=all_rows if include_deleted else current_rows,
                    day_offset_adjustment=(
                        0
                        if include_deleted
                        else int(without_deleted_day_offset_adjustment)
                    ),
                )
            )
            if (
                model_id == selected_model
                and include_deleted == bool(current_includes_deleted_reviews)
            ):
                baseline_key = key

    if not baseline_key:  # Defensive: the selected model was validated above.
        raise ValueError("The selected matrix baseline is unavailable.")
    return ImmediateStateComparisonPlan(
        comparison=MODELS_AND_DELETED_REVIEWS_COMPARISON,
        baseline_key=baseline_key,
        process_many_mode=normalize_process_many_mode(process_many_mode),
        current_card_ids=frozenset(int(card_id) for card_id in current_card_ids),
        current_review_count=len(current_rows),
        # Keep the two history policies adjacent for each model. Unlike the
        # standalone comparisons, Setup's matrix result does not use its
        # baseline for relative coloring, so moving that row first would only
        # make the table and progress sequence harder to scan.
        variants=tuple(variants),
    )


def current_review_rows_from_included_history(
    rows: Sequence[Mapping[str, Any]],
    current_card_ids: Iterable[int],
    *,
    source_day_offset_origin: int,
) -> tuple[list[dict[str, Any]], int]:
    """Select current-card rows and return the normal filtered-history day shift."""

    current_ids = frozenset(int(card_id) for card_id in current_card_ids)
    # Comparison variants temporarily adjust normalized day offsets.  Keep that
    # mutation isolated from the canonical compact history.
    current_rows = [dict(row) for row in rows if int(row["card_id"]) in current_ids]
    adjustment = int(source_day_offset_origin) - day_offset_origin_from_rows(current_rows)
    return current_rows, adjustment


def model_comparison_plan(
    *,
    rows: ComparisonReviewRows,
    current_card_ids: Iterable[int],
    model_ids: Sequence[str],
    current_model_id: str,
    include_deleted_reviews: bool,
    current_review_count: int,
    process_many_mode: str,
) -> ImmediateStateComparisonPlan:
    normalized_models = tuple(dict.fromkeys(str(model_id) for model_id in model_ids))
    if not normalized_models:
        raise ValueError("No RWKV models are available for comparison.")
    baseline_key = str(current_model_id)
    if baseline_key not in normalized_models:
        raise ValueError("The selected RWKV model is not available for comparison.")
    variants = tuple(
        ImmediateStateComparisonVariant(
            key=model_id,
            label=model_id,
            model_id=model_id,
            include_deleted_reviews=bool(include_deleted_reviews),
            rows=rows,
        )
        for model_id in normalized_models
    )
    return ImmediateStateComparisonPlan(
        comparison=MODELS_COMPARISON,
        baseline_key=baseline_key,
        process_many_mode=normalize_process_many_mode(process_many_mode),
        current_card_ids=frozenset(int(card_id) for card_id in current_card_ids),
        current_review_count=max(0, int(current_review_count)),
        variants=_baseline_first(variants, baseline_key),
    )


def run_immediate_state_comparison(
    plan: ImmediateStateComparisonPlan,
    progress: ProgressReporter,
    *,
    runtime_factory: RuntimeFactory | None = None,
    process_rows: Callable[..., PredictionRecordSet] | None = None,
    calculate_metrics: Callable[..., ScopedMetricResults] | None = None,
    release_runtime: Callable[[object], None] | None = None,
) -> ImmediateStateComparisonResult:
    """Build and score each disposable state while retaining only scalar metrics."""

    variants = tuple(plan.variants)
    _validate_plan(plan, variants)
    total_work = sum(len(variant.rows) + 1 for variant in variants)
    completed_work = 0
    measurements: list[ImmediateStateComparisonMeasurement] = []

    for index, variant in enumerate(variants, start=1):
        prefix = f"{variant.label} ({index}/{len(variants)})"
        variant_progress = _VariantProgress(
            progress,
            offset=completed_work,
            span=len(variant.rows),
            total_work=total_work,
            prefix=prefix,
        )
        evaluation = evaluate_immediate_state_variant(
            variant.rows,
            plan.current_card_ids,
            model_id=variant.model_id,
            process_many_mode=plan.process_many_mode,
            progress=variant_progress,
            day_offset_adjustment=variant.day_offset_adjustment,
            runtime_factory=runtime_factory,
            process_rows=process_rows,
            calculate_metrics=calculate_metrics,
            release_runtime=release_runtime,
        )
        measurements.append(
            ImmediateStateComparisonMeasurement(
                key=variant.key,
                label=variant.label,
                model_id=variant.model_id,
                include_deleted_reviews=variant.include_deleted_reviews,
                processed_review_count=evaluation.processed_review_count,
                evaluated_review_count=evaluation.evaluated_review_count,
                metrics=evaluation.metrics,
                expected_checkpoint_bytes=evaluation.expected_checkpoint_bytes,
            )
        )
        completed_work += len(variant.rows) + 1
        progress.update(completed_work, total_work, f"{prefix}: complete")

    return ImmediateStateComparisonResult(
        comparison=plan.comparison,
        baseline_key=plan.baseline_key,
        process_many_mode=plan.process_many_mode,
        current_card_count=len(plan.current_card_ids),
        current_review_count=plan.current_review_count,
        measurements=tuple(measurements),
    )


def evaluate_immediate_state_variant(
    rows: ComparisonReviewRows,
    current_card_ids: Iterable[int],
    *,
    model_id: str,
    process_many_mode: str,
    progress: ProgressReporter,
    day_offset_adjustment: int = 0,
    runtime_factory: RuntimeFactory | None = None,
    process_rows: Callable[..., PredictionRecordSet] | None = None,
    calculate_metrics: Callable[..., ScopedMetricResults] | None = None,
    release_runtime: Callable[[object], None] | None = None,
) -> ImmediateStateVariantEvaluation:
    """Build, evaluate, and release one disposable Immediate-only state.

    Both the comparison dialog and Setup Wizard use this shared in-process path,
    so resource cleanup and Evaluate-compatible metrics cannot drift apart.
    """

    make_runtime = runtime_factory or new_rwkvp_runtime
    process = process_rows or process_review_rows_with_predictions
    score = calculate_metrics or calculate_aligned_scoped_metrics
    release = release_runtime or release_runtime_resources
    scope = EvaluationScope(
        key=_COLLECTION_SCOPE_KEY,
        label="Current collection",
        kind="collection",
        search="",
        card_ids=frozenset(int(card_id) for card_id in current_card_ids),
    )
    runtime = None
    records = PredictionRecordSet.empty()
    effective_rows = rows
    rows_adjusted = False
    total = max(1, len(rows))
    try:
        progress.check_cancelled()
        progress.update(0, total, f"Loading {model_id}")
        if day_offset_adjustment:
            if isinstance(rows, PackedProcessReviewRows):
                effective_rows = [dict(row) for row in rows]
            _adjust_day_offsets(effective_rows, day_offset_adjustment)
            rows_adjusted = True
        runtime = make_runtime(
            model_id=str(model_id),
            process_many_mode=str(process_many_mode),
        )
        progress.check_cancelled()
        records = process(
            runtime,
            effective_rows,
            progress,
            label="Building disposable RWKV state",
            record_set=records,
            latest_curves_by_card=None,
            process_many_mode=str(process_many_mode),
            calculate_curves=False,
        )
        progress.check_cancelled()
        progress.update(len(rows), total, "Calculating RMSE(bins) and LogLoss")
        scoped = score(effective_rows, records.immediate_predictions, (scope,))
        expected_checkpoint_bytes = _expected_checkpoint_bytes(runtime, effective_rows)
        progress.check_cancelled()
        return ImmediateStateVariantEvaluation(
            processed_review_count=len(rows),
            # This count comes from the same aligned metric calculation as
            # Evaluate; a separate preprocessing count can disagree.
            evaluated_review_count=scoped.counts[scope.key],
            metrics=scoped.metrics[scope.key],
            expected_checkpoint_bytes=expected_checkpoint_bytes,
        )
    finally:
        # Prediction arrays can be large. Clear them before loading another
        # model/state, then explicitly close the native runtime on its worker.
        records.immediate_predictions.clear()
        records.predict_ahead_predictions.clear()
        try:
            if runtime is not None:
                release(runtime)
        finally:
            if rows_adjusted:
                _adjust_day_offsets(effective_rows, -day_offset_adjustment)


class _VariantProgress(ProgressReporter):
    def __init__(
        self,
        parent: ProgressReporter,
        *,
        offset: int,
        span: int,
        total_work: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self._parent = parent
        self._offset = max(0, int(offset))
        self._span = max(0, int(span))
        self._total_work = max(1, int(total_work))
        self._prefix = str(prefix)

    def check_cancelled(self) -> None:
        self._parent.check_cancelled()

    def update(self, current: int, total: int, label: str = "") -> None:
        source_total = max(1, int(total))
        source_current = min(source_total, max(0, int(current)))
        mapped = self._offset + round(self._span * source_current / source_total)
        text = f"{self._prefix}: {label} ({source_current:,}/{max(0, int(total)):,} reviews)"
        self._parent.update(mapped, self._total_work, text)


def _validate_plan(
    plan: ImmediateStateComparisonPlan,
    variants: tuple[ImmediateStateComparisonVariant, ...],
) -> None:
    if plan.comparison not in {
        DELETED_REVIEWS_COMPARISON,
        MODELS_COMPARISON,
        MODELS_AND_DELETED_REVIEWS_COMPARISON,
    }:
        raise ValueError(f"Unsupported state comparison: {plan.comparison!r}.")
    if not variants:
        raise ValueError("The state comparison has no variants.")
    keys = tuple(variant.key for variant in variants)
    if len(set(keys)) != len(keys):
        raise ValueError("State comparison variant keys must be unique.")
    if plan.baseline_key not in keys:
        raise ValueError("The state comparison baseline is missing.")
    if not plan.current_card_ids:
        raise ValueError("The collection has no current cards to evaluate.")
    if plan.current_review_count <= 0:
        raise ValueError("The collection has no current-card reviews to evaluate.")
    if any(not variant.rows for variant in variants):
        raise ValueError("The collection has no processable reviews to compare.")


def _baseline_first(
    variants: Sequence[ImmediateStateComparisonVariant],
    baseline_key: str,
) -> tuple[ImmediateStateComparisonVariant, ...]:
    return tuple(
        sorted(
            variants,
            key=lambda variant: (variant.key != baseline_key,),
        )
    )


def _adjust_day_offsets(rows: Sequence[dict[str, Any]], adjustment: int) -> None:
    changed = 0
    try:
        for row in rows:
            row["day_offset"] = int(row["day_offset"]) + int(adjustment)
            changed += 1
    except BaseException:
        for row in rows[:changed]:
            row["day_offset"] = int(row["day_offset"]) - int(adjustment)
        raise


def _expected_checkpoint_bytes(
    runtime: object,
    rows: Sequence[Mapping[str, Any]],
) -> int | None:
    expected_checkpoint_size = getattr(runtime, "expected_checkpoint_size", None)
    if not callable(expected_checkpoint_size):
        return None
    counts = rust_checkpoint_identity_counts(rows)
    return int(expected_checkpoint_size(**counts.as_kwargs()))
