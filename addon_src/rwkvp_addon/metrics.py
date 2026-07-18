from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .constants import DEFAULT_MODEL_ID
from .prediction_cache import (
    PER_REVIEW_CACHE_SPEC,
    PREDICT_AHEAD_CACHE_SPEC,
    EvaluationCacheValidation,
    PredictionCacheSpec,
    PredictionValueCache,
    StalePredictionCacheError,
    load_prediction_cache,
    load_prediction_value_cache,
)
from .profile_store import ProfileStore
from .progress import ProgressReporter
from .rwkv_processing import new_rwkvp_runtime, process_review_rows
from .rwkv_runtime_resources import release_runtime_resources


@dataclass(frozen=True)
class EvaluationScope:
    key: str
    label: str
    kind: str
    search: str
    card_ids: frozenset[int] | None = None
    deck_id: int | None = None
    preset_id: int | None = None
    preset_config_id: int | None = None

    def with_card_ids(self, card_ids: Iterable[int]) -> EvaluationScope:
        return replace(self, card_ids=frozenset(int(card_id) for card_id in card_ids))


class RWKVPredictionMode(Enum):
    PER_REVIEW = "per_review"
    PREDICT_AHEAD = "predict_ahead"


@dataclass(frozen=True)
class MetricResult:
    rmse_bins: float | None
    log_loss: float | None
    error: str | None = None


@dataclass(frozen=True)
class ScopedMetricResults:
    metrics: dict[str, MetricResult]
    counts: dict[str, int]


def calculate_rwkv_metrics(
    rows: list[dict[str, Any]],
    scopes: list[EvaluationScope],
    progress: ProgressReporter,
    *,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, MetricResult]:
    predictions = rwkv_review_predictions(rows, progress, model_id=model_id)
    return calculate_scoped_metrics(predictions, scopes)


def calculate_cached_rwkv_metrics(
    rows: list[dict[str, Any]],
    scopes: list[EvaluationScope],
    store: ProfileStore,
    progress: ProgressReporter,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    expected_processed_count: int | None = None,
    target_review_ids_by_scope: dict[str, set[int]] | None = None,
    prediction_mode: RWKVPredictionMode = RWKVPredictionMode.PER_REVIEW,
) -> dict[str, MetricResult]:
    predictions = cached_rwkv_review_predictions(
        rows,
        store,
        progress,
        model_id=model_id,
        expected_processed_count=expected_processed_count,
        prediction_mode=prediction_mode,
    )
    return calculate_scoped_metrics(
        predictions,
        scopes,
        target_review_ids_by_scope=target_review_ids_by_scope,
    )


def cached_rwkv_review_predictions(
    rows: list[dict[str, Any]],
    store: ProfileStore,
    progress: ProgressReporter,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    expected_processed_count: int | None = None,
    prediction_mode: RWKVPredictionMode = RWKVPredictionMode.PER_REVIEW,
    card_ids: Iterable[int] | None = None,
    validation: EvaluationCacheValidation | None = None,
) -> list[dict[str, Any]]:
    spec = prediction_cache_spec_for_mode(prediction_mode)

    progress.update(0, 1, f"Loading {spec.label}")
    cache = load_prediction_cache(
        spec.path(store),
        rows,
        model_id=model_id,
        cache_kind=spec.cache_kind,
        card_ids=card_ids,
        validation=validation,
    )
    processed_count = int(cache.metadata["processed_review_count"])
    if expected_processed_count is not None and processed_count != expected_processed_count:
        raise StalePredictionCacheError(
            "RWKV prediction cache is not aligned with the durable checkpoint. "
            "Run Update Checkpoint or Rebuild Checkpoint to refresh it."
        )
    if processed_count < len(rows):
        progress.update(
            1,
            1,
            f"Loaded {spec.label} through {processed_count} reviews",
        )
    else:
        progress.update(1, 1, f"Loaded {spec.label}")
    return cache.records


def cached_rwkv_review_prediction_values(
    rows: list[dict[str, Any]],
    store: ProfileStore,
    progress: ProgressReporter,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    expected_processed_count: int | None = None,
    prediction_mode: RWKVPredictionMode = RWKVPredictionMode.PER_REVIEW,
    validation: EvaluationCacheValidation | None = None,
) -> PredictionValueCache:
    """Load one evaluation prediction stream without duplicating row metadata."""

    spec = prediction_cache_spec_for_mode(prediction_mode)
    progress.update(0, 1, f"Loading {spec.label}")
    cache = load_prediction_value_cache(
        spec.path(store),
        rows,
        model_id=model_id,
        cache_kind=spec.cache_kind,
        validation=validation,
    )
    processed_count = int(cache.metadata["processed_review_count"])
    if expected_processed_count is not None and processed_count != expected_processed_count:
        raise StalePredictionCacheError(
            "RWKV prediction cache is not aligned with the durable checkpoint. "
            "Run Update Checkpoint or Rebuild Checkpoint to refresh it."
        )
    if processed_count < len(rows):
        progress.update(
            1,
            1,
            f"Loaded {spec.label} through {processed_count} reviews",
        )
    else:
        progress.update(1, 1, f"Loaded {spec.label}")
    return cache


def prediction_cache_spec_for_mode(
    prediction_mode: RWKVPredictionMode,
) -> PredictionCacheSpec:
    if prediction_mode == RWKVPredictionMode.PREDICT_AHEAD:
        return PREDICT_AHEAD_CACHE_SPEC
    return PER_REVIEW_CACHE_SPEC


def rwkv_review_predictions(
    rows: list[dict[str, Any]],
    progress: ProgressReporter,
    *,
    model_id: str = DEFAULT_MODEL_ID,
) -> list[dict[str, Any]]:
    runtime = new_rwkvp_runtime(model_id=model_id)
    try:
        return process_review_rows(
            runtime,
            rows,
            progress,
            label="Evaluating RWKV review predictions",
        )
    finally:
        release_runtime_resources(runtime)


def calculate_scoped_metrics(
    prediction_rows: list[dict[str, Any]],
    scopes: list[EvaluationScope],
    *,
    target_review_ids_by_scope: dict[str, set[int]] | None = None,
) -> dict[str, MetricResult]:
    results: dict[str, MetricResult] = {}
    for scope in scopes:
        target_ids = (
            None
            if target_review_ids_by_scope is None
            else target_review_ids_by_scope.get(scope.key)
        )
        scoped = scoped_metric_rows(
            prediction_rows,
            scope,
            target_ids,
        )
        try:
            predictions = [float(row["prediction"]) for row in scoped]
            rmse_bins, log_loss = _calculate_precomputed_metrics(
                scoped,
                predictions,
                range(len(scoped)),
            )
            results[scope.key] = MetricResult(
                rmse_bins=rmse_bins,
                log_loss=log_loss,
            )
        except Exception as exc:
            results[scope.key] = MetricResult(None, None, str(exc))
    return results


def count_scoped_metric_rows(
    prediction_rows: list[dict[str, Any]],
    scopes: list[EvaluationScope],
    *,
    target_review_ids_by_scope: dict[str, set[int]] | None = None,
) -> dict[str, int]:
    return {
        scope.key: len(
            scoped_metric_rows(
                prediction_rows,
                scope,
                None
                if target_review_ids_by_scope is None
                else target_review_ids_by_scope.get(scope.key),
            )
        )
        for scope in scopes
    }


def calculate_aligned_scoped_metrics(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[float | None],
    scopes: Sequence[EvaluationScope],
    *,
    target_review_ids_by_scope: dict[str, set[int]] | None = None,
) -> ScopedMetricResults:
    """Score aligned predictions without materializing per-review dictionaries.

    This is the specialized Evaluate path.  The cache's prediction array is
    already aligned and its shared fields are present in ``rows``.  Target IDs
    let us visit only each scope's FSRS test rows instead of rescanning the
    complete history for every scope.  The accumulator mirrors the precomputed
    field path in the benchmark's ``include_sameday=False`` metric contract.
    """

    if len(predictions) > len(rows):
        raise ValueError("Aligned RWKV predictions exceed the available review history.")

    processed_rows = rows[: len(predictions)]
    index_by_review_id: dict[int, int] | None = None
    metrics: dict[str, MetricResult] = {}
    counts: dict[str, int] = {}
    all_indices = range(len(predictions))

    for scope in scopes:
        target_ids = (
            None
            if target_review_ids_by_scope is None
            else target_review_ids_by_scope.get(scope.key)
        )
        if target_ids is None:
            candidate_indices: Iterable[int] = all_indices
        else:
            if index_by_review_id is None:
                index_by_review_id = {
                    int(row["review_id"]): index for index, row in enumerate(processed_rows)
                }
            candidate_indices = sorted(
                index_by_review_id[review_id]
                for review_id in target_ids
                if review_id in index_by_review_id
            )
        selected_indices = [
            index
            for index in candidate_indices
            if row_matches_scope(processed_rows[index], scope)
            and _has_finite_prediction_value(predictions[index])
        ]
        counts[scope.key] = len(selected_indices)
        try:
            rmse_bins, log_loss = _calculate_precomputed_metrics(
                processed_rows,
                predictions,
                selected_indices,
            )
            metrics[scope.key] = MetricResult(
                rmse_bins=rmse_bins,
                log_loss=log_loss,
            )
        except Exception as exc:
            metrics[scope.key] = MetricResult(None, None, str(exc))

    return ScopedMetricResults(metrics=metrics, counts=counts)


def scoped_metric_rows(
    prediction_rows: list[dict[str, Any]],
    scope: EvaluationScope,
    target_review_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in prediction_rows
        if row_matches_scope(row, scope)
        and _has_finite_prediction(row)
        and (target_review_ids is None or int(row["review_id"]) in target_review_ids)
    ]


def row_matches_scope(row: dict[str, Any], scope: EvaluationScope) -> bool:
    if scope.card_ids is not None:
        return int(row["card_id"]) in scope.card_ids
    if scope.deck_id is not None:
        return row.get("deck_id") == scope.deck_id
    if scope.preset_id is not None:
        return row.get("preset_id") == scope.preset_id
    return True


def _has_finite_prediction(row: dict[str, Any]) -> bool:
    prediction = row.get("prediction")
    if prediction is None:
        return False
    try:
        value = float(prediction)
    except (TypeError, ValueError):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


_FLOAT_EPS = 2.220446049250313e-16


def _has_finite_prediction_value(prediction: float | None) -> bool:
    if prediction is None:
        return False
    try:
        return math.isfinite(float(prediction))
    except (TypeError, ValueError):
        return False


def _calculate_precomputed_metrics(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[float | None],
    indices: Sequence[int],
) -> tuple[float, float]:
    grouped: dict[tuple[float, float, float], list[float]] = {}
    log_loss_sum = 0.0
    evaluated_count = 0

    for index in indices:
        prediction = float(predictions[index])  # selected indices are finite/non-null
        if prediction < 0.0 or prediction > 1.0:
            raise ValueError(f"prediction must be between 0 and 1, got {prediction}")

        row = rows[index]
        try:
            rating = int(row["rating"])
        except KeyError as exc:
            raise ValueError("Each review must include either rating or reality/y.") from exc
        if rating not in {1, 2, 3, 4}:
            continue

        elapsed_days = float(row["elapsed_days"])
        if elapsed_days == 0.0:
            continue
        delta_t = max(0.0, elapsed_days)
        if delta_t <= 0.0:
            continue

        review_count_value = row.get("review_count", row.get("i"))
        prior_lapses_value = row.get("prior_lapses", row.get("rmse_bins_lapse"))
        review_count = 1 if review_count_value is None else int(float(review_count_value))
        prior_lapses = 0 if prior_lapses_value is None else int(float(prior_lapses_value))
        actual = 0 if rating == 1 else 1

        clipped = min(max(prediction, _FLOAT_EPS), 1.0 - _FLOAT_EPS)
        log_loss_sum -= actual * math.log(clipped) + (1 - actual) * math.log(1.0 - clipped)

        key = (
            _elapsed_days_bin(elapsed_days),
            _review_count_bin(review_count),
            _lapse_count_bin(prior_lapses),
        )
        stats = grouped.setdefault(key, [0.0, 0.0, 0.0])
        stats[0] += actual
        stats[1] += prediction
        stats[2] += 1.0
        evaluated_count += 1

    if evaluated_count <= 0:
        raise ValueError("No evaluable reviews after preprocessing.")

    weighted_square_error = 0.0
    total_weight = 0.0
    # Evaluation-cache records do not store custom weights, so the reference
    # calculator's group weight is exactly the number of rows in each bin.
    for actual_sum, prediction_sum, count in grouped.values():
        actual_mean = actual_sum / count
        prediction_mean = prediction_sum / count
        weighted_square_error += count * (actual_mean - prediction_mean) ** 2
        total_weight += count

    if total_weight <= 0.0:
        raise ValueError("RMSE(bins) requires a positive total weight.")
    return (
        math.sqrt(weighted_square_error / total_weight),
        log_loss_sum / evaluated_count,
    )


def _elapsed_days_bin(elapsed_days: float) -> float:
    value = max(elapsed_days, 1e-6)
    return round(2.48 * 3.62 ** math.floor(math.log(value) / math.log(3.62)), 2)


def _review_count_bin(review_count: int) -> float:
    return round(
        1.99 * 1.89 ** math.floor(math.log(review_count) / math.log(1.89)),
        0,
    )


def _lapse_count_bin(prior_lapses: int) -> float:
    if prior_lapses == 0:
        return 0.0
    return round(
        1.65 * 1.73 ** math.floor(math.log(prior_lapses) / math.log(1.73)),
        0,
    )
