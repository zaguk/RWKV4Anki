from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .dataset_export import CheckpointReadiness
from .metrics import (
    RWKVPredictionMode,
    cached_rwkv_review_prediction_values,
    cached_rwkv_review_predictions,
    prediction_cache_spec_for_mode,
)
from .prediction_cache import (
    PredictionTailSnapshot,
    StalePredictionCacheError,
    prediction_record,
)
from .profile_store import ProfileStore
from .progress import ProgressReporter
from .review_rows import ReviewData


@dataclass(frozen=True)
class EvaluationHistoryRevision:
    """Lightweight identity for results produced from one review snapshot.

    Normal Anki review-history changes append or remove revlog rows, so the
    processable row count and boundary IDs prevent a modeless Evaluate window
    from mixing results produced before and after ordinary review/undo activity.
    Checkpoint consistency remains the authoritative full-history validation.
    """

    processable_review_count: int
    source_revlog_count: int
    first_review_id: int | None
    last_review_id: int | None


@dataclass(frozen=True)
class EvaluationPredictionSnapshot:
    predictions: tuple[float | None, ...]
    durable_processed_review_count: int
    history_revision: EvaluationHistoryRevision

    @property
    def processed_review_count(self) -> int:
        return len(self.predictions)


def evaluation_history_revision(
    review_data: ReviewData,
) -> EvaluationHistoryRevision:
    rows = review_data.rows
    revlogs = review_data.revlogs
    boundary_rows = revlogs or rows
    return EvaluationHistoryRevision(
        processable_review_count=len(rows),
        source_revlog_count=len(revlogs),
        first_review_id=(None if not boundary_rows else int(boundary_rows[0]["review_id"])),
        last_review_id=(None if not boundary_rows else int(boundary_rows[-1]["review_id"])),
    )


def load_current_evaluation_prediction_snapshot(
    readiness: CheckpointReadiness,
    store: ProfileStore,
    manager,
    progress: ProgressReporter,
    *,
    prediction_mode: RWKVPredictionMode,
) -> EvaluationPredictionSnapshot:
    """Combine the immutable checkpoint cache with its replayed in-memory tail."""

    rows = readiness.review_data.rows
    durable_count, tail_predictions = _validated_snapshot_tail(
        readiness,
        prediction_mode=prediction_mode,
    )

    durable_cache = cached_rwkv_review_prediction_values(
        rows,
        store,
        progress,
        model_id=manager.model_id,
        expected_processed_count=durable_count,
        prediction_mode=prediction_mode,
        validation=manager.evaluation_cache_validation(),
    )
    if len(durable_cache.predictions) != durable_count:
        raise StalePredictionCacheError(
            "RWKV evaluation cache values are not aligned with its durable checkpoint."
        )

    combined = (*durable_cache.predictions, *tail_predictions)
    if len(combined) != len(rows):
        raise StalePredictionCacheError(
            "RWKV evaluation prediction snapshot is not aligned with current history."
        )
    return EvaluationPredictionSnapshot(
        predictions=combined,
        durable_processed_review_count=durable_count,
        history_revision=evaluation_history_revision(readiness.review_data),
    )


def load_current_evaluation_prediction_rows(
    readiness: CheckpointReadiness,
    store: ProfileStore,
    manager,
    progress: ProgressReporter,
    *,
    prediction_mode: RWKVPredictionMode,
    card_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Load selected durable rows and append selected values from the current tail."""

    rows = readiness.review_data.rows
    durable_count, tail_predictions = _validated_snapshot_tail(
        readiness,
        prediction_mode=prediction_mode,
    )
    allowed = None if card_ids is None else {int(card_id) for card_id in card_ids}
    durable_rows = cached_rwkv_review_predictions(
        rows,
        store,
        progress,
        model_id=manager.model_id,
        expected_processed_count=durable_count,
        prediction_mode=prediction_mode,
        card_ids=allowed,
        validation=manager.evaluation_cache_validation(),
    )
    transient_rows = [
        prediction_record(row, prediction)
        for row, prediction in zip(
            rows[durable_count:],
            tail_predictions,
            strict=True,
        )
        if allowed is None or int(row["card_id"]) in allowed
    ]
    return [*durable_rows, *transient_rows]


def _validated_snapshot_tail(
    readiness: CheckpointReadiness,
    *,
    prediction_mode: RWKVPredictionMode,
) -> tuple[int, tuple[float | None, ...]]:
    rows = readiness.review_data.rows
    durable_count = readiness.durable_processed_review_count
    if durable_count is None:
        raise StalePredictionCacheError(
            "RWKV checkpoint does not report a durable processed-review count."
        )
    durable_count = int(durable_count)
    if durable_count < 0 or durable_count > len(rows):
        raise StalePredictionCacheError(
            "RWKV durable checkpoint count is not aligned with current review history."
        )

    tail = readiness.transient_prediction_tail
    if tail is None:
        tail = PredictionTailSnapshot.empty(durable_count)
    if tail.start_index != durable_count:
        raise StalePredictionCacheError(
            "RWKV transient evaluation predictions do not begin at the durable checkpoint boundary."
        )

    expected_tail_count = len(rows) - durable_count
    if len(tail.immediate_predictions) != expected_tail_count:
        raise StalePredictionCacheError(
            "RWKV transient evaluation predictions do not cover the complete current review tail."
        )
    spec = prediction_cache_spec_for_mode(prediction_mode)
    tail_predictions = tail.predictions_for(spec)
    if len(tail_predictions) != expected_tail_count:
        raise StalePredictionCacheError(
            f"RWKV transient {spec.label} do not cover the complete current review tail."
        )
    return durable_count, tail_predictions
