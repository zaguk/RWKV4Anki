from __future__ import annotations

import time
from dataclasses import replace

from .adaptive_retention import (
    AdaptiveRetentionSettings,
    adaptive_retention_card_data_for_card_ids,
)
from .anki_api import fsrs_difficulties_for_card_ids
from .dataset_export import (
    load_review_data_for_checkpoint,
    open_checkpoint_runtime_from_load,
)
from .live_review_candidates import (
    LiveReviewCandidateBootstrap,
    build_live_review_candidate_bootstrap,
    live_review_candidates_with_adaptive_retention,
    predict_live_review_candidates,
)
from .review_rows import checkpoint_scope_cards_for_card_ids
from .rwkv_processing import predict_many_batched
from .vendor_bootstrap import require_rwkv_interval


def prepare_live_review_candidates_for_deck(
    col,
    *,
    source_deck_id: int,
    store,
    manager,
    progress,
    target_timestamp_seconds: float | None = None,
    retentions=None,
    same_day_only: bool = False,
    extra_search: str | None = None,
    adaptive_retention_settings: AdaptiveRetentionSettings | None = None,
    defer_initial_prediction: bool = False,
) -> LiveReviewCandidateBootstrap:
    progress.update(0, 1, "Finding RWKV Live Session candidates...")
    bootstrap = build_live_review_candidate_bootstrap(
        col,
        source_deck_id=int(source_deck_id),
        retentions=retentions,
        same_day_only=same_day_only,
        extra_search=extra_search,
    )
    progress.update(1, 1, "Found RWKV Live Session candidates.")
    if not bootstrap.candidates:
        return bootstrap

    review_load = load_review_data_for_checkpoint(
        col,
        store,
        manager,
        progress,
        allow_incremental=True,
        allow_persisted_tail_context=True,
    )
    readiness, runtime = open_checkpoint_runtime_from_load(
        manager,
        review_load,
        progress,
        scope_cards=checkpoint_scope_cards_for_card_ids(
            (candidate.card_id for candidate in bootstrap.candidates),
            review_load.review_data,
        ),
    )
    try:
        candidates = _with_adaptive_retention_data(
            col,
            manager=manager,
            candidates=bootstrap.candidates,
            adaptive_retention_settings=adaptive_retention_settings,
        )
        initial_prediction_deferred = bool(defer_initial_prediction)
        if not initial_prediction_deferred:
            prediction_time = (
                time.time()
                if target_timestamp_seconds is None
                else float(target_timestamp_seconds)
            )
            candidates = predict_live_review_candidates(
                candidates,
                review_data=readiness.review_data,
                target_timestamp_seconds=prediction_time,
                predictor=lambda rows: predict_many_batched(
                    runtime.predict_many,
                    rows,
                    progress,
                    label="Predicting RWKV Live Session candidates",
                    chunk_size=getattr(
                        runtime,
                        "predict_many_progress_chunk_size",
                        None,
                    ),
                ),
            )
    except BaseException:
        runtime.close()
        raise
    return replace(
        bootstrap,
        candidates=candidates,
        review_data=readiness.review_data,
        runtime_session=runtime,
        initial_prediction_deferred=initial_prediction_deferred,
    )


def refresh_live_review_candidates_for_deck(
    col,
    *,
    source_deck_id: int,
    review_data,
    manager,
    progress,
    runtime=None,
    target_timestamp_seconds: float | None = None,
    retentions=None,
    same_day_only: bool = False,
    extra_search: str | None = None,
    adaptive_retention_settings: AdaptiveRetentionSettings | None = None,
    defer_prediction: bool = False,
) -> LiveReviewCandidateBootstrap:
    runtime = manager if runtime is None else runtime
    progress.update(0, 1, "Finding RWKV Live Session candidates...")
    bootstrap = build_live_review_candidate_bootstrap(
        col,
        source_deck_id=int(source_deck_id),
        retentions=retentions,
        same_day_only=same_day_only,
        extra_search=extra_search,
    )
    progress.update(1, 1, "Found RWKV Live Session candidates.")
    if not bootstrap.candidates:
        return replace(
            bootstrap,
            review_data=review_data,
            initial_prediction_deferred=bool(defer_prediction),
        )

    candidates = _with_adaptive_retention_data(
        col,
        manager=manager,
        candidates=bootstrap.candidates,
        adaptive_retention_settings=adaptive_retention_settings,
    )
    if not defer_prediction:
        prediction_time = (
            time.time()
            if target_timestamp_seconds is None
            else float(target_timestamp_seconds)
        )
        candidates = predict_live_review_candidates(
            candidates,
            review_data=review_data,
            target_timestamp_seconds=prediction_time,
            predictor=lambda rows: predict_many_batched(
                runtime.predict_many,
                rows,
                progress,
                label="Rechecking RWKV Live Session candidates",
                chunk_size=getattr(
                    runtime,
                    "predict_many_progress_chunk_size",
                    None,
                ),
            ),
        )
    return replace(
        bootstrap,
        candidates=candidates,
        review_data=review_data,
        initial_prediction_deferred=bool(defer_prediction),
    )


def _with_adaptive_retention_data(
    col,
    *,
    manager,
    candidates,
    adaptive_retention_settings: AdaptiveRetentionSettings | None,
):
    if adaptive_retention_settings is None or not adaptive_retention_settings.enabled:
        return tuple(candidates)
    if not bool(getattr(manager, "calculate_curves", True)):
        return tuple(candidates)
    card_ids = tuple(int(candidate.card_id) for candidate in candidates)
    if not card_ids:
        return tuple(candidates)
    adaptive_data = adaptive_retention_card_data_for_card_ids(
        card_ids,
        latest_curves_by_card=manager.latest_curves_for_cards(card_ids),
        fsrs_difficulties_by_card=fsrs_difficulties_for_card_ids(col, card_ids),
        interval_for_curve=require_rwkv_interval(),
    )
    return live_review_candidates_with_adaptive_retention(
        candidates,
        adaptive_retention_by_card=adaptive_data,
        adaptive_retention_settings=adaptive_retention_settings,
    )
