from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .adaptive_retention import (
    AdaptiveRetentionCardData,
    AdaptiveRetentionSettings,
)
from .anki_api import (
    DeckRetention,
    active_card_search_for_deck,
    card_schedule_info_for_ids,
    deck_retentions_for_subtree,
    find_cards,
)
from .filtered_deck_sort import FilteredDeckSortInfo
from .live_review_engine import LiveReviewCandidate
from .review_rows import ReviewData, prediction_rows_for_card_ids
from .search_filters import required_search_with_filter

LIVE_REVIEW_CARD_SEARCH_EXCLUSIONS = "-is:buried -deck:filtered"
FALLBACK_DESIRED_RETENTION = 0.9
UNPREDICTED_RETRIEVABILITY = float("nan")


@dataclass(frozen=True)
class LiveReviewCandidateBootstrap:
    source_deck_id: int
    search: str
    card_ids: tuple[int, ...]
    candidates: tuple[LiveReviewCandidate, ...]
    review_data: ReviewData | None = None
    runtime_session: object | None = None
    initial_prediction_deferred: bool = False


def live_review_search_for_deck(
    col,
    source_deck_id: int,
    *,
    same_day_only: bool = False,
    extra_search: str | None = None,
) -> str:
    search = (
        f"{active_card_search_for_deck(col, int(source_deck_id))} "
        f"{LIVE_REVIEW_CARD_SEARCH_EXCLUSIONS}"
    )
    if same_day_only:
        search = f"{search} rated:1"
    return required_search_with_filter(search, extra_search)


def build_live_review_candidate_bootstrap(
    col,
    *,
    source_deck_id: int,
    card_ids: Iterable[int] | None = None,
    retentions: Iterable[DeckRetention] | None = None,
    same_day_only: bool = False,
    extra_search: str | None = None,
    adaptive_retention_by_card: Mapping[int, AdaptiveRetentionCardData] | None = None,
    adaptive_retention_settings: AdaptiveRetentionSettings | None = None,
) -> LiveReviewCandidateBootstrap:
    search = live_review_search_for_deck(
        col,
        int(source_deck_id),
        same_day_only=same_day_only,
        extra_search=extra_search,
    )
    resolved_card_ids = (
        tuple(int(card_id) for card_id in card_ids)
        if card_ids is not None
        else tuple(find_cards(col, search))
    )
    resolved_retentions = (
        list(retentions)
        if retentions is not None
        else deck_retentions_for_subtree(col, int(source_deck_id))
    )
    candidates = tuple(
        live_review_candidates_for_card_ids(
            col,
            source_deck_id=int(source_deck_id),
            card_ids=resolved_card_ids,
            retentions=resolved_retentions,
            adaptive_retention_by_card=adaptive_retention_by_card,
            adaptive_retention_settings=adaptive_retention_settings,
        )
    )
    return LiveReviewCandidateBootstrap(
        source_deck_id=int(source_deck_id),
        search=search,
        card_ids=resolved_card_ids,
        candidates=candidates,
    )


def live_review_candidates_for_card_ids(
    col,
    *,
    source_deck_id: int,
    card_ids: Iterable[int],
    retentions: Iterable[DeckRetention],
    adaptive_retention_by_card: Mapping[int, AdaptiveRetentionCardData] | None = None,
    adaptive_retention_settings: AdaptiveRetentionSettings | None = None,
) -> list[LiveReviewCandidate]:
    card_id_list = [int(card_id) for card_id in card_ids]
    if not card_id_list:
        return []

    schedule_infos = card_schedule_info_for_ids(col, card_id_list)
    retention_by_deck = {int(retention.deck_id): retention for retention in retentions}
    adaptive_by_card = {
        int(key): value for key, value in (adaptive_retention_by_card or {}).items()
    }
    rows: list[
        tuple[
            int,
            int | None,
            float,
            float,
            FilteredDeckSortInfo,
            AdaptiveRetentionCardData | None,
        ]
    ] = []
    for card_id in card_id_list:
        schedule_info = schedule_infos.get(card_id)
        if schedule_info is None:
            continue
        card_source_deck_id = schedule_info.source_deck_id
        retention = retention_by_deck.get(int(card_source_deck_id or 0))
        if retention is None:
            retention = retention_by_deck.get(int(source_deck_id))
        desired_retention = (
            float(retention.desired_retention)
            if retention is not None
            else FALLBACK_DESIRED_RETENTION
        )
        same_day_desired_retention = (
            float(retention.same_day_desired_retention)
            if retention is not None and retention.same_day_desired_retention is not None
            else desired_retention
        )
        rows.append(
            (
                card_id,
                card_source_deck_id,
                desired_retention,
                same_day_desired_retention,
                FilteredDeckSortInfo(
                    card_id=card_id,
                    modified_secs=schedule_info.modified_secs,
                ),
                adaptive_by_card.get(card_id),
            )
        )

    return [
        LiveReviewCandidate(
            card_id=card_id,
            source_deck_id=card_source_deck_id,
            desired_retention=desired_retention,
            same_day_desired_retention=same_day_desired_retention,
            predicted_retrievability=UNPREDICTED_RETRIEVABILITY,
            sort_info=sort_info,
            rwkv_stability_days=(
                adaptive_data.rwkv_stability_days if adaptive_data else None
            ),
            fsrs_difficulty=adaptive_data.fsrs_difficulty if adaptive_data else None,
            adaptive_retention=adaptive_retention_settings,
        )
        for (
            card_id,
            card_source_deck_id,
            desired_retention,
            same_day_desired_retention,
            sort_info,
            adaptive_data,
        ) in rows
    ]


def live_review_candidates_with_adaptive_retention(
    candidates: Iterable[LiveReviewCandidate],
    *,
    adaptive_retention_by_card: Mapping[int, AdaptiveRetentionCardData],
    adaptive_retention_settings: AdaptiveRetentionSettings | None,
) -> tuple[LiveReviewCandidate, ...]:
    adaptive_by_card = {
        int(card_id): data for card_id, data in adaptive_retention_by_card.items()
    }
    return tuple(
        replace(
            candidate,
            rwkv_stability_days=(
                adaptive_data.rwkv_stability_days if adaptive_data else None
            ),
            fsrs_difficulty=adaptive_data.fsrs_difficulty if adaptive_data else None,
            adaptive_retention=adaptive_retention_settings,
        )
        for candidate in candidates
        for adaptive_data in (adaptive_by_card.get(int(candidate.card_id)),)
    )


def predict_live_review_candidates(
    candidates: Iterable[LiveReviewCandidate],
    *,
    review_data: ReviewData,
    target_timestamp_seconds: float,
    predictor: Callable[[list[dict[str, Any]]], Iterable[float]],
) -> tuple[LiveReviewCandidate, ...]:
    candidate_list = tuple(candidates)
    if not candidate_list:
        return ()

    rows = prediction_rows_for_card_ids(
        [candidate.card_id for candidate in candidate_list],
        review_data,
        target_timestamp_seconds=target_timestamp_seconds,
    )
    predictions = list(predictor(rows))
    if len(predictions) != len(candidate_list):
        raise ValueError("RWKV live review prediction count did not match candidates.")

    return tuple(
        replace(
            candidate,
            predicted_retrievability=float(prediction),
            metadata={
                **dict(candidate.metadata),
                "prediction_timestamp_seconds": float(target_timestamp_seconds),
                "elapsed_days": float(row.get("elapsed_days", -1)),
            },
        )
        for candidate, row, prediction in zip(
            candidate_list,
            rows,
            predictions,
            strict=True,
        )
    )
