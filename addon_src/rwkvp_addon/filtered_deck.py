from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from .adaptive_retention import (
    AdaptiveRetentionCardData,
    AdaptiveRetentionSettings,
    active_desired_retention_with_adaptive,
)
from .anki_api import DeckRetention
from .filtered_deck_sort import (
    FilteredDeckOrder,
    FilteredDeckSortInfo,
    filtered_deck_order_from_index,
    filtered_deck_sort_key,
)
from .retrievability import CardPrediction
from .review_load_policy import (
    DEFAULT_MINIMUM_RETENTION_STEP,
    is_same_day_elapsed,
    select_with_minimum_retention,
)
from .review_rows import ReviewData, prediction_rows_for_card_ids
from .rwkv_modes import (
    RetrievabilityMode,
    generated_deck_name_prefix_for_mode,
)
from .rwkv_modes import (
    top_level_deck_label as _top_level_deck_label,
)
from .search_filters import required_search_with_filter


@dataclass(frozen=True)
class FilteredDeckSettings:
    limit: int
    order_index: int
    same_day_only: bool = False
    minimum: int = 0
    minimum_retention_extra_quantum: float = DEFAULT_MINIMUM_RETENTION_STEP
    adaptive_retention: AdaptiveRetentionSettings | None = None


@dataclass(frozen=True)
class FilteredDeckCandidate:
    card_id: int
    deck_id: int | None
    desired_retention: float
    predicted_retrievability: float
    elapsed_days: float = -1
    sort_info: FilteredDeckSortInfo | None = None
    same_day_desired_retention: float | None = None
    rwkv_stability_days: float | None = None
    fsrs_difficulty: float | None = None
    adaptive_retention: AdaptiveRetentionSettings | None = None

    @property
    def active_desired_retention(self) -> float:
        return active_desired_retention_with_adaptive(
            self.desired_retention,
            self.same_day_desired_retention,
            elapsed_days=self.elapsed_days,
            adaptive_settings=self.adaptive_retention,
            rwkv_stability_days=self.rwkv_stability_days,
            fsrs_difficulty=self.fsrs_difficulty,
        )


@dataclass(frozen=True)
class FilteredDeckPlan:
    source_deck_id: int
    source_deck_name: str
    settings: FilteredDeckSettings
    candidates: list[FilteredDeckCandidate]
    selected: list[FilteredDeckCandidate]
    expected_retrievability: float | None
    deck_name: str


class EmptyFilteredDeckPlanError(ValueError):
    pass


def build_filtered_deck_plan(
    *,
    source_deck_id: int,
    source_deck_name: str,
    card_ids: Iterable[int],
    review_data: ReviewData,
    retentions: Iterable[DeckRetention],
    target_timestamp_seconds: float,
    predictor,
    settings: FilteredDeckSettings,
    card_sort_info: Mapping[int, FilteredDeckSortInfo] | None = None,
    adaptive_retention_by_card: Mapping[int, AdaptiveRetentionCardData] | None = None,
    deck_name_prefix: str | None = None,
) -> FilteredDeckPlan:
    card_id_list = [int(card_id) for card_id in card_ids]
    sort_info_by_card = {int(key): value for key, value in (card_sort_info or {}).items()}
    adaptive_by_card = {
        int(key): value for key, value in (adaptive_retention_by_card or {}).items()
    }
    rows = prediction_rows_for_card_ids(
        card_id_list,
        review_data,
        target_timestamp_seconds=target_timestamp_seconds,
    )
    predictions = predictor(rows)
    retention_by_deck = {retention.deck_id: retention for retention in retentions}

    predicted_candidates: list[FilteredDeckCandidate] = []
    for card_id, row, prediction in zip(card_id_list, rows, predictions, strict=True):
        deck_id = row.get("deck_id")
        retention = retention_by_deck.get(deck_id) or retention_by_deck.get(source_deck_id)
        desired = retention.desired_retention if retention else 0.9
        same_day_desired = (
            retention.same_day_desired_retention
            if retention and retention.same_day_desired_retention is not None
            else desired
        )
        adaptive_data = adaptive_by_card.get(card_id)
        predicted_candidates.append(
            FilteredDeckCandidate(
                card_id=card_id,
                deck_id=deck_id,
                desired_retention=desired,
                same_day_desired_retention=same_day_desired,
                predicted_retrievability=float(prediction),
                elapsed_days=float(row.get("elapsed_days", -1)),
                sort_info=sort_info_by_card.get(card_id)
                or FilteredDeckSortInfo(card_id=card_id),
                rwkv_stability_days=(
                    adaptive_data.rwkv_stability_days if adaptive_data else None
                ),
                fsrs_difficulty=(
                    adaptive_data.fsrs_difficulty if adaptive_data else None
                ),
                adaptive_retention=settings.adaptive_retention,
            )
        )

    candidates, selected, _final_extra = select_candidates_with_minimum(
        predicted_candidates,
        settings=settings,
    )
    expected = (
        sum(candidate.predicted_retrievability for candidate in selected) / len(selected)
        if selected
        else None
    )
    deck_name = generated_deck_name_from_prefix(
        deck_name_prefix or generated_deck_name_prefix(source_deck_name),
        expected,
    )
    return FilteredDeckPlan(
        source_deck_id=source_deck_id,
        source_deck_name=source_deck_name,
        settings=settings,
        candidates=candidates,
        selected=selected,
        expected_retrievability=expected,
        deck_name=deck_name,
    )


def filtered_deck_candidate_search(
    base_search: str,
    settings: FilteredDeckSettings,
    *,
    extra_search: str | None = None,
) -> str:
    search = base_search.strip()
    if settings.same_day_only:
        search = f"{search} rated:1" if search else "rated:1"
    return required_search_with_filter(search, extra_search)


def ensure_filtered_deck_plan_has_selected_cards(plan: FilteredDeckPlan) -> None:
    if not plan.selected:
        raise EmptyFilteredDeckPlanError(
            "No cards matched the selected retention threshold."
        )


def retentions_with_desired_values(
    retentions: Iterable[DeckRetention],
    desired_values: Iterable[str],
    same_day_desired_values: Iterable[str] | None = None,
) -> list[DeckRetention]:
    retention_list = list(retentions)
    value_list = list(desired_values)
    if len(retention_list) != len(value_list):
        raise ValueError("Desired retention values do not match the displayed deck rows.")
    same_day_value_list = (
        list(same_day_desired_values)
        if same_day_desired_values is not None
        else list(value_list)
    )
    if len(retention_list) != len(same_day_value_list):
        raise ValueError("Desired retention values do not match the displayed deck rows.")
    return [
        replace(
            retention,
            desired_retention=parse_desired_retention(value),
            same_day_desired_retention=parse_desired_retention(same_day_value),
        )
        for retention, value, same_day_value in zip(
            retention_list,
            value_list,
            same_day_value_list,
            strict=True,
        )
    ]


def parse_desired_retention(value: str) -> float:
    text = value.strip()
    try:
        retention = float(text)
    except ValueError as exc:
        raise ValueError("Desired retention must be a number between 0 and 1.") from exc
    if not 0 < retention <= 1:
        raise ValueError("Desired retention must be greater than 0 and at most 1.")
    return retention


def sort_candidates(
    candidates: list[FilteredDeckCandidate],
    order_index: int,
) -> list[FilteredDeckCandidate]:
    rows = list(candidates)
    order = filtered_deck_order_from_index(order_index)
    random_keys = (
        {row.card_id: random.random() for row in rows}
        if order == FilteredDeckOrder.RANDOM
        else {}
    )
    return sorted(
        rows,
        key=lambda row: _candidate_sort_key(
            row,
            order=order,
            random_key=random_keys.get(row.card_id, 0.0),
        ),
    )


def select_candidates_with_minimum(
    candidates: Iterable[FilteredDeckCandidate],
    *,
    settings: FilteredDeckSettings,
) -> tuple[list[FilteredDeckCandidate], list[FilteredDeckCandidate], float]:
    candidate_list = list(candidates)
    order = filtered_deck_order_from_index(settings.order_index)
    random_keys = (
        {row.card_id: random.random() for row in candidate_list}
        if order == FilteredDeckOrder.RANDOM
        else {}
    )
    return select_with_minimum_retention(
        candidate_list,
        limit=settings.limit,
        minimum=settings.minimum,
        extra_quantum=settings.minimum_retention_extra_quantum,
        sort_key=lambda candidate: _candidate_sort_key(
            candidate,
            order=order,
            random_key=random_keys.get(candidate.card_id, 0.0),
        ),
        prediction=lambda candidate: candidate.predicted_retrievability,
        desired_retention=lambda candidate: candidate.active_desired_retention,
        allow_widening=lambda candidate: not _candidate_is_intraday(candidate),
    )


def _candidate_is_intraday(candidate: FilteredDeckCandidate) -> bool:
    return is_same_day_elapsed(candidate.elapsed_days)


def generated_deck_name(source_deck_name: str, expected_retrievability: float | None) -> str:
    return generated_deck_name_from_prefix(
        generated_deck_name_prefix(source_deck_name),
        expected_retrievability,
    )


def generated_deck_name_prefix(source_deck_name: str) -> str:
    return generated_deck_name_prefix_for_mode(
        RetrievabilityMode.IMMEDIATE,
        source_deck_name,
    )


def generated_curve_deck_name_prefix(source_deck_name: str) -> str:
    return generated_deck_name_prefix_for_mode(
        RetrievabilityMode.FORGETTING_CURVE,
        source_deck_name,
    )


def top_level_deck_label(source_deck_name: str) -> str:
    return _top_level_deck_label(source_deck_name)


def generated_deck_name_from_prefix(
    prefix: str,
    expected_retrievability: float | None,
) -> str:
    expected = "none" if expected_retrievability is None else f"{expected_retrievability:.3f}"
    return f"{prefix}{expected}"


def predictions_from_plan(plan: FilteredDeckPlan) -> list[CardPrediction]:
    return [
        CardPrediction(
            card_id=candidate.card_id,
            deck_id=candidate.deck_id,
            preset_id=None,
            retrievability=candidate.predicted_retrievability,
        )
        for candidate in plan.selected
    ]


def _candidate_sort_key(
    candidate: FilteredDeckCandidate,
    *,
    order: FilteredDeckOrder,
    random_key: float,
) -> tuple[float | int, ...]:
    return filtered_deck_sort_key(
        order=order,
        card_id=candidate.card_id,
        predicted_retrievability=candidate.predicted_retrievability,
        desired_retention=candidate.active_desired_retention,
        sort_info=candidate.sort_info,
        random_key=random_key,
    )
