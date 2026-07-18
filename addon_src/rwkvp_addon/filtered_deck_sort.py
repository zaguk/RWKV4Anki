from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import IntEnum


class FilteredDeckOrder(IntEnum):
    """Supported order indexes from anki/proto/anki/decks.proto.

    The numeric values are intentionally sparse. They are persisted in profile
    settings and passed to Anki, so renumbering the reduced set would silently
    change existing selections.
    """

    RANDOM = 1
    RETRIEVABILITY_ASCENDING = 8
    RETRIEVABILITY_DESCENDING = 9
    RELATIVE_OVERDUENESS = 10


FILTERED_DECK_ORDER_OPTIONS: tuple[tuple[int, str], ...] = (
    (int(FilteredDeckOrder.RETRIEVABILITY_ASCENDING), "Ascending retrievability"),
    (int(FilteredDeckOrder.RETRIEVABILITY_DESCENDING), "Descending retrievability"),
    (int(FilteredDeckOrder.RELATIVE_OVERDUENESS), "Relative overdueness"),
    (int(FilteredDeckOrder.RANDOM), "Random"),
)


@dataclass(frozen=True)
class FilteredDeckSortInfo:
    card_id: int
    modified_secs: int = 0


def filtered_deck_order_from_index(order_index: int) -> FilteredDeckOrder:
    try:
        return FilteredDeckOrder(int(order_index))
    except (TypeError, ValueError):
        return FilteredDeckOrder.RETRIEVABILITY_ASCENDING


def filtered_deck_sort_key(
    *,
    order: FilteredDeckOrder,
    card_id: int,
    predicted_retrievability: float,
    desired_retention: float,
    sort_info: FilteredDeckSortInfo | None = None,
    random_key: float = 0.0,
    tiebreaker: int | None = None,
) -> tuple[float | int, ...]:
    normalized_card_id = int(card_id)
    resolved_info = sort_info or FilteredDeckSortInfo(card_id=normalized_card_id)
    resolved_tiebreaker = (
        filtered_deck_tiebreaker(normalized_card_id, resolved_info)
        if tiebreaker is None
        else int(tiebreaker)
    )

    if order == FilteredDeckOrder.RANDOM:
        return (float(random_key), resolved_tiebreaker)
    if order == FilteredDeckOrder.RETRIEVABILITY_ASCENDING:
        return (float(predicted_retrievability), resolved_tiebreaker)
    if order == FilteredDeckOrder.RETRIEVABILITY_DESCENDING:
        return (-float(predicted_retrievability), resolved_tiebreaker)
    if order == FilteredDeckOrder.RELATIVE_OVERDUENESS:
        desired = max(float(desired_retention), 0.0001)
        prediction = max(float(predicted_retrievability), 0.0001)
        return (prediction / desired, resolved_tiebreaker)
    raise AssertionError(f"Unhandled filtered-deck order: {order!r}")


def filtered_deck_tiebreaker(
    card_id: int,
    sort_info: FilteredDeckSortInfo | None = None,
) -> int:
    normalized_card_id = int(card_id)
    resolved_info = sort_info or FilteredDeckSortInfo(card_id=normalized_card_id)
    return _anki_tiebreaker(normalized_card_id, resolved_info)


def _anki_tiebreaker(card_id: int, sort_info: FilteredDeckSortInfo) -> int:
    # Mirrors Anki's fnvhash(c.id, c.mod) tie-breaker closely enough for local
    # candidate selection without querying the database directly.
    return _fnvhash_i64(int(card_id), int(sort_info.modified_secs))


def _fnvhash_i64(*values: int) -> int:
    hash_value = 0xCBF29CE484222325
    for value in values:
        for byte in int(value).to_bytes(8, sys.byteorder, signed=True):
            hash_value ^= byte
            hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    if hash_value >= 0x8000000000000000:
        return hash_value - 0x10000000000000000
    return hash_value
