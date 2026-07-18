"""
Parity helpers for comparing RWKV-SRS against Anki's FSRS time-series evaluator.

Anki's public `evaluate_params()` returns aggregate metrics but does not expose
the individual review IDs selected as test rows by fsrs-rs time-series splits.
RWKV-SRS evaluation needs that same test-row set so both predictors are scored on
the same reviews.

This module reconstructs only that selection layer. It mirrors these Anki source
paths in `/workspace/anki-addon-development/references/anki/`:

- `rslib/src/scheduler/fsrs/params.rs::evaluate_params()`
- `rslib/src/scheduler/fsrs/params.rs::fsrs_items_for_training()`
- `rslib/src/scheduler/fsrs/params.rs::reviews_for_fsrs()`

Keep all review filtering and time-series target reconstruction here. Callers
should provide already-exported research dataset rows plus scope/card options;
GUI and Anki API modules should not duplicate this behavior.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Any

from .review_rows import ReviewData

DEFAULT_TIME_SERIES_SPLITS = 5

LEARNING = 0
FILTERED = 3
MANUAL = 4


@dataclass(frozen=True)
class FsrsTargetScope:
    key: str
    card_ids: frozenset[int] | None = None
    ignore_revlogs_before_ms: int = 0


ANKI_FSRS_PARITY_SOURCES = (
    "rslib/src/scheduler/fsrs/params.rs::evaluate_params()",
    "rslib/src/scheduler/fsrs/params.rs::fsrs_items_for_training()",
    "rslib/src/scheduler/fsrs/params.rs::reviews_for_fsrs()",
)


def fsrs_time_series_target_review_ids_by_scope(
    review_data: ReviewData,
    scopes: Sequence[FsrsTargetScope],
    *,
    processed_review_count: int | None = None,
) -> dict[str, set[int]]:
    return _review_ids_by_scope(
        review_data,
        scopes,
        time_series=True,
        processed_review_count=processed_review_count,
    )


def fsrs_training_item_review_ids_by_scope(
    review_data: ReviewData,
    scopes: Sequence[FsrsTargetScope],
    *,
    processed_review_count: int | None = None,
) -> dict[str, set[int]]:
    return _review_ids_by_scope(
        review_data,
        scopes,
        time_series=False,
        processed_review_count=processed_review_count,
    )


def _review_ids_by_scope(
    review_data: ReviewData,
    scopes: Sequence[FsrsTargetScope],
    *,
    time_series: bool,
    processed_review_count: int | None,
) -> dict[str, set[int]]:
    revlogs = revlogs_for_processed_review_prefix(review_data, processed_review_count)
    if not scopes:
        return {}
    if len(scopes) == 1:
        scope = scopes[0]
        selected = (
            fsrs_time_series_target_review_ids(
                revlogs,
                card_ids=scope.card_ids,
                ignore_revlogs_before_ms=scope.ignore_revlogs_before_ms,
            )
            if time_series
            else fsrs_training_item_review_ids(
                revlogs,
                card_ids=scope.card_ids,
                ignore_revlogs_before_ms=scope.ignore_revlogs_before_ms,
            )
        )
        return {scope.key: set(selected)}

    prepared = _PreparedFsrsTargetIndex(revlogs)
    results: dict[str, set[int]] = {}
    for scope in scopes:
        training_ids = prepared.training_item_review_ids(scope)
        selected = (
            time_series_test_items(training_ids, n_splits=DEFAULT_TIME_SERIES_SPLITS)
            if time_series
            else training_ids
        )
        results[scope.key] = set(selected)
    return results


class _PreparedFsrsTargetIndex:
    """One sorted history index shared by every requested evaluation scope.

    A card may belong to collection, deck, and preset scopes simultaneously.
    Its training-item sequence depends only on the card history and that
    scope's cutoff, so memoizing ``(card_id, cutoff)`` preserves the exact
    per-scope time-series split while avoiding a complete history sort/scan for
    every table row.
    """

    def __init__(self, revlogs: Sequence[dict[str, Any]]) -> None:
        ordered = sorted(
            revlogs,
            key=lambda item: (int(item["card_id"]), int(item["review_id"])),
        )
        self._entries_by_card = {
            int(card_id): tuple(entries)
            for card_id, entries in groupby(
                ordered,
                key=lambda item: int(item["card_id"]),
            )
        }
        self._training_ids_by_card_cutoff: dict[tuple[int, int], tuple[int, ...]] = {}

    def training_item_review_ids(self, scope: FsrsTargetScope) -> list[int]:
        card_ids: Iterable[int]
        if scope.card_ids is None:
            card_ids = self._entries_by_card
        else:
            card_ids = (int(card_id) for card_id in scope.card_ids)

        cutoff = int(scope.ignore_revlogs_before_ms)
        item_ids: list[int] = []
        for card_id in card_ids:
            entries = self._entries_by_card.get(card_id)
            if entries is None:
                continue
            cache_key = (card_id, cutoff)
            cached = self._training_ids_by_card_cutoff.get(cache_key)
            if cached is None:
                cached = tuple(
                    _training_item_review_ids_for_card(
                        list(entries),
                        ignore_revlogs_before_ms=cutoff,
                    )
                )
                self._training_ids_by_card_cutoff[cache_key] = cached
            item_ids.extend(cached)
        item_ids.sort()
        return item_ids


def revlogs_for_processed_review_prefix(
    review_data: ReviewData,
    processed_review_count: int | None,
) -> list[dict[str, Any]]:
    if processed_review_count is None or processed_review_count >= len(review_data.rows):
        return review_data.revlogs
    if processed_review_count <= 0:
        return []
    last_review_id = int(review_data.rows[processed_review_count - 1]["review_id"])
    return [row for row in review_data.revlogs if int(row["review_id"]) <= last_review_id]


def fsrs_time_series_target_review_ids(
    revlogs: Sequence[dict[str, Any]],
    *,
    card_ids: Iterable[int] | None = None,
    ignore_revlogs_before_ms: int = 0,
    n_splits: int = DEFAULT_TIME_SERIES_SPLITS,
) -> set[int]:
    target_ids = fsrs_training_item_review_ids(
        revlogs,
        card_ids=card_ids,
        ignore_revlogs_before_ms=ignore_revlogs_before_ms,
    )
    return set(time_series_test_items(target_ids, n_splits=n_splits))


def fsrs_training_item_review_ids(
    revlogs: Sequence[dict[str, Any]],
    *,
    card_ids: Iterable[int] | None = None,
    ignore_revlogs_before_ms: int = 0,
) -> list[int]:
    allowed_cards = None if card_ids is None else {int(card_id) for card_id in card_ids}
    scoped = (
        row
        for row in sorted(revlogs, key=lambda item: (int(item["card_id"]), int(item["review_id"])))
        if allowed_cards is None or int(row["card_id"]) in allowed_cards
    )
    item_ids: list[int] = []
    for _card_id, card_entries in groupby(scoped, key=lambda item: int(item["card_id"])):
        item_ids.extend(
            _training_item_review_ids_for_card(
                list(card_entries),
                ignore_revlogs_before_ms=ignore_revlogs_before_ms,
            )
        )
    item_ids.sort()
    return item_ids


def time_series_test_items(items: Sequence[int], *, n_splits: int) -> list[int]:
    item_count = len(items)
    if n_splits <= 0:
        return list(items)
    if item_count <= n_splits:
        return []

    test_size = item_count // (n_splits + 1)
    if test_size <= 0:
        return []

    first_test_start = item_count - n_splits * test_size
    selected: list[int] = []
    for test_start in range(first_test_start, item_count, test_size):
        selected.extend(items[test_start : test_start + test_size])
    return selected


def _training_item_review_ids_for_card(
    entries: list[dict[str, Any]],
    *,
    ignore_revlogs_before_ms: int,
) -> list[int]:
    if not entries:
        return []

    first_of_last_learn_entries: int | None = None
    first_user_grade_idx: int | None = None
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if _is_cramming(entry):
            continue

        within_cutoff = int(entry["review_id"]) > int(ignore_revlogs_before_ms)
        user_graded = _has_rating(entry)
        interday = int(entry.get("interval", 0)) >= 1 or int(entry.get("interval", 0)) <= -86400
        if user_graded and within_cutoff and interday:
            first_user_grade_idx = index

        if user_graded and int(entry.get("review_kind", 0)) == LEARNING:
            first_of_last_learn_entries = index
        elif _is_reset(entry):
            if first_of_last_learn_entries is not None:
                break
            if first_user_grade_idx is not None:
                break
            return []

    if first_of_last_learn_entries is not None:
        if int(entries[first_of_last_learn_entries]["review_id"]) < int(ignore_revlogs_before_ms):
            return []
        entries = entries[first_of_last_learn_entries:]
    else:
        return []

    entries = [entry for entry in entries if _has_rating_and_affects_scheduling(entry)]
    if len(entries) < 2:
        return []

    target_ids: list[int] = []
    previous_days_elapsed: int | None = None
    for index, entry in enumerate(entries):
        days_elapsed = int(entry["days_elapsed"])
        delta_t = 0 if previous_days_elapsed is None else previous_days_elapsed - days_elapsed
        if index >= 1 and delta_t > 0:
            target_ids.append(int(entry["review_id"]))
        previous_days_elapsed = days_elapsed
    return target_ids


def _has_rating(entry: dict[str, Any]) -> bool:
    return int(entry.get("button_chosen", entry.get("rating", 0))) > 0


def _has_rating_and_affects_scheduling(entry: dict[str, Any]) -> bool:
    return _has_rating(entry) and not _is_cramming(entry)


def _is_cramming(entry: dict[str, Any]) -> bool:
    return int(entry.get("review_kind", 0)) == FILTERED and int(entry.get("ease_factor", 0)) == 0


def _is_reset(entry: dict[str, Any]) -> bool:
    return int(entry.get("review_kind", 0)) == MANUAL and int(entry.get("ease_factor", 0)) == 0
