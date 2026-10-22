"""Prepare V5o-a Anki history while keeping source provenance explicit."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .compact_review_data import PackedProcessReviewRows, PackedRevlogRows, concatenate_revlog_rows
from .compact_v5o_history import V5O_SOURCE_RECORD, PackedV5oReviewRows
from .review_rows import (
    ANKI_REVLOG_KIND_LEARNING,
    CardInfo,
    DeckInfo,
    LastReviewInfo,
    ReviewData,
    ReviewRowBuildContext,
    _append_review_values_to_context,
    _card_metadata,
    _rwkv_processable_review_values,
    days_elapsed_for_review,
)
from .review_type_normalization import FilteredReviewNormalizationPolicy
from .v5o_review_rows import exact_i64


def build_v5o_review_data(
    revlogs: Iterable[Any],
    cards: dict[int, CardInfo],
    decks: dict[int, DeckInfo],
    next_day_at: int,
    *,
    exclude_deleted_card_revlogs: bool = True,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy | None = None,
    base: ReviewData | None = None,
    context: ReviewRowBuildContext | None = None,
    input_is_chronological: bool = False,
) -> ReviewData:
    """Preserve answers, optionally including cards absent from the collection.

    Anki revlog IDs are stable source keys, so they also serve as ordinals; a
    native/reference source can instead supply an explicit source_ordinal.
    Removing a deleted card never renumbers another card's source identity.
    The source learning marker is determined before constructing compatibility
    metric rows, whose missing-predecessor sentinel alone cannot identify it.
    """

    if base is not None and base.history_generation != "v5o":
        raise ValueError("V5o history cannot extend a different generation's prepared rows")
    policy = (
        base.filtered_review_normalization_policy
        if base is not None
        else filtered_review_normalization_policy or FilteredReviewNormalizationPolicy.disabled()
    )
    retained = (
        entry for entry in revlogs
        if not exclude_deleted_card_revlogs or exact_i64(entry.cid, "card_id") in cards
    )
    if not input_is_chronological:
        retained = sorted(retained, key=lambda entry: (
            exact_i64(entry.id, "review_finish_ms"),
            exact_i64(getattr(entry, "source_ordinal", entry.id), "source_ordinal"),
        ))
    if base is not None and context is None:
        context = base.row_build_context
        if context is None:
            raise ValueError("Incremental V5o history requires its source row context")
    if context is not None and context.filtered_review_normalization_policy != policy:
        raise ValueError("Review-row context uses a different Filtered-review policy.")
    updated_context = ReviewRowBuildContext(
        day_offset_origin=context.day_offset_origin if context else None,
        previous_by_card=dict(context.previous_by_card) if context else {},
        previous_review_kind_by_card=dict(context.previous_review_kind_by_card) if context else {},
        positive_day_counts_by_card=dict(context.positive_day_counts_by_card) if context else {},
        prior_lapses_by_card=dict(context.prior_lapses_by_card) if context else {},
        filtered_review_normalization_policy=policy,
        filtered_review_phase_by_card=(
            dict(context.filtered_review_phase_by_card) if context else {}
        ),
    )
    process_rows = PackedProcessReviewRows()
    revlog_rows = PackedRevlogRows()
    source_records = bytearray()
    last_by_card = {} if base is None else dict(base.last_by_card)
    lapses = {card_id: info.lapse_count for card_id, info in last_by_card.items()}
    latest_by_card = {}
    metadata_by_card = {}
    for source in retained:
        review_id = exact_i64(source.id, "review_finish_ms")
        ordinal = exact_i64(getattr(source, "source_ordinal", source.id), "source_ordinal")
        card_id = int(source.cid)
        rating = int(getattr(source, "button_chosen", 0))
        taken_millis = int(getattr(source, "taken_millis", 0))
        review_kind = int(getattr(source, "review_kind", 0))
        interval = int(getattr(source, "interval", 0))
        last_interval = int(getattr(source, "last_interval", 0))
        ease_factor = int(getattr(source, "ease_factor", 0))
        metadata = metadata_by_card.get(card_id)
        if metadata is None:
            metadata = metadata_by_card[card_id] = _card_metadata(card_id, cards, decks)
        note_id, deck_id, preset_id = metadata
        revlog_rows._append_prepared_values(
            review_id=review_id, card_id=card_id, note_id=note_id, deck_id=deck_id,
            preset_id=preset_id, rating=rating, duration=max(0.0, float(taken_millis)),
            taken_millis=taken_millis, review_kind=review_kind, interval=interval,
            last_interval=last_interval, ease_factor=ease_factor,
            days_elapsed=days_elapsed_for_review(review_id, next_day_at),
        )
        if not _rwkv_processable_review_values(rating, review_kind, ease_factor):
            continue
        first = _original_first_review(
            source, has_previous=card_id in updated_context.previous_by_card,
        )
        duration = exact_i64(source.taken_millis, "duration_ms")
        if duration < 0:
            raise ValueError("duration_ms must be nonnegative")
        exact_i64(review_id - duration, "review_start_ms")
        _row, updated_context, day_offset, elapsed_days = _append_review_values_to_context(
            review_id=review_id, card_id=card_id, note_id=note_id, deck_id=deck_id,
            preset_id=preset_id, rating=rating, taken_millis=duration, review_kind=review_kind,
            interval=interval, last_interval=last_interval, ease_factor=ease_factor,
            next_day_at=next_day_at, context=updated_context, is_learning_start=None,
            packed_rows=process_rows,
        )
        source_records.extend(V5O_SOURCE_RECORD.pack(ordinal, first))
        if rating == 1 and elapsed_days != 0:
            lapses[card_id] = lapses.get(card_id, 0) + 1
        latest_by_card[card_id] = (review_id, day_offset, interval)
    rows = PackedV5oReviewRows(process_rows, memoryview(source_records))
    revlog_rows.seal()
    if updated_context.day_offset_origin is None:
        updated_context = replace(updated_context, day_offset_origin=0)
    for card_id, (review_id, day_offset, interval) in latest_by_card.items():
        last_by_card[card_id] = LastReviewInfo(
            review_id, day_offset, review_id / 1000.0, interval, lapses.get(card_id, 0),
        )
    if base is None:
        complete_rows = rows
        complete_revlogs = revlog_rows
    else:
        complete_rows = (
            PackedV5oReviewRows.concatenate(base.rows, rows)
            if isinstance(base.rows, PackedV5oReviewRows) else [*base.rows, *rows]
        )
        complete_revlogs = concatenate_revlog_rows(base.revlogs, revlog_rows)
    return ReviewData(
        rows=complete_rows,
        revlogs=complete_revlogs,
        cards=cards,
        decks=decks,
        last_by_card=last_by_card,
        next_day_at=next_day_at,
        day_offset_origin=int(updated_context.day_offset_origin or 0),
        filtered_review_normalization_policy=policy,
        row_build_context=updated_context,
        history_generation="v5o",
    )


def add_v5o_source_facts(row: dict[str, Any], source: Any, *, has_previous: bool) -> None:
    """Attach the same source provenance during full replay and Live answers."""

    row["is_original_first_review"] = _original_first_review(source, has_previous=has_previous)
    row["source_ordinal"] = exact_i64(
        getattr(source, "source_ordinal", source.id), "source_ordinal"
    )
    row["review_finish_ms"] = exact_i64(source.id, "review_finish_ms")
    row["duration_ms"] = exact_i64(source.taken_millis, "duration_ms")
    if row["duration_ms"] < 0:
        raise ValueError("duration_ms must be nonnegative")
    row["review_start_ms"] = exact_i64(
        row["review_finish_ms"] - row["duration_ms"], "review_start_ms"
    )


def _original_first_review(source: Any, *, has_previous: bool) -> bool:
    first = getattr(source, "is_original_first_review", None)
    if first is None:
        first = not has_previous and int(source.review_kind) == ANKI_REVLOG_KIND_LEARNING
    if type(first) is not bool:
        raise TypeError("is_original_first_review must be a Boolean")
    if first and has_previous:
        raise ValueError("An original first review cannot follow a retained answer")
    return first
