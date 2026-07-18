from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compact_review_data import PackedProcessReviewRows, PackedRevlogRows
from .review_schema import (
    prediction_input_row,
    processing_review_row,
    revlog_metadata_row,
)
from .review_type_normalization import (
    FilteredReviewNormalizationPolicy,
    resolve_rwkv_review_state,
)

SECONDS_PER_DAY = 86_400
NEW_CARD_ELAPSED = -1
RWKV_DURATION_MAX_MS = 60_000
ANKI_REVLOG_KIND_LEARNING = 0
ANKI_REVLOG_KIND_REVIEW = 1
ANKI_REVLOG_KIND_RELEARNING = 2
ANKI_REVLOG_KIND_FILTERED = 3
ANKI_REVLOG_KIND_MANUAL = 4
ANKI_REVLOG_KIND_RESCHEDULED = 5
RWKV_PROCESSABLE_REVIEW_KINDS = {
    ANKI_REVLOG_KIND_LEARNING,
    ANKI_REVLOG_KIND_REVIEW,
    ANKI_REVLOG_KIND_RELEARNING,
    ANKI_REVLOG_KIND_FILTERED,
}


@dataclass(frozen=True)
class CardInfo:
    card_id: int
    note_id: int | None
    deck_id: int | None
    preset_id: int | None


@dataclass(frozen=True)
class DeckInfo:
    deck_id: int
    parent_id: int | None
    preset_id: int | None


@dataclass(frozen=True)
class LastReviewInfo:
    review_id: int
    day_offset: int
    timestamp_seconds: float
    interval: int
    lapse_count: int


@dataclass(frozen=True)
class LearningSequenceTrimResult:
    revlogs: list[Any]
    learning_start_review_ids: frozenset[int]
    original_processable_count: int
    dropped_processable_count: int


@dataclass(frozen=True)
class IncrementalLearningSequenceTrimResult:
    revlogs: list[Any]
    learning_start_review_ids: frozenset[int]
    invalidating_learning_start_review_ids: frozenset[int]
    original_processable_count: int
    dropped_processable_count: int


@dataclass(frozen=True)
class ReviewData:
    rows: Sequence[Mapping[str, Any]]
    revlogs: Sequence[Mapping[str, Any]]
    cards: dict[int, CardInfo]
    decks: dict[int, DeckInfo]
    last_by_card: dict[int, LastReviewInfo]
    next_day_at: int
    day_offset_origin: int
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy = field(
        default_factory=FilteredReviewNormalizationPolicy.disabled
    )
    # A cold Live Session can use a virtual durable prefix plus a materialized
    # tail.  These fields let its per-card history state start without scanning
    # the virtual prefix.  Ordinary complete histories leave them unset.
    row_build_context: ReviewRowBuildContext | None = None
    latest_revlog_id_by_card: dict[int, int] | None = None
    complete_history: bool = True


@dataclass(frozen=True)
class ReviewRowBuildContext:
    day_offset_origin: int | None
    previous_by_card: dict[int, tuple[int, int]]
    previous_review_kind_by_card: dict[int, int]
    positive_day_counts_by_card: dict[int, int]
    prior_lapses_by_card: dict[int, int]
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy = field(
        default_factory=FilteredReviewNormalizationPolicy.disabled
    )
    filtered_review_phase_by_card: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CompactReviewHistoryBuildResult:
    """Final compact outputs from one full-history construction pass."""

    rows: PackedProcessReviewRows
    revlogs: PackedRevlogRows
    last_by_card: dict[int, LastReviewInfo]
    day_offset_origin: int
    original_processable_count: int
    dropped_processable_count: int


def load_research_dataset(path: str | Path):
    from anki import stats_pb2

    dataset = stats_pb2.Dataset()
    dataset.ParseFromString(Path(path).read_bytes())
    return dataset


def review_data_from_dataset(
    dataset: Any,
    *,
    exclude_deleted_card_revlogs: bool = False,
    trim_learning_sequences: bool = True,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy | None = None,
) -> ReviewData:
    normalization_policy = (
        filtered_review_normalization_policy or FilteredReviewNormalizationPolicy.disabled()
    )
    cards = _cards_from_dataset(dataset)
    decks = _decks_from_dataset(dataset)
    source_revlogs = (
        _revlogs_for_current_cards(dataset.revlogs, cards)
        if exclude_deleted_card_revlogs
        else dataset.revlogs
    )
    history = build_compact_review_history(
        source_revlogs,
        cards,
        decks,
        int(dataset.next_day_at),
        trim_learning_sequences=trim_learning_sequences,
        filtered_review_normalization_policy=normalization_policy,
    )
    return ReviewData(
        rows=history.rows,
        revlogs=history.revlogs,
        cards=cards,
        decks=decks,
        last_by_card=history.last_by_card,
        next_day_at=int(dataset.next_day_at),
        day_offset_origin=history.day_offset_origin,
        filtered_review_normalization_policy=normalization_policy,
    )


def _revlogs_for_current_cards(
    revlogs: Iterable[Any],
    cards: dict[int, CardInfo],
) -> list[Any]:
    current_card_ids = {card_id for card_id, card in cards.items() if card.note_id is not None}
    return [revlog for revlog in revlogs if int(revlog.cid) in current_card_ids]


def build_compact_review_history(
    revlogs: Iterable[Any],
    cards: dict[int, CardInfo],
    decks: dict[int, DeckInfo],
    next_day_at: int,
    *,
    trim_learning_sequences: bool = True,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy | None = None,
    input_is_chronological: bool = False,
) -> CompactReviewHistoryBuildResult:
    """Build both compact full-history tables in one chronological output pass.

    Benchmark-compatible trimming depends on the final Learning start for each
    card, so boundary discovery remains a separate read-only phase. Once those
    boundaries are known, metadata rows, process rows, the day origin, and the
    latest-review map are all produced together without intermediate retained
    row lists or a second scan of the packed process table.
    """

    source = revlogs if isinstance(revlogs, Sequence) else list(revlogs)
    ordered_revlogs = (
        source if input_is_chronological else sorted(source, key=lambda entry: int(entry.id))
    )
    retained_start_by_card: dict[int, int] | None = None
    original_processable_count = 0
    if trim_learning_sequences:
        (
            retained_start_by_card,
            original_processable_count,
        ) = _learning_sequence_retention_boundaries(ordered_revlogs)

    normalization_policy = (
        filtered_review_normalization_policy or FilteredReviewNormalizationPolicy.disabled()
    )
    context = ReviewRowBuildContext(
        day_offset_origin=None,
        previous_by_card={},
        previous_review_kind_by_card={},
        positive_day_counts_by_card={},
        prior_lapses_by_card={},
        filtered_review_normalization_policy=normalization_policy,
        filtered_review_phase_by_card={},
    )
    process_rows = PackedProcessReviewRows()
    metadata_rows = PackedRevlogRows()
    latest_by_card: dict[int, tuple[int, int, int]] = {}
    lapse_counts_by_card: dict[int, int] = {}
    retained_processable_count = 0

    for revlog in ordered_revlogs:
        review_id = int(revlog.id)
        card_id = int(revlog.cid)
        retained_start = (
            retained_start_by_card.get(card_id) if retained_start_by_card is not None else None
        )
        if retained_start_by_card is not None and (
            retained_start is None or review_id < retained_start
        ):
            continue

        rating = int(getattr(revlog, "button_chosen", 0))
        taken_millis = int(getattr(revlog, "taken_millis", 0))
        review_kind = int(getattr(revlog, "review_kind", 0))
        interval = int(getattr(revlog, "interval", 0))
        last_interval = int(getattr(revlog, "last_interval", 0))
        ease_factor = int(getattr(revlog, "ease_factor", 0))
        note_id, deck_id, preset_id = _card_metadata(card_id, cards, decks)
        metadata_rows.append_values(
            review_id=review_id,
            card_id=card_id,
            note_id=note_id,
            deck_id=deck_id,
            preset_id=preset_id,
            rating=rating,
            duration=max(0.0, float(taken_millis)),
            taken_millis=taken_millis,
            review_kind=review_kind,
            interval=interval,
            last_interval=last_interval,
            ease_factor=ease_factor,
            days_elapsed=days_elapsed_for_review(review_id, next_day_at),
        )

        if not _rwkv_processable_review_values(rating, review_kind, ease_factor):
            continue
        retained_processable_count += 1
        (
            _row,
            context,
            day_offset,
            elapsed_days,
        ) = _append_review_values_to_context(
            review_id=review_id,
            card_id=card_id,
            note_id=note_id,
            deck_id=deck_id,
            preset_id=preset_id,
            rating=rating,
            taken_millis=taken_millis,
            review_kind=review_kind,
            interval=interval,
            last_interval=last_interval,
            ease_factor=ease_factor,
            next_day_at=next_day_at,
            context=context,
            is_learning_start=(
                review_id == retained_start if retained_start_by_card is not None else None
            ),
            packed_rows=process_rows,
        )
        if rating == 1 and elapsed_days != 0:
            lapse_counts_by_card[card_id] = lapse_counts_by_card.get(card_id, 0) + 1
        latest_by_card[card_id] = (review_id, day_offset, interval)

    process_rows.seal()
    metadata_rows.seal()
    day_offset_origin = int(context.day_offset_origin or 0)
    last_by_card = {
        card_id: LastReviewInfo(
            review_id=review_id,
            day_offset=day_offset,
            timestamp_seconds=review_id / 1000.0,
            interval=interval,
            lapse_count=lapse_counts_by_card.get(card_id, 0),
        )
        for card_id, (review_id, day_offset, interval) in latest_by_card.items()
    }
    if retained_start_by_card is None:
        original_processable_count = retained_processable_count
    return CompactReviewHistoryBuildResult(
        rows=process_rows,
        revlogs=metadata_rows,
        last_by_card=last_by_card,
        day_offset_origin=day_offset_origin,
        original_processable_count=original_processable_count,
        dropped_processable_count=(original_processable_count - retained_processable_count),
    )


def _learning_sequence_retention_boundaries(
    chronological_revlogs: Iterable[Any],
) -> tuple[dict[int, int], int]:
    """Return each retained card's final benchmark Learning-start review id."""

    retained_start_by_card: dict[int, int] = {}
    previous_review_kind_by_card: dict[int, int] = {}
    processable_count = 0
    for entry in chronological_revlogs:
        rating = int(getattr(entry, "button_chosen", 0))
        review_kind = int(getattr(entry, "review_kind", 0))
        ease_factor = int(getattr(entry, "ease_factor", 0))
        if not _rwkv_processable_review_values(rating, review_kind, ease_factor):
            continue
        processable_count += 1
        card_id = int(entry.cid)
        if (
            review_kind == ANKI_REVLOG_KIND_LEARNING
            and previous_review_kind_by_card.get(card_id) != ANKI_REVLOG_KIND_LEARNING
        ):
            retained_start_by_card[card_id] = int(entry.id)
        previous_review_kind_by_card[card_id] = review_kind
    return retained_start_by_card, processable_count


def build_revlog_rows(
    revlogs: Iterable[Any],
    cards: dict[int, CardInfo],
    decks: dict[int, DeckInfo],
    next_day_at: int,
    *,
    compact: bool = False,
) -> list[dict[str, Any]] | PackedRevlogRows:
    rows: list[dict[str, Any]] | PackedRevlogRows = PackedRevlogRows() if compact else []
    for revlog in sorted(revlogs, key=lambda entry: int(entry.id)):
        card_id = int(revlog.cid)
        note_id, deck_id, preset_id = _card_metadata(card_id, cards, decks)
        review_id = int(revlog.id)
        rating = int(getattr(revlog, "button_chosen", 0))
        taken_millis = int(getattr(revlog, "taken_millis", 0))
        review_kind = int(getattr(revlog, "review_kind", 0))
        interval = int(getattr(revlog, "interval", 0))
        last_interval = int(getattr(revlog, "last_interval", 0))
        ease_factor = int(getattr(revlog, "ease_factor", 0))
        days_elapsed = days_elapsed_for_review(review_id, next_day_at)
        if isinstance(rows, PackedRevlogRows):
            rows.append_values(
                review_id=review_id,
                card_id=card_id,
                note_id=note_id,
                deck_id=deck_id,
                preset_id=preset_id,
                rating=rating,
                duration=max(0.0, float(taken_millis)),
                taken_millis=taken_millis,
                review_kind=review_kind,
                interval=interval,
                last_interval=last_interval,
                ease_factor=ease_factor,
                days_elapsed=days_elapsed,
            )
            continue
        rows.append(
            revlog_metadata_row(
                review_id=review_id,
                card_id=card_id,
                note_id=note_id,
                deck_id=deck_id,
                preset_id=preset_id,
                rating=rating,
                duration=float(taken_millis),
                taken_millis=taken_millis,
                state=review_kind,
                interval=interval,
                last_interval=last_interval,
                ease_factor=ease_factor,
                days_elapsed=days_elapsed,
            )
        )
    return rows.seal() if isinstance(rows, PackedRevlogRows) else rows


def build_review_rows(
    revlogs: Iterable[Any],
    cards: dict[int, CardInfo],
    decks: dict[int, DeckInfo],
    next_day_at: int,
    *,
    context: ReviewRowBuildContext | None = None,
    trim_learning_sequences: bool = False,
    learning_start_review_ids: frozenset[int] | None = None,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy | None = None,
    input_is_chronological: bool = False,
    input_is_processable: bool = False,
    compact: bool = False,
) -> list[dict[str, Any]] | PackedProcessReviewRows:
    rows, _context = build_review_rows_with_context(
        revlogs,
        cards,
        decks,
        next_day_at,
        context=context,
        trim_learning_sequences=trim_learning_sequences,
        learning_start_review_ids=learning_start_review_ids,
        filtered_review_normalization_policy=filtered_review_normalization_policy,
        input_is_chronological=input_is_chronological,
        input_is_processable=input_is_processable,
        compact=compact,
    )
    return rows


def build_review_rows_with_context(
    revlogs: Iterable[Any],
    cards: dict[int, CardInfo],
    decks: dict[int, DeckInfo],
    next_day_at: int,
    *,
    context: ReviewRowBuildContext | None = None,
    trim_learning_sequences: bool = False,
    learning_start_review_ids: frozenset[int] | None = None,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy | None = None,
    input_is_chronological: bool = False,
    input_is_processable: bool = False,
    compact: bool = False,
) -> tuple[list[dict[str, Any]] | PackedProcessReviewRows, ReviewRowBuildContext]:
    if trim_learning_sequences and context is not None:
        raise ValueError("Learning-sequence trimming requires a full review history.")
    if trim_learning_sequences and learning_start_review_ids is not None:
        raise ValueError("Pass either trim_learning_sequences or learning_start_review_ids.")
    if (
        context is not None
        and filtered_review_normalization_policy is not None
        and context.filtered_review_normalization_policy != filtered_review_normalization_policy
    ):
        raise ValueError("Review-row context uses a different Filtered-review policy.")

    if trim_learning_sequences:
        trim = trim_benchmark_learning_sequences(revlogs)
        revlogs = trim.revlogs
        learning_start_review_ids = trim.learning_start_review_ids
        input_is_chronological = True
        input_is_processable = True

    normalization_policy = (
        context.filtered_review_normalization_policy
        if context is not None
        else filtered_review_normalization_policy or FilteredReviewNormalizationPolicy.disabled()
    )
    rows: list[dict[str, Any]] | PackedProcessReviewRows = (
        PackedProcessReviewRows() if compact else []
    )
    mutable_context = ReviewRowBuildContext(
        day_offset_origin=context.day_offset_origin if context else None,
        previous_by_card=dict(context.previous_by_card) if context else {},
        previous_review_kind_by_card=(
            dict(context.previous_review_kind_by_card) if context else {}
        ),
        positive_day_counts_by_card=(dict(context.positive_day_counts_by_card) if context else {}),
        prior_lapses_by_card=dict(context.prior_lapses_by_card) if context else {},
        filtered_review_normalization_policy=normalization_policy,
        filtered_review_phase_by_card=(
            dict(context.filtered_review_phase_by_card) if context else {}
        ),
    )

    ordered_revlogs = (
        revlogs if input_is_chronological else sorted(revlogs, key=lambda entry: int(entry.id))
    )
    for revlog in ordered_revlogs:
        row, mutable_context = append_review_row_to_context(
            revlog,
            cards,
            decks,
            next_day_at,
            mutable_context,
            learning_start_review_ids=learning_start_review_ids,
            input_is_processable=input_is_processable,
            packed_rows=rows if isinstance(rows, PackedProcessReviewRows) else None,
        )
        if row is not None:
            rows.append(row)

    if mutable_context.day_offset_origin is None:
        mutable_context = ReviewRowBuildContext(
            day_offset_origin=0,
            previous_by_card=mutable_context.previous_by_card,
            previous_review_kind_by_card=mutable_context.previous_review_kind_by_card,
            positive_day_counts_by_card=mutable_context.positive_day_counts_by_card,
            prior_lapses_by_card=mutable_context.prior_lapses_by_card,
            filtered_review_normalization_policy=(
                mutable_context.filtered_review_normalization_policy
            ),
            filtered_review_phase_by_card=(mutable_context.filtered_review_phase_by_card),
        )
    if isinstance(rows, PackedProcessReviewRows):
        rows.seal()
    return rows, mutable_context


def append_review_row_to_context(
    revlog: Any,
    cards: dict[int, CardInfo],
    decks: dict[int, DeckInfo],
    next_day_at: int,
    context: ReviewRowBuildContext,
    *,
    learning_start_review_ids: frozenset[int] | None = None,
    input_is_processable: bool = False,
    packed_rows: PackedProcessReviewRows | None = None,
) -> tuple[dict[str, Any] | None, ReviewRowBuildContext]:
    if not input_is_processable and not rwkv_processable_revlog(revlog):
        return None, context

    rating = int(getattr(revlog, "button_chosen", 0))
    card_id = int(revlog.cid)
    review_id = int(revlog.id)
    note_id, deck_id, preset_id = _card_metadata(card_id, cards, decks)
    row, context, _day_offset, _elapsed_days = _append_review_values_to_context(
        review_id=review_id,
        card_id=card_id,
        note_id=note_id,
        deck_id=deck_id,
        preset_id=preset_id,
        rating=rating,
        taken_millis=int(getattr(revlog, "taken_millis", 0)),
        review_kind=int(getattr(revlog, "review_kind", 0)),
        interval=int(getattr(revlog, "interval", 0)),
        last_interval=int(getattr(revlog, "last_interval", 0)),
        ease_factor=int(getattr(revlog, "ease_factor", 0)),
        next_day_at=next_day_at,
        context=context,
        is_learning_start=(
            review_id in learning_start_review_ids
            if learning_start_review_ids is not None
            else None
        ),
        packed_rows=packed_rows,
    )
    return row, context


def _append_review_values_to_context(
    *,
    review_id: int,
    card_id: int,
    note_id: int | None,
    deck_id: int | None,
    preset_id: int | None,
    rating: int,
    taken_millis: int,
    review_kind: int,
    interval: int,
    last_interval: int,
    ease_factor: int,
    next_day_at: int,
    context: ReviewRowBuildContext,
    is_learning_start: bool | None,
    packed_rows: PackedProcessReviewRows | None,
) -> tuple[dict[str, Any] | None, ReviewRowBuildContext, int, int]:
    review_ts = review_id / 1000.0
    raw_day_offset = day_offset_for_timestamp(review_ts, next_day_at)
    day_offset_origin = context.day_offset_origin
    if day_offset_origin is None:
        day_offset_origin = raw_day_offset
        context = ReviewRowBuildContext(
            day_offset_origin=day_offset_origin,
            previous_by_card=context.previous_by_card,
            previous_review_kind_by_card=context.previous_review_kind_by_card,
            positive_day_counts_by_card=context.positive_day_counts_by_card,
            prior_lapses_by_card=context.prior_lapses_by_card,
            filtered_review_normalization_policy=(context.filtered_review_normalization_policy),
            filtered_review_phase_by_card=context.filtered_review_phase_by_card,
        )
    day_offset = raw_day_offset - day_offset_origin
    previous = context.previous_by_card.get(card_id)
    if previous is None:
        elapsed_days = NEW_CARD_ELAPSED
        elapsed_seconds = NEW_CARD_ELAPSED
    else:
        previous_day, previous_review_id = previous
        elapsed_days = day_offset - previous_day
        elapsed_seconds = int((review_id - previous_review_id) // 1000)

    benchmark_state = rwkv_benchmark_state_for_review_kind(
        review_kind,
        previous_review_kind=context.previous_review_kind_by_card.get(card_id),
        is_learning_start=is_learning_start,
    )
    state, filtered_review_phase = resolve_rwkv_review_state(
        benchmark_state=benchmark_state,
        is_filtered=review_kind == ANKI_REVLOG_KIND_FILTERED,
        elapsed_days=elapsed_days,
        rating=rating,
        previous_phase=context.filtered_review_phase_by_card.get(card_id),
        normalize_filtered=(context.filtered_review_normalization_policy.applies_to(review_id)),
    )
    duration = rwkv_duration_millis(taken_millis)
    prior_lapses = context.prior_lapses_by_card.get(card_id, 0)
    if elapsed_days > 0:
        context.positive_day_counts_by_card[card_id] = (
            context.positive_day_counts_by_card.get(card_id, 0) + 1
        )
    review_count = context.positive_day_counts_by_card.get(card_id, 0) + 1

    if packed_rows is None:
        row = processing_review_row(
            review_id=review_id,
            card_id=card_id,
            note_id=note_id,
            deck_id=deck_id,
            preset_id=preset_id,
            raw_day_offset=raw_day_offset,
            day_offset=day_offset,
            elapsed_days=elapsed_days,
            elapsed_seconds=elapsed_seconds,
            rating=rating,
            duration=duration,
            taken_millis=taken_millis,
            state=state,
            review_kind=review_kind,
            interval=interval,
            last_interval=last_interval,
            ease_factor=ease_factor,
            review_count=review_count,
            prior_lapses=prior_lapses,
        )
    else:
        packed_rows.append_values(
            review_id=review_id,
            card_id=card_id,
            note_id=note_id,
            deck_id=deck_id,
            preset_id=preset_id,
            raw_day_offset=raw_day_offset,
            day_offset=day_offset,
            elapsed_days=elapsed_days,
            elapsed_seconds=elapsed_seconds,
            rating=rating,
            duration=duration,
            taken_millis=taken_millis,
            state=state,
            review_kind=review_kind,
            interval=interval,
            last_interval=last_interval,
            ease_factor=ease_factor,
            review_count=review_count,
            prior_lapses=prior_lapses,
        )
        row = None
    if rating == 1 and elapsed_days > 0:
        context.prior_lapses_by_card[card_id] = prior_lapses + 1
    context.previous_by_card[card_id] = (day_offset, review_id)
    context.previous_review_kind_by_card[card_id] = review_kind
    context.filtered_review_phase_by_card[card_id] = filtered_review_phase
    return row, context, day_offset, elapsed_days


def rwkv_benchmark_state_for_review_kind(
    review_kind: int,
    *,
    previous_review_kind: int | None,
    is_learning_start: bool | None = None,
) -> int:
    """Map Anki revlog kind to the benchmark/training state convention.

    The original parquet builder filtered non-processable rows first, then used:
    first learning row in a learning sequence -> 0, every other raw kind -> kind + 1.
    This keeps the first retained learning review as the model's new-card state,
    while later learning rows become state 1, review rows become state 2, and
    relearning rows become state 3.
    """

    raw_kind = int(review_kind)
    if is_learning_start is not None:
        if raw_kind == ANKI_REVLOG_KIND_LEARNING and is_learning_start:
            return 0
        return raw_kind + 1
    if raw_kind == ANKI_REVLOG_KIND_LEARNING and previous_review_kind != ANKI_REVLOG_KIND_LEARNING:
        return 0
    return raw_kind + 1


def trim_benchmark_learning_sequences(
    revlogs: Iterable[Any],
    *,
    input_is_card_ordered: bool = False,
) -> LearningSequenceTrimResult:
    """Reproduce the original parquet builder's learning-sequence trimming.

    The research export orders revlogs by ``cid, id``. The builder identifies
    learning starts in that order, drops rows before the last learning-start
    sequence for each card, then drops cards whose retained history does not
    begin with a learning start. The returned revlogs are sorted chronologically
    for the add-on's normal row construction path.
    """

    processable = (entry for entry in revlogs if rwkv_processable_revlog(entry))
    entries = (
        list(processable)
        if input_is_card_ordered
        else sorted(
            processable,
            key=lambda entry: (int(entry.cid), int(entry.id)),
        )
    )
    row_infos: list[tuple[Any, bool, int]] = []
    per_card_counts: dict[int, int] = {}
    sequence_group = 0
    previous_review_kind: int | None = None
    last_learning_start_group_by_card: dict[int, int] = {}

    for entry in entries:
        card_id = int(entry.cid)
        review_kind = int(getattr(entry, "review_kind", 0))
        per_card_counts[card_id] = per_card_counts.get(card_id, 0) + 1
        is_learning_start = review_kind == ANKI_REVLOG_KIND_LEARNING and (
            previous_review_kind != ANKI_REVLOG_KIND_LEARNING or per_card_counts[card_id] == 1
        )
        if is_learning_start:
            sequence_group += 1
            last_learning_start_group_by_card[card_id] = sequence_group
        row_infos.append((entry, is_learning_start, sequence_group))
        previous_review_kind = review_kind

    retained_by_card: dict[int, list[tuple[Any, bool]]] = {}
    for entry, is_learning_start, row_sequence_group in row_infos:
        card_id = int(entry.cid)
        last_learning_start_group = last_learning_start_group_by_card.get(card_id, 0)
        if last_learning_start_group <= row_sequence_group:
            retained_by_card.setdefault(card_id, []).append((entry, is_learning_start))

    retained_entries: list[Any] = []
    learning_start_review_ids: set[int] = set()
    for retained in retained_by_card.values():
        if not retained or not retained[0][1]:
            continue
        for entry, is_learning_start in retained:
            retained_entries.append(entry)
            if is_learning_start:
                learning_start_review_ids.add(int(entry.id))

    retained_entries.sort(key=lambda entry: int(entry.id))
    return LearningSequenceTrimResult(
        revlogs=retained_entries,
        learning_start_review_ids=frozenset(learning_start_review_ids),
        original_processable_count=len(entries),
        dropped_processable_count=len(entries) - len(retained_entries),
    )


def trim_incremental_benchmark_learning_sequences(
    revlogs: Iterable[Any],
    context: ReviewRowBuildContext,
) -> IncrementalLearningSequenceTrimResult:
    """Return the tail rows that can be safely appended under benchmark trimming.

    Full benchmark trimming can rewrite all retained history for a card when a
    later learning-start sequence appears. Incremental checkpoint updates cannot
    remove already-processed recurrent state, so such rows are reported as
    invalidating. For cards with no retained RWKV history yet, non-learning rows
    are ignored until a new learning-start row appears.
    """

    entries = sorted(
        (entry for entry in revlogs if rwkv_processable_revlog(entry)),
        key=lambda entry: (int(entry.cid), int(entry.id)),
    )
    previous_kind_by_card = dict(context.previous_review_kind_by_card)
    retained_cards = set(context.previous_by_card)
    retained_entries: list[Any] = []
    learning_start_review_ids: set[int] = set()
    invalidating_learning_start_review_ids: set[int] = set()
    dropped_count = 0

    for entry in entries:
        card_id = int(entry.cid)
        review_id = int(entry.id)
        review_kind = int(getattr(entry, "review_kind", 0))
        previous_kind = previous_kind_by_card.get(card_id)
        is_learning_start = (
            review_kind == ANKI_REVLOG_KIND_LEARNING and previous_kind != ANKI_REVLOG_KIND_LEARNING
        )

        if card_id in retained_cards:
            if is_learning_start:
                invalidating_learning_start_review_ids.add(review_id)
                dropped_count += 1
            else:
                retained_entries.append(entry)
        elif is_learning_start:
            retained_cards.add(card_id)
            retained_entries.append(entry)
            learning_start_review_ids.add(review_id)
        else:
            dropped_count += 1

        previous_kind_by_card[card_id] = review_kind

    retained_entries.sort(key=lambda entry: int(entry.id))
    return IncrementalLearningSequenceTrimResult(
        revlogs=retained_entries,
        learning_start_review_ids=frozenset(learning_start_review_ids),
        invalidating_learning_start_review_ids=frozenset(invalidating_learning_start_review_ids),
        original_processable_count=len(entries),
        dropped_processable_count=dropped_count,
    )


def revlogs_for_trimmed_learning_sequences(
    revlogs: Iterable[Any],
    retained_processable_revlogs: Iterable[Any],
) -> list[Any]:
    """Keep raw metadata rows inside cards' retained benchmark-trimmed spans."""

    first_retained_review_id_by_card: dict[int, int] = {}
    for entry in retained_processable_revlogs:
        card_id = int(entry.cid)
        review_id = int(entry.id)
        current = first_retained_review_id_by_card.get(card_id)
        if current is None or review_id < current:
            first_retained_review_id_by_card[card_id] = review_id

    return [
        entry
        for entry in revlogs
        if (
            int(entry.cid) in first_retained_review_id_by_card
            and int(entry.id) >= first_retained_review_id_by_card[int(entry.cid)]
        )
    ]


def revlogs_for_incremental_trimmed_learning_sequences(
    revlogs: Iterable[Any],
    retained_processable_revlogs: Iterable[Any],
    retained_card_ids: Iterable[int],
) -> list[Any]:
    """Keep raw tail metadata rows compatible with incremental trim decisions."""

    retained_cards = {int(card_id) for card_id in retained_card_ids}
    first_retained_review_id_by_card: dict[int, int] = {}
    for entry in retained_processable_revlogs:
        card_id = int(entry.cid)
        if card_id in retained_cards:
            continue
        review_id = int(entry.id)
        current = first_retained_review_id_by_card.get(card_id)
        if current is None or review_id < current:
            first_retained_review_id_by_card[card_id] = review_id

    rows: list[Any] = []
    for entry in revlogs:
        card_id = int(entry.cid)
        review_id = int(entry.id)
        if card_id in retained_cards:
            rows.append(entry)
            continue
        first_retained_review_id = first_retained_review_id_by_card.get(card_id)
        if first_retained_review_id is not None and review_id >= first_retained_review_id:
            rows.append(entry)
    return rows


def rwkv_processable_revlog(revlog: Any) -> bool:
    """Return whether a revlog should enter RWKV's process stream.

    This follows the original Anki Revlogs dataset builder's coarse eligibility:
    user-rated reviews are processable, except filtered-deck reviews with no
    scheduling effect. Other Anki revlog kinds are excluded because they are not
    recall attempts and the benchmark parquet data has no examples for those
    states. Raw revlog metadata can still keep those rows for FSRS evaluation
    parity; this helper only gates RWKV model input.
    """

    return _rwkv_processable_review_values(
        int(getattr(revlog, "button_chosen", 0)),
        int(getattr(revlog, "review_kind", 0)),
        int(getattr(revlog, "ease_factor", 0)),
    )


def _rwkv_processable_review_values(
    rating: int,
    review_kind: int,
    ease_factor: int,
) -> bool:
    if rating not in {1, 2, 3, 4}:
        return False
    if review_kind not in RWKV_PROCESSABLE_REVIEW_KINDS:
        return False
    return not (review_kind == ANKI_REVLOG_KIND_FILTERED and ease_factor == 0)


def rwkv_processable_revlog_where_sql(table: str) -> str:
    kinds = ", ".join(str(int(kind)) for kind in sorted(RWKV_PROCESSABLE_REVIEW_KINDS))
    return (
        f"{table}.ease BETWEEN 1 AND 4 "
        f"AND {table}.type IN ({kinds}) "
        f"AND NOT ({table}.type = {ANKI_REVLOG_KIND_FILTERED} AND {table}.factor = 0)"
    )


def rwkv_duration_millis(taken_millis: Any) -> float:
    """Clamp RWKV input duration to the benchmark dataset's documented range."""

    duration = float(taken_millis)
    if not math.isfinite(duration):
        raise ValueError("Review duration must be finite.")
    return min(float(RWKV_DURATION_MAX_MS), max(0.0, duration))


def review_row_build_context_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    day_offset_origin: int,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy | None = None,
) -> ReviewRowBuildContext:
    normalization_policy = (
        filtered_review_normalization_policy or FilteredReviewNormalizationPolicy.disabled()
    )
    previous_by_card: dict[int, tuple[int, int]] = {}
    previous_review_kind_by_card: dict[int, int] = {}
    positive_day_counts_by_card: dict[int, int] = {}
    prior_lapses_by_card: dict[int, int] = {}
    filtered_review_phase_by_card: dict[int, int] = {}

    for (
        review_id,
        card_id,
        day_offset,
        elapsed_days,
        rating,
        review_kind,
        state,
        has_review_kind,
    ) in _review_context_values(rows):
        if elapsed_days > 0:
            positive_day_counts_by_card[card_id] = positive_day_counts_by_card.get(card_id, 0) + 1
        if rating == 1 and elapsed_days > 0:
            prior_lapses_by_card[card_id] = prior_lapses_by_card.get(card_id, 0) + 1
        previous_by_card[card_id] = (day_offset, review_id)
        previous_review_kind_by_card[card_id] = review_kind
        is_filtered = (
            review_kind == ANKI_REVLOG_KIND_FILTERED
            if has_review_kind
            else state == ANKI_REVLOG_KIND_FILTERED + 1
        )
        _emitted_state, phase = resolve_rwkv_review_state(
            benchmark_state=(ANKI_REVLOG_KIND_FILTERED + 1 if is_filtered else state),
            is_filtered=is_filtered,
            elapsed_days=elapsed_days,
            rating=rating,
            previous_phase=filtered_review_phase_by_card.get(card_id),
            normalize_filtered=normalization_policy.applies_to(review_id),
        )
        filtered_review_phase_by_card[card_id] = phase

    return ReviewRowBuildContext(
        day_offset_origin=day_offset_origin,
        previous_by_card=previous_by_card,
        previous_review_kind_by_card=previous_review_kind_by_card,
        positive_day_counts_by_card=positive_day_counts_by_card,
        prior_lapses_by_card=prior_lapses_by_card,
        filtered_review_normalization_policy=normalization_policy,
        filtered_review_phase_by_card=filtered_review_phase_by_card,
    )


def _review_context_values(
    rows: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[int, int, int, float, int, int, int, bool]]:
    packed_values = getattr(rows, "iter_context_values", None)
    if callable(packed_values):
        return packed_values()

    def values() -> Iterator[tuple[int, int, int, float, int, int, int, bool]]:
        for row in rows:
            has_review_kind = "review_kind" in row
            state = int(row.get("state", 0))
            review_kind = int(row.get("review_kind", row.get("state", ANKI_REVLOG_KIND_LEARNING)))
            yield (
                int(row["review_id"]),
                int(row["card_id"]),
                int(row["day_offset"]),
                float(row["elapsed_days"]),
                int(row["rating"]),
                review_kind,
                state,
                has_review_kind,
            )

    return values()


def day_offset_origin_from_rows(rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        return 0
    return int(rows[0].get("raw_day_offset", rows[0]["day_offset"]))


def rebased_day_offset_origin(
    rows: Iterable[Mapping[str, Any]],
    *,
    current_origin: int | None,
    next_day_at: int,
    previous_next_day_at: int | None = None,
) -> int | None:
    """Return an origin that preserves normalized days under a new cutoff.

    Anki's raw day offsets are relative to the current ``day_cutoff``.  That
    cutoff advances during a long-running process, while the normalized day
    offsets already processed into RWKV state must not change.  Prefer the
    number of study days between the old and new cutoffs (rounded so DST's
    23/25-hour days still count as one); a historical row is the fallback when
    the prior cutoff is unavailable.  Neither path rewrites durable history.
    """

    if current_origin is not None and previous_next_day_at is not None:
        cutoff_delta = int(next_day_at) - int(previous_next_day_at)
        elapsed_study_days = int(round(cutoff_delta / SECONDS_PER_DAY))
        return int(current_origin) - elapsed_study_days

    for row in rows:
        try:
            review_timestamp = int(row["review_id"]) / 1000.0
            normalized_day = int(row["day_offset"])
        except (KeyError, TypeError, ValueError):
            continue
        return day_offset_for_timestamp(review_timestamp, int(next_day_at)) - normalized_day
    return current_origin


def build_last_review_map(rows: Iterable[Mapping[str, Any]]) -> dict[int, LastReviewInfo]:
    latest_by_card: dict[int, tuple[int, int, int]] = {}
    lapses: dict[int, int] = {}
    for review_id, card_id, day_offset, elapsed_days, rating, interval in last_review_scalar_values(
        rows
    ):
        if rating == 1 and elapsed_days != 0:
            lapses[card_id] = lapses.get(card_id, 0) + 1
        latest_by_card[card_id] = (review_id, day_offset, interval)
    return {
        card_id: LastReviewInfo(
            review_id=review_id,
            day_offset=day_offset,
            timestamp_seconds=review_id / 1000.0,
            interval=interval,
            lapse_count=lapses.get(card_id, 0),
        )
        for card_id, (review_id, day_offset, interval) in latest_by_card.items()
    }


def last_review_scalar_values(
    rows: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[int, int, int, float, int, int]]:
    packed_values = getattr(rows, "iter_last_review_values", None)
    if callable(packed_values):
        return packed_values()
    return (
        (
            int(row["review_id"]),
            int(row["card_id"]),
            int(row["day_offset"]),
            float(row["elapsed_days"]),
            int(row["rating"]),
            int(row.get("interval", 0)),
        )
        for row in rows
    )


def prediction_row_for_card(
    card: CardInfo,
    last_review: LastReviewInfo | None,
    *,
    target_timestamp_seconds: float,
    next_day_at: int,
    day_offset_origin: int = 0,
) -> dict[str, Any]:
    raw_target_day = day_offset_for_timestamp(target_timestamp_seconds, next_day_at)
    target_day = raw_target_day - day_offset_origin
    if last_review is None:
        elapsed_days = NEW_CARD_ELAPSED
        elapsed_seconds = NEW_CARD_ELAPSED
    else:
        elapsed_days = target_day - last_review.day_offset
        elapsed_seconds = target_timestamp_seconds - last_review.timestamp_seconds

    return prediction_input_row(
        review_id=int(target_timestamp_seconds * 1000),
        card_id=card.card_id,
        note_id=card.note_id,
        deck_id=card.deck_id,
        preset_id=card.preset_id,
        raw_day_offset=raw_target_day,
        day_offset=target_day,
        elapsed_days=elapsed_days,
        elapsed_seconds=elapsed_seconds,
    )


def prediction_rows_for_card_ids(
    card_ids: Iterable[int],
    review_data: ReviewData,
    *,
    target_timestamp_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for card_id in card_ids:
        card = review_data.cards.get(int(card_id))
        if card is None:
            card = CardInfo(int(card_id), None, None, None)
        rows.append(
            prediction_row_for_card(
                card,
                review_data.last_by_card.get(int(card_id)),
                target_timestamp_seconds=target_timestamp_seconds,
                next_day_at=review_data.next_day_at,
                day_offset_origin=review_data.day_offset_origin,
            )
        )
    return rows


def checkpoint_scope_cards_for_card_ids(
    card_ids: Iterable[int],
    review_data: ReviewData,
) -> list[dict[str, int | None]]:
    scope: list[dict[str, int | None]] = []
    for card_id_value in card_ids:
        card_id = int(card_id_value)
        card = review_data.cards.get(card_id)
        scope.append(
            {
                "card_id": card_id,
                "note_id": None if card is None else card.note_id,
                "deck_id": None if card is None else card.deck_id,
                "preset_id": None if card is None else card.preset_id,
            }
        )
    return scope


def day_offset_for_timestamp(timestamp_seconds: float, next_day_at: int) -> int:
    return math.floor((timestamp_seconds - next_day_at) / SECONDS_PER_DAY)


def days_elapsed_for_review(review_id: int, next_day_at: int) -> int:
    return max(0, (int(next_day_at) - int(review_id) // 1000) // SECONDS_PER_DAY)


def _card_metadata(
    card_id: int,
    cards: dict[int, CardInfo],
    decks: dict[int, DeckInfo],
) -> tuple[int | None, int | None, int | None]:
    card = cards.get(card_id)
    deck_id = card.deck_id if card else None
    deck = decks.get(deck_id) if deck_id is not None else None
    return (
        card.note_id if card else None,
        deck_id,
        deck.preset_id if deck else (card.preset_id if card else None),
    )


def _cards_from_dataset(dataset: Any) -> dict[int, CardInfo]:
    cards: dict[int, CardInfo] = {}
    deck_entries = _decks_from_dataset(dataset)
    for entry in dataset.cards:
        deck_id = int(entry.deck_id) if int(entry.deck_id) else None
        deck = deck_entries.get(deck_id) if deck_id is not None else None
        cards[int(entry.id)] = CardInfo(
            card_id=int(entry.id),
            note_id=int(entry.note_id) if int(entry.note_id) else None,
            deck_id=deck_id,
            preset_id=deck.preset_id if deck else None,
        )
    return cards


def _decks_from_dataset(dataset: Any) -> dict[int, DeckInfo]:
    decks: dict[int, DeckInfo] = {}
    for entry in dataset.decks:
        deck_id = int(entry.id)
        decks[deck_id] = DeckInfo(
            deck_id=deck_id,
            parent_id=int(entry.parent_id) if int(entry.parent_id) else None,
            preset_id=int(entry.preset_id) if int(entry.preset_id) else None,
        )
    return decks
