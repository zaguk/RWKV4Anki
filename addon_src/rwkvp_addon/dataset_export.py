from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from .checkpoint_progress import update_checkpoint_review_history
from .compact_review_data import (
    concatenate_process_review_rows,
    concatenate_revlog_rows,
    set_first_raw_day_offset,
)
from .prediction_cache import PredictionTailSnapshot
from .profile_store import ProfileStore
from .review_rows import (
    CardInfo,
    DeckInfo,
    IncrementalLearningSequenceTrimResult,
    LastReviewInfo,
    ReviewData,
    ReviewRowBuildContext,
    build_compact_review_history,
    build_review_rows_with_context,
    build_revlog_rows,
    last_review_scalar_values,
    load_research_dataset,
    rebased_day_offset_origin,
    review_data_from_dataset,
    review_row_build_context_from_rows,
    revlogs_for_incremental_trimmed_learning_sequences,
    trim_incremental_benchmark_learning_sequences,
)
from .review_tail_context import (
    CheckpointTailRows,
    CollectionRevision,
    load_review_tail_context,
    read_collection_revision,
)
from .review_type_normalization import (
    FilteredReviewNormalizationPolicy,
    filtered_review_normalization_policy_for_store,
)


@dataclass(frozen=True)
class DirectReviewDataStats:
    revlog_count: int
    processable_review_count: int
    card_count: int
    deck_count: int


class _DirectRevlogEntry(NamedTuple):
    id: int
    cid: int
    button_chosen: int
    interval: int
    last_interval: int
    ease_factor: int
    taken_millis: int
    review_kind: int


@dataclass(frozen=True)
class ReviewDataLoad:
    review_data: ReviewData
    exported: bool
    latest_review_id: int | None
    incremental: bool = False
    collection_revision: CollectionRevision | None = None
    persisted_tail_context: bool = False


@dataclass(frozen=True)
class IncrementalCardReviewDataLoad:
    review_data: ReviewData
    review_rows: Sequence[Mapping[str, Any]]
    revlog_rows: Sequence[Mapping[str, Any]]
    context: ReviewRowBuildContext


@dataclass(frozen=True)
class CheckpointReadiness:
    review_data: ReviewData
    exported: bool
    latest_review_id: int | None
    checkpoint_result: object | None
    durable_processed_review_count: int | None
    transient_prediction_tail: PredictionTailSnapshot | None = None


class LearningSequenceTrimInvalidationError(RuntimeError):
    """Raised when an incremental tail would require rewriting prior card history."""


def export_review_data(
    col,
    store: ProfileStore,
    *,
    exclude_deleted_card_revlogs: bool = True,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy | None = None,
) -> ReviewData:
    normalization_policy = filtered_review_normalization_policy or _policy_for_store_or_disabled(
        store
    )
    store.ensure()
    target = store.cache_dir / f"research-dataset-{int(time.time())}.pb"
    col.export_dataset_for_research(str(target))
    dataset = load_research_dataset(target)
    return review_data_from_dataset(
        dataset,
        exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
        trim_learning_sequences=True,
        filtered_review_normalization_policy=normalization_policy,
    )


def load_review_data_directly(
    col,
    *,
    exclude_deleted_card_revlogs: bool = True,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy | None = None,
) -> ReviewData:
    """Read Anki review data directly from collection APIs/DB without protobuf export.

    This intentionally diverges from Anki's research-dataset export when
    exclude_deleted_card_revlogs is enabled, because the add-on does not need
    deleted-card history for user checkpoint state. Otherwise it mirrors the
    research export shape:
    - revlogs use the same rating/filter shape as `get_revlog_entries_for_export_dataset()`
    - cards use original deck ids when present, matching `get_card_entry.sql`
    - decks include normal decks with their preset id and immediate parent id
    """

    normalization_policy = (
        filtered_review_normalization_policy or FilteredReviewNormalizationPolicy.disabled()
    )
    next_day_at = int(col.sched.day_cutoff)
    cards = _cards_from_db(col)
    decks = _decks_from_collection(col)
    cards = _cards_with_preset_ids(cards, decks)
    revlog_entries = _revlogs_from_db(
        col,
        exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
    )
    history = build_compact_review_history(
        revlog_entries,
        cards,
        decks,
        next_day_at,
        filtered_review_normalization_policy=normalization_policy,
        input_is_chronological=True,
    )
    return ReviewData(
        rows=history.rows,
        revlogs=history.revlogs,
        cards=cards,
        decks=decks,
        last_by_card=history.last_by_card,
        next_day_at=next_day_at,
        day_offset_origin=history.day_offset_origin,
        filtered_review_normalization_policy=normalization_policy,
    )


def load_incremental_review_data_directly(
    col,
    base: ReviewData,
    *,
    since_revlog_id: int,
    exclude_deleted_card_revlogs: bool = True,
) -> ReviewData:
    next_day_at = int(col.sched.day_cutoff)
    cards = _cards_from_db(col)
    decks = _decks_from_collection(col)
    cards = _cards_with_preset_ids(cards, decks)
    tail_revlog_entries = _revlogs_from_db(
        col,
        after_review_id=since_revlog_id,
        exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
    )
    context = _incremental_context_from_base(base, next_day_at=next_day_at)
    trim = _incremental_learning_sequence_trim_or_raise(
        tail_revlog_entries,
        context,
    )
    tail_revlog_entries = revlogs_for_incremental_trimmed_learning_sequences(
        tail_revlog_entries,
        trim.revlogs,
        context.previous_by_card,
    )
    tail_revlogs = build_revlog_rows(
        tail_revlog_entries,
        cards,
        decks,
        next_day_at,
        compact=True,
    )
    tail_rows, updated_context = build_review_rows_with_context(
        trim.revlogs,
        cards,
        decks,
        next_day_at,
        context=context,
        learning_start_review_ids=trim.learning_start_review_ids,
        input_is_chronological=True,
        input_is_processable=True,
        compact=True,
    )
    day_offset_origin = int(updated_context.day_offset_origin or 0)
    combined_rows = concatenate_process_review_rows(base.rows, tail_rows)
    _store_current_origin_on_first_row(combined_rows, day_offset_origin)

    return ReviewData(
        rows=combined_rows,
        revlogs=concatenate_revlog_rows(base.revlogs, tail_revlogs),
        cards=cards,
        decks=decks,
        last_by_card=extend_last_review_map(base.last_by_card, tail_rows),
        next_day_at=next_day_at,
        day_offset_origin=day_offset_origin,
        filtered_review_normalization_policy=(base.filtered_review_normalization_policy),
    )


def load_incremental_review_data_for_card_directly(
    col,
    base: ReviewData,
    *,
    card_id: int,
    context: ReviewRowBuildContext | None = None,
    exclude_deleted_card_revlogs: bool = True,
) -> IncrementalCardReviewDataLoad:
    """Load newly added revlog rows for one card using checkpoint row semantics."""

    next_day_at = int(col.sched.day_cutoff)
    cards = _cards_from_db(col)
    decks = _decks_from_collection(col)
    cards = _cards_with_preset_ids(cards, decks)
    tail_revlog_entries = _revlogs_from_db(
        col,
        after_review_id=_latest_revlog_id_for_card(base, int(card_id)),
        card_id=int(card_id),
        exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
    )
    resolved_context = context
    if resolved_context is None and base.rows:
        resolved_context = _incremental_context_from_base(
            base,
            next_day_at=next_day_at,
        )
    elif resolved_context is not None:
        resolved_context = ReviewRowBuildContext(
            day_offset_origin=rebased_day_offset_origin(
                (
                    {
                        "review_id": review_id,
                        "day_offset": day_offset,
                    }
                    for day_offset, review_id in resolved_context.previous_by_card.values()
                ),
                current_origin=resolved_context.day_offset_origin,
                next_day_at=next_day_at,
                previous_next_day_at=base.next_day_at,
            ),
            previous_by_card=resolved_context.previous_by_card,
            previous_review_kind_by_card=resolved_context.previous_review_kind_by_card,
            positive_day_counts_by_card=resolved_context.positive_day_counts_by_card,
            prior_lapses_by_card=resolved_context.prior_lapses_by_card,
            filtered_review_normalization_policy=(
                resolved_context.filtered_review_normalization_policy
            ),
            filtered_review_phase_by_card=(resolved_context.filtered_review_phase_by_card),
        )
    if resolved_context is None:
        resolved_context = ReviewRowBuildContext(
            day_offset_origin=None,
            previous_by_card={},
            previous_review_kind_by_card={},
            positive_day_counts_by_card={},
            prior_lapses_by_card={},
            filtered_review_normalization_policy=(base.filtered_review_normalization_policy),
            filtered_review_phase_by_card={},
        )
    trim = _incremental_learning_sequence_trim_or_raise(
        tail_revlog_entries,
        resolved_context,
    )
    tail_revlog_entries = revlogs_for_incremental_trimmed_learning_sequences(
        tail_revlog_entries,
        trim.revlogs,
        resolved_context.previous_by_card,
    )
    tail_revlogs = build_revlog_rows(
        tail_revlog_entries,
        cards,
        decks,
        next_day_at,
        compact=True,
    )
    tail_rows, updated_context = build_review_rows_with_context(
        trim.revlogs,
        cards,
        decks,
        next_day_at,
        context=resolved_context,
        learning_start_review_ids=trim.learning_start_review_ids,
        input_is_chronological=True,
        input_is_processable=True,
        compact=True,
    )
    day_offset_origin = int(updated_context.day_offset_origin or 0)
    combined_rows = concatenate_process_review_rows(base.rows, tail_rows)
    _store_current_origin_on_first_row(combined_rows, day_offset_origin)
    review_data = ReviewData(
        rows=combined_rows,
        revlogs=concatenate_revlog_rows(base.revlogs, tail_revlogs),
        cards=cards,
        decks=decks,
        last_by_card=extend_last_review_map(base.last_by_card, tail_rows),
        next_day_at=next_day_at,
        day_offset_origin=day_offset_origin,
        filtered_review_normalization_policy=(base.filtered_review_normalization_policy),
    )
    return IncrementalCardReviewDataLoad(
        review_data=review_data,
        review_rows=tail_rows,
        revlog_rows=tail_revlogs,
        context=updated_context,
    )


def direct_review_data_stats(review_data: ReviewData) -> DirectReviewDataStats:
    return DirectReviewDataStats(
        revlog_count=len(review_data.revlogs),
        processable_review_count=len(review_data.rows),
        card_count=len(review_data.cards),
        deck_count=len(review_data.decks),
    )


def load_review_data_for_checkpoint(
    col,
    store: ProfileStore,
    manager,
    progress,
    *,
    force_export: bool = False,
    allow_incremental: bool = False,
    allow_persisted_tail_context: bool = False,
) -> ReviewDataLoad:
    exclude_deleted_card_revlogs = bool(getattr(manager, "exclude_deleted_card_revlogs", True))
    normalization_policy = getattr(manager, "filtered_review_normalization_policy", None)
    if normalization_policy is None:
        normalization_policy = _policy_for_store_or_disabled(store)
    collection_revision = read_collection_revision(col)
    latest_review_id = (
        collection_revision.latest_review_id
        if collection_revision is not None
        else latest_collection_review_id(col)
    )
    if not force_export:
        cached_for_revision = getattr(manager, "cached_review_data_for_revision", None)
        cached = (
            cached_for_revision(collection_revision)
            if collection_revision is not None and callable(cached_for_revision)
            else manager.cached_review_data_if_current(latest_review_id)
        )
        if cached is not None:
            update_checkpoint_review_history(progress, "Review data is already loaded")
            return ReviewDataLoad(
                cached,
                exported=False,
                latest_review_id=latest_review_id,
                collection_revision=collection_revision,
            )
        if allow_incremental:
            source_for_revision = getattr(
                manager,
                "incremental_review_data_source_for_revision",
                None,
            )
            source = (
                source_for_revision(collection_revision)
                if collection_revision is not None and callable(source_for_revision)
                else manager.incremental_review_data_source(latest_review_id)
            )
            if source is not None:
                base, since_revlog_id = source
                update_checkpoint_review_history(progress, "Reading new reviews")
                try:
                    review_data = load_incremental_review_data_directly(
                        col,
                        base,
                        since_revlog_id=since_revlog_id,
                        exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
                    )
                except Exception:
                    update_checkpoint_review_history(progress, "Preparing review history")
                    review_data = load_review_data_from_collection(
                        col,
                        store,
                        exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
                        filtered_review_normalization_policy=normalization_policy,
                    )
                    update_checkpoint_review_history(progress, "Review history ready")
                    return ReviewDataLoad(
                        review_data,
                        exported=True,
                        latest_review_id=latest_review_id,
                        collection_revision=_stable_collection_revision(
                            col,
                            collection_revision,
                        ),
                    )
                update_checkpoint_review_history(progress, "New reviews ready")
                return ReviewDataLoad(
                    review_data,
                    exported=True,
                    latest_review_id=latest_review_id,
                    incremental=True,
                    collection_revision=_stable_collection_revision(
                        col,
                        collection_revision,
                    ),
                )
        if allow_persisted_tail_context and collection_revision is not None:
            persisted = _load_review_data_from_persisted_tail_context(
                col,
                store,
                manager,
                collection_revision=collection_revision,
                exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
                filtered_review_normalization_policy=normalization_policy,
            )
            if persisted is not None:
                update_checkpoint_review_history(progress, "New reviews ready")
                return persisted

    update_checkpoint_review_history(progress, "Preparing review history")
    review_data = load_review_data_from_collection(
        col,
        store,
        exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
        filtered_review_normalization_policy=normalization_policy,
    )
    update_checkpoint_review_history(progress, "Review history ready")
    stable_revision = _stable_collection_revision(col, collection_revision)
    return ReviewDataLoad(
        review_data,
        exported=True,
        latest_review_id=(
            stable_revision.latest_review_id if stable_revision is not None else latest_review_id
        ),
        collection_revision=stable_revision,
    )


def load_review_data_from_collection(
    col,
    store: ProfileStore,
    *,
    exclude_deleted_card_revlogs: bool = True,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy | None = None,
) -> ReviewData:
    normalization_policy = filtered_review_normalization_policy or _policy_for_store_or_disabled(
        store
    )
    try:
        return load_review_data_directly(
            col,
            exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
            filtered_review_normalization_policy=normalization_policy,
        )
    except MemoryError:
        raise
    except Exception:
        return export_review_data(
            col,
            store,
            exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
            filtered_review_normalization_policy=normalization_policy,
        )


def _stable_collection_revision(
    col,
    before: CollectionRevision | None,
) -> CollectionRevision | None:
    if before is None:
        return None
    after = read_collection_revision(col)
    return after if after == before else None


def _load_review_data_from_persisted_tail_context(
    col,
    store: ProfileStore,
    manager,
    *,
    collection_revision: CollectionRevision,
    exclude_deleted_card_revlogs: bool,
    filtered_review_normalization_policy: FilteredReviewNormalizationPolicy,
) -> ReviewDataLoad | None:
    """Build only the rows newer than a verified durable checkpoint prefix."""

    fingerprint_reader = getattr(manager, "checkpoint_history_fingerprint", None)
    accept_tail = getattr(manager, "accept_persisted_review_tail", None)
    if not callable(fingerprint_reader) or not callable(accept_tail):
        return None
    fingerprint = fingerprint_reader()
    if fingerprint is None:
        return None
    persisted = load_review_tail_context(
        store.review_tail_context_path,
        model_id=str(getattr(manager, "model_id", "")),
        exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
        filtered_review_normalization_policy=filtered_review_normalization_policy,
        checkpoint_history_fingerprint=fingerprint,
        collection_revision=collection_revision,
    )
    if persisted is None:
        return None

    try:
        next_day_at = int(col.sched.day_cutoff)
        cards = _cards_from_db(col)
        decks = _decks_from_collection(col)
        cards = _cards_with_preset_ids(cards, decks)
        raw_tail_entries = _revlogs_from_db(
            col,
            after_review_id=persisted.last_review_id,
            exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
        )
        context = ReviewRowBuildContext(
            day_offset_origin=rebased_day_offset_origin(
                (),
                current_origin=persisted.context.day_offset_origin,
                next_day_at=next_day_at,
                previous_next_day_at=persisted.next_day_at,
            ),
            previous_by_card=dict(persisted.context.previous_by_card),
            previous_review_kind_by_card=dict(persisted.context.previous_review_kind_by_card),
            positive_day_counts_by_card=dict(persisted.context.positive_day_counts_by_card),
            prior_lapses_by_card=dict(persisted.context.prior_lapses_by_card),
            filtered_review_normalization_policy=filtered_review_normalization_policy,
            filtered_review_phase_by_card=dict(persisted.context.filtered_review_phase_by_card),
        )
        trim = _incremental_learning_sequence_trim_or_raise(raw_tail_entries, context)
        retained_tail_entries = revlogs_for_incremental_trimmed_learning_sequences(
            raw_tail_entries,
            trim.revlogs,
            context.previous_by_card,
        )
        tail_revlogs = build_revlog_rows(
            retained_tail_entries,
            cards,
            decks,
            next_day_at,
            compact=True,
        )
        tail_rows, updated_context = build_review_rows_with_context(
            trim.revlogs,
            cards,
            decks,
            next_day_at,
            context=context,
            learning_start_review_ids=trim.learning_start_review_ids,
            input_is_chronological=True,
            input_is_processable=True,
            compact=True,
        )
        latest_revlog_id_by_card = dict(persisted.latest_revlog_id_by_card)
        for entry in raw_tail_entries:
            latest_revlog_id_by_card[int(entry.cid)] = int(entry.id)
        rows = CheckpointTailRows(
            durable_count=persisted.processed_review_count,
            first_review_id=persisted.first_review_id,
            first_card_id=persisted.first_card_id,
            last_review_id=persisted.last_review_id,
            last_card_id=persisted.last_card_id,
            day_offset_origin=int(updated_context.day_offset_origin or 0),
            tail=tail_rows,
        )
        if read_collection_revision(col) != collection_revision:
            return None
        if not bool(accept_tail(rows, persisted.checkpoint_history_fingerprint)):
            return None
    except MemoryError:
        raise
    except Exception:
        # The persisted object is optional acceleration data. Any race,
        # unsupported fake, malformed tail, or learning-trim invalidation must
        # return to the complete-history/normal-consistency path.
        return None

    review_data = ReviewData(
        rows=rows,
        revlogs=tail_revlogs,
        cards=cards,
        decks=decks,
        last_by_card=extend_last_review_map(persisted.last_by_card, tail_rows),
        next_day_at=next_day_at,
        day_offset_origin=int(updated_context.day_offset_origin or 0),
        filtered_review_normalization_policy=filtered_review_normalization_policy,
        row_build_context=updated_context,
        latest_revlog_id_by_card=latest_revlog_id_by_card,
        complete_history=False,
    )
    return ReviewDataLoad(
        review_data=review_data,
        exported=True,
        latest_review_id=collection_revision.latest_review_id,
        incremental=True,
        collection_revision=collection_revision,
        persisted_tail_context=True,
    )


def _incremental_learning_sequence_trim_or_raise(
    revlogs: list[_DirectRevlogEntry],
    context: ReviewRowBuildContext,
) -> IncrementalLearningSequenceTrimResult:
    trim = trim_incremental_benchmark_learning_sequences(revlogs, context)
    if trim.invalidating_learning_start_review_ids:
        review_ids = ", ".join(
            str(review_id) for review_id in sorted(trim.invalidating_learning_start_review_ids)
        )
        raise LearningSequenceTrimInvalidationError(
            "A later learning-start sequence would require rebuilding existing "
            f"RWKV card state. Review ids: {review_ids}."
        )
    return trim


def _incremental_context_from_base(
    base: ReviewData,
    *,
    next_day_at: int,
) -> ReviewRowBuildContext:
    if base.rows:
        return review_row_build_context_from_rows(
            base.rows,
            day_offset_origin=int(
                rebased_day_offset_origin(
                    base.rows,
                    current_origin=base.day_offset_origin,
                    next_day_at=next_day_at,
                    previous_next_day_at=base.next_day_at,
                )
                or 0
            ),
            filtered_review_normalization_policy=(base.filtered_review_normalization_policy),
        )
    return ReviewRowBuildContext(
        day_offset_origin=None,
        previous_by_card={},
        previous_review_kind_by_card={},
        positive_day_counts_by_card={},
        prior_lapses_by_card={},
        filtered_review_normalization_policy=(base.filtered_review_normalization_policy),
        filtered_review_phase_by_card={},
    )


def _policy_for_store_or_disabled(store: object) -> FilteredReviewNormalizationPolicy:
    if not isinstance(store, ProfileStore):
        return FilteredReviewNormalizationPolicy.disabled()
    return filtered_review_normalization_policy_for_store(store)


def _store_current_origin_on_first_row(
    rows: Sequence[Mapping[str, Any]],
    day_offset_origin: int,
) -> None:
    """Keep manifest origin derivation current without rewriting history rows.

    ``raw_day_offset`` is add-on metadata rather than an RWKV input or history
    fingerprint field.  Copying only the first row avoids mutating the cached
    base and lets durable checkpoint writes recover the rebased origin without
    copying every historical review dictionary.
    """

    if not rows:
        return
    if set_first_raw_day_offset(rows, day_offset_origin):
        return
    if not isinstance(rows, list):
        raise TypeError("Review rows do not support a day-offset-origin update.")
    rows[0] = {**rows[0], "raw_day_offset": int(day_offset_origin)}


def _remember_review_data_from_load(manager, review_load: ReviewDataLoad) -> None:
    manager.remember_review_data(
        review_load.review_data,
        latest_collection_review_id=review_load.latest_review_id,
    )
    revision = review_load.collection_revision
    if revision is None:
        return
    remember_revision = getattr(manager, "remember_review_data_collection_revision", None)
    if callable(remember_revision):
        remember_revision(review_load.review_data, revision)
    persist_context = getattr(manager, "persist_review_tail_context", None)
    if callable(persist_context):
        persist_context(review_load.review_data, revision)


def ensure_checkpoint_ready_from_load(
    manager,
    review_load: ReviewDataLoad,
    progress,
    *,
    force_save: bool = False,
    wait_for_save: bool = False,
    capture_prediction_tail: bool = False,
) -> CheckpointReadiness:
    result = None
    transient_prediction_tail = None
    durable_processed_review_count = None
    if wait_for_save and manager.save_in_progress:
        progress.update(0, 1, "Waiting for checkpoint/cache write")
        manager.wait_for_pending_save()
    durable_count_before = manager.durable_processed_review_count()
    needs_runtime = (
        review_load.exported
        or force_save
        or (capture_prediction_tail and durable_count_before != len(review_load.review_data.rows))
    )
    lease = None
    try:
        if needs_runtime:
            lease, result = manager.open_scoped_runtime(
                review_load.review_data.rows,
                (),
                progress,
                force_save=force_save,
                check_consistency=not review_load.incremental,
            )
            _remember_review_data_from_load(manager, review_load)
        if wait_for_save and manager.save_in_progress:
            progress.update(0, 1, "Waiting for checkpoint/cache write")
            manager.wait_for_pending_save()
        if capture_prediction_tail:
            durable_count = manager.durable_processed_review_count()
            if lease is not None:
                transient_prediction_tail = lease.evaluation_prediction_tail()
            else:
                transient_prediction_tail = PredictionTailSnapshot.empty(
                    0 if durable_count is None else durable_count
                )
        durable_processed_review_count = manager.durable_processed_review_count()
    finally:
        if lease is not None:
            lease.close()
    return CheckpointReadiness(
        review_data=review_load.review_data,
        exported=review_load.exported,
        latest_review_id=review_load.latest_review_id,
        checkpoint_result=result,
        durable_processed_review_count=durable_processed_review_count,
        transient_prediction_tail=transient_prediction_tail,
    )


def open_checkpoint_runtime_from_load(
    manager,
    review_load: ReviewDataLoad,
    progress,
    *,
    scope_cards,
    force_save: bool = False,
    wait_for_save: bool = False,
) -> tuple[CheckpointReadiness, object]:
    lease, result = manager.open_scoped_runtime(
        review_load.review_data.rows,
        scope_cards,
        progress,
        force_save=force_save,
        check_consistency=not review_load.incremental,
    )
    try:
        _remember_review_data_from_load(manager, review_load)
        if wait_for_save and manager.save_in_progress:
            progress.update(0, 1, "Waiting for checkpoint/cache write")
            manager.wait_for_pending_save()
        readiness = CheckpointReadiness(
            review_data=review_load.review_data,
            exported=review_load.exported,
            latest_review_id=review_load.latest_review_id,
            checkpoint_result=result,
            durable_processed_review_count=manager.durable_processed_review_count(),
        )
    except BaseException:
        lease.close()
        raise
    return readiness, lease


def initialize_or_update_checkpoint_from_load(
    manager,
    review_load: ReviewDataLoad,
    progress,
    *,
    rebuild: bool,
    force_save: bool = False,
) -> CheckpointReadiness:
    if rebuild:
        progress.update(0, 1, "Preparing checkpoint rebuild")
        result = manager.initialize_or_rebuild(review_load.review_data.rows, progress)
    else:
        progress.update(0, 1, "Checking checkpoint and cached evaluation data")
        lease = None
        try:
            lease, result = manager.open_scoped_runtime(
                review_load.review_data.rows,
                (),
                progress,
                force_save=force_save,
            )
        finally:
            if lease is not None:
                lease.close()
    _remember_review_data_from_load(manager, review_load)
    if rebuild:
        manager.release_runtime(preserve_review_data=True)
    return CheckpointReadiness(
        review_data=review_load.review_data,
        exported=review_load.exported,
        latest_review_id=review_load.latest_review_id,
        checkpoint_result=result,
        durable_processed_review_count=manager.durable_processed_review_count(),
    )


def latest_collection_review_id(col) -> int | None:
    try:
        # This read-only fast path avoids exporting the full research dataset
        # when the in-memory checkpoint has already seen the latest revlog row.
        # If the query is unavailable, callers conservatively fall back to export.
        value = col.db.scalar("select max(id) from revlog")
    except Exception:
        return None
    return None if value is None else int(value)


def collection_review_id_bounds(col) -> tuple[int | None, int | None]:
    try:
        # A single read-only aggregate keeps Evaluate's date controls cheap to
        # initialize without exporting or parsing the research dataset.
        row = col.db.first("select min(id), max(id) from revlog")
    except Exception:
        return None, None
    if row is None:
        return None, None
    first_review_id, last_review_id = row
    return (
        None if first_review_id is None else int(first_review_id),
        None if last_review_id is None else int(last_review_id),
    )


def latest_collection_review_timestamp_seconds(col) -> float | None:
    review_id = latest_collection_review_id(col)
    return None if review_id is None else review_id / 1000.0


def extend_last_review_map(
    last_by_card: dict[int, LastReviewInfo],
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, LastReviewInfo]:
    last = dict(last_by_card)
    lapses = {card_id: info.lapse_count for card_id, info in last.items()}
    latest_by_card: dict[int, tuple[int, int, int]] = {}
    for review_id, card_id, day_offset, elapsed_days, rating, interval in last_review_scalar_values(
        rows
    ):
        if rating == 1 and elapsed_days != 0:
            lapses[card_id] = lapses.get(card_id, 0) + 1
        latest_by_card[card_id] = (review_id, day_offset, interval)
    for card_id, (review_id, day_offset, interval) in latest_by_card.items():
        last[card_id] = LastReviewInfo(
            review_id=review_id,
            day_offset=day_offset,
            timestamp_seconds=review_id / 1000.0,
            interval=interval,
            lapse_count=lapses.get(card_id, 0),
        )
    return last


def _cards_from_db(col) -> dict[int, CardInfo]:
    rows = col.db.all(
        """
        select id,
               nid,
               case when odid = 0 then did else odid end as did
        from cards
        """
    )
    return {
        int(card_id): CardInfo(
            card_id=int(card_id),
            note_id=int(note_id) if int(note_id) else None,
            deck_id=int(deck_id) if int(deck_id) else None,
            preset_id=None,
        )
        for card_id, note_id, deck_id in rows
    }


def load_card_info_for_card_directly(
    col,
    card_id: int,
    decks: dict[int, DeckInfo],
) -> CardInfo | None:
    row = col.db.first(
        """
        select id,
               nid,
               case when odid = 0 then did else odid end as did
        from cards
        where id = ?
        """,
        int(card_id),
    )
    if row is None:
        return None
    resolved_card_id, note_id, deck_id = row
    normalized_deck_id = int(deck_id) if int(deck_id) else None
    deck = decks.get(normalized_deck_id) if normalized_deck_id is not None else None
    return CardInfo(
        card_id=int(resolved_card_id),
        note_id=int(note_id) if int(note_id) else None,
        deck_id=normalized_deck_id,
        preset_id=deck.preset_id if deck else None,
    )


def _cards_with_preset_ids(
    cards: dict[int, CardInfo],
    decks: dict[int, DeckInfo],
) -> dict[int, CardInfo]:
    return {
        card_id: CardInfo(
            card_id=card.card_id,
            note_id=card.note_id,
            deck_id=card.deck_id,
            preset_id=(
                decks[card.deck_id].preset_id
                if card.deck_id is not None and card.deck_id in decks
                else None
            ),
        )
        for card_id, card in cards.items()
    }


def _revlogs_from_db(
    col,
    *,
    after_review_id: int | None = None,
    card_id: int | None = None,
    exclude_deleted_card_revlogs: bool = True,
) -> list[_DirectRevlogEntry]:
    from_clause = "from revlog r"
    if exclude_deleted_card_revlogs:
        from_clause += """
        join cards c on c.id = r.cid
        join notes n on n.id = c.nid
        """
    after_clause = "" if after_review_id is None else "and r.id > ?"
    card_clause = "" if card_id is None else "and r.cid = ?"
    args = []
    if after_review_id is not None:
        args.append(int(after_review_id))
    if card_id is not None:
        args.append(int(card_id))
    rows = col.db.all(
        f"""
        select r.id, r.cid, r.ease, r.ivl, r.lastIvl, r.factor, r.time, r.type
        {from_clause}
        where ((r.ease between 1 and 4) or (r.ease = 0 and r.factor = 0))
        {after_clause}
        {card_clause}
        order by r.id
        """,
        *args,
    )
    return [
        _DirectRevlogEntry(
            int(review_id),
            int(card_id),
            int(ease),
            int(interval),
            int(last_interval),
            int(ease_factor),
            int(taken_millis),
            int(review_kind),
        )
        for (
            review_id,
            card_id,
            ease,
            interval,
            last_interval,
            ease_factor,
            taken_millis,
            review_kind,
        ) in rows
    ]


def load_new_revlog_entries_for_card_directly(
    col,
    *,
    card_id: int,
    after_review_id: int | None,
    exclude_deleted_card_revlogs: bool = True,
) -> list[_DirectRevlogEntry]:
    return _revlogs_from_db(
        col,
        after_review_id=after_review_id,
        card_id=int(card_id),
        exclude_deleted_card_revlogs=exclude_deleted_card_revlogs,
    )


def _latest_revlog_id_for_card(review_data: ReviewData, card_id: int) -> int | None:
    review_ids = [
        int(row["review_id"]) for row in review_data.revlogs if int(row["card_id"]) == int(card_id)
    ]
    return max(review_ids) if review_ids else None


def _decks_from_collection(col) -> dict[int, DeckInfo]:
    legacy_decks = list(col.decks.all())
    name_to_id = {str(deck["name"]): int(deck["id"]) for deck in legacy_decks}
    decks: dict[int, DeckInfo] = {}
    for deck in legacy_decks:
        if int(deck.get("dyn", 0)):
            continue
        deck_id = int(deck["id"])
        preset_id = _legacy_deck_preset_id(deck)
        if preset_id is None:
            continue
        parent_name = _immediate_parent_name(str(deck["name"]))
        parent_id = name_to_id.get(parent_name) if parent_name else None
        decks[deck_id] = DeckInfo(
            deck_id=deck_id,
            parent_id=int(parent_id) if parent_id else None,
            preset_id=preset_id,
        )
    return decks


def _legacy_deck_preset_id(deck: dict[str, Any]) -> int | None:
    preset_id = deck.get("conf")
    if preset_id is None:
        return None
    preset_id = int(preset_id)
    return preset_id if preset_id else None


def _immediate_parent_name(deck_name: str) -> str | None:
    if "\x1f" in deck_name:
        return deck_name.rsplit("\x1f", 1)[0]
    if "::" in deck_name:
        return deck_name.rsplit("::", 1)[0]
    return None
