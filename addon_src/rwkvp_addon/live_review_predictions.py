from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from .dataset_export import (
    load_card_info_for_card_directly,
    load_new_revlog_entries_for_card_directly,
)
from .filtered_deck_sort import (
    FilteredDeckOrder,
    filtered_deck_order_from_index,
    filtered_deck_tiebreaker,
)
from .live_review_engine import (
    LiveNativePredictionValue,
    LivePredictionResult,
    LivePredictionStatus,
    LivePredictionToken,
    LiveReviewCandidate,
    LiveReviewEngine,
    _candidate_static_values,
)
from .review_load_policy import minimum_retention_extra_required
from .review_rows import (
    NEW_CARD_ELAPSED,
    LastReviewInfo,
    ReviewData,
    ReviewRowBuildContext,
    append_review_row_to_context,
    day_offset_for_timestamp,
    rebased_day_offset_origin,
    review_row_build_context_from_rows,
)
from .vendor_bootstrap import require_rwkv_live_candidate_seed


@dataclass(frozen=True)
class LiveReviewNativePredictionJob:
    token: LivePredictionToken
    target_timestamp_seconds: float
    target_day_offset: float
    hot_card_ids: tuple[int, ...]
    hot_seeds: tuple[Any, ...]
    include_hot_card_ids: tuple[int, ...]
    exclude_card_ids: tuple[int, ...]
    exclude_refresh_card_ids: tuple[int, ...] = ()
    select_limit: int = 2


@dataclass(frozen=True)
class LiveReviewNativePredictionJobResult:
    token: LivePredictionToken
    selected_values: tuple[LiveNativePredictionValue, ...]
    hot_values: tuple[LiveNativePredictionValue, ...]
    included_hot_card_ids: tuple[int, ...]
    refreshed_count: int
    eligible_count: int
    target_timestamp_seconds: float


@dataclass(frozen=True)
class LiveStaleRecheckResult:
    batches: int
    checked_count: int
    eligible_count: int
    remaining_count: int
    eligible_card_ids: tuple[int, ...] = ()

    @property
    def found_eligible(self) -> bool:
        return bool(self.eligible_card_ids)


@dataclass(frozen=True)
class LiveNativeCandidateReconciliation:
    candidates: tuple[LiveReviewCandidate, ...]
    result: Any | None
    dropped_out_of_scope_card_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class LiveNativeReconciliationResult:
    """A widened selection paired with the reconciliation's full work count."""

    generation: int
    refreshed_count: int
    eligible_count: int
    active_count: int
    selected: tuple[Any, ...]
    next_retention_extra: float | None


@dataclass(frozen=True)
class _ReviewRollbackSnapshot:
    review_id: int
    card_id: int
    previous_day_offset_origin: int | None
    previous_latest_revlog_id: int | None
    had_last_review: bool
    previous_last_review: LastReviewInfo | None
    had_previous_by_card: bool
    previous_by_card: tuple[int, int] | None
    had_previous_review_kind: bool
    previous_review_kind: int | None
    had_positive_day_count: bool
    previous_positive_day_count: int | None
    had_prior_lapses: bool
    previous_prior_lapses: int | None
    had_filtered_review_phase: bool
    previous_filtered_review_phase: int | None


class LiveReviewHistoryState:
    """Prediction history used during live review.

    Startup can afford to derive this state from the full review history. The
    answer path cannot, so this object updates only per-card dictionaries after
    reading the answered card's new revlog rows.
    """

    def __init__(self, review_data: ReviewData) -> None:
        self._base_rows = review_data.rows
        self._base_revlogs = review_data.revlogs
        self._complete_history = bool(review_data.complete_history)
        self.cards = dict(review_data.cards)
        self.decks = dict(review_data.decks)
        self.last_by_card = dict(review_data.last_by_card)
        self.next_day_at = int(review_data.next_day_at)
        self.context = review_data.row_build_context or (
            review_row_build_context_from_rows(
                review_data.rows,
                day_offset_origin=review_data.day_offset_origin,
                filtered_review_normalization_policy=(
                    review_data.filtered_review_normalization_policy
                ),
            )
            if review_data.rows
            else ReviewRowBuildContext(
                # ReviewData carries the checkpoint-relative origin explicitly.
                # Preserve it even when a partial/synthetic scope has no history
                # rows of its own; otherwise elapsed_days is calculated in raw
                # Anki-day coordinates and can become invalid for the native
                # live-session API.
                day_offset_origin=(
                    int(review_data.day_offset_origin) if review_data.last_by_card else None
                ),
                previous_by_card={},
                previous_review_kind_by_card={},
                positive_day_counts_by_card={},
                prior_lapses_by_card={},
                filtered_review_normalization_policy=(
                    review_data.filtered_review_normalization_policy
                ),
                filtered_review_phase_by_card={},
            )
        )
        self.latest_revlog_id_by_card = dict(
            review_data.latest_revlog_id_by_card
            if review_data.latest_revlog_id_by_card is not None
            else _latest_revlog_ids_by_card(review_data)
        )
        self._rollback_by_review_id: dict[int, _ReviewRollbackSnapshot] = {}
        self._prediction_rows_by_card: dict[int, dict[str, Any]] = {}

    @property
    def review_data(self) -> ReviewData:
        return ReviewData(
            rows=self._base_rows,
            revlogs=self._base_revlogs,
            cards=self.cards,
            decks=self.decks,
            last_by_card=self.last_by_card,
            next_day_at=self.next_day_at,
            day_offset_origin=self.day_offset_origin,
            filtered_review_normalization_policy=(
                self.context.filtered_review_normalization_policy
            ),
            row_build_context=self.context,
            latest_revlog_id_by_card=dict(self.latest_revlog_id_by_card),
            complete_history=self._complete_history,
        )

    @property
    def day_offset_origin(self) -> int:
        return 0 if self.context.day_offset_origin is None else self.context.day_offset_origin

    def prediction_target_day_offset(self, target_timestamp_seconds: float) -> float:
        raw_target_day = day_offset_for_timestamp(
            float(target_timestamp_seconds),
            self.next_day_at,
        )
        return float(raw_target_day - self.day_offset_origin)

    def load_new_review_rows_for_card(self, col, card_id: int) -> list[dict[str, Any]]:
        normalized_card_id = int(card_id)
        next_day_at = int(col.sched.day_cutoff)
        self._rebase_day_offset_origin(next_day_at)
        self.next_day_at = next_day_at
        card = load_card_info_for_card_directly(
            col,
            normalized_card_id,
            self.decks,
        )
        if card is not None:
            self.cards[normalized_card_id] = card
            self._prediction_rows_by_card.pop(normalized_card_id, None)

        rows: list[dict[str, Any]] = []
        entries = load_new_revlog_entries_for_card_directly(
            col,
            card_id=normalized_card_id,
            after_review_id=self.latest_revlog_id_by_card.get(normalized_card_id),
        )
        for entry in entries:
            entry_card_id = int(entry.cid)
            review_id = int(entry.id)
            previous_latest = self.latest_revlog_id_by_card.get(entry_card_id)
            if previous_latest is None or review_id > previous_latest:
                self.latest_revlog_id_by_card[entry_card_id] = review_id

        # Live-review state is transient and intentionally allowed to drift from
        # the durable benchmark-trimmed history for a short session. If a later
        # learning-start sequence would require rewriting old durable card state,
        # the normal checkpoint update path will invalidate the checkpoint and
        # prompt for a rebuild. During live review, process the answered row so
        # the session can continue instead of stopping on an ephemeral mismatch.
        for entry in entries:
            entry_card_id = int(entry.cid)
            review_id = int(entry.id)
            snapshot = self._snapshot(entry_card_id, review_id)
            row, self.context = append_review_row_to_context(
                entry,
                self.cards,
                self.decks,
                self.next_day_at,
                self.context,
            )
            if row is None:
                continue
            self._rollback_by_review_id[review_id] = snapshot
            self._update_last_review(row)
            rows.append(row)
        return rows

    def prediction_rows_for_card_ids(
        self,
        card_ids: tuple[int, ...],
        *,
        target_timestamp_seconds: float,
    ) -> tuple[dict[str, Any], ...]:
        """Update and return session-owned prediction rows for one bounded job."""

        rows, _elapsed_days = self.prediction_inputs_for_card_ids(
            card_ids,
            target_timestamp_seconds=target_timestamp_seconds,
        )
        return tuple(rows)

    def prediction_inputs_for_card_ids(
        self,
        card_ids: tuple[int, ...],
        *,
        target_timestamp_seconds: float,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Return reusable native rows plus aligned engine-owned elapsed values."""

        target = float(target_timestamp_seconds)
        raw_target_day = day_offset_for_timestamp(target, self.next_day_at)
        target_day = raw_target_day - self.day_offset_origin
        review_id = int(target * 1000)
        cached_rows = self._prediction_rows_by_card
        cards = self.cards
        last_by_card = self.last_by_card
        rows: list[dict[str, Any]] = []
        elapsed_days: list[float] = []

        for card_id in card_ids:
            row = cached_rows.get(card_id)
            created = row is None
            if created:
                card = cards.get(card_id)
                row = {
                    "review_id": review_id,
                    "card_id": card_id,
                    "note_id": None if card is None else card.note_id,
                    "deck_id": None if card is None else card.deck_id,
                    "preset_id": None if card is None else card.preset_id,
                    "day_offset": target_day,
                    "elapsed_days": NEW_CARD_ELAPSED,
                    "elapsed_seconds": NEW_CARD_ELAPSED,
                    "raw_day_offset": raw_target_day,
                }
                cached_rows[card_id] = row
            else:
                row["review_id"] = review_id
                day_changed = (
                    int(row["raw_day_offset"]) != raw_target_day
                    or int(row["day_offset"]) != target_day
                )
                if day_changed:
                    row["day_offset"] = target_day
                    row["raw_day_offset"] = raw_target_day

            last_review = last_by_card.get(card_id)
            if last_review is None:
                if created or day_changed:
                    row["elapsed_days"] = NEW_CARD_ELAPSED
                row["elapsed_seconds"] = NEW_CARD_ELAPSED
            else:
                if created or day_changed:
                    row["elapsed_days"] = target_day - last_review.day_offset
                elapsed_seconds = target - last_review.timestamp_seconds
                row["elapsed_seconds"] = (
                    int(elapsed_seconds) if elapsed_seconds.is_integer() else elapsed_seconds
                )
            rows.append(row)
            elapsed_days.append(float(row["elapsed_days"]))
        return rows, elapsed_days

    def _rebase_day_offset_origin(self, next_day_at: int) -> None:
        if int(next_day_at) == self.next_day_at:
            return
        anchors: Iterable[dict[str, Any]] = self._base_rows
        if not self._base_rows:
            anchors = (
                {
                    "review_id": review_id,
                    "day_offset": day_offset,
                }
                for day_offset, review_id in self.context.previous_by_card.values()
            )
        origin = rebased_day_offset_origin(
            anchors,
            current_origin=self.context.day_offset_origin,
            next_day_at=next_day_at,
            previous_next_day_at=self.next_day_at,
        )
        self.context = replace(self.context, day_offset_origin=origin)

    def rollback_review(self, review_id: int) -> bool:
        normalized_review_id = int(review_id)
        snapshot = self._rollback_by_review_id.pop(normalized_review_id, None)
        if snapshot is None:
            return False

        current_last = self.last_by_card.get(snapshot.card_id)
        if current_last is not None and int(current_last.review_id) != normalized_review_id:
            raise ValueError("Cannot roll back a non-latest live review for a card.")

        self._restore_context(snapshot)
        if snapshot.previous_latest_revlog_id is None:
            self.latest_revlog_id_by_card.pop(snapshot.card_id, None)
        else:
            self.latest_revlog_id_by_card[snapshot.card_id] = snapshot.previous_latest_revlog_id
        if snapshot.had_last_review:
            assert snapshot.previous_last_review is not None
            self.last_by_card[snapshot.card_id] = snapshot.previous_last_review
        else:
            self.last_by_card.pop(snapshot.card_id, None)
        self._prediction_rows_by_card.pop(snapshot.card_id, None)
        return True

    def _snapshot(
        self,
        card_id: int,
        review_id: int,
    ) -> _ReviewRollbackSnapshot:
        normalized_card_id = int(card_id)
        return _ReviewRollbackSnapshot(
            review_id=int(review_id),
            card_id=normalized_card_id,
            previous_day_offset_origin=self.context.day_offset_origin,
            previous_latest_revlog_id=self.latest_revlog_id_by_card.get(normalized_card_id),
            had_last_review=normalized_card_id in self.last_by_card,
            previous_last_review=self.last_by_card.get(normalized_card_id),
            had_previous_by_card=normalized_card_id in self.context.previous_by_card,
            previous_by_card=self.context.previous_by_card.get(normalized_card_id),
            had_previous_review_kind=(
                normalized_card_id in self.context.previous_review_kind_by_card
            ),
            previous_review_kind=self.context.previous_review_kind_by_card.get(normalized_card_id),
            had_positive_day_count=(normalized_card_id in self.context.positive_day_counts_by_card),
            previous_positive_day_count=self.context.positive_day_counts_by_card.get(
                normalized_card_id
            ),
            had_prior_lapses=normalized_card_id in self.context.prior_lapses_by_card,
            previous_prior_lapses=self.context.prior_lapses_by_card.get(normalized_card_id),
            had_filtered_review_phase=(
                normalized_card_id in self.context.filtered_review_phase_by_card
            ),
            previous_filtered_review_phase=(
                self.context.filtered_review_phase_by_card.get(normalized_card_id)
            ),
        )

    def _restore_context(self, snapshot: _ReviewRollbackSnapshot) -> None:
        self.context = ReviewRowBuildContext(
            day_offset_origin=snapshot.previous_day_offset_origin,
            previous_by_card=self.context.previous_by_card,
            previous_review_kind_by_card=self.context.previous_review_kind_by_card,
            positive_day_counts_by_card=self.context.positive_day_counts_by_card,
            prior_lapses_by_card=self.context.prior_lapses_by_card,
            filtered_review_normalization_policy=(
                self.context.filtered_review_normalization_policy
            ),
            filtered_review_phase_by_card=(self.context.filtered_review_phase_by_card),
        )
        _restore_optional_map_value(
            self.context.previous_by_card,
            snapshot.card_id,
            snapshot.had_previous_by_card,
            snapshot.previous_by_card,
        )
        _restore_optional_map_value(
            self.context.previous_review_kind_by_card,
            snapshot.card_id,
            snapshot.had_previous_review_kind,
            snapshot.previous_review_kind,
        )
        _restore_optional_map_value(
            self.context.positive_day_counts_by_card,
            snapshot.card_id,
            snapshot.had_positive_day_count,
            snapshot.previous_positive_day_count,
        )
        _restore_optional_map_value(
            self.context.prior_lapses_by_card,
            snapshot.card_id,
            snapshot.had_prior_lapses,
            snapshot.previous_prior_lapses,
        )
        _restore_optional_map_value(
            self.context.filtered_review_phase_by_card,
            snapshot.card_id,
            snapshot.had_filtered_review_phase,
            snapshot.previous_filtered_review_phase,
        )

    def _update_last_review(self, row: dict[str, Any]) -> None:
        card_id = int(row["card_id"])
        previous = self.last_by_card.get(card_id)
        previous_lapses = previous.lapse_count if previous is not None else 0
        lapse_count = previous_lapses
        if int(row["rating"]) == 1 and float(row["elapsed_days"]) != 0:
            lapse_count += 1
        self.last_by_card[card_id] = LastReviewInfo(
            review_id=int(row["review_id"]),
            day_offset=int(row["day_offset"]),
            timestamp_seconds=int(row["review_id"]) / 1000.0,
            interval=int(row.get("interval", 0)),
            lapse_count=lapse_count,
        )
        self._prediction_rows_by_card.pop(card_id, None)


class LiveReviewPredictionCoordinator:
    def __init__(
        self,
        *,
        engine: LiveReviewEngine,
        review_data: ReviewData,
        clock: Callable[[], float] | None = None,
        runtime_session: Any | None = None,
        native_seed_factory: Callable[..., Any] | None = None,
        native_profiling: bool = False,
    ) -> None:
        self.engine = engine
        self.history = LiveReviewHistoryState(review_data)
        self.clock = clock or time.time
        self._in_flight = False
        self._runtime_session = runtime_session
        self._native_seed_factory = native_seed_factory
        self._native_profiling = bool(native_profiling)
        self._native_session: Any | None = None
        self._native_initial_result: Any | None = None
        self._native_start_selection: Any | None = None
        self._native_reconciliation_result: Any | None = None
        self._native_reconciliation_dropped_card_ids: tuple[int, ...] = ()
        # Native refreshes intentionally do not return their large membership
        # tuple on the hot path. ``None`` means it can be fetched lazily from
        # the diagnostic API if (and only if) a quiet exhaustion scan needs to
        # advance beyond the ordinary post-answer refresh.
        self._last_refresh_card_ids: tuple[int, ...] | None = ()

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    @property
    def review_data(self) -> ReviewData:
        return self.history.review_data

    @property
    def review_context(self) -> ReviewRowBuildContext:
        return self.history.context

    @property
    def native_session_active(self) -> bool:
        return self._native_session is not None

    @property
    def native_initial_result(self) -> Any | None:
        return self._native_initial_result

    @property
    def native_start_selection(self) -> Any | None:
        return self._native_start_selection

    @property
    def native_current_universe_result(self) -> Any | None:
        return self._native_reconciliation_result or self._native_initial_result

    @property
    def native_reconciliation_dropped_card_ids(self) -> tuple[int, ...]:
        return self._native_reconciliation_dropped_card_ids

    @property
    def native_profile(self) -> dict[str, Any] | None:
        if self._native_session is None:
            return None
        return dict(self._native_session.profile())

    def activate_native_session(
        self,
        *,
        target_timestamp_seconds: float | None = None,
    ) -> bool:
        """Move generic live candidate ranking into RWKV-SRS.

        Every currently exposed order is generic SRS policy supported by the
        native helper; Anki-specific orders remain outside this boundary.
        """

        if self._native_session is not None:
            return True
        runtime = self._runtime_session
        factory = getattr(runtime, "predict_many_live_session", None)
        if not callable(factory) or not self.engine.active:
            return False
        refresh_limit = int(self.engine.settings.prediction_refresh_limit)
        if refresh_limit <= 0:
            return False
        order = native_live_order(self.engine.settings.order_index)
        if order is None:
            return False
        card_ids = self.engine.candidate_universe_card_ids
        if not card_ids:
            return False
        target = (
            self.clock() if target_timestamp_seconds is None else float(target_timestamp_seconds)
        )
        seeds = self._native_seeds(card_ids, target_timestamp_seconds=target)
        if not seeds:
            return False
        live = factory(
            seeds,
            initial_target_timestamp_seconds=target,
            initial_target_day_offset=self.history.prediction_target_day_offset(target),
            order=order,
            refresh_limit=refresh_limit,
            profiling=self._native_profiling,
            initial_select_limit=2,
        )
        self._native_session = live
        initial_result = getattr(live, "initial_result", None)
        if initial_result is None:
            self._native_session = None
            live.close()
            raise RuntimeError(
                "The bundled RWKV-SRS live session does not expose its initial result."
            )
        self._native_initial_result = initial_result
        self.engine.enable_native_prediction_selection()
        start_selection = self._initial_native_selection(
            live,
            initial_result,
            seeds=seeds,
        )
        self._native_start_selection = start_selection
        # The native constructor predicts the complete seed universe once.
        # Retain the existing tuple so an unlikely startup exhaustion check
        # cannot immediately predict the same cards again.
        self._last_refresh_card_ids = card_ids
        selected_values = self._native_values_for_selections(
            tuple(start_selection.selected),
            target_timestamp_seconds=target,
        )
        token = LivePredictionToken(
            session_generation=self.engine.session_generation,
            prediction_generation=self.engine.prediction_generation,
            candidate_card_ids=tuple(value.card_id for value in selected_values),
        )
        applied = self.engine.apply_native_refresh_result(
            token,
            selected_values=selected_values,
            refreshed_count=int(initial_result.refreshed_count),
            target_timestamp_seconds=target,
        )
        if applied.status != LivePredictionStatus.APPLIED:
            self.deactivate_native_session()
            return False
        return True

    def deactivate_native_session(self) -> None:
        live = self._native_session
        self._native_session = None
        self._native_initial_result = None
        self._native_start_selection = None
        self._native_reconciliation_result = None
        self._native_reconciliation_dropped_card_ids = ()
        self._last_refresh_card_ids = ()
        self.engine.disable_native_prediction_selection()
        if live is not None:
            live.close()

    def process_answer(self, row: dict[str, Any]) -> tuple[float, Any]:
        if self._native_session is None:
            raise RuntimeError("native live prediction session is not active")
        if (
            self.engine.settings.allow_same_day_repeats
            and int(self.engine.settings.same_day_reentry_delay_reviews) > 0
        ):
            process_and_exclude = getattr(
                self._native_session,
                "process_answer_and_exclude",
                None,
            )
            if callable(process_and_exclude):
                return process_and_exclude(dict(row))
            result = self._native_session.process_answer(
                dict(row),
                requeue_after_prediction=True,
            )
            self._native_session.exclude_card(int(row["card_id"]))
            return result
        return self._native_session.process_answer(
            dict(row),
            requeue_after_prediction=bool(self.engine.settings.allow_same_day_repeats),
        )

    def undo_last_process(self) -> int:
        if self._native_session is None:
            raise RuntimeError("native live prediction session is not active")
        return int(self._native_session.undo_last_process())

    def reconcile_candidate_universe(
        self,
        candidates: Iterable[LiveReviewCandidate],
        *,
        target_timestamp_seconds: float | None = None,
    ) -> LiveNativeCandidateReconciliation:
        """Reconcile a complete source-search result without retiring Rust rank."""

        live = self._native_session
        if live is None:
            raise RuntimeError("native live prediction session is not active")
        target = (
            self.clock() if target_timestamp_seconds is None else float(target_timestamp_seconds)
        )
        materialized = tuple(candidates)
        requested_ids = {int(candidate.card_id) for candidate in materialized}
        contained_card_ids = getattr(
            self._runtime_session,
            "contained_card_ids",
            None,
        )
        contains_card = getattr(self._runtime_session, "contains_card", None)
        if callable(contained_card_ids):
            in_scope = set(contained_card_ids(requested_ids))
            supported = tuple(
                candidate for candidate in materialized if int(candidate.card_id) in in_scope
            )
            dropped = tuple(
                int(candidate.card_id)
                for candidate in materialized
                if int(candidate.card_id) not in in_scope
            )
        elif callable(contains_card):
            supported_values: list[LiveReviewCandidate] = []
            dropped_values: list[int] = []
            for candidate in materialized:
                card_id = int(candidate.card_id)
                if contains_card(card_id):
                    supported_values.append(candidate)
                else:
                    dropped_values.append(card_id)
            supported = tuple(supported_values)
            dropped = tuple(dropped_values)
        else:
            supported = materialized
            dropped = ()

        if not supported:
            self._native_reconciliation_dropped_card_ids = dropped
            return LiveNativeCandidateReconciliation(
                candidates=(),
                result=None,
                dropped_out_of_scope_card_ids=dropped,
            )
        hot_card_ids = tuple(hot.card_id for hot in self.engine.hot_registry)
        quarantined_hot_card_ids = set(self.engine.quarantined_hot_card_ids)
        native_supported = tuple(
            candidate
            for candidate in supported
            if int(candidate.card_id) not in quarantined_hot_card_ids
        )
        exclude_card_ids = self.engine.native_selection_exclusion_card_ids(
            hot_card_ids=hot_card_ids,
        )
        target_day_offset = self.history.prediction_target_day_offset(target)
        desired_card_ids = tuple(int(candidate.card_id) for candidate in native_supported)
        reconcile_membership = getattr(live, "reconcile_membership", None)
        if not callable(reconcile_membership):
            raise RuntimeError(
                "The bundled RWKV-SRS live session does not provide reconcile_membership()."
            )
        changed_candidates = self._native_membership_changed_candidates(native_supported)
        seeds = self._native_seeds_for_candidates(
            changed_candidates,
            target_timestamp_seconds=target,
        )
        reconciled_result = reconcile_membership(
            desired_card_ids,
            seeds,
            target_timestamp_seconds=target,
            target_day_offset=target_day_offset,
            select_limit=2,
            exclude_card_ids=exclude_card_ids,
            retention_extra=self.engine._active_minimum_retention_extra(),
        )
        # Preserve the Python policy mirror before minimum-review widening can
        # change its retention-extra state. ``replace_candidate_universe()``
        # repeats this call, but the engine deliberately retains only the first
        # snapshot attached to the latest answer.
        self.engine._capture_candidate_universe_for_latest_undo()
        widening_exclusions = exclude_card_ids
        if not reconciled_result.selected and self.engine._minimum_fill_active():
            widening_exclusions = (
                *exclude_card_ids,
                *self._intraday_card_ids_at_day(
                    desired_card_ids,
                    target_day_offset=target_day_offset,
                ),
            )
        selection_result = self._initial_native_selection(
            live,
            reconciled_result,
            seeds=seeds,
            exclude_card_ids=widening_exclusions,
        )
        result = LiveNativeReconciliationResult(
            generation=int(selection_result.generation),
            refreshed_count=int(reconciled_result.refreshed_count),
            eligible_count=int(selection_result.eligible_count),
            active_count=int(selection_result.active_count),
            selected=tuple(selection_result.selected),
            next_retention_extra=(
                None
                if selection_result.next_retention_extra is None
                else float(selection_result.next_retention_extra)
            ),
        )

        # The native commit succeeded. Install the matching small Python policy
        # mirror, then apply only the compact selections needed to build Anki's
        # next two-card buffer.
        self.engine.replace_candidate_universe(supported)
        selected_values = self._native_values_for_selections(
            tuple(result.selected),
            target_timestamp_seconds=target,
        )
        hot_values = self._native_values_for_card_ids(hot_card_ids)
        included_hot_card_ids = self.engine.native_hot_card_ids_pending_include()
        token = LivePredictionToken(
            session_generation=self.engine.session_generation,
            prediction_generation=self.engine.prediction_generation,
            candidate_card_ids=tuple(value.card_id for value in selected_values),
        )
        applied = self.engine.apply_native_refresh_result(
            token,
            selected_values=selected_values,
            hot_values=hot_values,
            refreshed_count=int(result.refreshed_count),
            target_timestamp_seconds=target,
            included_hot_card_ids=included_hot_card_ids,
        )
        if applied.status != LivePredictionStatus.APPLIED:
            raise RuntimeError("native live candidate reconciliation became stale")
        self._native_reconciliation_result = result
        self._native_reconciliation_dropped_card_ids = dropped
        self._last_refresh_card_ids = desired_card_ids
        return LiveNativeCandidateReconciliation(
            candidates=supported,
            result=result,
            dropped_out_of_scope_card_ids=dropped,
        )

    def begin_refresh(
        self,
        *,
        target_timestamp_seconds: float | None = None,
        exclude_refresh_card_ids: Iterable[int] = (),
        select_limit: int = 2,
    ) -> LiveReviewNativePredictionJob | None:
        if self._in_flight or not self.engine.active:
            return None

        # A refresh attempt supersedes the previous pass for purposes of the
        # bounded quiet-expansion scan. Leave this empty unless a result is
        # successfully applied below.
        self._last_refresh_card_ids = ()

        target = (
            self.clock() if target_timestamp_seconds is None else float(target_timestamp_seconds)
        )
        if self._native_session is not None:
            hot_card_ids = self.engine.native_refresh_hot_card_ids()
            include_hot_card_ids = self.engine.native_hot_card_ids_pending_include()
            token = self.engine.begin_prediction(hot_card_ids)
            hot_seeds = self._native_seeds(
                hot_card_ids,
                target_timestamp_seconds=target,
            )
            self._in_flight = True
            return LiveReviewNativePredictionJob(
                token=token,
                target_timestamp_seconds=target,
                target_day_offset=self.history.prediction_target_day_offset(target),
                hot_card_ids=hot_card_ids,
                hot_seeds=hot_seeds,
                include_hot_card_ids=include_hot_card_ids,
                exclude_card_ids=self.engine.native_selection_exclusion_card_ids(
                    hot_card_ids=hot_card_ids,
                ),
                exclude_refresh_card_ids=tuple(
                    int(card_id) for card_id in exclude_refresh_card_ids
                ),
                select_limit=max(0, int(select_limit)),
            )

        return None

    def run_job(
        self,
        job: LiveReviewNativePredictionJob,
    ) -> LiveReviewNativePredictionJobResult:
        return self._run_native_job(job)

    def apply_result(
        self,
        result: LiveReviewNativePredictionJobResult,
    ) -> LivePredictionResult:
        self._in_flight = False
        applied = self.engine.apply_native_refresh_result(
            result.token,
            selected_values=result.selected_values,
            hot_values=result.hot_values,
            included_hot_card_ids=result.included_hot_card_ids,
            refreshed_count=result.refreshed_count,
            target_timestamp_seconds=result.target_timestamp_seconds,
        )
        if applied.status == LivePredictionStatus.APPLIED:
            # Defer the diagnostic ID transfer until an exhaustion scan needs
            # it. Normal reviews keep the compact native result.
            self._last_refresh_card_ids = None
        return applied

    def finish_failed_refresh(self) -> LivePredictionResult:
        self._in_flight = False
        self._last_refresh_card_ids = ()
        if not self.engine.active:
            return LivePredictionResult(LivePredictionStatus.INACTIVE)
        return LivePredictionResult(LivePredictionStatus.STALE)

    def extend_quiet_recheck_exclusions(self, checked_card_ids: set[int]) -> None:
        """Exclude the immediately preceding normal refresh from quiet scans.

        Native refresh membership is deliberately retrieved only after the
        ordinary refresh found no usable candidates. The cached tuple is then
        reused by every configured quiet attempt in this refill.
        """

        card_ids = self._last_refresh_card_ids
        if card_ids is None:
            live = self._native_session
            if live is None:
                card_ids = ()
            else:
                debug = live.last_refresh_debug()
                card_ids = tuple(int(card_id) for card_id in debug.get("membership_card_ids", ()))
            self._last_refresh_card_ids = card_ids
        checked_card_ids.update(card_ids)

    def recheck_stale_candidates(
        self,
        *,
        target_timestamp_seconds: float | None = None,
        max_batches: int = 1,
        needed_count: int = 1,
        checked_card_ids: set[int] | None = None,
    ) -> LiveStaleRecheckResult:
        checked: set[int] = checked_card_ids if checked_card_ids is not None else set()
        batches = 0
        checked_count = 0
        eligible_count = 0
        eligible_card_ids: tuple[int, ...] = ()
        if (
            not self.engine.active
            or self._native_session is None
            or int(self.engine.settings.prediction_refresh_limit) <= 0
            or int(max_batches) <= 0
        ):
            return self._record_stale_recheck_result(
                batches=0,
                checked_count=0,
                eligible_count=0,
                eligible_card_ids=(),
            )

        target_eligible_count = max(1, int(needed_count))
        for _batch_index in range(int(max_batches)):
            before_checked = len(checked)
            job = self.begin_refresh(
                target_timestamp_seconds=target_timestamp_seconds,
                exclude_refresh_card_ids=checked,
                select_limit=target_eligible_count,
            )
            if job is None:
                break
            try:
                result = self.run_job(job)
                applied = self.apply_result(result)
            except Exception:
                self.finish_failed_refresh()
                raise
            if applied.status != LivePredictionStatus.APPLIED:
                break
            self.extend_quiet_recheck_exclusions(checked)
            batch_checked_count = len(checked) - before_checked
            batches += 1
            checked_count += batch_checked_count
            eligible_count += int(result.eligible_count)
            eligible_card_ids = tuple(
                int(value.card_id) for value in result.selected_values
            )
            if eligible_card_ids or batch_checked_count == 0:
                break

        return self._record_stale_recheck_result(
            batches=batches,
            checked_count=checked_count,
            eligible_count=eligible_count,
            eligible_card_ids=eligible_card_ids,
        )

    def update_review_data(
        self,
        review_data: ReviewData,
        context: ReviewRowBuildContext,
    ) -> None:
        self.history = LiveReviewHistoryState(review_data)
        self.history.context = context

    def load_new_review_rows_for_card(self, col, card_id: int) -> list[dict[str, Any]]:
        return self.history.load_new_review_rows_for_card(col, card_id)

    def rollback_review(self, review_id: int) -> bool:
        return self.history.rollback_review(review_id)

    def _run_native_job(
        self,
        job: LiveReviewNativePredictionJob,
    ) -> LiveReviewNativePredictionJobResult:
        live = self._native_session
        if live is None:
            raise RuntimeError("native live prediction session is not active")
        if job.hot_seeds:
            upsert_and_include = getattr(
                live,
                "upsert_and_include_candidates",
                None,
            )
            if job.include_hot_card_ids and callable(upsert_and_include):
                upsert_and_include(
                    job.hot_seeds,
                    job.include_hot_card_ids,
                )
            else:
                live.upsert_candidates(job.hot_seeds)
                for card_id in job.include_hot_card_ids:
                    live.include_card(int(card_id))
        result = live.refresh(
            target_timestamp_seconds=job.target_timestamp_seconds,
            target_day_offset=job.target_day_offset,
            select_limit=job.select_limit,
            exclude_card_ids=job.exclude_card_ids,
            exclude_refresh_card_ids=job.exclude_refresh_card_ids,
            retention_extra=self.engine._active_minimum_retention_extra(),
        )
        selected_values = self._native_values_for_selections(
            tuple(result.selected),
            target_timestamp_seconds=job.target_timestamp_seconds,
        )
        hot_values = self._native_values_for_card_ids(job.hot_card_ids)
        return LiveReviewNativePredictionJobResult(
            token=job.token,
            selected_values=selected_values,
            hot_values=hot_values,
            included_hot_card_ids=job.include_hot_card_ids,
            refreshed_count=int(result.refreshed_count),
            eligible_count=int(result.eligible_count),
            target_timestamp_seconds=job.target_timestamp_seconds,
        )

    def _native_values_for_selections(
        self,
        selections: tuple[Any, ...],
        *,
        target_timestamp_seconds: float,
    ) -> tuple[LiveNativePredictionValue, ...]:
        """Convert the compact refresh result without diagnostic round trips."""

        card_ids = tuple(int(selection.card_id) for selection in selections)
        if not card_ids:
            return ()
        _rows, elapsed_days = self.history.prediction_inputs_for_card_ids(
            card_ids,
            target_timestamp_seconds=float(target_timestamp_seconds),
        )
        return tuple(
            LiveNativePredictionValue(
                card_id=card_id,
                predicted_retrievability=float(selection.retrievability),
                elapsed_days=float(elapsed),
            )
            for card_id, selection, elapsed in zip(
                card_ids,
                selections,
                elapsed_days,
                strict=True,
            )
        )

    def _native_values_for_card_ids(
        self,
        card_ids: tuple[int, ...],
    ) -> tuple[LiveNativePredictionValue, ...]:
        live = self._native_session
        if live is None:
            return ()
        values: list[LiveNativePredictionValue] = []
        for card_id in card_ids:
            snapshot = live.candidate(int(card_id))
            if snapshot is None:
                continue
            values.append(
                LiveNativePredictionValue(
                    card_id=int(snapshot.card_id),
                    predicted_retrievability=float(snapshot.retrievability),
                    elapsed_days=float(snapshot.elapsed_days),
                )
            )
        return tuple(values)

    def _native_seeds(
        self,
        card_ids: Iterable[int],
        *,
        target_timestamp_seconds: float,
    ) -> tuple[Any, ...]:
        normalized_ids = tuple(int(card_id) for card_id in card_ids)
        if not normalized_ids:
            return ()
        return self._native_seeds_for_candidates(
            self.engine.candidates_for_card_ids(normalized_ids),
            target_timestamp_seconds=target_timestamp_seconds,
            card_ids=normalized_ids,
        )

    def _native_seeds_for_candidates(
        self,
        candidates: Iterable[LiveReviewCandidate],
        *,
        target_timestamp_seconds: float,
        card_ids: Iterable[int] | None = None,
    ) -> tuple[Any, ...]:
        candidate_values = tuple(candidates)
        candidates_by_card = {int(candidate.card_id): candidate for candidate in candidate_values}
        normalized_ids = (
            tuple(int(card_id) for card_id in card_ids)
            if card_ids is not None
            else tuple(candidates_by_card)
        )
        if not normalized_ids:
            return ()
        rows, _elapsed_days = self.history.prediction_inputs_for_card_ids(
            normalized_ids,
            target_timestamp_seconds=float(target_timestamp_seconds),
        )
        seed_factory = self._native_seed_factory or require_rwkv_live_candidate_seed()
        random_order = (
            filtered_deck_order_from_index(self.engine.settings.order_index)
            == FilteredDeckOrder.RANDOM
        )
        seeds: list[Any] = []
        for card_id, row in zip(normalized_ids, rows, strict=True):
            candidate = candidates_by_card.get(card_id)
            if candidate is None:
                continue
            same_day_target, normal_target = _candidate_static_values(candidate)
            # The add-on's Anki tie-breaker is a signed i64. Shift it into the
            # upstream u64 domain so its numeric ordering remains unchanged.
            signed_tie = int(filtered_deck_tiebreaker(int(candidate.card_id), candidate.sort_info))
            random_key = (
                _native_random_key(self.engine._random_sort_key_for_card(int(candidate.card_id)))
                if random_order
                else 0
            )
            seeds.append(
                seed_factory(
                    row=row,
                    target_retrievability=float(normal_target),
                    intraday_target_retrievability=float(same_day_target),
                    tie_breaker=signed_tie + 2**63,
                    random_key=random_key,
                )
            )
        return tuple(seeds)

    def _native_membership_changed_candidates(
        self,
        candidates: Iterable[LiveReviewCandidate],
    ) -> tuple[LiveReviewCandidate, ...]:
        """Return only candidates whose native seed facts cannot be reused."""

        current_by_card = self.engine._candidates_by_card
        changed: list[LiveReviewCandidate] = []
        for candidate in candidates:
            card_id = int(candidate.card_id)
            current = current_by_card.get(card_id)
            if current is None:
                changed.append(candidate)
                continue
            if current.source_deck_id != candidate.source_deck_id:
                changed.append(candidate)
                continue
            if _candidate_static_values(current) != _candidate_static_values(candidate):
                changed.append(candidate)
                continue
            current_tie = filtered_deck_tiebreaker(
                int(current.card_id),
                current.sort_info,
            )
            replacement_tie = filtered_deck_tiebreaker(
                card_id,
                candidate.sort_info,
            )
            if current_tie != replacement_tie:
                changed.append(candidate)
        return tuple(changed)

    def _intraday_card_ids_at_day(
        self,
        card_ids: Iterable[int],
        *,
        target_day_offset: float,
    ) -> tuple[int, ...]:
        """Return existing same-day cards without rebuilding prediction rows."""

        target_day = float(target_day_offset)
        last_by_card = self.history.last_by_card
        intraday: list[int] = []
        for card_id in card_ids:
            normalized = int(card_id)
            last_review = last_by_card.get(normalized)
            if last_review is None:
                continue
            elapsed_days = target_day - float(last_review.day_offset)
            if 0.0 <= elapsed_days < 1.0:
                intraday.append(normalized)
        return tuple(intraday)

    def _initial_native_selection(
        self,
        live: Any,
        initial_result: Any,
        *,
        seeds: tuple[Any, ...],
        exclude_card_ids: Iterable[int] = (),
    ) -> Any:
        """Apply minimum-review widening to the constructor rank without inference."""

        if initial_result.selected or not self.engine._minimum_fill_active():
            return initial_result

        excluded = set(int(card_id) for card_id in exclude_card_ids)
        excluded.update(_intraday_seed_card_ids(seeds))
        result = live.current_selection(
            select_limit=2,
            exclude_card_ids=excluded,
        )
        extra = float(self.engine.minimum_retention_extra)
        quantum = float(self.engine.settings.minimum_retention_extra_quantum)
        while not result.selected and result.next_retention_extra is not None:
            next_extra = minimum_retention_extra_required(
                float(result.next_retention_extra),
                0.0,
                extra_quantum=quantum,
            )
            if next_extra is None or next_extra <= extra:
                break
            extra = float(next_extra)
            self.engine.minimum_retention_extra = extra
            live.set_retention_extra(extra)
            result = live.current_selection(
                select_limit=2,
                exclude_card_ids=excluded,
            )
        return result

    def _record_stale_recheck_result(
        self,
        *,
        batches: int,
        checked_count: int,
        eligible_count: int,
        eligible_card_ids: tuple[int, ...],
    ) -> LiveStaleRecheckResult:
        self.engine.record_stale_recheck(
            batches=batches,
            checked_count=checked_count,
            eligible_count=eligible_count,
        )
        return LiveStaleRecheckResult(
            batches=max(0, int(batches)),
            checked_count=max(0, int(checked_count)),
            eligible_count=max(0, int(eligible_count)),
            remaining_count=len(self.engine.stale_queue),
            eligible_card_ids=tuple(int(card_id) for card_id in eligible_card_ids),
        )


def _latest_revlog_ids_by_card(review_data: ReviewData) -> dict[int, int]:
    latest: dict[int, int] = {}
    for row in review_data.revlogs:
        card_id = int(row["card_id"])
        review_id = int(row["review_id"])
        if review_id > latest.get(card_id, -1):
            latest[card_id] = review_id
    return latest


def _restore_optional_map_value(
    target: dict[int, Any],
    key: int,
    existed: bool,
    value: Any,
) -> None:
    if existed:
        target[int(key)] = value
    else:
        target.pop(int(key), None)


def native_live_order(order_index: int) -> str | None:
    order = filtered_deck_order_from_index(order_index)
    if order == FilteredDeckOrder.RETRIEVABILITY_ASCENDING:
        return "retrievability_ascending"
    if order == FilteredDeckOrder.RETRIEVABILITY_DESCENDING:
        return "retrievability_descending"
    if order == FilteredDeckOrder.RELATIVE_OVERDUENESS:
        return "relative_overdueness"
    if order == FilteredDeckOrder.RANDOM:
        return "random"
    return None


def _native_random_key(value: float) -> int:
    """Map a Python random() key monotonically into the native u64 domain."""

    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("Live Session random ordering key must be finite.")
    resolved = min(math.nextafter(1.0, 0.0), max(0.0, resolved))
    return min(2**64 - 1, int(resolved * 2**64))


def _intraday_seed_card_ids(seeds: Iterable[Any]) -> tuple[int, ...]:
    card_ids: list[int] = []
    for seed in seeds:
        row = getattr(seed, "row", None)
        if row is None:
            continue
        try:
            elapsed_days = float(row["elapsed_days"])
            card_id = int(row["card_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0.0 <= elapsed_days < 1.0:
            card_ids.append(card_id)
    return tuple(card_ids)
