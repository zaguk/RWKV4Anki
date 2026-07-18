from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .anki_api import cid_search, is_filtered_deck
from .constants import LIVE_REVIEW_RUST_UNDO_LIMIT
from .live_review_engine import (
    LiveAnswerResult,
    LiveAnswerStatus,
    LiveReviewCandidate,
    LiveReviewEngine,
    LiveReviewSettings,
    LiveSelectionStatus,
    LiveUndoStatus,
)
from .live_review_stats import LiveRetentionSummary, summarize_live_retention_records

LIVE_FILTERED_DECK_NAME = "- RWKV Live Session"
# Anki filtered decks evaluate at most two search terms. One cid per term
# preserves order while leaving one backup card behind the visible card. The
# per-term order is irrelevant because each term can pull at most one fixed card.
LIVE_FILTERED_DECK_ORDER = 0
LIVE_FILTERED_DECK_BUFFER_SIZE = 2
LIVE_FILTERED_DECK_MAX_REFILL_BUILDS = 5

class LiveBridgeStatus(Enum):
    REFILLED = "refilled"
    EMPTY = "empty"
    IGNORED = "ignored"
    PAUSED_FOR_UNDO = "paused_for_undo"
    STOPPED_LIMIT = "stopped_limit"
    TRACKED_DECK_UNAVAILABLE = "tracked_deck_unavailable"


@dataclass(frozen=True)
class LiveBridgeResult:
    status: LiveBridgeStatus
    deck_id: int | None
    selected_card_id: int | None = None
    requested_card_id: int | None = None
    buffered_card_ids: tuple[int, ...] = ()
    skipped_card_ids: tuple[int, ...] = ()
    elapsed_ms: float = 0.0
    merged_undo: bool = False
    reviews_done: int = 0
    transient_processed_review_count: int = 0
    undone_review_id: int | None = None
    undone_card_id: int | None = None


@dataclass
class LiveReviewBridgeSession:
    pending_card_ids: list[int] = field(default_factory=list)
    engine: LiveReviewEngine | None = None
    deck_name: str = LIVE_FILTERED_DECK_NAME
    review_limit: int | None = None
    filtered_deck_id: int | None = None
    fsrs_comparison_enabled: bool = True
    active: bool = True
    current_card_id: int | None = None
    buffered_card_ids: list[int] = field(default_factory=list)
    reviews_done: int = 0
    transient_processed_review_count: int = 0
    refill_latencies_ms: list[float] = field(default_factory=list)
    pre_answer_buffers_by_review_id: dict[int, tuple[int, ...]] = field(
        default_factory=dict,
    )
    pre_answer_buffer_review_ids: list[int] = field(default_factory=list)
    pre_answer_undo_targets_by_review_id: dict[int, int] = field(default_factory=dict)
    pending_pre_answer_undo_target: int | None = None
    current_pre_answer_undo_target: int | None = None
    runtime_session: Any | None = None

    def capture_shown_card(
        self,
        card: Any,
        *,
        fsrs_prediction: float | None = None,
        pre_answer_undo_target: int | None = None,
    ) -> bool:
        if not self.active or self.filtered_deck_id is None:
            return False
        card_deck_id = _int_attr(card, "did")
        if card_deck_id != self.filtered_deck_id:
            self.current_card_id = None
            self.current_pre_answer_undo_target = None
            self.pending_pre_answer_undo_target = None
            return False
        self.current_card_id = _card_id(card)
        if pre_answer_undo_target is None:
            pre_answer_undo_target = self.pending_pre_answer_undo_target
        self.current_pre_answer_undo_target = _positive_int_or_none(
            pre_answer_undo_target
        )
        self.pending_pre_answer_undo_target = None
        if self.engine is not None:
            self.engine.snapshot_shown_candidate(
                self.current_card_id,
                fsrs_prediction=fsrs_prediction,
            )
        return True

    def handle_answered_card(
        self,
        col,
        card: Any,
        _ease: int,
        *,
        review_row: dict[str, Any] | None = None,
        process_review_row: Callable[[dict[str, Any]], tuple[float, Any]] | None = None,
        curve_probability_func: Callable[[Any, float], float] | None = None,
        clock: Callable[[], float] | None = None,
        refresh_predictions_before_refill: Callable[[], Any] | None = None,
        stale_recheck: Callable[[set[int], int], bool] | None = None,
        merge_undo: bool = True,
    ) -> LiveBridgeResult:
        if not self.active:
            return self._result(LiveBridgeStatus.IGNORED)
        if self.current_card_id is None or _card_id(card) != self.current_card_id:
            return self._result(LiveBridgeStatus.IGNORED)
        if not self._tracked_deck_is_usable(col):
            self.active = False
            return self._result(LiveBridgeStatus.TRACKED_DECK_UNAVAILABLE)

        self.current_card_id = None
        answered_card_id = _card_id(card)
        pre_answer_undo_target = self.current_pre_answer_undo_target
        self.current_pre_answer_undo_target = None
        pre_answer_buffer = tuple(self.buffered_card_ids)
        self._forget_buffered_card(answered_card_id)
        answer = self._record_engine_answer(
            answered_card_id,
            review_row=review_row,
        )
        answer_status = answer.status
        if answer_status == LiveAnswerStatus.IGNORED:
            return self._result(LiveBridgeStatus.IGNORED)
        self._remember_pre_answer_buffer(
            answer.review_id,
            pre_answer_buffer,
            pre_answer_undo_target=pre_answer_undo_target,
        )
        if review_row is not None:
            self._process_answer_review_row(
                review_row,
                process_review_row=process_review_row,
                curve_probability_func=curve_probability_func,
                clock=clock,
            )

        undo_target = _last_undo_step(col) if merge_undo else None

        if (
            answer_status == LiveAnswerStatus.STOPPED_LIMIT
            or self.review_limit is not None
            and self.reviews_done >= self.review_limit
        ):
            self.active = False
            result = self.empty_live_filtered_deck(col)
            bridge_result = LiveBridgeResult(
                status=LiveBridgeStatus.STOPPED_LIMIT,
                deck_id=result.deck_id,
                elapsed_ms=result.elapsed_ms,
                reviews_done=self.reviews_done,
                transient_processed_review_count=self.transient_processed_review_count,
            )
            return _with_merged_undo_if_possible(
                col,
                bridge_result,
                target=undo_target,
                merge_undo=merge_undo,
            )

        self._release_unshown_buffered_cards()
        if refresh_predictions_before_refill is not None:
            refresh_predictions_before_refill()

        result = self.refill_buffer(
            col,
            force_rebuild=True,
            merge_undo_target=undo_target,
            merge_undo=merge_undo,
            stale_recheck=stale_recheck,
        )
        return _with_merged_undo_if_possible(
            col,
            result,
            target=undo_target,
            merge_undo=merge_undo,
        )

    def handle_undo(
        self,
        col,
        *,
        undo_process: Callable[[], int],
        stale_recheck: Callable[[set[int], int], bool] | None = None,
    ) -> LiveBridgeResult:
        if not self.active or self.engine is None:
            return self._result(LiveBridgeStatus.IGNORED)

        undoable_review_ids = tuple(self.engine.undoable_review_ids)
        undone_review_id = _first_missing_review_id(
            col,
            reversed(undoable_review_ids),
        )
        if undone_review_id is not None:
            undone_card_id = _card_id_for_undoable_review_id(
                self.engine.undoable_review_rows,
                undone_review_id,
            )
            if not undoable_review_ids or int(undone_review_id) != int(
                undoable_review_ids[-1]
            ):
                self.engine.record_undo(undone_review_id)
                self.active = False
                result = self.empty_live_filtered_deck(col)
                return LiveBridgeResult(
                    status=LiveBridgeStatus.PAUSED_FOR_UNDO,
                    deck_id=result.deck_id,
                    elapsed_ms=result.elapsed_ms,
                    reviews_done=self.reviews_done,
                    transient_processed_review_count=(
                        self.transient_processed_review_count
                    ),
                    undone_review_id=undone_review_id,
                    undone_card_id=undone_card_id,
                )
            undo_process()
            undo = self.engine.record_undo(undone_review_id)
            if undo.status == LiveUndoStatus.PAUSED_STALE:
                self.active = False
                result = self.empty_live_filtered_deck(col)
                return LiveBridgeResult(
                    status=LiveBridgeStatus.PAUSED_FOR_UNDO,
                    deck_id=result.deck_id,
                    elapsed_ms=result.elapsed_ms,
                    reviews_done=self.reviews_done,
                    transient_processed_review_count=(
                        self.transient_processed_review_count
                    ),
                    undone_review_id=undone_review_id,
                    undone_card_id=undone_card_id,
            )
            self.reviews_done = int(undo.reviews_done)
            self.current_card_id = None
            repair_undo_target = _last_undo_step(col)
            restored = self._restore_pre_answer_buffer_after_undo(
                col,
                review_id=undone_review_id,
                required_card_id=undone_card_id,
                merge_undo_target=repair_undo_target,
            )
            if restored is not None:
                return replace(
                    restored,
                    undone_review_id=undone_review_id,
                    undone_card_id=undone_card_id,
                )
            synced = self._sync_existing_buffer_after_undo(
                col,
                required_card_id=undone_card_id,
                merge_undo_target=repair_undo_target,
            )
            if synced is not None:
                return replace(
                    synced,
                    undone_review_id=undone_review_id,
                    undone_card_id=undone_card_id,
                )

            self.buffered_card_ids = []
            result = self.refill_buffer(
                col,
                stale_recheck=stale_recheck,
                merge_undo_target=repair_undo_target,
                merge_undo=repair_undo_target is not None,
            )
            return replace(
                result,
                undone_review_id=undone_review_id,
                undone_card_id=undone_card_id,
            )

        return self._result(LiveBridgeStatus.IGNORED)

    def refill_buffer(
        self,
        col,
        *,
        force_rebuild: bool = False,
        merge_undo_target: int | None = None,
        merge_undo: bool = False,
        stale_recheck: Callable[[set[int], int], bool] | None = None,
    ) -> LiveBridgeResult:
        if not self.active:
            return self._result(LiveBridgeStatus.IGNORED)
        if not self._tracked_deck_is_usable(col):
            self.active = False
            return self._result(LiveBridgeStatus.TRACKED_DECK_UNAVAILABLE)

        self._reconcile_buffered_cards(col)
        # Keep already-buffered cards if Anki still has them in the live deck.
        # This can allow one widened-threshold minimum-fill card to remain after
        # the session reaches its minimum review count. That bounded one-card
        # grace is intentional; avoiding it would add extra rebuild churn for a
        # negligible policy gain.
        skipped: list[int] = []
        skipped_set: set[int] = set()
        elapsed_ms = 0.0
        merged_undo = False
        rebuild_count = 0
        stale_recheck_calls = 0
        stale_recheck_checked_card_ids: set[int] = set()
        quiet_refresh_attempt_limit = (
            max(0, int(self.engine.settings.quiet_refresh_attempts))
            if self.engine is not None
            else 0
        )

        while True:
            proposed_card_ids = self._proposed_buffer_card_ids()
            if (
                not proposed_card_ids
                and stale_recheck is not None
                and stale_recheck_calls < quiet_refresh_attempt_limit
            ):
                stale_recheck_calls += 1
                needed_count = LIVE_FILTERED_DECK_BUFFER_SIZE
                if stale_recheck(stale_recheck_checked_card_ids, needed_count):
                    continue
                # A quiet attempt that found nothing still advances the bounded
                # scan through ``stale_recheck_checked_card_ids``. Keep trying
                # until the user-selected attempt budget is exhausted; only then
                # return EMPTY and let the GUI open its visible full recheck.
                if stale_recheck_calls < quiet_refresh_attempt_limit:
                    continue
            if not proposed_card_ids:
                break
            if proposed_card_ids == self.buffered_card_ids and not force_rebuild:
                self.refill_latencies_ms.append(elapsed_ms)
                self._remember_pending_pre_answer_undo_target(col)
                return self._refilled_result(skipped, elapsed_ms)

            started = time.perf_counter()
            rebuild_count += 1
            result = update_live_filtered_deck_with_cards(
                col,
                deck_id=self.filtered_deck_id,
                name=self.deck_name,
                card_ids=proposed_card_ids,
            )
            merged_undo = (
                _merge_undo_entries_if_requested(
                    col,
                    target=merge_undo_target,
                    merge_undo=merge_undo,
                )
                or merged_undo
            )
            elapsed_ms += (time.perf_counter() - started) * 1000
            self.filtered_deck_id = int(result.id)

            pulled_card_ids: list[int] = []
            rejected_card_ids: list[int] = []
            for candidate_id in proposed_card_ids:
                if _card_is_in_deck(col, candidate_id, self.filtered_deck_id):
                    pulled_card_ids.append(candidate_id)
                else:
                    rejected_card_ids.append(candidate_id)

            self.buffered_card_ids = pulled_card_ids[:LIVE_FILTERED_DECK_BUFFER_SIZE]
            for candidate_id in rejected_card_ids:
                if candidate_id not in skipped_set:
                    skipped.append(candidate_id)
                    skipped_set.add(candidate_id)
                if self.engine is not None:
                    self.engine.mark_card_unavailable(candidate_id)

            if (
                len(self.buffered_card_ids) >= LIVE_FILTERED_DECK_BUFFER_SIZE
                or not rejected_card_ids
            ):
                self.refill_latencies_ms.append(elapsed_ms)
                self._remember_pending_pre_answer_undo_target(col)
                return self._refilled_result(
                    skipped,
                    elapsed_ms,
                    merged_undo=merged_undo,
                )
            if rebuild_count >= LIVE_FILTERED_DECK_MAX_REFILL_BUILDS:
                self.refill_latencies_ms.append(elapsed_ms)
                if self.buffered_card_ids:
                    self._remember_pending_pre_answer_undo_target(col)
                    return self._refilled_result(
                        skipped,
                        elapsed_ms,
                        merged_undo=merged_undo,
                    )
                break

        self.active = False
        self.refill_latencies_ms.append(elapsed_ms)
        self.empty_live_filtered_deck(col)
        merged_undo = (
            _merge_undo_entries_if_requested(
                col,
                target=merge_undo_target,
                merge_undo=merge_undo,
            )
            or merged_undo
        )
        return LiveBridgeResult(
            status=LiveBridgeStatus.EMPTY,
            deck_id=self.filtered_deck_id,
            skipped_card_ids=tuple(skipped),
            elapsed_ms=elapsed_ms,
            merged_undo=merged_undo,
            reviews_done=self.reviews_done,
            transient_processed_review_count=self.transient_processed_review_count,
        )

    def empty_live_filtered_deck(self, col) -> LiveBridgeResult:
        if self.filtered_deck_id is None:
            self.buffered_card_ids = []
            return self._result(LiveBridgeStatus.EMPTY)
        started = time.perf_counter()
        if is_filtered_deck(col, self.filtered_deck_id):
            col.sched.empty_filtered_deck(int(self.filtered_deck_id))
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.buffered_card_ids = []
        self.current_card_id = None
        self.current_pre_answer_undo_target = None
        self.pending_pre_answer_undo_target = None
        return LiveBridgeResult(
            status=LiveBridgeStatus.EMPTY,
            deck_id=self.filtered_deck_id,
            elapsed_ms=elapsed_ms,
            reviews_done=self.reviews_done,
            transient_processed_review_count=self.transient_processed_review_count,
        )

    def restart_with_candidates(
        self,
        col,
        candidates: list[LiveReviewCandidate],
        *,
        merge_undo_target: int | None = None,
        merge_undo: bool = False,
        candidate_universe_prepared: bool = False,
    ) -> LiveBridgeResult:
        if self.engine is None:
            return self._result(LiveBridgeStatus.EMPTY)
        self.active = True
        self.current_card_id = None
        self.buffered_card_ids = []
        if not candidate_universe_prepared:
            self.engine.replace_candidate_universe(candidates)
        return self.refill_buffer(
            col,
            force_rebuild=True,
            merge_undo_target=merge_undo_target,
            merge_undo=merge_undo,
        )

    def close(self) -> None:
        self.active = False
        self.current_card_id = None
        self.current_pre_answer_undo_target = None
        self.pending_pre_answer_undo_target = None
        self.buffered_card_ids = []
        self.pending_card_ids = []
        self.pre_answer_buffers_by_review_id.clear()
        self.pre_answer_buffer_review_ids.clear()
        self.pre_answer_undo_targets_by_review_id.clear()
        if self.engine is not None:
            self.engine.close()
        runtime_session = self.runtime_session
        self.runtime_session = None
        close_runtime = getattr(runtime_session, "close", None)
        if close_runtime is not None:
            close_runtime()

    def diagnostics(self, col=None, *, now: float | None = None) -> dict[str, Any]:
        filtered_deck_card_ids: tuple[int, ...] = ()
        if col is not None and self.filtered_deck_id is not None:
            filtered_deck_card_ids = _card_ids_in_deck(
                col,
                self.filtered_deck_id,
                limit=LIVE_FILTERED_DECK_BUFFER_SIZE * 5,
            )
        return {
            "active": bool(self.active),
            "filtered_deck_id": self.filtered_deck_id,
            "deck_name": self.deck_name,
            "review_limit": self.review_limit,
            "reviews_done": int(self.reviews_done),
            "transient_processed_review_count": int(
                self.transient_processed_review_count
            ),
            "current_card_id": self.current_card_id,
            "buffered_card_ids": [int(card_id) for card_id in self.buffered_card_ids],
            "pending_card_count": len(self.pending_card_ids),
            "pending_card_head": _int_head(self.pending_card_ids),
            "pre_answer_buffer_snapshot_count": len(
                self.pre_answer_buffers_by_review_id,
            ),
            "filtered_deck_card_ids": [int(card_id) for card_id in filtered_deck_card_ids],
            "last_refill_latency_ms": (
                self.refill_latencies_ms[-1] if self.refill_latencies_ms else None
            ),
            "engine": self.engine.diagnostics(now=now) if self.engine is not None else None,
        }

    def retention_summary(self) -> LiveRetentionSummary:
        if self.engine is None:
            return summarize_live_retention_records(())
        return self.engine.retention_summary()

    def _release_unshown_buffered_cards(self) -> None:
        if not self.buffered_card_ids:
            return
        released = tuple(self.buffered_card_ids)
        self.buffered_card_ids = []
        if self.engine is not None:
            self.engine.release_unshown_emitted_cards(released)

    def _proposed_buffer_card_ids(self) -> list[int]:
        proposed = list(self.buffered_card_ids)
        needed_count = LIVE_FILTERED_DECK_BUFFER_SIZE - len(proposed)
        if needed_count <= 0:
            return proposed
        excluded = set(proposed)
        if self.engine is not None:
            for card_id in self.engine.ready_queue:
                normalized = int(card_id)
                if normalized in excluded:
                    continue
                proposed.append(normalized)
                excluded.add(normalized)
                if len(proposed) >= LIVE_FILTERED_DECK_BUFFER_SIZE:
                    return proposed
            needed_count = LIVE_FILTERED_DECK_BUFFER_SIZE - len(proposed)
            result = self.engine.next_buffer(needed_count)
            if result.status == LiveSelectionStatus.INACTIVE:
                self.active = False
                return proposed
            if result.status == LiveSelectionStatus.READY:
                for card_id in result.card_ids:
                    normalized = int(card_id)
                    if normalized in excluded:
                        continue
                    proposed.append(normalized)
                    excluded.add(normalized)
            return proposed
        while len(proposed) < LIVE_FILTERED_DECK_BUFFER_SIZE:
            candidate_id = self._next_candidate_excluding(excluded)
            if candidate_id is None:
                break
            proposed.append(candidate_id)
            excluded.add(candidate_id)
        return proposed

    def _next_candidate_excluding(self, excluded: set[int]) -> int | None:
        while self.pending_card_ids:
            candidate_id = int(self.pending_card_ids.pop(0))
            if candidate_id not in excluded:
                return candidate_id
        return None

    def _forget_buffered_card(self, card_id: int) -> bool:
        before = tuple(self.buffered_card_ids)
        self.buffered_card_ids = [
            buffered_id
            for buffered_id in self.buffered_card_ids
            if buffered_id != int(card_id)
        ]
        return tuple(self.buffered_card_ids) != before

    def _reconcile_buffered_cards(self, col) -> None:
        if self.filtered_deck_id is None:
            return
        retained: list[int] = []
        for card_id in self.buffered_card_ids:
            if _card_is_in_deck(col, card_id, self.filtered_deck_id):
                retained.append(card_id)
            elif self.engine is not None:
                # A card can disappear from the live filtered deck without being
                # answered, most commonly via bury/suspend. We conservatively
                # mark it unavailable instead of trying to repair that operation
                # inside suspend/bury hooks. A future improvement could add a
                # targeted post-operation refill/undo repair path, but repeated
                # bury/suspend remains a known edge case for now.
                self.engine.mark_card_unavailable(card_id)
        self.buffered_card_ids = retained

    def _remember_pre_answer_buffer(
        self,
        review_id: int | None,
        buffer_snapshot: tuple[int, ...],
        *,
        pre_answer_undo_target: int | None,
    ) -> None:
        if review_id is None:
            return
        snapshot = tuple(
            _dedupe_card_ids(list(buffer_snapshot))[:LIVE_FILTERED_DECK_BUFFER_SIZE],
        )
        if not snapshot:
            return
        normalized = int(review_id)
        if normalized not in self.pre_answer_buffers_by_review_id:
            self.pre_answer_buffer_review_ids.append(normalized)
        self.pre_answer_buffers_by_review_id[normalized] = snapshot
        normalized_target = _positive_int_or_none(pre_answer_undo_target)
        if normalized_target is not None:
            self.pre_answer_undo_targets_by_review_id[normalized] = normalized_target
        while len(self.pre_answer_buffer_review_ids) > LIVE_REVIEW_RUST_UNDO_LIMIT:
            stale_review_id = self.pre_answer_buffer_review_ids.pop(0)
            self.pre_answer_buffers_by_review_id.pop(stale_review_id, None)
            self.pre_answer_undo_targets_by_review_id.pop(stale_review_id, None)

    def _pop_pre_answer_buffer(
        self,
        review_id: int,
    ) -> tuple[tuple[int, ...], int | None] | None:
        normalized = int(review_id)
        snapshot = self.pre_answer_buffers_by_review_id.pop(normalized, None)
        undo_target = self.pre_answer_undo_targets_by_review_id.pop(normalized, None)
        self.pre_answer_buffer_review_ids = [
            stored_id
            for stored_id in self.pre_answer_buffer_review_ids
            if stored_id != normalized
        ]
        if snapshot is None:
            return None
        return snapshot, undo_target

    def _restore_pre_answer_buffer_after_undo(
        self,
        col,
        *,
        review_id: int,
        required_card_id: int | None,
        merge_undo_target: int | None = None,
    ) -> LiveBridgeResult | None:
        pre_answer = self._pop_pre_answer_buffer(review_id)
        if pre_answer is None or self.filtered_deck_id is None:
            return None
        snapshot, saved_undo_target = pre_answer
        effective_merge_target = saved_undo_target or merge_undo_target
        restored = self._refill_after_undo_with_required_first(
            col,
            required_card_id=required_card_id,
            merge_undo_target=effective_merge_target,
        )
        if restored is not None:
            return restored

        ordered = list(snapshot)
        if required_card_id is not None:
            ordered = _dedupe_card_ids([int(required_card_id), *ordered])
        ordered = ordered[:LIVE_FILTERED_DECK_BUFFER_SIZE]
        if not ordered:
            return None

        rebuilt = self._rebuild_undo_deck_with_cards(
            col,
            ordered,
            merge_undo_target=effective_merge_target,
        )
        if rebuilt is None:
            return None
        pulled, elapsed_ms, merged_undo = rebuilt
        if required_card_id is not None and int(required_card_id) not in pulled:
            return None
        return self._accept_undo_buffer(
            col,
            pulled,
            required_card_id=required_card_id,
            elapsed_ms=elapsed_ms,
            merged_undo=merged_undo,
            merge_undo_target=effective_merge_target,
        )

    def _sync_existing_buffer_after_undo(
        self,
        col,
        *,
        required_card_id: int | None,
        merge_undo_target: int | None = None,
    ) -> LiveBridgeResult | None:
        if self.filtered_deck_id is None:
            return None
        existing = _card_ids_in_deck(
            col,
            self.filtered_deck_id,
            limit=LIVE_FILTERED_DECK_BUFFER_SIZE,
        )
        if not existing:
            return None
        ordered = list(existing)
        if required_card_id is not None and int(required_card_id) not in existing:
            return None
        if required_card_id is not None:
            # After Anki undoes an answer, the restored filtered deck can contain
            # the undone card without making it the next card. Force the undone
            # card to the front so Undo returns the user to the card they just
            # reverted instead of advancing to another buffered card.
            ordered = _dedupe_card_ids([int(required_card_id), *ordered])
            restored = self._refill_after_undo_with_required_first(
                col,
                required_card_id=required_card_id,
                merge_undo_target=merge_undo_target,
            )
            if restored is not None:
                return restored

        rebuilt = self._rebuild_undo_deck_with_cards(
            col,
            ordered,
            merge_undo_target=merge_undo_target,
        )
        if rebuilt is None:
            return None
        pulled, elapsed_ms, merged_undo = rebuilt
        if required_card_id is not None and int(required_card_id) not in pulled:
            return None
        return self._accept_undo_buffer(
            col,
            pulled,
            required_card_id=required_card_id,
            elapsed_ms=elapsed_ms,
            merged_undo=merged_undo,
            merge_undo_target=merge_undo_target,
        )

    def _accept_undo_buffer(
        self,
        col,
        pulled: list[int],
        *,
        required_card_id: int | None,
        elapsed_ms: float,
        merged_undo: bool,
        merge_undo_target: int | None,
    ) -> LiveBridgeResult | None:
        restored = pulled[:LIVE_FILTERED_DECK_BUFFER_SIZE]
        if self.engine is not None and not self.engine.sync_existing_buffer(restored):
            required = None if required_card_id is None else int(required_card_id)
            if required is None or required not in restored:
                return None
            if not self.engine.sync_existing_buffer([required]):
                return None
            # The old second buffered card may no longer be below DR after the
            # answer-time refresh. Undo only needs the undone card first; refill
            # the second slot from current valid candidates instead of requiring
            # the exact pre-answer second card to survive.
            restored = [required]

        self.buffered_card_ids = restored
        if len(self.buffered_card_ids) < LIVE_FILTERED_DECK_BUFFER_SIZE:
            refill = self.refill_buffer(
                col,
                force_rebuild=True,
                merge_undo_target=merge_undo_target,
                merge_undo=merge_undo_target is not None,
            )
            return replace(
                refill,
                elapsed_ms=elapsed_ms + refill.elapsed_ms,
                merged_undo=merged_undo or refill.merged_undo,
            )

        self.refill_latencies_ms.append(elapsed_ms)
        self._remember_pending_pre_answer_undo_target(col)
        return self._refilled_result([], elapsed_ms, merged_undo=merged_undo)

    def _rebuild_undo_deck_with_cards(
        self,
        col,
        ordered_card_ids: list[int],
        *,
        merge_undo_target: int | None,
    ) -> tuple[list[int], float, bool] | None:
        if self.filtered_deck_id is None:
            return None
        started = time.perf_counter()
        result = update_live_filtered_deck_with_cards(
            col,
            deck_id=self.filtered_deck_id,
            name=self.deck_name,
            card_ids=ordered_card_ids,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.filtered_deck_id = int(result.id)
        merged_undo = _merge_undo_entries_if_requested(
            col,
            target=merge_undo_target,
            merge_undo=merge_undo_target is not None,
        )
        pulled = [
            card_id
            for card_id in ordered_card_ids
            if _card_is_in_deck(col, card_id, self.filtered_deck_id)
        ]
        return pulled, elapsed_ms, merged_undo

    def _refill_after_undo_with_required_first(
        self,
        col,
        *,
        required_card_id: int | None,
        merge_undo_target: int | None,
    ) -> LiveBridgeResult | None:
        if required_card_id is None or self.engine is None:
            return None
        required = int(required_card_id)
        if not self.engine.restore_existing_buffer([required]):
            return None

        proposed = _dedupe_card_ids([required, *self.buffered_card_ids])
        while len(proposed) < LIVE_FILTERED_DECK_BUFFER_SIZE:
            result = self.engine.next_buffer(
                LIVE_FILTERED_DECK_BUFFER_SIZE - len(proposed)
            )
            if result.status == LiveSelectionStatus.INACTIVE:
                self.active = False
                break
            if result.status != LiveSelectionStatus.READY:
                break
            previous_len = len(proposed)
            proposed = _dedupe_card_ids([*proposed, *result.card_ids])
            if len(proposed) == previous_len:
                break
        proposed = proposed[:LIVE_FILTERED_DECK_BUFFER_SIZE]

        rebuilt = self._rebuild_undo_deck_with_cards(
            col,
            proposed,
            merge_undo_target=merge_undo_target,
        )
        if rebuilt is None:
            return None
        pulled, elapsed_ms, merged_undo = rebuilt
        if required not in pulled:
            return None

        self.buffered_card_ids = pulled[:LIVE_FILTERED_DECK_BUFFER_SIZE]
        if not self.engine.sync_existing_buffer(self.buffered_card_ids):
            return None
        self.refill_latencies_ms.append(elapsed_ms)
        self._remember_pending_pre_answer_undo_target(col)
        return self._refilled_result([], elapsed_ms, merged_undo=merged_undo)

    def _remember_pending_pre_answer_undo_target(self, col) -> None:
        self.pending_pre_answer_undo_target = _last_undo_step(col)

    def _record_engine_answer(
        self,
        card_id: int,
        *,
        review_row: dict[str, Any] | None,
    ) -> LiveAnswerResult:
        if self.engine is None:
            self.reviews_done += 1
            return LiveAnswerResult(
                status=LiveAnswerStatus.RECORDED,
                card_id=int(card_id),
                reviews_done=self.reviews_done,
            )
        answer = self.engine.record_answer(
            card_id,
            review_row=review_row,
        )
        self.reviews_done = int(answer.reviews_done)
        return answer

    def _process_answer_review_row(
        self,
        row: dict[str, Any],
        *,
        process_review_row: Callable[[dict[str, Any]], tuple[float, Any]] | None,
        curve_probability_func: Callable[[Any, float], float] | None,
        clock: Callable[[], float] | None,
    ) -> None:
        if process_review_row is None:
            if self.engine is not None:
                self.engine.discard_hot_release(row)
            return
        del clock, curve_probability_func
        process_review_row(dict(row))
        self.transient_processed_review_count += 1
        if self.engine is not None and not self.engine.release_processed_hot_card(row):
            self.engine.discard_hot_release(row)

    def _tracked_deck_is_usable(self, col) -> bool:
        if self.filtered_deck_id is None:
            existing_deck_id = _deck_id_for_name(col, self.deck_name)
            if existing_deck_id is not None:
                if not is_filtered_deck(col, existing_deck_id):
                    raise ValueError(
                        "A normal deck already exists with the RWKV live filtered deck name."
                    )
                self.filtered_deck_id = existing_deck_id
            return True
        return is_filtered_deck(col, self.filtered_deck_id)

    def _result(self, status: LiveBridgeStatus) -> LiveBridgeResult:
        return LiveBridgeResult(
            status=status,
            deck_id=self.filtered_deck_id,
            reviews_done=self.reviews_done,
            transient_processed_review_count=self.transient_processed_review_count,
        )

    def _refilled_result(
        self,
        skipped: list[int],
        elapsed_ms: float,
        *,
        merged_undo: bool = False,
    ) -> LiveBridgeResult:
        first_card_id = self.buffered_card_ids[0]
        return LiveBridgeResult(
            status=LiveBridgeStatus.REFILLED,
            deck_id=self.filtered_deck_id,
            selected_card_id=first_card_id,
            requested_card_id=first_card_id,
            buffered_card_ids=tuple(self.buffered_card_ids),
            skipped_card_ids=tuple(skipped),
            elapsed_ms=elapsed_ms,
            merged_undo=merged_undo,
            reviews_done=self.reviews_done,
            transient_processed_review_count=self.transient_processed_review_count,
        )


def create_live_review_bridge_with_candidates(
    candidates: list[LiveReviewCandidate],
    *,
    deck_name: str = LIVE_FILTERED_DECK_NAME,
    review_limit: int | None = None,
    settings: LiveReviewSettings | None = None,
    fsrs_comparison_enabled: bool = True,
) -> LiveReviewBridgeSession:
    """Create a bridge without selecting cards or mutating Anki's collection."""

    engine_settings = settings or LiveReviewSettings(review_limit=review_limit)
    if review_limit is not None and engine_settings.review_limit != review_limit:
        engine_settings = replace(engine_settings, review_limit=review_limit)
    engine = LiveReviewEngine.from_candidates(
        candidates,
        settings=engine_settings,
    )
    return LiveReviewBridgeSession(
        engine=engine,
        deck_name=deck_name,
        review_limit=engine_settings.review_limit,
        fsrs_comparison_enabled=fsrs_comparison_enabled,
    )


def update_live_filtered_deck_with_card(
    col,
    *,
    deck_id: int | None,
    name: str,
    card_id: int,
):
    return update_live_filtered_deck_with_cards(
        col,
        deck_id=deck_id,
        name=name,
        card_ids=[int(card_id)],
    )


def update_live_filtered_deck_with_cards(
    col,
    *,
    deck_id: int | None,
    name: str,
    card_ids: list[int],
):
    if not card_ids:
        raise ValueError("No cards were selected for the live filtered deck.")
    existing_deck_id = _deck_id_for_name(col, name)
    if deck_id is None and existing_deck_id is not None:
        if not is_filtered_deck(col, existing_deck_id):
            raise ValueError(
                "A normal deck already exists with the RWKV live filtered deck name."
            )
        deck_id = existing_deck_id

    if deck_id is not None and not is_filtered_deck(col, int(deck_id)):
        raise ValueError("The tracked RWKV live deck is no longer a filtered deck.")

    deck = col.sched.get_or_create_filtered_deck(deck_id=int(deck_id or 0))
    deck.name = name
    deck.allow_empty = True
    del deck.config.search_terms[:]
    term_type = type(deck.config).SearchTerm
    deck.config.search_terms.extend(
        [
            term_type(
                search=cid_search([int(card_id)]),
                limit=1,
                order=LIVE_FILTERED_DECK_ORDER,
            )
            for card_id in card_ids[:LIVE_FILTERED_DECK_BUFFER_SIZE]
        ]
    )
    deck.config.reschedule = True
    return col.sched.add_or_update_filtered_deck(deck)


def _deck_id_for_name(col, name: str) -> int | None:
    try:
        deck_id = col.decks.id_for_name(name)
    except Exception:
        return None
    return int(deck_id) if deck_id else None


def _card_is_in_deck(col, card_id: int, deck_id: int) -> bool:
    try:
        card = col.get_card(int(card_id))
    except Exception:
        return False
    return _int_attr(card, "did") == int(deck_id) and (_int_attr(card, "queue") or 0) >= 0


def _int_head(values: list[int] | tuple[int, ...], *, limit: int = 20) -> list[int]:
    return [int(value) for value in values[: int(limit)]]


def _dedupe_card_ids(values: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        normalized = int(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _card_id(card: Any) -> int:
    return int(card.id)


def _int_attr(obj: Any, name: str) -> int | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _last_undo_step(col) -> int | None:
    try:
        step = getattr(col.undo_status(), "last_step", None)
    except Exception:
        return None
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def _merge_undo_entries(col, target: int) -> bool:
    try:
        col.merge_undo_entries(int(target))
    except Exception:
        return False
    return True


def _merge_undo_entries_if_requested(
    col,
    *,
    target: int | None,
    merge_undo: bool,
) -> bool:
    return bool(
        merge_undo
        and target is not None
        and _merge_undo_entries(col, int(target))
    )


def _first_missing_review_id(col, review_ids) -> int | None:
    ids = [int(review_id) for review_id in review_ids]
    if not ids:
        return None
    existing = _existing_review_ids(col, ids)
    for review_id in ids:
        if review_id not in existing:
            return review_id
    return None


def _card_id_for_undoable_review_id(
    rows: tuple[dict[str, Any], ...],
    review_id: int,
) -> int | None:
    normalized_review_id = int(review_id)
    for row in rows:
        try:
            if int(row["review_id"]) == normalized_review_id:
                return int(row["card_id"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _existing_review_ids(col, review_ids: list[int]) -> set[int]:
    existing: set[int] = set()
    for chunk in _chunks(review_ids, 500):
        try:
            rows = col.db.list(
                f"select id from revlog where id in {_sql_id_list(chunk)}"
            )
        except Exception:
            return set(review_ids)
        existing.update(int(row) for row in rows)
    return existing


def _sql_id_list(ids: list[int]) -> str:
    return "(" + ",".join(str(int(value)) for value in ids) + ")"


def _chunks(values: list[int], chunk_size: int):
    for start in range(0, len(values), max(1, int(chunk_size))):
        yield values[start : start + chunk_size]


def _card_ids_in_deck(
    col,
    deck_id: int,
    *,
    limit: int,
) -> tuple[int, ...]:
    try:
        rows = col.db.list(
            """
            select id
            from cards
            where did = ?
              and queue >= 0
            order by due, id
            limit ?
            """,
            int(deck_id),
            int(limit),
        )
    except Exception:
        return ()
    return tuple(int(row) for row in rows)


def _with_merged_undo(result: LiveBridgeResult) -> LiveBridgeResult:
    return replace(result, merged_undo=True)


def _with_merged_undo_if_possible(
    col,
    result: LiveBridgeResult,
    *,
    target: int | None,
    merge_undo: bool,
) -> LiveBridgeResult:
    mergeable_statuses = {
        LiveBridgeStatus.REFILLED,
        LiveBridgeStatus.EMPTY,
        LiveBridgeStatus.STOPPED_LIMIT,
    }
    if (
        not result.merged_undo
        and merge_undo
        and target is not None
        and result.status in mergeable_statuses
        and _merge_undo_entries(col, target)
    ):
        return _with_merged_undo(result)
    return result
