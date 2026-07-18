from __future__ import annotations

import heapq
import math
import random
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, TypeVar

from .adaptive_retention import (
    AdaptiveRetentionSettings,
    active_desired_retention_with_adaptive,
)
from .constants import LIVE_REVIEW_RUST_UNDO_LIMIT
from .filtered_deck_sort import (
    FilteredDeckOrder,
    FilteredDeckSortInfo,
)
from .live_review_stats import (
    LiveRetentionRecord,
    LiveRetentionSummary,
    retention_record_for_answer,
    summarize_live_retention_records,
)
from .review_load_policy import (
    DEFAULT_MINIMUM_RETENTION_STEP,
    is_same_day_elapsed,
    prediction_below_retention,
)

DEFAULT_HOT_PREDICT_LIMIT = 50
DEFAULT_HOT_CURVE_OFFSETS_SECONDS = (10.0, 60.0, 600.0)
DEFAULT_HOT_CURVE_MARGIN = 0.10
DEFAULT_HOT_COLD_RECHECK_SECONDS = 3600.0
DEFAULT_PREDICTION_REFRESH_LIMIT = 247
DEFAULT_QUIET_REFRESH_ATTEMPTS = 5
DEFAULT_SAME_DAY_REENTRY_DELAY_REVIEWS = 2
_MISSING = object()
_T = TypeVar("_T")


class LiveSelectionStatus(Enum):
    READY = "ready"
    EMPTY = "empty"
    INACTIVE = "inactive"


class LiveAnswerStatus(Enum):
    RECORDED = "recorded"
    IGNORED = "ignored"
    STOPPED_LIMIT = "stopped_limit"


class LiveUndoStatus(Enum):
    HANDLED_UNDOABLE = "handled_undoable"
    PAUSED_STALE = "paused_stale"
    IGNORED = "ignored"


class LivePredictionStatus(Enum):
    APPLIED = "applied"
    STALE = "stale"
    INACTIVE = "inactive"


@dataclass(frozen=True)
class LiveReviewSettings:
    hot_predict_limit: int = DEFAULT_HOT_PREDICT_LIMIT
    hot_curve_offsets_seconds: tuple[float, ...] = DEFAULT_HOT_CURVE_OFFSETS_SECONDS
    hot_curve_margin: float = DEFAULT_HOT_CURVE_MARGIN
    hot_cold_recheck_seconds: float = DEFAULT_HOT_COLD_RECHECK_SECONDS
    prediction_refresh_limit: int = DEFAULT_PREDICTION_REFRESH_LIMIT
    quiet_refresh_attempts: int = DEFAULT_QUIET_REFRESH_ATTEMPTS
    review_limit: int | None = None
    minimum_review_limit: int = 0
    minimum_retention_extra_quantum: float = DEFAULT_MINIMUM_RETENTION_STEP
    allow_same_day_repeats: bool = False
    same_day_reentry_delay_reviews: int = DEFAULT_SAME_DAY_REENTRY_DELAY_REVIEWS
    order_index: int = int(FilteredDeckOrder.RETRIEVABILITY_ASCENDING)
    adaptive_retention: AdaptiveRetentionSettings | None = None

    def __post_init__(self) -> None:
        _require_non_negative("hot_predict_limit", self.hot_predict_limit)
        _require_non_negative("hot_curve_margin", self.hot_curve_margin)
        _require_non_negative(
            "hot_cold_recheck_seconds",
            self.hot_cold_recheck_seconds,
        )
        _require_non_negative("prediction_refresh_limit", self.prediction_refresh_limit)
        _require_non_negative("quiet_refresh_attempts", self.quiet_refresh_attempts)
        _require_non_negative("order_index", self.order_index)
        _require_non_negative("minimum_review_limit", self.minimum_review_limit)
        _require_non_negative(
            "same_day_reentry_delay_reviews",
            self.same_day_reentry_delay_reviews,
        )
        _require_non_negative(
            "minimum_retention_extra_quantum",
            self.minimum_retention_extra_quantum,
        )
        for offset in self.hot_curve_offsets_seconds:
            _require_non_negative("hot_curve_offsets_seconds", offset)
        if self.review_limit is not None:
            _require_non_negative("review_limit", self.review_limit)
            if int(self.minimum_review_limit) > int(self.review_limit):
                raise ValueError("minimum_review_limit must not exceed review_limit.")


@dataclass(frozen=True)
class LiveReviewCandidate:
    card_id: int
    source_deck_id: int | None
    desired_retention: float
    predicted_retrievability: float
    same_day_desired_retention: float | None = None
    rwkv_stability_days: float | None = None
    fsrs_difficulty: float | None = None
    adaptive_retention: AdaptiveRetentionSettings | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sort_info: FilteredDeckSortInfo | None = None

    @property
    def active_desired_retention(self) -> float:
        return active_desired_retention_with_adaptive(
            self.desired_retention,
            self.same_day_desired_retention,
            elapsed_days=self.metadata.get("elapsed_days", -1),
            adaptive_settings=self.adaptive_retention,
            rwkv_stability_days=self.rwkv_stability_days,
            fsrs_difficulty=self.fsrs_difficulty,
        )

    @property
    def hot_desired_retention(self) -> float:
        return active_desired_retention_with_adaptive(
            self.desired_retention,
            self.same_day_desired_retention,
            elapsed_days=0.0,
            adaptive_settings=self.adaptive_retention,
            rwkv_stability_days=self.rwkv_stability_days,
            fsrs_difficulty=self.fsrs_difficulty,
        )

    @property
    def eligible(self) -> bool:
        prediction = float(self.predicted_retrievability)
        desired = float(self.active_desired_retention)
        return math.isfinite(prediction) and math.isfinite(desired) and prediction < desired


@dataclass(frozen=True)
class LiveHotCard:
    card_id: int
    predicted_retrievability: float | None = None
    desired_retention: float = 0.9
    next_check_at: float | None = None
    release_after_review_count: int = 0
    native_include_pending: bool = False


@dataclass(frozen=True)
class _LiveHotRelease:
    candidate: LiveReviewCandidate


class _LiveCandidateTable:
    """Slot-aligned dynamic state for the Live Review candidate universe.

    ``LiveReviewCandidate`` remains the public immutable value type, but its
    prediction timestamp and elapsed fields change for thousands of cards on
    every GPU refresh.  Keeping those values here avoids mutating a frozen
    dataclass and its metadata dictionary for every result.  Boundary methods
    materialize an up-to-date candidate only when the caller actually needs
    one.
    """

    __slots__ = (
        "below_retention",
        "candidates",
        "elapsed_days",
        "prediction_timestamps",
        "predictions",
        "slot_by_card_id",
    )

    def __init__(self) -> None:
        self.slot_by_card_id: dict[int, int] = {}
        self.candidates: list[LiveReviewCandidate] = []
        self.predictions: list[float] = []
        self.elapsed_days: list[float] = []
        self.prediction_timestamps: list[float | None] = []
        self.below_retention: list[bool] = []

    def rebuild(self, candidates: Mapping[int, LiveReviewCandidate]) -> None:
        self.clear()
        for card_id, candidate in candidates.items():
            self._append(int(card_id), candidate)

    def clear(self) -> None:
        self.slot_by_card_id.clear()
        self.candidates.clear()
        self.predictions.clear()
        self.elapsed_days.clear()
        self.prediction_timestamps.clear()
        self.below_retention.clear()

    def upsert(self, candidate: LiveReviewCandidate) -> int:
        card_id = int(candidate.card_id)
        slot = self.slot_by_card_id.get(card_id)
        if slot is None:
            return self._append(card_id, candidate)
        self.candidates[slot] = candidate
        self.predictions[slot] = float(candidate.predicted_retrievability)
        self.elapsed_days[slot] = _candidate_elapsed_days(candidate)
        self.prediction_timestamps[slot] = _candidate_prediction_timestamp(candidate)
        self.below_retention[slot] = bool(candidate.eligible)
        return slot

    def apply_prediction(
        self,
        card_id: int,
        *,
        prediction: float,
        elapsed_days: float,
        target_timestamp_seconds: float,
    ) -> int | None:
        slot = self.slot_by_card_id.get(int(card_id))
        if slot is None:
            return None
        self.predictions[slot] = float(prediction)
        self.elapsed_days[slot] = float(elapsed_days)
        self.prediction_timestamps[slot] = float(target_timestamp_seconds)
        return slot

    def prediction_for(self, card_id: int, default: float = math.nan) -> float:
        slot = self.slot_by_card_id.get(int(card_id))
        if slot is None:
            return float(default)
        return self.predictions[slot]

    def elapsed_days_for(self, card_id: int, default: float = -1.0) -> float:
        slot = self.slot_by_card_id.get(int(card_id))
        if slot is None:
            return float(default)
        return self.elapsed_days[slot]

    def materialize(
        self,
        candidate: LiveReviewCandidate,
    ) -> LiveReviewCandidate:
        slot = self.slot_by_card_id.get(int(candidate.card_id))
        if slot is None or self.prediction_timestamps[slot] is None:
            return candidate
        metadata = dict(candidate.metadata)
        metadata["elapsed_days"] = self.elapsed_days[slot]
        timestamp = self.prediction_timestamps[slot]
        if timestamp is not None:
            metadata["prediction_timestamp_seconds"] = timestamp
        return replace(
            candidate,
            predicted_retrievability=self.predictions[slot],
            metadata=metadata,
        )

    def _append(self, card_id: int, candidate: LiveReviewCandidate) -> int:
        slot = len(self.candidates)
        self.slot_by_card_id[card_id] = slot
        self.candidates.append(candidate)
        self.predictions.append(float(candidate.predicted_retrievability))
        self.elapsed_days.append(_candidate_elapsed_days(candidate))
        self.prediction_timestamps.append(_candidate_prediction_timestamp(candidate))
        self.below_retention.append(bool(candidate.eligible))
        return slot


@dataclass(frozen=True)
class LivePredictionToken:
    session_generation: int
    prediction_generation: int
    candidate_card_ids: tuple[int, ...]


@dataclass(frozen=True)
class LiveSelectionResult:
    status: LiveSelectionStatus
    card_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class LiveAnswerResult:
    status: LiveAnswerStatus
    card_id: int | None
    review_id: int | None = None
    reviews_done: int = 0
    undoable_review_count: int = 0
    transient_applied_review_count: int = 0


@dataclass(frozen=True)
class LiveUndoResult:
    status: LiveUndoStatus
    review_id: int | None
    reviews_done: int
    undoable_review_count: int
    needs_targeted_repair: bool = False


@dataclass(frozen=True)
class LivePredictionResult:
    status: LivePredictionStatus
    applied_count: int = 0


@dataclass(frozen=True)
class LiveNativePredictionValue:
    """One compact native-session value mirrored for add-on-only policy."""

    card_id: int
    predicted_retrievability: float
    elapsed_days: float


@dataclass(frozen=True)
class _LiveCandidateUniverseUndoSnapshot:
    candidates: tuple[LiveReviewCandidate, ...]
    transport_card_ids: tuple[int, ...]
    hot_registry: dict[int, LiveHotCard]
    emitted_card_ids: set[int]
    emitted_card_order: tuple[int, ...]
    unavailable_card_ids: set[int]
    hot_candidate_templates_by_card: dict[int, LiveReviewCandidate]
    random_sort_keys: dict[int, float]
    minimum_retention_extra: float
    native_selection_card_ids: tuple[int, ...]
    native_refreshed_count: int


@dataclass
class LiveReviewEngine:
    settings: LiveReviewSettings = field(default_factory=LiveReviewSettings)
    active: bool = True
    reviews_done: int = 0
    session_generation: int = 0
    prediction_generation: int = 0
    transient_applied_review_count: int = 0
    _candidates_by_card: dict[int, LiveReviewCandidate] = field(default_factory=dict)
    _candidate_table: _LiveCandidateTable = field(default_factory=_LiveCandidateTable)
    _hot_registry: dict[int, LiveHotCard] = field(default_factory=dict)
    _answered_card_ids: set[int] = field(default_factory=set)
    _emitted_card_ids: set[int] = field(default_factory=set)
    _emitted_card_order: list[int] = field(default_factory=list)
    _unavailable_card_ids: set[int] = field(default_factory=set)
    _undoable_review_rows: deque[dict[str, Any]] = field(default_factory=deque)
    _undoable_candidates_by_review_id: dict[int, LiveReviewCandidate | None] = field(
        default_factory=dict
    )
    _minimum_retention_extra_by_review_id: dict[int, float] = field(
        default_factory=dict
    )
    _candidate_universe_snapshots_by_review_id: dict[
        int, _LiveCandidateUniverseUndoSnapshot
    ] = field(default_factory=dict)
    _hot_releases_by_review_id: dict[int, _LiveHotRelease] = field(default_factory=dict)
    _native_hot_includes_by_review_id: dict[int, set[int]] = field(default_factory=dict)
    _answered_desired_retention_by_card: dict[int, float] = field(default_factory=dict)
    # Answered same-day cards are removed from normal candidates while their
    # reviews remain undoable. Keep the original candidate so immediate
    # undoable_process() can release the card back into hot prediction after
    # RWKV has actually seen the review.
    _hot_candidate_templates_by_card: dict[int, LiveReviewCandidate] = field(default_factory=dict)
    _shown_candidates_by_card: dict[int, LiveReviewCandidate] = field(default_factory=dict)
    _shown_fsrs_predictions_by_card: dict[int, float] = field(default_factory=dict)
    _retention_records_by_review_id: dict[int, LiveRetentionRecord] = field(default_factory=dict)
    _skipped_retention_review_ids: set[int] = field(default_factory=set)
    _random_sort_keys: dict[int, float] = field(default_factory=dict)
    _candidate_transport_order_index: tuple[int, ...] = ()
    stale_recheck_batches: int = 0
    stale_recheck_checked_count: int = 0
    stale_recheck_eligible_count: int = 0
    minimum_retention_extra: float = 0.0
    needs_targeted_repair: bool = False
    _native_selection_enabled: bool = False
    _native_selection_card_ids: tuple[int, ...] = ()
    _native_refreshed_count: int = 0

    @classmethod
    def from_candidates(
        cls,
        candidates: Iterable[LiveReviewCandidate],
        *,
        settings: LiveReviewSettings | None = None,
    ) -> LiveReviewEngine:
        engine = cls(settings=settings or LiveReviewSettings())
        engine.replace_candidate_universe(candidates)
        if engine._review_limit_reached():
            engine.close()
        return engine

    @property
    def ready_queue(self) -> tuple[int, ...]:
        return tuple(
            card_id
            for card_id in self._emitted_card_order
            if card_id in self._emitted_card_ids
            and self._card_is_available_below_active_retention(card_id)
        )

    @property
    def hot_queue(self) -> tuple[int, ...]:
        return tuple(hot.card_id for hot in self.hot_registry)

    @property
    def hot_registry(self) -> tuple[LiveHotCard, ...]:
        return tuple(
            sorted(
                self._hot_registry.values(),
                key=self._hot_prediction_sort_key,
            )
        )

    @property
    def backlog_queue(self) -> tuple[int, ...]:
        return self._selectable_universe_card_ids()

    @property
    def stale_queue(self) -> tuple[int, ...]:
        return self._stale_universe_card_ids()

    @property
    def undoable_review_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._undoable_review_rows)

    @property
    def undoable_review_ids(self) -> tuple[int, ...]:
        return tuple(
            review_id
            for row in self._undoable_review_rows
            if (review_id := _review_id_from_row(row)) is not None
        )

    @property
    def undoable_review_count(self) -> int:
        return len(self._undoable_review_rows)

    def retention_summary(self) -> LiveRetentionSummary:
        return summarize_live_retention_records(
            self._retention_records_by_review_id.values(),
            skipped_count=len(self._skipped_retention_review_ids),
        )

    def snapshot_shown_candidate(
        self,
        card_id: int,
        *,
        fsrs_prediction: float | None = None,
    ) -> bool:
        if not self.active:
            return False
        normalized = int(card_id)
        candidate = self._candidates_by_card.get(
            normalized,
        ) or self._hot_candidate_templates_by_card.get(normalized)
        if candidate is None:
            return False
        candidate = self._candidate_table.materialize(candidate)
        # Coordinator-owned candidates are updated in place after a successful
        # prediction generation. Preserve the exact values that were visible
        # when this card was shown for retention reporting.
        self._shown_candidates_by_card[normalized] = replace(
            candidate,
            metadata=dict(candidate.metadata),
        )
        fsrs_prediction_value = _finite_float_or_none(fsrs_prediction)
        if fsrs_prediction_value is None:
            self._shown_fsrs_predictions_by_card.pop(normalized, None)
        else:
            self._shown_fsrs_predictions_by_card[normalized] = fsrs_prediction_value
        return True

    def diagnostics(self, *, now: float | None = None) -> dict[str, Any]:
        """Return a compact queue snapshot for investigating live-review exhaustion."""
        del now
        selectable_ids = [
            card_id
            for card_id in self._candidates_by_card
            if self._card_is_available_below_active_retention(card_id)
        ]
        ready_selectable = list(self.ready_queue)
        backlog_selectable = list(self.backlog_queue)
        stale_ids = list(self.stale_queue)
        hot_needs_prediction = [
            hot.card_id
            for hot in self.hot_registry
            if hot.card_id not in self._emitted_card_ids
            and hot.card_id not in self._unavailable_card_ids
            and _finite_float_or_none(hot.predicted_retrievability) is None
        ]
        return {
            "active": bool(self.active),
            "reviews_done": int(self.reviews_done),
            "review_limit": self.settings.review_limit,
            "minimum_review_limit": self.settings.minimum_review_limit,
            "minimum_retention_extra": float(self.minimum_retention_extra),
            "undoable_review_count": int(self.undoable_review_count),
            "transient_applied_review_count": int(self.transient_applied_review_count),
            "candidate_count": len(self._candidates_by_card),
            "selectable_candidate_count": len(selectable_ids),
            "ineligible_candidate_count": max(
                0,
                len(self._candidates_by_card) - len(selectable_ids),
            ),
            "ready_queue_count": len(ready_selectable),
            "ready_selectable_count": len(ready_selectable),
            "backlog_queue_count": len(backlog_selectable),
            "backlog_selectable_count": len(backlog_selectable),
            "stale_queue_count": len(stale_ids),
            "hot_registry_count": len(self._hot_registry),
            "hot_needs_prediction_count": len(hot_needs_prediction),
            "hot_release_count": len(self._hot_releases_by_review_id),
            "answered_count": len(self._answered_card_ids),
            "emitted_count": len(self._emitted_card_ids),
            "unavailable_count": len(self._unavailable_card_ids),
            "hot_template_count": len(self._hot_candidate_templates_by_card),
            "session_generation": int(self.session_generation),
            "prediction_generation": int(self.prediction_generation),
            "allow_same_day_repeats": bool(self.settings.allow_same_day_repeats),
            "stale_recheck_batches": int(self.stale_recheck_batches),
            "stale_recheck_checked_count": int(self.stale_recheck_checked_count),
            "stale_recheck_eligible_count": int(self.stale_recheck_eligible_count),
            "stale_recheck_remaining_count": len(stale_ids),
            "ready_head": _head(ready_selectable),
            "backlog_head": _head(backlog_selectable),
            "stale_head": _head(stale_ids),
            "hot_head": _head(hot.card_id for hot in self.hot_registry),
            "hot_needs_prediction_head": _head(hot_needs_prediction),
            "unavailable_head": _head(sorted(self._unavailable_card_ids)),
        }

    def begin_prediction(self, card_ids: Iterable[int]) -> LivePredictionToken:
        if not self.active:
            return LivePredictionToken(
                session_generation=self.session_generation,
                prediction_generation=self.prediction_generation,
                candidate_card_ids=(),
            )
        self.prediction_generation += 1
        normalized_card_ids = (
            card_ids if isinstance(card_ids, tuple) else tuple(int(card_id) for card_id in card_ids)
        )
        return LivePredictionToken(
            session_generation=self.session_generation,
            prediction_generation=self.prediction_generation,
            candidate_card_ids=normalized_card_ids,
        )

    def candidates_for_card_ids(
        self,
        card_ids: Iterable[int],
    ) -> tuple[LiveReviewCandidate, ...]:
        if not self.active:
            return ()
        candidates: list[LiveReviewCandidate] = []
        for card_id in card_ids:
            normalized_card_id = int(card_id)
            candidate = self._candidates_by_card.get(
                normalized_card_id
            ) or self._hot_candidate_templates_by_card.get(normalized_card_id)
            if candidate is not None:
                candidates.append(self._candidate_table.materialize(candidate))
        return tuple(candidates)

    @property
    def candidate_universe_card_ids(self) -> tuple[int, ...]:
        """Return stable candidate insertion order for one-time native seeding."""

        if not self.active:
            return ()
        return tuple(self._candidate_transport_order_index)

    @property
    def native_selection_enabled(self) -> bool:
        return self._native_selection_enabled

    def enable_native_prediction_selection(self) -> None:
        if not self.active:
            return
        self._native_selection_enabled = True
        self._native_selection_card_ids = ()
        self._native_refreshed_count = 0

    def disable_native_prediction_selection(self) -> None:
        self._native_selection_enabled = False
        self._native_selection_card_ids = ()
        self._native_refreshed_count = 0

    def native_refresh_hot_card_ids(self) -> tuple[int, ...]:
        if not self._native_selection_enabled:
            return ()
        return self.hot_prediction_card_ids(
            limit=min(
                int(self.settings.hot_predict_limit),
                int(self.settings.prediction_refresh_limit),
            )
        )

    def native_hot_card_ids_pending_include(self) -> tuple[int, ...]:
        """Return released hot cards that still need native reinclusion."""

        if not self._native_selection_enabled:
            return ()
        return tuple(
            hot.card_id
            for hot in self._hot_registry.values()
            if hot.native_include_pending and self._hot_card_is_released(hot)
        )

    @property
    def quarantined_hot_card_ids(self) -> tuple[int, ...]:
        return tuple(
            hot.card_id
            for hot in self._hot_registry.values()
            if not self._hot_card_is_released(hot)
        )

    def native_selection_exclusion_card_ids(
        self,
        *,
        hot_card_ids: Iterable[int] = (),
    ) -> tuple[int, ...]:
        """Return the small add-on-owned exclusion set for compact selection.

        Hot cards are ranked separately so they retain the add-on's historical
        priority over ordinary candidates.  Emitted and unavailable cards must
        also remain outside a newly returned two-card buffer.
        """

        # Hot cards are selected by the add-on's separate same-day priority
        # rule. Include the complete registry here, rather than only the cards
        # currently due for prediction, so quarantined cards cannot leak into
        # native ordinary selection.
        excluded = set(self._hot_registry)
        excluded.update(int(card_id) for card_id in hot_card_ids)
        excluded.update(self._emitted_card_ids)
        excluded.update(self._unavailable_card_ids)
        if not self.settings.allow_same_day_repeats:
            # Answered candidates are normally removed from the native index by
            # process_answer(). A full reconciliation can seed them again from
            # the fresh source search, so retain the same session-level policy
            # through the caller-owned exclusion set.
            excluded.update(self._answered_card_ids)
        return tuple(excluded)

    def apply_native_refresh_result(
        self,
        token: LivePredictionToken,
        *,
        selected_values: Sequence[LiveNativePredictionValue],
        hot_values: Sequence[LiveNativePredictionValue] = (),
        refreshed_count: int,
        target_timestamp_seconds: float,
        included_hot_card_ids: Iterable[int] = (),
    ) -> LivePredictionResult:
        """Mirror only compact Rust results needed by Anki-owned behavior.

        Rust remains authoritative for ordinary candidate prediction, rank,
        and eligibility. Python mirrors the returned normal selections plus at
        most ``hot_predict_limit`` same-day candidates, which are the only rows
        needed for Card Info and the add-on's hot-card priority rule.
        """

        if not self.active:
            return LivePredictionResult(LivePredictionStatus.INACTIVE)
        if (
            not self._native_selection_enabled
            or token.session_generation != self.session_generation
            or token.prediction_generation != self.prediction_generation
        ):
            return LivePredictionResult(LivePredictionStatus.STALE)

        newly_included_hot_card_ids: set[int] = set()
        for card_id in included_hot_card_ids:
            normalized = int(card_id)
            hot = self._hot_registry.get(normalized)
            if hot is not None and hot.native_include_pending:
                self._hot_registry[normalized] = replace(
                    hot,
                    native_include_pending=False,
                )
                newly_included_hot_card_ids.add(normalized)
        if newly_included_hot_card_ids and self._undoable_review_rows:
            review_id = _review_id_from_row(self._undoable_review_rows[-1])
            if review_id is not None:
                self._native_hot_includes_by_review_id.setdefault(
                    review_id,
                    set(),
                ).update(newly_included_hot_card_ids)

        candidate_table = self._candidate_table
        candidates_by_card = self._candidates_by_card
        hot_templates = self._hot_candidate_templates_by_card
        special_card_ids = set(self._hot_registry)
        special_card_ids.update(hot_templates)
        special_card_ids.update(self._unavailable_card_ids)
        refreshed_special_candidates: dict[int, LiveReviewCandidate] = {}
        target = float(target_timestamp_seconds)

        for value in (*selected_values, *hot_values):
            card_id = int(value.card_id)
            candidate = candidates_by_card.get(card_id) or hot_templates.get(card_id)
            if candidate is None:
                continue
            slot = candidate_table.slot_by_card_id.get(card_id)
            if slot is None:
                slot = candidate_table.upsert(candidate)
            candidate_table.apply_prediction(
                card_id,
                prediction=float(value.predicted_retrievability),
                elapsed_days=float(value.elapsed_days),
                target_timestamp_seconds=target,
            )
            materialized = candidate_table.materialize(candidate)
            candidate_table.below_retention[slot] = self._candidate_is_within_active_retention(
                materialized
            )
            if card_id in special_card_ids:
                candidates_by_card[card_id] = materialized
                candidate_table.upsert(materialized)
                refreshed_special_candidates[card_id] = materialized

        self._reconcile_refreshed_special_candidates(refreshed_special_candidates)
        hot_ids = set(self._hot_registry)
        self._native_selection_card_ids = tuple(
            card_id
            for value in selected_values
            if (card_id := int(value.card_id)) not in hot_ids
            and self._card_is_available_below_active_retention(card_id)
            and card_id not in self._emitted_card_ids
        )
        self._native_refreshed_count = max(0, int(refreshed_count))
        self.session_generation += 1
        return LivePredictionResult(
            status=LivePredictionStatus.APPLIED,
            applied_count=self._native_refreshed_count,
        )

    def record_stale_recheck(
        self,
        *,
        batches: int,
        checked_count: int,
        eligible_count: int,
    ) -> None:
        self.stale_recheck_batches = max(0, int(batches))
        self.stale_recheck_checked_count = max(0, int(checked_count))
        self.stale_recheck_eligible_count = max(0, int(eligible_count))

    def _reconcile_refreshed_special_candidates(
        self,
        refreshed_candidates: Mapping[int, LiveReviewCandidate],
    ) -> None:
        for card_id, candidate in refreshed_candidates.items():
            self._hot_candidate_templates_by_card.pop(card_id, None)
            self._unavailable_card_ids.discard(card_id)
            if card_id in self._hot_registry:
                self._hot_registry[card_id] = self._updated_hot_card(
                    card_id,
                    predicted_retrievability=candidate.predicted_retrievability,
                )

    def replace_candidate_universe(
        self,
        candidates: Iterable[LiveReviewCandidate],
    ) -> None:
        """Replace current pull candidates while preserving live undo state."""
        self._capture_candidate_universe_for_latest_undo()
        normalized_candidates = {
            normalized.card_id: normalized
            for normalized in (_normalize_candidate(candidate) for candidate in candidates)
        }
        self.active = True
        self._candidates_by_card = normalized_candidates
        self._candidate_table.rebuild(normalized_candidates)
        valid_card_ids = set(normalized_candidates)
        self._candidate_transport_order_index = tuple(normalized_candidates)
        self._emitted_card_ids.clear()
        self._emitted_card_order.clear()
        self._unavailable_card_ids.clear()
        self._native_selection_card_ids = ()
        self._native_refreshed_count = 0
        self._hot_registry = {
            card_id: hot
            for card_id, hot in self._hot_registry.items()
            if int(card_id) in valid_card_ids
        }
        self._hot_candidate_templates_by_card = {
            card_id: candidate
            for card_id, candidate in self._hot_candidate_templates_by_card.items()
            if int(card_id) in valid_card_ids
        }
        self.session_generation += 1
        self.prediction_generation += 1
        if self._review_limit_reached():
            self.close()
            return

    def add_hot_cards(
        self,
        card_ids: Iterable[int],
        *,
        predicted_retrievability: float | None = None,
        desired_retention: float | None = None,
        next_check_at: float | None = None,
        release_after_review_count: int | None = None,
        native_include_pending: bool = False,
    ) -> None:
        if not self.active:
            return
        for card_id in card_ids:
            normalized = int(card_id)
            if normalized in self._emitted_card_ids:
                continue
            self._hot_registry[normalized] = LiveHotCard(
                card_id=normalized,
                predicted_retrievability=_finite_float_or_none(
                    predicted_retrievability,
                ),
                desired_retention=_finite_float_or_default(
                    desired_retention,
                    self._answered_desired_retention_by_card.get(normalized, 0.9),
                ),
                next_check_at=_finite_float_or_none(next_check_at),
                release_after_review_count=max(
                    0,
                    int(
                        self.reviews_done
                        if release_after_review_count is None
                        else release_after_review_count
                    ),
                ),
                native_include_pending=bool(native_include_pending),
            )

    def add_hot_card_from_curve_probabilities(
        self,
        card_id: int,
        *,
        now: float,
        desired_retention: float,
        curve_retrievability_by_offset: Mapping[float, float],
    ) -> None:
        self.add_hot_cards(
            [card_id],
            desired_retention=desired_retention,
            next_check_at=estimate_hot_next_check_at(
                now=now,
                desired_retention=desired_retention,
                curve_retrievability_by_offset=curve_retrievability_by_offset,
                settings=self.settings,
            ),
        )

    def update_hot_card_from_curve_probabilities(
        self,
        card_id: int,
        *,
        now: float,
        curve_retrievability_by_offset: Mapping[float, float],
    ) -> None:
        normalized = int(card_id)
        if (
            not self.active
            or not self.settings.allow_same_day_repeats
            or normalized not in self._hot_registry
        ):
            return
        self._hot_registry[normalized] = self._updated_hot_card(
            normalized,
            next_check_at=estimate_hot_next_check_at(
                now=now,
                desired_retention=self._answered_desired_retention_by_card.get(
                    normalized,
                    0.9,
                ),
                curve_retrievability_by_offset=curve_retrievability_by_offset,
                settings=self.settings,
            ),
        )

    def release_processed_hot_card(
        self,
        review_row: Mapping[str, Any],
        *,
        predicted_retrievability: float | None = None,
    ) -> bool:
        if not self.active or not self.settings.allow_same_day_repeats:
            return False
        review_id = _review_id_from_row(review_row)
        if review_id is None:
            return False
        release = self._hot_releases_by_review_id.pop(review_id, None)
        if release is None:
            return False
        candidate = release.candidate
        card_id = int(review_row["card_id"])
        if int(candidate.card_id) != card_id:
            return False
        self._hot_candidate_templates_by_card[card_id] = candidate
        desired_retention = float(candidate.hot_desired_retention)
        self._answered_desired_retention_by_card[card_id] = desired_retention
        self.add_hot_cards(
            [card_id],
            predicted_retrievability=predicted_retrievability,
            desired_retention=desired_retention,
            release_after_review_count=(
                self.reviews_done
                + int(self.settings.same_day_reentry_delay_reviews)
            ),
            native_include_pending=(
                int(self.settings.same_day_reentry_delay_reviews) > 0
            ),
        )
        return True

    def release_processed_hot_card_from_curve_probabilities(
        self,
        review_row: Mapping[str, Any],
        *,
        now: float,
        curve_retrievability_by_offset: Mapping[float, float],
    ) -> bool:
        del now, curve_retrievability_by_offset
        return self.release_processed_hot_card(review_row)

    def discard_hot_release(self, review_row: Mapping[str, Any]) -> None:
        review_id = _review_id_from_row(review_row)
        if review_id is not None:
            self._hot_releases_by_review_id.pop(review_id, None)

    def hot_prediction_card_ids(
        self,
        *,
        now: float | None = None,
        limit: int | None = None,
        exclude: Iterable[int] = (),
    ) -> tuple[int, ...]:
        del now
        if not self.active:
            return ()
        resolved_limit = self.settings.hot_predict_limit if limit is None else limit
        if resolved_limit <= 0:
            return ()
        excluded = {int(card_id) for card_id in exclude}
        sorted_hot_cards = _ordered_limited(
            (
                hot
                for hot in self._hot_registry.values()
                if hot.card_id not in self._emitted_card_ids
                and hot.card_id not in self._unavailable_card_ids
                and hot.card_id not in excluded
                and self._hot_card_is_released(hot)
            ),
            key=self._hot_prediction_sort_key,
            limit=int(resolved_limit),
        )
        return tuple(hot.card_id for hot in sorted_hot_cards)

    def reschedule_hot_cards(
        self,
        card_ids: Iterable[int],
        *,
        next_check_at: float,
    ) -> None:
        if not self.active:
            return
        for card_id in card_ids:
            normalized = int(card_id)
            if normalized not in self._hot_registry:
                continue
            self._hot_registry[normalized] = self._updated_hot_card(
                normalized,
                next_check_at=next_check_at,
            )

    def next_buffer(self, size: int) -> LiveSelectionResult:
        if not self.active:
            return LiveSelectionResult(LiveSelectionStatus.INACTIVE)
        if self._review_limit_reached():
            self.close()
            return LiveSelectionResult(LiveSelectionStatus.INACTIVE)
        if size <= 0:
            return LiveSelectionResult(LiveSelectionStatus.EMPTY)
        selected = self._next_unemitted_card_ids(int(size))
        if not selected:
            return LiveSelectionResult(LiveSelectionStatus.EMPTY)
        self._mark_emitted(selected)
        return LiveSelectionResult(LiveSelectionStatus.READY, selected)

    def mark_card_unavailable(self, card_id: int) -> None:
        if not self.active:
            return
        normalized = int(card_id)
        self._unavailable_card_ids.add(normalized)
        self._remove_card_from_queues(normalized)
        self._shown_candidates_by_card.pop(normalized, None)
        self._shown_fsrs_predictions_by_card.pop(normalized, None)
        self._discard_emitted(normalized)
        self._native_selection_card_ids = tuple(
            candidate_id
            for candidate_id in self._native_selection_card_ids
            if candidate_id != normalized
        )
        self.session_generation += 1

    def release_unshown_emitted_cards(self, card_ids: Iterable[int]) -> None:
        if not self.active:
            return
        changed = False
        for card_id in card_ids:
            normalized = int(card_id)
            if normalized in self._emitted_card_ids:
                self._discard_emitted(normalized)
                changed = True
        if not changed:
            return
        self.session_generation += 1

    def restore_existing_buffer(self, card_ids: Iterable[int]) -> bool:
        if not self.active:
            return False
        normalized_card_ids = _dedupe(card_ids)
        if not normalized_card_ids:
            return False
        if any(
            not self._card_is_available_below_active_retention(card_id)
            for card_id in normalized_card_ids
        ):
            return False

        self._mark_emitted(normalized_card_ids)
        self.session_generation += 1
        return True

    def sync_existing_buffer(self, card_ids: Iterable[int]) -> bool:
        if not self.active:
            return False
        normalized_card_ids = _dedupe(card_ids)
        if not normalized_card_ids:
            return False
        if any(
            not self._card_is_available_below_active_retention(card_id)
            for card_id in normalized_card_ids
        ):
            return False

        synced_emitted_card_ids = set(normalized_card_ids)
        if (
            synced_emitted_card_ids == self._emitted_card_ids
            and list(normalized_card_ids) == self._emitted_card_order
        ):
            return True
        self._emitted_card_ids = synced_emitted_card_ids
        self._emitted_card_order = list(normalized_card_ids)
        self.session_generation += 1
        return True

    def record_answer(
        self,
        card_id: int,
        *,
        review_row: Mapping[str, Any] | None = None,
        hot_next_check_at: float | None = None,
    ) -> LiveAnswerResult:
        normalized_card_id = int(card_id)
        if not self.active:
            return LiveAnswerResult(
                status=LiveAnswerStatus.IGNORED,
                card_id=normalized_card_id,
                reviews_done=self.reviews_done,
                undoable_review_count=self.undoable_review_count,
                transient_applied_review_count=self.transient_applied_review_count,
            )
        if normalized_card_id not in self._emitted_card_ids:
            return LiveAnswerResult(
                status=LiveAnswerStatus.IGNORED,
                card_id=normalized_card_id,
                reviews_done=self.reviews_done,
                undoable_review_count=self.undoable_review_count,
                transient_applied_review_count=self.transient_applied_review_count,
            )

        previous_minimum_retention_extra = float(self.minimum_retention_extra)
        answered_candidate = self._candidates_by_card.get(normalized_card_id)
        if answered_candidate is not None:
            answered_candidate = self._candidate_table.materialize(answered_candidate)
        shown_candidate = self._shown_candidates_by_card.pop(
            normalized_card_id,
            None,
        )
        shown_fsrs_prediction = self._shown_fsrs_predictions_by_card.pop(
            normalized_card_id,
            None,
        )
        if answered_candidate is not None:
            self._answered_desired_retention_by_card[normalized_card_id] = float(
                answered_candidate.hot_desired_retention
            )

        self.reviews_done += 1
        self.session_generation += 1
        self._answered_card_ids.add(normalized_card_id)
        self._discard_emitted(normalized_card_id)
        self._candidates_by_card.pop(normalized_card_id, None)
        self._native_selection_card_ids = tuple(
            candidate_id
            for candidate_id in self._native_selection_card_ids
            if candidate_id != normalized_card_id
        )
        self._remove_card_from_queues(normalized_card_id)

        self._record_retention_record(
            review_row,
            answered_candidate=shown_candidate or answered_candidate,
            fsrs_prediction=shown_fsrs_prediction,
        )
        self._record_undoable_review_row(
            review_row,
            answered_candidate=answered_candidate,
            hot_next_check_at=hot_next_check_at,
            previous_minimum_retention_extra=previous_minimum_retention_extra,
        )
        if self._review_limit_reached():
            status = LiveAnswerStatus.STOPPED_LIMIT
        else:
            status = LiveAnswerStatus.RECORDED

        result = LiveAnswerResult(
            status=status,
            card_id=normalized_card_id,
            review_id=_review_id_from_row(review_row),
            reviews_done=self.reviews_done,
            undoable_review_count=self.undoable_review_count,
            transient_applied_review_count=self.transient_applied_review_count,
        )
        if status == LiveAnswerStatus.STOPPED_LIMIT:
            self.close()
            return _with_undoable_review_count(result, 0)
        self._reset_minimum_retention_if_satisfied()
        return result

    def record_undo(self, review_id: int) -> LiveUndoResult:
        if not self.active:
            return LiveUndoResult(
                status=LiveUndoStatus.IGNORED,
                review_id=int(review_id),
                reviews_done=self.reviews_done,
                undoable_review_count=self.undoable_review_count,
            )
        normalized = int(review_id)
        undoable_rows = list(self._undoable_review_rows)
        for index, row in enumerate(undoable_rows):
            if _review_id_from_row(row) == normalized:
                if index != len(undoable_rows) - 1:
                    self.close()
                    self.session_generation += 1
                    return LiveUndoResult(
                        status=LiveUndoStatus.PAUSED_STALE,
                        review_id=normalized,
                        reviews_done=self.reviews_done,
                        undoable_review_count=self.undoable_review_count,
                    )
                del undoable_rows[index]
                self._undoable_review_rows = deque(undoable_rows)
                # Candidate-index rebuilding depends on whether the minimum
                # review target is still active. Move the counter back before
                # restoring either the pre-reconciliation universe or the
                # answered card so both are indexed under the pre-answer policy.
                self.reviews_done = max(0, self.reviews_done - 1)
                self._restore_candidate_universe_for_undo(normalized)
                self._restore_undo_candidate(row)
                for card_id in self._native_hot_includes_by_review_id.pop(
                    normalized,
                    set(),
                ):
                    hot = self._hot_registry.get(card_id)
                    if hot is not None:
                        self._hot_registry[card_id] = replace(
                            hot,
                            native_include_pending=True,
                        )
                self.session_generation += 1
                self.needs_targeted_repair = True
                return LiveUndoResult(
                    status=LiveUndoStatus.HANDLED_UNDOABLE,
                    review_id=normalized,
                    reviews_done=self.reviews_done,
                    undoable_review_count=self.undoable_review_count,
                    needs_targeted_repair=True,
                )
        return LiveUndoResult(
            status=LiveUndoStatus.IGNORED,
            review_id=normalized,
            reviews_done=self.reviews_done,
            undoable_review_count=self.undoable_review_count,
        )

    def close(self) -> None:
        self.active = False
        self.session_generation += 1
        self.prediction_generation += 1
        self._candidates_by_card.clear()
        self._candidate_table.clear()
        self._hot_registry.clear()
        self._answered_card_ids.clear()
        self._emitted_card_ids.clear()
        self._emitted_card_order.clear()
        self._unavailable_card_ids.clear()
        self._undoable_review_rows.clear()
        self._undoable_candidates_by_review_id.clear()
        self._minimum_retention_extra_by_review_id.clear()
        self._candidate_universe_snapshots_by_review_id.clear()
        self._hot_releases_by_review_id.clear()
        self._native_hot_includes_by_review_id.clear()
        self._answered_desired_retention_by_card.clear()
        self._hot_candidate_templates_by_card.clear()
        self._shown_candidates_by_card.clear()
        self._shown_fsrs_predictions_by_card.clear()
        self._random_sort_keys.clear()
        self._candidate_transport_order_index = ()
        self.record_stale_recheck(batches=0, checked_count=0, eligible_count=0)
        self.minimum_retention_extra = 0.0
        self.needs_targeted_repair = False
        self.disable_native_prediction_selection()

    def _next_unemitted_card_ids(self, limit: int) -> tuple[int, ...]:
        if int(limit) <= 0:
            return ()
        selected: list[int] = []
        for card_id in self._hot_selectable_card_ids(limit=int(limit)):
            selected.append(card_id)
            if len(selected) >= int(limit):
                return tuple(selected)
        remaining = int(limit) - len(selected)
        if remaining > 0:
            selected.extend(
                self._selectable_universe_card_ids(
                    limit=remaining,
                    exclude=selected,
                )
            )
        return tuple(selected)

    def _hot_selectable_card_ids(
        self,
        *,
        limit: int | None = None,
        exclude: Iterable[int] = (),
    ) -> tuple[int, ...]:
        excluded = {int(card_id) for card_id in exclude}
        candidates = _ordered_limited(
            (
                card_id
                for card_id in self._hot_registry
                if card_id not in excluded
                and card_id not in self._emitted_card_ids
                and self._hot_card_is_released(self._hot_registry[card_id])
                and self._card_is_available_below_active_retention(card_id)
            ),
            key=self._retrievability_sort_key,
            limit=limit,
        )
        return tuple(candidates)

    def _selectable_universe_card_ids(
        self,
        *,
        limit: int | None = None,
        exclude: Iterable[int] = (),
    ) -> tuple[int, ...]:
        excluded = {int(card_id) for card_id in exclude}
        if not self._native_selection_enabled:
            return ()
        return self._take_indexed_card_ids(
            self._native_selection_card_ids,
            limit=limit,
            predicate=lambda card_id: (
                card_id not in excluded
                and card_id not in self._emitted_card_ids
                and card_id not in self._hot_registry
                and self._card_is_available_below_active_retention(card_id)
            ),
        )

    def _stale_universe_card_ids(self) -> tuple[int, ...]:
        return self._take_indexed_card_ids(
            self._candidate_transport_order_index,
            limit=None,
            predicate=self._card_is_stale_recheckable,
        )

    def _mark_emitted(self, card_ids: Iterable[int]) -> None:
        for card_id in card_ids:
            normalized = int(card_id)
            if normalized not in self._emitted_card_ids:
                self._emitted_card_ids.add(normalized)
                self._emitted_card_order.append(normalized)

    def _discard_emitted(self, card_id: int) -> None:
        normalized = int(card_id)
        if normalized not in self._emitted_card_ids:
            return
        self._emitted_card_ids.discard(normalized)
        self._emitted_card_order = [
            candidate_id
            for candidate_id in self._emitted_card_order
            if candidate_id != normalized
        ]

    def _take_indexed_card_ids(
        self,
        index: Iterable[int],
        *,
        limit: int | None,
        predicate: Callable[[int], bool],
    ) -> tuple[int, ...]:
        resolved_limit = None if limit is None else int(limit)
        if resolved_limit is not None and resolved_limit <= 0:
            return ()
        selected: list[int] = []
        for card_id in index:
            normalized = int(card_id)
            if not predicate(normalized):
                continue
            selected.append(normalized)
            if resolved_limit is not None and len(selected) >= resolved_limit:
                break
        return tuple(selected)

    def _card_is_available_below_active_retention(self, card_id: int) -> bool:
        normalized = int(card_id)
        slot = self._candidate_table.slot_by_card_id.get(normalized)
        return (
            slot is not None
            and self._candidate_table.below_retention[slot]
            and self._card_is_available(normalized)
        )

    def _card_is_stale_recheckable(self, card_id: int) -> bool:
        normalized = int(card_id)
        slot = self._candidate_table.slot_by_card_id.get(normalized)
        return (
            slot is not None
            and not self._candidate_table.below_retention[slot]
            and self._card_is_available(normalized)
            and normalized not in self._emitted_card_ids
            and normalized not in self._hot_registry
        )

    def _random_sort_key_for_card(self, card_id: int) -> float:
        normalized = int(card_id)
        random_key = self._random_sort_keys.get(normalized)
        if random_key is None:
            random_key = random.random()
            self._random_sort_keys[normalized] = random_key
        return random_key

    def _retrievability_sort_key(self, card_id: int) -> tuple[float, int]:
        normalized = int(card_id)
        return (self._candidate_table.prediction_for(normalized), normalized)

    def _hot_prediction_sort_key(self, hot: LiveHotCard) -> tuple[float, ...]:
        prediction = _finite_float_or_none(hot.predicted_retrievability)
        if prediction is None:
            return (0.0, 0.0, float(hot.card_id))
        desired = _finite_float_or_default(hot.desired_retention, 0.9)
        return (
            1.0,
            prediction - desired,
            prediction,
            float(hot.card_id),
        )

    def _updated_hot_card(
        self,
        card_id: int,
        *,
        predicted_retrievability: float | None | object = _MISSING,
        desired_retention: float | None | object = _MISSING,
        next_check_at: float | None | object = _MISSING,
        release_after_review_count: int | object = _MISSING,
        native_include_pending: bool | object = _MISSING,
    ) -> LiveHotCard:
        normalized = int(card_id)
        current = self._hot_registry.get(normalized)
        return LiveHotCard(
            card_id=normalized,
            predicted_retrievability=(
                _finite_float_or_none(predicted_retrievability)
                if predicted_retrievability is not _MISSING
                else (None if current is None else current.predicted_retrievability)
            ),
            desired_retention=(
                _finite_float_or_default(desired_retention, 0.9)
                if desired_retention is not _MISSING
                else (
                    self._answered_desired_retention_by_card.get(normalized, 0.9)
                    if current is None
                    else current.desired_retention
                )
            ),
            next_check_at=(
                _finite_float_or_none(next_check_at)
                if next_check_at is not _MISSING
                else (None if current is None else current.next_check_at)
            ),
            release_after_review_count=(
                max(0, int(release_after_review_count))
                if release_after_review_count is not _MISSING
                else (
                    self.reviews_done
                    if current is None
                    else current.release_after_review_count
                )
            ),
            native_include_pending=(
                bool(native_include_pending)
                if native_include_pending is not _MISSING
                else (False if current is None else current.native_include_pending)
            ),
        )

    def _hot_card_is_released(self, hot: LiveHotCard) -> bool:
        return self.reviews_done >= int(hot.release_after_review_count)

    def _remove_card_from_queues(self, card_id: int) -> None:
        self._hot_registry.pop(int(card_id), None)

    def _review_limit_reached(self) -> bool:
        return (
            self.settings.review_limit is not None
            and self.reviews_done >= self.settings.review_limit
        )

    def _minimum_fill_active(self) -> bool:
        target = self._minimum_review_target()
        return target > 0 and self.reviews_done < target

    def _minimum_review_target(self) -> int:
        configured = max(0, int(self.settings.minimum_review_limit))
        if configured <= 0:
            return 0
        if self.settings.review_limit is None:
            return configured
        return min(configured, int(self.settings.review_limit))

    def _candidate_is_within_active_retention(
        self,
        candidate: LiveReviewCandidate,
    ) -> bool:
        return prediction_below_retention(
            candidate.predicted_retrievability,
            candidate.active_desired_retention,
            extra_retention=(
                0.0 if _candidate_is_intraday(candidate) else self._active_minimum_retention_extra()
            ),
        )

    def _active_minimum_retention_extra(self) -> float:
        if not self._minimum_fill_active():
            return 0.0
        return max(0.0, float(self.minimum_retention_extra))

    def _card_is_available(self, card_id: int) -> bool:
        normalized = int(card_id)
        return (
            normalized in self._candidates_by_card
            and normalized not in self._unavailable_card_ids
            and not (
                normalized in self._answered_card_ids and not self.settings.allow_same_day_repeats
            )
        )

    def _reset_minimum_retention_if_satisfied(self) -> None:
        if self._minimum_fill_active() or self.minimum_retention_extra <= 0:
            return
        self.minimum_retention_extra = 0.0

    def _record_undoable_review_row(
        self,
        review_row: Mapping[str, Any] | None,
        *,
        answered_candidate: LiveReviewCandidate | None,
        hot_next_check_at: float | None,
        previous_minimum_retention_extra: float,
    ) -> None:
        if review_row is None:
            return
        normalized = dict(review_row)
        self._undoable_review_rows.append(normalized)
        review_id = _review_id_from_row(normalized)
        if review_id is not None:
            self._undoable_candidates_by_review_id[review_id] = answered_candidate
            self._minimum_retention_extra_by_review_id[review_id] = float(
                previous_minimum_retention_extra
            )
            if self.settings.allow_same_day_repeats and answered_candidate is not None:
                self._hot_releases_by_review_id[review_id] = _LiveHotRelease(
                    candidate=answered_candidate,
                )
        self.transient_applied_review_count += 1
        self._trim_undoable_review_rows()

    def _capture_candidate_universe_for_latest_undo(self) -> None:
        """Attach the first full-universe replacement to the latest answer.

        RWKV-SRS uses the same boundary rule. Keeping a compact Python policy
        mirror here ensures an Anki undo restores metadata and availability for
        the same universe whose native rank was restored upstream.
        """

        if not self._undoable_review_rows:
            return
        review_id = _review_id_from_row(self._undoable_review_rows[-1])
        if (
            review_id is None
            or review_id in self._candidate_universe_snapshots_by_review_id
        ):
            return
        ordered_ids = tuple(
            card_id
            for card_id in self._candidate_transport_order_index
            if card_id in self._candidates_by_card
        )
        seen = set(ordered_ids)
        ordered_ids = (
            *ordered_ids,
            *(card_id for card_id in self._candidates_by_card if card_id not in seen),
        )
        candidates = tuple(
            self._candidate_table.materialize(self._candidates_by_card[card_id])
            for card_id in ordered_ids
        )
        self._candidate_universe_snapshots_by_review_id[review_id] = (
            _LiveCandidateUniverseUndoSnapshot(
                candidates=candidates,
                transport_card_ids=ordered_ids,
                hot_registry=dict(self._hot_registry),
                emitted_card_ids=set(self._emitted_card_ids),
                emitted_card_order=tuple(self._emitted_card_order),
                unavailable_card_ids=set(self._unavailable_card_ids),
                hot_candidate_templates_by_card=dict(
                    self._hot_candidate_templates_by_card
                ),
                random_sort_keys=dict(self._random_sort_keys),
                minimum_retention_extra=float(self.minimum_retention_extra),
                native_selection_card_ids=tuple(self._native_selection_card_ids),
                native_refreshed_count=int(self._native_refreshed_count),
            )
        )

    def _restore_candidate_universe_for_undo(self, review_id: int) -> None:
        snapshot = self._candidate_universe_snapshots_by_review_id.pop(
            int(review_id),
            None,
        )
        if snapshot is None:
            return
        candidates_by_card = {
            int(candidate.card_id): candidate for candidate in snapshot.candidates
        }
        self._candidates_by_card = candidates_by_card
        self._candidate_table.rebuild(candidates_by_card)
        self._candidate_transport_order_index = tuple(snapshot.transport_card_ids)
        self._hot_registry = dict(snapshot.hot_registry)
        self._emitted_card_ids = set(snapshot.emitted_card_ids)
        self._emitted_card_order = list(snapshot.emitted_card_order)
        self._unavailable_card_ids = set(snapshot.unavailable_card_ids)
        self._hot_candidate_templates_by_card = dict(
            snapshot.hot_candidate_templates_by_card
        )
        self._random_sort_keys = dict(snapshot.random_sort_keys)
        self.minimum_retention_extra = float(snapshot.minimum_retention_extra)
        self._native_selection_card_ids = tuple(
            snapshot.native_selection_card_ids
        )
        self._native_refreshed_count = int(snapshot.native_refreshed_count)

    def _record_retention_record(
        self,
        review_row: Mapping[str, Any] | None,
        *,
        answered_candidate: LiveReviewCandidate | None,
        fsrs_prediction: float | None = None,
    ) -> None:
        if review_row is None:
            return
        record = retention_record_for_answer(
            answered_candidate,
            review_row,
            fsrs_prediction=fsrs_prediction,
        )
        if record is None:
            review_id = _review_id_from_row(review_row)
            if review_id is not None:
                self._skipped_retention_review_ids.add(review_id)
            return
        self._retention_records_by_review_id[record.review_id] = record

    def _restore_undo_candidate(self, row: Mapping[str, Any]) -> None:
        review_id = _review_id_from_row(row)
        card_id = int(row["card_id"])
        candidate = (
            None
            if review_id is None
            else self._undoable_candidates_by_review_id.pop(review_id, None)
        )
        self._answered_card_ids.discard(card_id)
        self._answered_desired_retention_by_card.pop(card_id, None)
        self._hot_candidate_templates_by_card.pop(card_id, None)
        self._shown_candidates_by_card.pop(card_id, None)
        self._shown_fsrs_predictions_by_card.pop(card_id, None)
        if review_id is not None:
            self._hot_releases_by_review_id.pop(review_id, None)
            self._retention_records_by_review_id.pop(review_id, None)
            self._skipped_retention_review_ids.discard(review_id)
            previous_extra = self._minimum_retention_extra_by_review_id.pop(
                review_id,
                None,
            )
            if previous_extra is not None:
                self.minimum_retention_extra = float(previous_extra)
        self._remove_card_from_queues(card_id)
        if candidate is not None:
            restored_card_id = int(candidate.card_id)
            self._candidates_by_card[restored_card_id] = candidate
            self._candidate_table.upsert(candidate)
            self._unavailable_card_ids.discard(restored_card_id)
            if restored_card_id not in self._candidate_transport_order_index:
                self._candidate_transport_order_index = (
                    *self._candidate_transport_order_index,
                    restored_card_id,
                )
        self.transient_applied_review_count = max(
            0,
            self.transient_applied_review_count - 1,
        )

    def _trim_undoable_review_rows(self) -> None:
        while len(self._undoable_review_rows) > LIVE_REVIEW_RUST_UNDO_LIMIT:
            row = self._undoable_review_rows.popleft()
            review_id = _review_id_from_row(row)
            if review_id is not None:
                self._undoable_candidates_by_review_id.pop(review_id, None)
                self._minimum_retention_extra_by_review_id.pop(review_id, None)
                self._hot_releases_by_review_id.pop(review_id, None)
                self._native_hot_includes_by_review_id.pop(review_id, None)
                self._candidate_universe_snapshots_by_review_id.pop(review_id, None)

def _candidate_elapsed_days(candidate: LiveReviewCandidate) -> float:
    try:
        return float(candidate.metadata.get("elapsed_days", -1))
    except (AttributeError, TypeError, ValueError):
        return math.nan


def _candidate_prediction_timestamp(
    candidate: LiveReviewCandidate,
) -> float | None:
    try:
        value = float(candidate.metadata.get("prediction_timestamp_seconds"))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _candidate_static_values(
    candidate: LiveReviewCandidate,
) -> tuple[float, float]:
    desired = float(candidate.desired_retention)
    same_day = desired
    if candidate.same_day_desired_retention is not None:
        try:
            configured_same_day = float(candidate.same_day_desired_retention)
        except (TypeError, ValueError):
            configured_same_day = math.nan
        if math.isfinite(configured_same_day):
            same_day = configured_same_day
    non_intraday = float(
        active_desired_retention_with_adaptive(
            desired,
            candidate.same_day_desired_retention,
            elapsed_days=1.0,
            adaptive_settings=candidate.adaptive_retention,
            rwkv_stability_days=candidate.rwkv_stability_days,
            fsrs_difficulty=candidate.fsrs_difficulty,
        )
    )
    return same_day, non_intraday


def _normalize_candidate(candidate: LiveReviewCandidate) -> LiveReviewCandidate:
    return LiveReviewCandidate(
        card_id=int(candidate.card_id),
        source_deck_id=(
            None if candidate.source_deck_id is None else int(candidate.source_deck_id)
        ),
        desired_retention=float(candidate.desired_retention),
        same_day_desired_retention=(
            None
            if candidate.same_day_desired_retention is None
            else float(candidate.same_day_desired_retention)
        ),
        predicted_retrievability=float(candidate.predicted_retrievability),
        rwkv_stability_days=(
            None if candidate.rwkv_stability_days is None else float(candidate.rwkv_stability_days)
        ),
        fsrs_difficulty=(
            None if candidate.fsrs_difficulty is None else float(candidate.fsrs_difficulty)
        ),
        adaptive_retention=candidate.adaptive_retention,
        # The engine owns dynamic prediction metadata and may update it after a
        # successful coordinator generation. Never alias a caller-owned map.
        metadata=dict(candidate.metadata),
        sort_info=candidate.sort_info,
    )


def _finite_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_float_or_default(value: float | None, default: float) -> float:
    parsed = _finite_float_or_none(value)
    return float(default) if parsed is None else parsed


def _review_id_from_row(row: Mapping[str, Any] | None) -> int | None:
    if row is None or "review_id" not in row:
        return None
    return int(row["review_id"])


def _with_undoable_review_count(
    result: LiveAnswerResult,
    undoable_review_count: int,
) -> LiveAnswerResult:
    return LiveAnswerResult(
        status=result.status,
        card_id=result.card_id,
        review_id=result.review_id,
        reviews_done=result.reviews_done,
        undoable_review_count=int(undoable_review_count),
        transient_applied_review_count=result.transient_applied_review_count,
    )


def _head(values: Iterable[int], *, limit: int = 20) -> list[int]:
    items: list[int] = []
    for value in values:
        if len(items) >= int(limit):
            break
        items.append(int(value))
    return items


def estimate_hot_next_check_at(
    *,
    now: float,
    desired_retention: float,
    curve_retrievability_by_offset: Mapping[float, float],
    settings: LiveReviewSettings | None = None,
) -> float:
    resolved_settings = settings or LiveReviewSettings()
    threshold = hot_curve_threshold(
        desired_retention,
        margin=resolved_settings.hot_curve_margin,
    )
    probabilities = {
        float(offset): float(probability)
        for offset, probability in curve_retrievability_by_offset.items()
    }
    for offset in resolved_settings.hot_curve_offsets_seconds:
        probability = probabilities.get(float(offset))
        if probability is None or not math.isfinite(probability):
            continue
        if probability <= threshold:
            return float(now) + float(offset)
    return float(now) + float(resolved_settings.hot_cold_recheck_seconds)


def hot_curve_threshold(desired_retention: float, *, margin: float) -> float:
    desired = float(desired_retention)
    if not math.isfinite(desired):
        return 0.99
    return min(0.99, desired * (1.0 + float(margin)))


def _dedupe(card_ids: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for card_id in card_ids:
        normalized = int(card_id)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _ordered_limited(
    values: Iterable[_T],
    *,
    key: Callable[[_T], object],
    limit: int | None,
) -> list[_T]:
    if limit is None:
        return sorted(values, key=key)
    resolved_limit = int(limit)
    if resolved_limit <= 0:
        return []
    return heapq.nsmallest(resolved_limit, values, key=key)


def _require_non_negative(name: str, value: int | float | None) -> None:
    if value is not None and float(value) < 0:
        raise ValueError(f"{name} must not be negative.")


def _candidate_is_intraday(candidate: LiveReviewCandidate) -> bool:
    return is_same_day_elapsed(candidate.metadata.get("elapsed_days", -1))
