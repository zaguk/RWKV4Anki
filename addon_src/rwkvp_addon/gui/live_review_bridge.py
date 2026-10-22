from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace

from aqt import gui_hooks, mw
from aqt.utils import qconnect, tooltip

from ..adaptive_retention import AdaptiveRetentionSettings
from ..addon_config import (
    addon_config_for_mw,
    configured_runtime_model_id,
    live_review_prediction_refresh_limit,
    live_review_quiet_refresh_attempts,
    live_review_sameday_prediction_limit,
    live_review_sameday_reentry_delay_reviews,
    minimum_review_widening_extra,
)
from ..anki_api import (
    DeckRetention,
    current_deck_id,
    fsrs_retrievabilities_for_card_ids,
)
from ..filtered_deck_sort import FilteredDeckOrder
from ..live_review_bridge import (
    LIVE_FILTERED_DECK_NAME,
    LiveBridgeStatus,
    LiveReviewBridgeSession,
    LiveReviewDeckUndoError,
    create_live_review_bridge_with_candidates,
)
from ..live_review_engine import LivePredictionStatus, LiveReviewSettings
from ..live_review_history import (
    LiveReviewSessionContext,
    append_live_review_history_session,
)
from ..live_review_predictions import LiveReviewPredictionCoordinator
from ..live_review_startup import (
    prepare_live_review_candidates_for_deck,
    refresh_live_review_candidates_for_deck,
)
from ..live_review_termination import LiveReviewLifecycleTrace
from ..progress import CancelledError
from ..review_load_policy import can_satisfy_minimum_retention, is_same_day_elapsed
from ..runtime import manager_for_mw, store_for_mw
from .checkpoint_failure import handle_checkpoint_failure
from .common import ProgressStage, run_with_progress_stages, show_quiet_info
from .web_message import show_web_warning

_active_session: LiveReviewBridgeSession | None = None
_active_prediction_coordinator: LiveReviewPredictionCoordinator | None = None
_active_start_request: _LiveReviewStartRequest | None = None
_live_review_start_token: object | None = None
_live_redo_block: _LiveRedoBlock | None = None
_live_review_trace = LiveReviewLifecycleTrace(max_events=16)
_live_exhaustion_refresh_pending = False
_live_exhaustion_overview_ready = False
_live_exhaustion_overview_refresh_seen = False
_live_exhaustion_overview_probe_in_flight = False
_live_exhaustion_overview_probe_sequence = 0
_live_exhaustion_resume_callback: Callable[[], None] | None = None
_live_exhaustion_resume_generation = 0
_live_exhaustion_resume_guard_active = False
_live_exhaustion_rebuild_generation: int | None = None
_live_exhaustion_timeout_generation: int | None = None
_live_review_question_transition_pending = False
_live_retention_dialogs: list[object] = []
_live_review_hooks_installed = False
_LIVE_STATUS_ELEMENT_ID = "rwkv-live-review-status"
_LIVE_REDO_VERIFY_FAILURE_LIMIT = 3
_LIVE_EXHAUSTION_OVERVIEW_PROBE_MS = 100
_LIVE_EXHAUSTION_OVERVIEW_PROBE_TIMEOUT_MS = 1000
_LIVE_EXHAUSTION_RESUME_TIMEOUT_MS = 15_000
_LIVE_SESSION_TITLE = "RWKV Live Session"
_UNEXPECTED_REVIEWER_EXIT_MESSAGE = (
    "RWKV Live Session stopped unexpectedly while Anki was moving to the next "
    "Live Session card. This can happen after an undo or deck refresh. Start a "
    "new Live Session to continue."
)
_UNDO_REMOVED_LIVE_DECK_MESSAGE = (
    "RWKV Live Session stopped because Undo removed its temporary Live Session "
    "deck instead of undoing a reviewed card. If you intended to undo an earlier "
    "review, use Undo again after closing this window, then start a new Live Session."
)
_UNDO_DURING_EXHAUSTION_MESSAGE = (
    "RWKV Live Session stopped because Undo was used while its full "
    "card recheck was still in progress. Anki's Undo succeeded; start a new "
    "Live Session to reload RWKV state."
)
_UNDO_DECK_RECONCILIATION_MESSAGE = (
    "Anki applied Undo, but RWKV Live Session could not safely repair its "
    "temporary filtered deck without changing Anki's undo history. The Live "
    "Session was stopped and the review Undo was preserved. Start a new Live "
    "Session to continue."
)
_UNCLASSIFIED_REVIEWER_EXIT_MESSAGE = (
    "RWKV Live Session ended when Anki left the review screen before a normal "
    "Live Session completion condition was recorded. If you left review "
    "intentionally, no action is needed."
)
_UNKNOWN_REVIEWER_EXIT_MESSAGE = (
    "RWKV Live Session ended, but the add-on could not determine why Anki left "
    "the review screen. Diagnostic details are included below."
)
_MISSING_COMPLETION_REASON_MESSAGE = (
    "RWKV Live Session ended without a recorded completion reason."
)
_EXHAUSTION_CANCELLED_MESSAGE = (
    "RWKV Live Session ended because its full card recheck was cancelled."
)
_EXHAUSTION_RESUME_TIMEOUT_MESSAGE = (
    "RWKV Live Session stopped because Anki did not finish returning to review "
    "after the full card recheck. Diagnostic details are included below."
)
_EXHAUSTION_NAVIGATION_INTERRUPTED_MESSAGE = (
    "RWKV Live Session stopped because Anki left its finished screen while the "
    "full card recheck was restoring the Live deck."
)
_EXHAUSTION_PAGE_STATE_SCRIPT = r"""
(() => {
    if (document.querySelector(".congrats")) {
        return "congrats-ready";
    }
    const path = String(window.location && window.location.pathname || "");
    if (/(^|\/)congrats\/?$/.test(path)) {
        return "congrats-loading";
    }
    if (document.getElementById("study")) {
        return "overview-ready";
    }
    return "other";
})()
"""


@dataclass(frozen=True)
class _LiveReviewStartRequest:
    source_deck_id: int
    settings: LiveReviewSettings
    retentions: tuple[DeckRetention, ...] | None = None
    same_day_only: bool = False
    extra_search: str = ""
    fsrs_comparison_enabled: bool = True
    from_exhaustion: bool = False
    started_at_ms: int = 0

    def for_exhaustion_restart(self) -> _LiveReviewStartRequest:
        return _LiveReviewStartRequest(
            source_deck_id=self.source_deck_id,
            settings=self.settings,
            retentions=self.retentions,
            same_day_only=self.same_day_only,
            extra_search=self.extra_search,
            fsrs_comparison_enabled=self.fsrs_comparison_enabled,
            from_exhaustion=True,
            started_at_ms=self.started_at_ms,
        )


@dataclass(frozen=True)
class _LiveRedoBlock:
    review_id: int
    card_id: int | None = None
    undo_counter: int | None = None
    verify_failure_count: int = 0


def _close_runtime_session(runtime_session) -> None:
    close = getattr(runtime_session, "close", None)
    if close is not None:
        with suppress(Exception):
            close()


def _expect_live_review_question() -> None:
    global _live_review_question_transition_pending
    _live_review_question_transition_pending = True


def _complete_live_review_question_transition() -> None:
    global _live_exhaustion_refresh_pending, _live_review_question_transition_pending
    _record_live_review_trace("question_shown", phase="reviewing")
    _live_review_question_transition_pending = False
    _live_exhaustion_refresh_pending = False
    _cancel_pending_live_exhaustion_resume()


def _clear_live_review_transition_state() -> None:
    global _live_review_question_transition_pending
    _live_review_question_transition_pending = False
    _cancel_pending_live_exhaustion_resume()


def _cancel_pending_live_exhaustion_resume() -> None:
    global _live_exhaustion_rebuild_generation, _live_exhaustion_timeout_generation
    global _live_exhaustion_overview_probe_in_flight
    global _live_exhaustion_overview_probe_sequence
    global _live_exhaustion_overview_ready, _live_exhaustion_overview_refresh_seen
    global _live_exhaustion_resume_callback, _live_exhaustion_resume_generation
    global _live_exhaustion_resume_guard_active
    _live_exhaustion_overview_ready = False
    _live_exhaustion_overview_refresh_seen = False
    _live_exhaustion_overview_probe_in_flight = False
    _live_exhaustion_overview_probe_sequence += 1
    _live_exhaustion_resume_callback = None
    _live_exhaustion_resume_guard_active = False
    _live_exhaustion_rebuild_generation = None
    _live_exhaustion_timeout_generation = None
    _live_exhaustion_resume_generation += 1


def _current_minimum_retention_extra(session=None) -> float:
    active = _active_session if session is None else session
    try:
        return max(0.0, float(active.engine.minimum_retention_extra))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _record_live_review_trace(
    event: str,
    *,
    phase: str | None = None,
    session=None,
    **details,
) -> None:
    with suppress(Exception):
        _live_review_trace.record(
            event,
            phase=phase,
            minimum_retention_extra=_current_minimum_retention_extra(session),
            **details,
        )


def _live_runtime_session(session: LiveReviewBridgeSession):
    runtime_session = getattr(session, "runtime_session", None)
    return runtime_session if runtime_session is not None else manager_for_mw(mw)


def install_live_review_bridge_hooks() -> None:
    global _live_review_hooks_installed
    if _live_review_hooks_installed:
        return
    if hasattr(gui_hooks, "reviewer_did_show_question"):
        gui_hooks.reviewer_did_show_question.append(_on_reviewer_did_show_question)
    if hasattr(gui_hooks, "reviewer_did_answer_card"):
        gui_hooks.reviewer_did_answer_card.append(_on_reviewer_did_answer_card)
    if hasattr(gui_hooks, "reviewer_will_end"):
        gui_hooks.reviewer_will_end.append(_on_reviewer_will_end)
    if hasattr(gui_hooks, "overview_did_refresh"):
        gui_hooks.overview_did_refresh.append(_on_overview_did_refresh)
    if hasattr(gui_hooks, "state_did_undo"):
        gui_hooks.state_did_undo.append(_on_state_did_undo)
    if hasattr(gui_hooks, "undo_state_did_change"):
        gui_hooks.undo_state_did_change.append(_on_undo_state_did_change)
    if hasattr(gui_hooks, "operation_did_execute"):
        gui_hooks.operation_did_execute.append(_on_operation_did_execute)
    if hasattr(gui_hooks, "webview_will_set_content"):
        gui_hooks.webview_will_set_content.append(_on_webview_will_set_content)
    if hasattr(gui_hooks, "profile_will_close"):
        gui_hooks.profile_will_close.append(_on_profile_will_close)
    _live_review_hooks_installed = True


def live_review_bridge_hooks_installed() -> bool:
    return _live_review_hooks_installed


def show_active_review_prototype(
    deck_id: int | None = None,
    *,
    retentions: tuple[DeckRetention, ...] | None = None,
    review_limit: int | None = None,
    minimum_review_limit: int = 0,
    order_index: int | None = None,
    same_day_only: bool = False,
    extra_search: str | None = None,
    adaptive_retention_settings: AdaptiveRetentionSettings | None = None,
    random_card_percent: float = 0.0,
) -> None:
    if mw.col is None:
        return
    config = addon_config_for_mw(mw)
    source_deck_id = int(deck_id if deck_id is not None else current_deck_id(mw.col))
    request = _LiveReviewStartRequest(
        source_deck_id=source_deck_id,
        settings=LiveReviewSettings(
            allow_same_day_repeats=True,
            hot_predict_limit=live_review_sameday_prediction_limit(config),
            prediction_refresh_limit=live_review_prediction_refresh_limit(config),
            quiet_refresh_attempts=live_review_quiet_refresh_attempts(config),
            same_day_reentry_delay_reviews=(live_review_sameday_reentry_delay_reviews(config)),
            review_limit=review_limit,
            minimum_review_limit=int(minimum_review_limit),
            minimum_retention_extra_quantum=minimum_review_widening_extra(config),
            order_index=(
                int(FilteredDeckOrder.RETRIEVABILITY_ASCENDING)
                if order_index is None
                else int(order_index)
            ),
            adaptive_retention=adaptive_retention_settings,
            random_card_percent=0.0 if same_day_only else random_card_percent,
        ),
        retentions=retentions,
        same_day_only=bool(same_day_only),
        extra_search=str(extra_search or "").strip(),
        # FSRS comparison is inexpensive relative to the Live Session work and
        # makes every retention-history record directly comparable.
        fsrs_comparison_enabled=True,
        started_at_ms=int(time.time() * 1000),
    )
    _start_active_review_from_request(request)


def _start_active_review_from_request(request: _LiveReviewStartRequest) -> None:
    global _live_review_start_token
    if mw.col is None:
        return
    # A Browser Card Info load is a retained, exclusive checkpoint lease. Live
    # Session takes ownership when explicitly started so it never waits behind an
    # open Browser window indefinitely.
    from .browser_card_info import close_browser_card_info_runtime

    close_browser_card_info_runtime(asynchronously=True)
    store = store_for_mw(mw)
    manager = manager_for_mw(mw)
    _stop_live_review_session(mw.col, show_retention=False)
    collection = mw.col
    start_token = object()
    _live_review_start_token = start_token
    pending_runtime = None
    pending_session = None

    def start_context_is_current() -> bool:
        return _live_review_start_token is start_token and mw.col is collection

    def require_current_start() -> None:
        if not start_context_is_current():
            raise CancelledError()

    def release_pending_resources() -> None:
        nonlocal pending_runtime, pending_session
        runtime, session = pending_runtime, pending_session
        pending_runtime = None
        pending_session = None
        if session is not None and start_context_is_current():
            # The final cancellation poll can run after refill has moved cards
            # into the temporary deck. Return only this request's cards while
            # it still owns the same collection and no replacement has started.
            _empty_session_deck(session, collection)
        _close_runtime_session(session if session is not None else runtime)

    def prepare_op(col, progress, _previous):
        nonlocal pending_runtime
        require_current_start()
        bootstrap = prepare_live_review_candidates_for_deck(
            col,
            source_deck_id=request.source_deck_id,
            store=store,
            manager=manager,
            progress=progress,
            retentions=request.retentions,
            same_day_only=request.same_day_only,
            extra_search=request.extra_search,
            adaptive_retention_settings=request.settings.adaptive_retention,
            defer_initial_prediction=True,
        )
        # Retain operation ownership across the runner's final cancellation
        # poll and the handoff between background stages.
        pending_runtime = bootstrap.runtime_session
        require_current_start()
        return bootstrap

    def start_op(col, progress, bootstrap):
        nonlocal pending_runtime, pending_session
        require_current_start()
        if not bootstrap.candidates:
            release_pending_resources()
            return bootstrap, None, None, None
        progress.update(0, 1, "Creating RWKV Live Session deck...")
        try:
            session = create_live_review_bridge_with_candidates(
                list(bootstrap.candidates),
                deck_name=LIVE_FILTERED_DECK_NAME,
                settings=request.settings,
                fsrs_comparison_enabled=request.fsrs_comparison_enabled,
            )
            session.runtime_session = bootstrap.runtime_session
            pending_session = session
            pending_runtime = None
            coordinator = None
            if bootstrap.review_data is not None and session.engine is not None:
                coordinator = LiveReviewPredictionCoordinator(
                    engine=session.engine,
                    review_data=bootstrap.review_data,
                    runtime_session=_live_runtime_session(session),
                )
                native_active = coordinator.activate_native_session()
                if not native_active:
                    raise RuntimeError(
                        "RWKV-SRS could not create the deferred native Live Session."
                    )
            start_selection = coordinator.native_start_selection
            if start_selection is None or not start_selection.selected:
                coordinator.deactivate_native_session()
                release_pending_resources()
                return bootstrap, None, None, None
            result = session.refill_buffer(col)
        except BaseException:
            release_pending_resources()
            raise
        progress.update(1, 1, "Created RWKV Live Session deck.")
        return bootstrap, session, result, coordinator

    def success(payload) -> None:
        global _active_prediction_coordinator, _active_session, _active_start_request
        nonlocal pending_runtime, pending_session
        if not start_context_is_current():
            release_pending_resources()
            return
        bootstrap, session, result, coordinator = payload
        if not bootstrap.candidates or session is None or result is None:
            release_pending_resources()
            clear_live_review_session()
            _show_no_live_review_candidates(from_exhaustion=request.from_exhaustion)
            return
        if result.status != LiveBridgeStatus.REFILLED:
            release_pending_resources()
            clear_live_review_session()
            show_web_warning(
                "The RWKV Live Session could not place a card in its filtered deck.",
                title=_LIVE_SESSION_TITLE,
                parent=mw,
            )
            return
        _active_session = session
        _active_start_request = request
        _active_prediction_coordinator = coordinator
        pending_runtime = None
        pending_session = None
        _live_review_trace.reset(phase="starting")
        _record_live_review_trace(
            "session_started",
            phase="waiting_for_question",
            session=session,
            selected_card_id=result.selected_card_id,
            from_exhaustion=bool(request.from_exhaustion),
        )
        _update_live_review_status_overlay()
        tooltip(
            "RWKV Live Session started. "
            f"{_potential_review_count_text(bootstrap, request.settings, coordinator)} "
            f"First card: {result.selected_card_id}.",
            parent=mw,
        )
        _expect_live_review_question()
        _move_to_live_review_state(reset_timebox=True)

    def failure(exception: Exception) -> None:
        release_pending_resources()
        if not start_context_is_current():
            return
        _stop_live_review_session(mw.col, show_retention=False)
        handle_checkpoint_failure(
            exception,
            lambda: _start_active_review_from_request(request),
            parent=mw,
        )

    def cancel() -> None:
        release_pending_resources()
        if start_context_is_current():
            _stop_live_review_session(mw.col, show_retention=False)

    run_with_progress_stages(
        parent=mw,
        title=_LIVE_SESSION_TITLE,
        label="Preparing RWKV Live Session...",
        stages=[
            ProgressStage(prepare_op, uses_collection=True),
            ProgressStage(start_op, uses_collection=True),
        ],
        on_success=success,
        on_failure=failure,
        on_cancel=cancel,
    )


def active_live_review_session() -> LiveReviewBridgeSession | None:
    return _active_session




def clear_live_review_session() -> None:
    global _active_prediction_coordinator, _active_session, _active_start_request
    global _live_review_start_token
    global _live_exhaustion_refresh_pending
    _live_exhaustion_refresh_pending = False
    _clear_live_review_transition_state()
    _clear_live_review_redo_block()
    session = _active_session
    _active_session = None
    _active_prediction_coordinator = None
    _active_start_request = None
    _live_review_start_token = None
    close = getattr(session, "close", None)
    if close is not None:
        with suppress(Exception):
            close()
    _hide_live_review_status_overlay()


def stop_live_review_session(col=None) -> None:
    _stop_live_review_session(col, show_retention=True)


def _termination_diagnostic_details(session, *, reason_code: str) -> str:
    """Return a diagnostic snapshot without ever blocking session teardown."""

    try:
        return _build_termination_diagnostic_details(
            session,
            reason_code=reason_code,
        )
    except Exception as exc:
        return json.dumps(
            {
                "reason_code": str(reason_code),
                "diagnostic_error": f"{type(exc).__name__}: {exc}",
            },
            indent=2,
            sort_keys=True,
        )


def _build_termination_diagnostic_details(session, *, reason_code: str) -> str:
    _record_live_review_trace(
        "termination_captured",
        phase="terminating",
        session=session,
        reason_code=str(reason_code),
    )
    reviewer = getattr(mw, "reviewer", None)
    request = _active_start_request
    coordinator = _active_prediction_coordinator
    session_snapshot: object
    diagnostics = getattr(session, "diagnostics", None)
    if diagnostics is None:
        session_snapshot = {"available": False}
    else:
        try:
            session_snapshot = diagnostics(mw.col)
        except Exception as exc:
            session_snapshot = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def card_id(card) -> int | None:
        try:
            return int(card.id)
        except (AttributeError, TypeError, ValueError):
            return None

    payload = {
        "reason_code": str(reason_code),
        "anki": {
            "state": getattr(mw, "state", None),
            "reviewer_available": reviewer is not None,
            "reviewer_card_id": card_id(getattr(reviewer, "card", None)),
            "reviewer_previous_card_id": card_id(
                getattr(reviewer, "previous_card", None),
            ),
        },
        "lifecycle_flags": {
            "exhaustion_refresh_pending": bool(_live_exhaustion_refresh_pending),
            "overview_refresh_seen": bool(_live_exhaustion_overview_refresh_seen),
            "overview_ready": bool(_live_exhaustion_overview_ready),
            "overview_probe_in_flight": bool(_live_exhaustion_overview_probe_in_flight),
            "resume_callback_pending": _live_exhaustion_resume_callback is not None,
            "resume_guard_active": bool(_live_exhaustion_resume_guard_active),
            "rebuild_in_flight": (
                _live_exhaustion_rebuild_generation
                == _live_exhaustion_resume_generation
            ),
            "rebuild_generation": _live_exhaustion_rebuild_generation,
            "timeout_pending": (
                _live_exhaustion_timeout_generation
                == _live_exhaustion_resume_generation
            ),
            "timeout_generation": _live_exhaustion_timeout_generation,
            "question_transition_pending": bool(_live_review_question_transition_pending),
        },
        "request": (
            None
            if request is None
            else {
                "source_deck_id": int(request.source_deck_id),
                "from_exhaustion": bool(request.from_exhaustion),
                "same_day_only": bool(request.same_day_only),
                "minimum_review_limit": int(request.settings.minimum_review_limit),
                "review_limit": request.settings.review_limit,
            }
        ),
        "coordinator": {
            "available": coordinator is not None,
            "native_session_active": (
                _coordinator_native_session_active(coordinator)
                if coordinator is not None
                else False
            ),
        },
        "session": session_snapshot,
        "trace": _live_review_trace.snapshot(),
    }
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _stop_live_review_session(
    col=None,
    *,
    show_retention: bool,
    retention_message: str | None = None,
    diagnostic_details: str | None = None,
) -> bool:
    global _active_prediction_coordinator, _active_session, _active_start_request
    global _live_review_start_token
    global _live_exhaustion_refresh_pending
    session = _active_session
    if show_retention and not retention_message:
        retention_message = _UNKNOWN_REVIEWER_EXIT_MESSAGE
        diagnostic_details = diagnostic_details or _termination_diagnostic_details(
            session,
            reason_code="unclassified_stop",
        )
    _live_exhaustion_refresh_pending = False
    _clear_live_review_transition_state()
    request = _active_start_request
    _active_session = None
    _active_prediction_coordinator = None
    _active_start_request = None
    _live_review_start_token = None
    _clear_live_review_redo_block()
    _empty_session_deck(session, col)
    close = getattr(session, "close", None)
    if close is not None:
        with suppress(Exception):
            close()
    _hide_live_review_status_overlay()
    shown = False
    if show_retention:
        if request is None:
            shown = _show_live_review_retention_summary(
                session,
                message=retention_message,
                diagnostic_details=diagnostic_details,
            )
        else:
            shown = _show_live_review_retention_summary(
                session,
                message=retention_message,
                diagnostic_details=diagnostic_details,
                request=request,
            )
    return shown


def _stop_live_review_with_reason(
    col,
    *,
    message: str,
    diagnostic_details: str | None = None,
    quiet_if_no_summary: bool = False,
) -> bool:
    """Stop and ensure the reason remains visible without retention rows."""

    shown = _stop_live_review_session(
        col,
        show_retention=True,
        retention_message=message,
        diagnostic_details=diagnostic_details,
    )
    if not shown:
        presenter = show_quiet_info if quiet_if_no_summary else show_web_warning
        presenter(message, title=_LIVE_SESSION_TITLE, parent=mw)
    return shown


def _on_reviewer_will_end() -> None:
    # This hook is installed globally while the Live Session feature is enabled,
    # so Anki also invokes it when an ordinary study session ends.  Termination
    # reporting and Live-deck cleanup belong exclusively to an active RWKV Live
    # Session; without this ownership guard a normal queue exhaustion is
    # misreported as an RWKV failure.
    if _active_session is None:
        return
    if _live_exhaustion_refresh_pending and not _live_review_question_transition_pending:
        _hide_live_review_status_overlay()
        return
    message, diagnostic_details = _reviewer_exit_retention_report(_active_session)
    _stop_live_review_with_reason(
        mw.col,
        message=message,
        diagnostic_details=diagnostic_details,
        quiet_if_no_summary=(message == _UNCLASSIFIED_REVIEWER_EXIT_MESSAGE),
    )


def _on_overview_did_refresh(_overview) -> None:
    """Wait for Anki's temporary finished page to finish its initial request.

    An exhaustion recheck begins after Reviewer discovers an empty filtered
    deck. Anki consequently opens its Svelte congratulations page, which starts
    an asynchronous backend request. The hook fires when that navigation starts,
    not when the request finishes, so it is only a signal to begin probing.
    """

    global _live_exhaustion_overview_refresh_seen
    if not _live_exhaustion_refresh_pending:
        return
    _live_exhaustion_overview_refresh_seen = True
    _record_live_review_trace("overview_refresh_seen", phase="waiting_for_finished_page")
    _probe_live_exhaustion_overview(_live_exhaustion_resume_generation)


def _reviewer_exit_retention_report(session) -> tuple[str, str]:
    """Explain every reviewer exit that lacks a normal Live Session result."""

    if _live_review_question_transition_pending:
        reason_code = "question_transition_interrupted"
        message = _UNEXPECTED_REVIEWER_EXIT_MESSAGE
        return message, _termination_diagnostic_details(session, reason_code=reason_code)
    reviewer = getattr(mw, "reviewer", None)
    if reviewer is None or not hasattr(reviewer, "card"):
        reason_code = "reviewer_state_unavailable"
        message = _UNKNOWN_REVIEWER_EXIT_MESSAGE
        return message, _termination_diagnostic_details(session, reason_code=reason_code)
    if reviewer.card is not None:
        # Anki retains the current card when the user intentionally leaves the
        # reviewer (for example, by returning to the deck overview). A delayed
        # Anki overview transition after an exhaustion refresh has the same
        # observable shape, so report the evidence without claiming intent.
        reason_code = "reviewer_left_with_card"
        message = _UNCLASSIFIED_REVIEWER_EXIT_MESSAGE
        return message, _termination_diagnostic_details(session, reason_code=reason_code)
    # Reviewer.nextCard() clears card before asking the scheduler for another
    # card. If that request returns nothing, Anki exits review with card=None.
    reason_code = "review_queue_lost"
    message = _UNEXPECTED_REVIEWER_EXIT_MESSAGE
    return message, _termination_diagnostic_details(session, reason_code=reason_code)


def _on_profile_will_close() -> None:
    _stop_live_review_session(mw.col, show_retention=False)


def _empty_session_deck(session, col) -> None:
    if session is None or col is None:
        return
    empty = getattr(session, "empty_live_filtered_deck", None)
    if empty is None:
        return
    with suppress(Exception):
        empty(col)


def _on_reviewer_did_show_question(card) -> None:
    session = _active_session
    if session is None:
        return
    fsrs_prediction = _fsrs_prediction_for_live_card(session, card)
    if _capture_live_review_shown_card(
        session,
        card,
        fsrs_prediction=fsrs_prediction,
    ):
        _complete_live_review_question_transition()
        _update_live_review_status_overlay()
        # Do not schedule a background RWKV refresh while the user is viewing a
        # card. The answer hook performs the authoritative post-answer refresh
        # after undoable_process_one(), then rebuilds the live deck from that
        # refreshed state. Background refresh can be reconsidered later if we
        # need latency hiding for very large refresh budgets, but it creates
        # extra in-flight work that the answer path has to retire safely.


def _fsrs_prediction_for_live_card(
    session: LiveReviewBridgeSession,
    card,
) -> float | None:
    if mw.col is None or not getattr(session, "active", False):
        return None
    if not getattr(session, "fsrs_comparison_enabled", True):
        return None
    try:
        filtered_deck_id = int(session.filtered_deck_id)
        card_deck_id = int(card.did)
        card_id = int(card.id)
    except (TypeError, ValueError, AttributeError):
        return None
    if card_deck_id != filtered_deck_id:
        return None
    with suppress(Exception):
        return fsrs_retrievabilities_for_card_ids(mw.col, [card_id]).get(card_id)
    return None


def _on_reviewer_did_answer_card(_reviewer, card, ease: int) -> None:
    global _active_prediction_coordinator, _active_session
    if _active_session is None or mw.col is None:
        return
    minimum_retention_extra_before = _current_minimum_retention_extra(_active_session)
    try:
        session = _active_session
        coordinator = _active_prediction_coordinator
        review_row = (
            _load_answer_review_row(card)
            if getattr(session, "current_card_id", None) == int(card.id)
            else None
        )
        result = session.handle_answered_card(
            mw.col,
            card,
            int(ease),
            review_row=review_row,
            process_review_row=(
                getattr(coordinator, "process_answer", None) if coordinator is not None else None
            ),
            refresh_predictions_before_refill=(
                (
                    lambda: _refresh_live_predictions_before_refill(
                        session,
                        coordinator,
                    )
                )
                if coordinator is not None
                else None
            ),
            stale_recheck=(
                lambda checked_card_ids, needed_count: _recheck_stale_live_candidates(
                    session,
                    checked_card_ids=checked_card_ids,
                    needed_count=needed_count,
                )
            ),
        )
    except Exception as exc:
        _stop_live_review_session(mw.col, show_retention=False)
        show_web_warning(
            f"RWKV Live Session stopped because the answered review could not be processed: {exc}",
            title=_LIVE_SESSION_TITLE,
            parent=mw,
        )
        return
    minimum_retention_extra_after = _current_minimum_retention_extra(session)
    if (
        result.status != LiveBridgeStatus.REFILLED
        or minimum_retention_extra_after != minimum_retention_extra_before
    ):
        _record_live_review_trace(
            "answer_result",
            phase="answer_applied",
            session=session,
            card_id=int(card.id),
            status=getattr(result.status, "value", str(result.status)),
            reviews_done=_session_reviews_done(session),
            minimum_retention_extra_before=minimum_retention_extra_before,
        )
    if result.status == LiveBridgeStatus.IGNORED:
        return
    _clear_live_review_redo_block()
    if result.status == LiveBridgeStatus.REFILLED:
        _expect_live_review_question()
        _update_live_review_status_overlay()
        _schedule_live_review_reviewer_reload()
        return
    if result.status == LiveBridgeStatus.STOPPED_LIMIT:
        _hide_live_review_status_overlay()
        message = _review_limit_reached_message(getattr(result, "reviews_done", None))
        tooltip("RWKV Live Session maximum review limit reached.", parent=mw)
        shown = _clear_active_session_after_stop_with_options(
            show_retention=True,
            retention_message=message,
        )
        if not shown:
            show_quiet_info(message, title=_LIVE_SESSION_TITLE, parent=mw)
        return
    elif result.status == LiveBridgeStatus.EMPTY:
        request = _active_start_request
        merge_undo_target = _current_undo_step(mw.col)
        _hide_live_review_status_overlay()
        _schedule_live_review_exhaustion_refresh(
            request,
            merge_undo_target=merge_undo_target,
        )
        return
    elif result.status == LiveBridgeStatus.PAUSED_FOR_UNDO:
        _hide_live_review_status_overlay()
        show_web_warning(
            "RWKV Live Session stopped because an undo reached reviews already "
            "processed into the live RWKV state. Start a new live session to reload.",
            title=_LIVE_SESSION_TITLE,
            parent=mw,
        )
    else:
        _hide_live_review_status_overlay()
        show_web_warning(
            "RWKV Live Session stopped because its filtered deck is unavailable.",
            title=_LIVE_SESSION_TITLE,
            parent=mw,
        )
    _clear_active_session_after_stop()


def _on_state_did_undo(_changes) -> None:
    global _active_prediction_coordinator, _active_session
    if _active_session is None or mw.col is None:
        return
    # Anki has already committed this Undo before this hook runs.  A full
    # exhaustion refresh, however, may still have collection/background work
    # in flight for the pre-Undo history and candidate universe.  Trying to
    # reconcile both operations would let a stale callback rebuild the Live
    # deck after Undo.  End only this Live Session; the successful Anki Undo is
    # preserved and a new session will reload the authoritative collection.
    if _live_exhaustion_refresh_pending:
        session = _active_session
        diagnostic_details = _termination_diagnostic_details(
            session,
            reason_code="undo_during_exhaustion_refresh",
        )
        _stop_live_review_with_reason(
            mw.col,
            message=_UNDO_DURING_EXHAUSTION_MESSAGE,
            diagnostic_details=diagnostic_details,
        )
        return
    try:
        coordinator = _active_prediction_coordinator
        rolled_back_review_ids: set[int] = set()

        def rollback_live_history(review_id: int) -> bool:
            if coordinator is None:
                return False
            normalized = int(review_id)
            rolled_back = bool(coordinator.rollback_review(normalized))
            if rolled_back:
                rolled_back_review_ids.add(normalized)
            return rolled_back

        result = _active_session.handle_undo(
            mw.col,
            undo_process=coordinator.undo_last_process,
            rollback_review=rollback_live_history,
            stale_recheck=(
                lambda checked_card_ids, needed_count: _recheck_stale_live_candidates(
                    _active_session,
                    checked_card_ids=checked_card_ids,
                    needed_count=needed_count,
                )
            ),
        )
        _record_live_review_trace(
            "undo_result",
            phase="undo_applied",
            session=_active_session,
            status=getattr(result.status, "value", str(result.status)),
            undone_review_id=getattr(result, "undone_review_id", None),
            undone_card_id=getattr(result, "undone_card_id", None),
        )
        # Compatibility fallback for test doubles or an older bridge object
        # that accepts but does not invoke the callback. The production bridge
        # performs this before any undo-time refill or prediction.
        if (
            result.undone_review_id is not None
            and result.status != LiveBridgeStatus.PAUSED_FOR_UNDO
            and int(result.undone_review_id) not in rolled_back_review_ids
            and not rollback_live_history(int(result.undone_review_id))
        ):
            raise RuntimeError(
                "RWKV could not remove the undone review from its live prediction history."
            )
        if (
            result.undone_review_id is not None
            and result.status != LiveBridgeStatus.PAUSED_FOR_UNDO
        ):
            _set_live_review_redo_block(
                review_id=result.undone_review_id,
                card_id=getattr(result, "undone_card_id", None),
                undo_counter=getattr(_changes, "counter", None),
            )
    except LiveReviewDeckUndoError as exc:
        session = _active_session
        _record_live_review_trace(
            "undo_deck_reconciliation_failed",
            phase="undo_applied",
            session=session,
            error=f"{type(exc).__name__}: {exc}",
        )
        diagnostic_details = _termination_diagnostic_details(
            session,
            reason_code="undo_deck_reconciliation_failed",
        )
        _stop_live_review_with_reason(
            mw.col,
            message=f"{_UNDO_DECK_RECONCILIATION_MESSAGE}\n\nTechnical detail: {exc}",
            diagnostic_details=diagnostic_details,
        )
        return
    except Exception as exc:
        _stop_live_review_session(mw.col, show_retention=False)
        show_web_warning(
            "RWKV Live Session stopped because undo could not be reconciled with "
            f"the prediction state: {exc}",
            title=_LIVE_SESSION_TITLE,
            parent=mw,
        )
        return
    if result.status == LiveBridgeStatus.IGNORED:
        return
    if result.status == LiveBridgeStatus.REFILLED:
        _expect_live_review_question()
        _update_live_review_status_overlay()
        _schedule_live_review_reviewer_reload()
        return
    if result.status == LiveBridgeStatus.PAUSED_FOR_UNDO:
        _hide_live_review_status_overlay()
        show_web_warning(
            "RWKV Live Session stopped because undo reverted a review that had "
            "already been processed into the live RWKV state. Start a new live "
            "session to reload.",
            title=_LIVE_SESSION_TITLE,
            parent=mw,
        )
    elif result.status == LiveBridgeStatus.EMPTY:
        request = _active_start_request
        _hide_live_review_status_overlay()
        _schedule_live_review_exhaustion_refresh(request)
        return
    elif result.status == LiveBridgeStatus.TRACKED_DECK_UNAVAILABLE:
        _hide_live_review_status_overlay()
        shown = _clear_active_session_after_stop_with_options(
            show_retention=True,
            retention_message=_UNDO_REMOVED_LIVE_DECK_MESSAGE,
        )
        if not shown:
            show_web_warning(
                _UNDO_REMOVED_LIVE_DECK_MESSAGE,
                title=_LIVE_SESSION_TITLE,
                parent=mw,
            )
        return
    else:
        _hide_live_review_status_overlay()
        show_web_warning(
            "RWKV Live Session stopped after undo because its filtered deck is unavailable.",
            title=_LIVE_SESSION_TITLE,
            parent=mw,
        )
    _clear_active_session_after_stop()


def _on_operation_did_execute(_changes, _handler) -> None:
    global _live_redo_block
    if _live_redo_block is None:
        return
    if _active_session is None or not getattr(_active_session, "active", False):
        _clear_live_review_redo_block()
        return
    review_exists = _live_review_review_row_exists(
        _live_redo_block.review_id,
        card_id=_live_redo_block.card_id,
    )
    if review_exists is False:
        if not _anki_redo_available():
            _clear_live_review_redo_block()
            return
        _disable_redo_action()
        return
    if review_exists is None:
        _live_redo_block = _LiveRedoBlock(
            review_id=_live_redo_block.review_id,
            card_id=_live_redo_block.card_id,
            undo_counter=_live_redo_block.undo_counter,
            verify_failure_count=_live_redo_block.verify_failure_count + 1,
        )
        if _live_redo_block.verify_failure_count >= _LIVE_REDO_VERIFY_FAILURE_LIMIT:
            _stop_live_review_session(mw.col, show_retention=False)
            show_web_warning(
                "RWKV Live Session stopped because redo state could not be "
                "verified. Start a new live session to reload.",
                title=_LIVE_SESSION_TITLE,
                parent=mw,
            )
            return
        _disable_redo_action()
        return

    _stop_live_review_session(mw.col, show_retention=False)
    show_web_warning(
        "RWKV Live Session stopped because an undone live answer appears to have "
        "been redone. Start a new live session to reload.",
        title=_LIVE_SESSION_TITLE,
        parent=mw,
    )


def _on_undo_state_did_change(_info) -> None:
    if _live_redo_block is not None:
        _disable_redo_action()


def _load_answer_review_row(card):
    coordinator = _active_prediction_coordinator
    if coordinator is None:
        raise ValueError("RWKV Live Session prediction state is unavailable.")
    if mw.col is None:
        raise ValueError("Anki collection is unavailable.")
    rows = coordinator.load_new_review_rows_for_card(mw.col, int(card.id))
    if len(rows) > 1:
        raise ValueError("More than one new review row was found for the answered card.")
    if not rows:
        raise ValueError("No new review row was found for the answered card.")
    return rows[0]


def _recheck_stale_live_candidates(
    session: LiveReviewBridgeSession | None,
    *,
    checked_card_ids: set[int] | None = None,
    needed_count: int = 1,
) -> bool:
    coordinator = _active_prediction_coordinator
    if session is None or coordinator is None or not getattr(session, "active", False):
        return False
    # Each quiet pass is an expansion beyond the ordinary post-answer refresh,
    # not a repeat of it. Native membership IDs cross the worker boundary only
    # here, after the compact normal result proved insufficient.
    coordinator.extend_quiet_recheck_exclusions(checked_card_ids)
    result = coordinator.recheck_stale_candidates(
        checked_card_ids=checked_card_ids,
        needed_count=needed_count,
        max_batches=1,
    )
    return result.found_eligible


def _clear_active_session_after_stop(*, preserve_start_request: bool = False) -> None:
    _clear_active_session_after_stop_with_options(
        preserve_start_request=preserve_start_request,
        show_retention=False,
    )


def _clear_active_session_after_stop_with_options(
    *,
    preserve_start_request: bool = False,
    show_retention: bool = False,
    retention_message: str | None = None,
) -> bool:
    global _active_prediction_coordinator, _active_session, _active_start_request
    global _live_exhaustion_refresh_pending
    _live_exhaustion_refresh_pending = False
    _clear_live_review_transition_state()
    session = _active_session
    request = _active_start_request
    close = getattr(session, "close", None)
    if close is not None:
        with suppress(Exception):
            close()
    _active_session = None
    _active_prediction_coordinator = None
    if not preserve_start_request:
        _active_start_request = None
    _clear_live_review_redo_block()
    if show_retention:
        if request is None:
            return _show_live_review_retention_summary(
                session,
                message=retention_message,
            )
        return _show_live_review_retention_summary(
            session,
            message=retention_message,
            request=request,
        )
    return False


def _show_live_review_retention_summary(
    session,
    *,
    message: str | None = None,
    diagnostic_details: str | None = None,
    request: _LiveReviewStartRequest | None = None,
) -> bool:
    if session is None:
        return False
    if not message:
        message = _MISSING_COMPLETION_REASON_MESSAGE
        diagnostic_details = diagnostic_details or _termination_diagnostic_details(
            session,
            reason_code="missing_completion_reason",
        )
    retention_summary = getattr(session, "retention_summary", None)
    if retention_summary is None:
        return False
    try:
        summary = retention_summary()
    except Exception:
        return False
    if getattr(summary, "review_count", 0) <= 0:
        return False
    _persist_live_review_retention_summary(session, summary, request=request)
    try:
        from .live_review_retention_dialog import LiveReviewRetentionDialog

        dialog = LiveReviewRetentionDialog(
            mw,
            summary,
            message=message,
            diagnostic_details=diagnostic_details,
            include_fsrs=getattr(session, "fsrs_comparison_enabled", True),
        )
        _live_retention_dialogs.append(dialog)
        finished = getattr(dialog, "finished", None)
        if finished is not None:

            def dialog_finished(_result, *, dialog=dialog) -> None:
                _forget_live_retention_dialog(dialog)

            qconnect(
                finished,
                dialog_finished,
            )
        dialog.show()
        return True
    except Exception:
        return False


def _persist_live_review_retention_summary(
    session,
    summary,
    *,
    request: _LiveReviewStartRequest | None,
) -> None:
    if (
        session is None
        or request is None
        or getattr(session, "_live_review_history_session_id", None)
    ):
        return
    with suppress(Exception):
        store = store_for_mw(mw)
        context = _live_review_history_context(
            session,
            request=request,
            ended_at_ms=int(time.time() * 1000),
        )
        session_id = append_live_review_history_session(
            store.live_review_history_path,
            summary,
            context,
        )
        if session_id is not None:
            session._live_review_history_session_id = session_id


def _live_review_history_context(
    session,
    *,
    request: _LiveReviewStartRequest | None,
    ended_at_ms: int,
) -> LiveReviewSessionContext:
    source_deck_id = int(request.source_deck_id) if request is not None else None
    settings = request.settings if request is not None else None
    config = addon_config_for_mw(mw)
    return LiveReviewSessionContext(
        started_at_ms=(
            int(request.started_at_ms)
            if request is not None and int(request.started_at_ms) > 0
            else ended_at_ms
        ),
        ended_at_ms=ended_at_ms,
        source_deck_id=source_deck_id,
        source_deck_name=_source_deck_name(source_deck_id),
        same_day_only=bool(request.same_day_only) if request is not None else False,
        allow_same_day_repeats=(
            bool(settings.allow_same_day_repeats) if settings is not None else False
        ),
        review_limit=settings.review_limit if settings is not None else None,
        minimum_review_limit=(int(settings.minimum_review_limit) if settings is not None else 0),
        order_index=int(settings.order_index) if settings is not None else None,
        model_id=configured_runtime_model_id(config),
        fsrs_comparison_enabled=bool(getattr(session, "fsrs_comparison_enabled", True)),
    )


def _source_deck_name(source_deck_id: int | None) -> str | None:
    if source_deck_id is None or mw.col is None:
        return None
    decks = getattr(mw.col, "decks", None)
    name = getattr(decks, "name", None)
    if callable(name):
        with suppress(Exception):
            return str(name(int(source_deck_id)))
    return None


def _forget_live_retention_dialog(dialog) -> None:
    with suppress(ValueError):
        _live_retention_dialogs.remove(dialog)


def _show_no_live_review_candidates(*, from_exhaustion: bool) -> None:
    message = (
        "No additional eligible review cards below desired retention were found."
        if from_exhaustion
        else "No eligible review cards below desired retention were found for "
        "the RWKV Live Session."
    )
    if from_exhaustion:
        show_quiet_info(message, title=_LIVE_SESSION_TITLE, parent=mw)
    else:
        show_web_warning(message, title=_LIVE_SESSION_TITLE, parent=mw)


def _potential_review_count_text(
    bootstrap,
    settings: LiveReviewSettings,
    coordinator: LiveReviewPredictionCoordinator | None = None,
) -> str:
    if bootstrap.initial_prediction_deferred and coordinator is not None:
        initial = coordinator.native_current_universe_result
        if initial is not None:
            below_dr = int(initial.eligible_count)
            if int(settings.minimum_review_limit) <= 0:
                noun = "review" if below_dr == 1 else "reviews"
                return f"Found {below_dr} potential {noun}."
            active = int(initial.active_count)
            noun = "card" if active == 1 else "cards"
            return (
                f"Found {active} candidate {noun} ({below_dr} currently below desired retention)."
            )
    below_dr = sum(1 for candidate in bootstrap.candidates if candidate.eligible)
    if int(settings.minimum_review_limit) <= 0:
        noun = "review" if below_dr == 1 else "reviews"
        return f"Found {below_dr} potential {noun}."
    widenable = sum(
        1
        for candidate in bootstrap.candidates
        if _candidate_can_be_selected_with_minimum(candidate)
    )
    noun = "review" if widenable == 1 else "reviews"
    return f"Found {widenable} potential {noun} ({below_dr} below desired retention)."


def _candidate_can_be_selected_with_minimum(candidate) -> bool:
    return can_satisfy_minimum_retention(
        candidate.predicted_retrievability,
        candidate.active_desired_retention,
        is_intraday=_candidate_is_intraday(candidate),
    )


def _candidate_is_intraday(candidate) -> bool:
    try:
        elapsed_days = candidate.metadata.get("elapsed_days", -1)
    except AttributeError:
        return False
    return is_same_day_elapsed(elapsed_days)


def _review_limit_reached_message(reviews_done: int | None) -> str:
    try:
        count = int(reviews_done)
    except (TypeError, ValueError):
        count = 0
    if count == 1:
        return "Maximum review limit reached. Nice work - you completed 1 RWKV Live review."
    return (
        "Maximum review limit reached. Nice work - you completed "
        f"{max(0, count)} RWKV Live reviews."
    )


def _live_exhaustion_complete_message(session) -> str:
    review_count = _session_reviews_done(session)
    minimum = _session_minimum_review_limit(session)
    review_word = "review" if review_count == 1 else "reviews"
    if minimum <= 0:
        return (
            "Nice work - you completed "
            f"{review_count} RWKV Live {review_word}. No additional cards below "
            "desired retention were found."
        )

    minimum_word = "review" if minimum == 1 else "reviews"
    if review_count >= minimum:
        verb = "met" if review_count == minimum else "exceeded"
        completed = (
            ""
            if review_count == minimum
            else f" by completing {review_count} RWKV Live {review_word}"
        )
        return (
            "Nice work - you "
            f"{verb} your minimum review threshold of {minimum} RWKV Live "
            f"{minimum_word}{completed}. No additional cards below desired "
            "retention were found."
        )
    return (
        "No additional cards below desired retention were found before reaching "
        f"your minimum review threshold of {minimum} RWKV Live {minimum_word}."
    )


def _session_reviews_done(session) -> int:
    try:
        return max(0, int(getattr(session, "reviews_done", 0)))
    except (TypeError, ValueError):
        return 0


def _session_minimum_review_limit(session) -> int:
    try:
        engine = getattr(session, "engine", None)
        settings = getattr(engine, "settings", None)
        return max(0, int(getattr(settings, "minimum_review_limit", 0)))
    except (TypeError, ValueError):
        return 0


def _defer_live_review_resume_until_overview_settles(
    callback: Callable[[], None],
) -> None:
    global _live_exhaustion_resume_callback, _live_exhaustion_resume_guard_active
    generation = _live_exhaustion_resume_generation
    _live_exhaustion_resume_callback = callback
    _live_exhaustion_resume_guard_active = True
    _record_live_review_trace(
        "finished_page_wait_started",
        phase="waiting_for_finished_page",
        generation=generation,
    )
    watchdog_scheduled = _schedule_live_exhaustion_timer(
        _LIVE_EXHAUSTION_RESUME_TIMEOUT_MS,
        lambda generation=generation: _live_exhaustion_resume_timed_out(generation),
    )
    if not watchdog_scheduled:
        _live_exhaustion_resume_timed_out(generation)
        return
    if _live_exhaustion_overview_ready:
        _schedule_live_exhaustion_timer(
            0,
            lambda generation=generation: _run_pending_live_exhaustion_resume(
                generation,
            ),
        )
        return
    if _live_exhaustion_overview_refresh_seen:
        _probe_live_exhaustion_overview(generation)


def _schedule_live_exhaustion_timer(
    delay_ms: int,
    callback: Callable[[], None],
) -> bool:
    progress = getattr(mw, "progress", None)
    single_shot = getattr(progress, "single_shot", None)
    if single_shot is None:
        if int(delay_ms) <= 0:
            callback()
            return True
        return False
    try:
        single_shot(
            max(0, int(delay_ms)),
            callback,
            True,
        )
        return True
    except Exception:
        if int(delay_ms) <= 0:
            callback()
            return True
        return False


def _probe_live_exhaustion_overview(generation: int) -> None:
    global _live_exhaustion_overview_probe_in_flight
    global _live_exhaustion_overview_probe_sequence
    if (
        generation != _live_exhaustion_resume_generation
        or not _live_exhaustion_refresh_pending
        or _live_exhaustion_overview_probe_in_flight
    ):
        return
    if getattr(mw, "state", None) != "overview":
        return
    web = getattr(mw, "web", None)
    evaluate = getattr(web, "evalWithCallback", None)
    if evaluate is None:
        return
    _live_exhaustion_overview_probe_in_flight = True
    _live_exhaustion_overview_probe_sequence += 1
    probe_sequence = _live_exhaustion_overview_probe_sequence

    def complete(page_state) -> None:
        _complete_live_exhaustion_overview_probe(
            generation,
            probe_sequence,
            page_state,
        )

    try:
        _schedule_live_exhaustion_timer(
            _LIVE_EXHAUSTION_OVERVIEW_PROBE_TIMEOUT_MS,
            lambda generation=generation, probe_sequence=probe_sequence: (
                _expire_live_exhaustion_overview_probe(
                    generation,
                    probe_sequence,
                )
            ),
        )
        evaluate(_EXHAUSTION_PAGE_STATE_SCRIPT, complete)
    except Exception:
        _live_exhaustion_overview_probe_in_flight = False
        _live_exhaustion_overview_probe_sequence += 1
        _schedule_live_exhaustion_timer(
            _LIVE_EXHAUSTION_OVERVIEW_PROBE_MS,
            lambda generation=generation: _probe_live_exhaustion_overview(generation),
        )


def _complete_live_exhaustion_overview_probe(
    generation: int,
    probe_sequence: int,
    page_state,
) -> None:
    global _live_exhaustion_overview_probe_in_flight, _live_exhaustion_overview_ready
    if (
        generation != _live_exhaustion_resume_generation
        or probe_sequence != _live_exhaustion_overview_probe_sequence
    ):
        return
    _live_exhaustion_overview_probe_in_flight = False
    if not _live_exhaustion_refresh_pending:
        return
    normalized_state = str(page_state or "")
    _record_live_review_trace(
        "finished_page_probe",
        phase="waiting_for_finished_page",
        page_state=normalized_state,
    )
    # The hook fires immediately after Anki starts loading ``/congrats``. The
    # old ordinary Overview DOM (including ``#study``) can still be visible for
    # a moment, so only the mounted Svelte congratulations component is proof
    # that its awaited ``congratsInfo()`` request has returned.
    if normalized_state == "congrats-ready":
        _live_exhaustion_overview_ready = True
        _record_live_review_trace(
            "finished_page_ready",
            phase="rebuilding_live_deck",
            page_state=normalized_state,
        )
        if _live_exhaustion_resume_callback is not None:
            _schedule_live_exhaustion_timer(
                0,
                lambda generation=generation: _run_pending_live_exhaustion_resume(
                    generation,
                ),
            )
        return
    _schedule_live_exhaustion_timer(
        _LIVE_EXHAUSTION_OVERVIEW_PROBE_MS,
        lambda generation=generation: _probe_live_exhaustion_overview(generation),
    )


def _expire_live_exhaustion_overview_probe(
    generation: int,
    probe_sequence: int,
) -> None:
    global _live_exhaustion_overview_probe_in_flight
    global _live_exhaustion_overview_probe_sequence
    if (
        generation != _live_exhaustion_resume_generation
        or probe_sequence != _live_exhaustion_overview_probe_sequence
        or not _live_exhaustion_overview_probe_in_flight
    ):
        return
    _live_exhaustion_overview_probe_in_flight = False
    # Invalidate the callback immediately.  It may arrive after this timeout
    # but before the retry has had a chance to allocate its own sequence.
    _live_exhaustion_overview_probe_sequence += 1
    _record_live_review_trace(
        "finished_page_probe_expired",
        phase="waiting_for_finished_page",
    )
    _schedule_live_exhaustion_timer(
        _LIVE_EXHAUSTION_OVERVIEW_PROBE_MS,
        lambda generation=generation: _probe_live_exhaustion_overview(generation),
    )


def _run_pending_live_exhaustion_resume(generation: int) -> None:
    global _live_exhaustion_resume_callback
    if (
        generation != _live_exhaustion_resume_generation
        or not _live_exhaustion_overview_ready
    ):
        return
    callback = _live_exhaustion_resume_callback
    _live_exhaustion_resume_callback = None
    if callback is not None:
        callback()


def _live_exhaustion_resume_timed_out(generation: int) -> None:
    global _live_exhaustion_timeout_generation
    if (
        generation != _live_exhaustion_resume_generation
        or not _live_exhaustion_resume_guard_active
        or not _live_exhaustion_refresh_pending
    ):
        return
    if _live_exhaustion_rebuild_generation == generation:
        # ``restart_with_candidates()`` owns the native session on Anki's
        # collection worker.  Closing that session from the UI watchdog would
        # race the worker (and can hang native teardown on some platforms).
        # Mark the timeout and let the task's main-thread completion callback
        # perform teardown after the worker has released it.
        _live_exhaustion_timeout_generation = generation
        _record_live_review_trace(
            "resume_watchdog_expired_during_rebuild",
            phase="waiting_for_live_deck_rebuild",
        )
        return
    _stop_live_exhaustion_after_timeout(generation)


def _stop_live_exhaustion_after_timeout(generation: int) -> None:
    if (
        generation != _live_exhaustion_resume_generation
        or not _live_exhaustion_resume_guard_active
        or not _live_exhaustion_refresh_pending
    ):
        return
    session = _active_session
    diagnostic_details = _termination_diagnostic_details(
        session,
        reason_code="return_to_review_timeout",
    )
    _stop_live_review_with_reason(
        mw.col,
        message=_EXHAUSTION_RESUME_TIMEOUT_MESSAGE,
        diagnostic_details=diagnostic_details,
    )


def _release_live_exhaustion_rebuild(generation: int) -> bool:
    """Release the rebuild guard only when its owning generation completes."""

    global _live_exhaustion_rebuild_generation
    if _live_exhaustion_rebuild_generation != generation:
        return False
    _live_exhaustion_rebuild_generation = None
    return True


def _schedule_live_review_exhaustion_refresh(
    request: _LiveReviewStartRequest | None,
    *,
    merge_undo_target: int | None = None,
) -> None:
    global _live_exhaustion_refresh_pending
    _clear_live_review_transition_state()
    _live_exhaustion_refresh_pending = True
    _record_live_review_trace(
        "exhaustion_refresh_scheduled",
        phase="exhaustion_refresh_pending",
        merge_undo=merge_undo_target is not None,
    )

    def refresh() -> None:
        _refresh_live_review_after_exhaustion(
            request,
            merge_undo_target=merge_undo_target,
        )

    progress = getattr(mw, "progress", None)
    single_shot = getattr(progress, "single_shot", None)
    if single_shot is None:
        refresh()
        return
    with suppress(Exception):
        single_shot(0, refresh, True)
        return
    refresh()


def _schedule_live_review_reviewer_reload() -> None:
    session = _active_session
    expected_card_id = None
    with suppress(AttributeError, IndexError, TypeError, ValueError):
        expected_card_id = int(session.buffered_card_ids[0])

    def reload_reviewer() -> None:
        if (
            session is None
            or _active_session is not session
            or not getattr(session, "active", False)
        ):
            return
        with suppress(AttributeError, IndexError, TypeError, ValueError):
            if (
                expected_card_id is not None
                and int(session.buffered_card_ids[0]) != expected_card_id
            ):
                return
        reviewer = getattr(mw, "reviewer", None)
        next_card = getattr(reviewer, "nextCard", None)
        if next_card is None:
            return
        current_card = getattr(reviewer, "card", None)
        try:
            current_card_id = int(current_card.id)
        except (AttributeError, TypeError, ValueError):
            current_card_id = None
        if expected_card_id is not None and current_card_id == expected_card_id:
            _capture_expected_reviewer_card(session, current_card, expected_card_id)
            return
        with suppress(Exception):
            next_card()

    progress = getattr(mw, "progress", None)
    single_shot = getattr(progress, "single_shot", None)
    if single_shot is None:
        reload_reviewer()
        return
    with suppress(Exception):
        single_shot(0, reload_reviewer, True)
        return
    reload_reviewer()


def _capture_expected_reviewer_card(
    session: LiveReviewBridgeSession,
    card,
    expected_card_id: int | None,
) -> bool:
    if expected_card_id is None or card is None:
        return False
    try:
        if int(card.id) != int(expected_card_id):
            return False
    except (AttributeError, TypeError, ValueError):
        return False
    capture = getattr(session, "capture_shown_card", None)
    if capture is None:
        return False
    captured = _capture_live_review_shown_card(
        session,
        card,
        fsrs_prediction=_fsrs_prediction_for_live_card(session, card),
    )
    if captured:
        _complete_live_review_question_transition()
    return captured


def _capture_live_review_shown_card(
    session: LiveReviewBridgeSession,
    card,
    *,
    fsrs_prediction: float | None,
) -> bool:
    capture = getattr(session, "capture_shown_card", None)
    if capture is None:
        return False
    try:
        return bool(
            capture(
                card,
                fsrs_prediction=fsrs_prediction,
            )
        )
    except Exception:
        return False
    return False


def _refresh_live_review_after_exhaustion(
    request: _LiveReviewStartRequest | None,
    *,
    merge_undo_target: int | None = None,
) -> None:
    global _active_prediction_coordinator, _active_session, _active_start_request
    global _live_exhaustion_refresh_pending
    session = _active_session
    coordinator = _active_prediction_coordinator
    refresh_generation = _live_exhaustion_resume_generation
    if request is None or session is None or coordinator is None or mw.col is None:
        _live_exhaustion_refresh_pending = False
        tooltip("RWKV Live Session has no more eligible cards.", parent=mw)
        clear_live_review_session()
        return

    manager = manager_for_mw(mw)
    refresh_target_timestamp_seconds = time.time()

    def refresh_op(col, progress, _previous):
        if not refresh_context_is_current():
            return None
        return refresh_live_review_candidates_for_deck(
            col,
            source_deck_id=request.source_deck_id,
            review_data=coordinator.review_data,
            manager=manager,
            runtime=_live_runtime_session(session),
            progress=progress,
            retentions=request.retentions,
            same_day_only=request.same_day_only,
            extra_search=request.extra_search,
            adaptive_retention_settings=request.settings.adaptive_retention,
            target_timestamp_seconds=refresh_target_timestamp_seconds,
            defer_prediction=True,
        )

    def reconcile_op(_col, progress, bootstrap):
        if not refresh_context_is_current():
            return bootstrap, False
        if not bootstrap.candidates or not _coordinator_native_session_active(coordinator):
            return bootstrap, False
        progress.update(0, 1, "Reconciling RWKV Live Session candidates...")
        reconciliation = coordinator.reconcile_candidate_universe(
            bootstrap.candidates,
            target_timestamp_seconds=refresh_target_timestamp_seconds,
        )
        bootstrap = replace(
            bootstrap,
            card_ids=tuple(int(candidate.card_id) for candidate in reconciliation.candidates),
            candidates=reconciliation.candidates,
        )
        if not bootstrap.candidates or reconciliation.result is None:
            return bootstrap, False
        progress.update(1, 1, "Reconciled RWKV Live Session candidates.")
        return bootstrap, True

    def refresh_context_is_current() -> bool:
        # Exhaustion refresh runs through background progress stages. If the user
        # stops/restarts live review while that work is pending, stale callbacks
        # must not rebuild a filtered deck or resurrect the old session.
        return (
            _live_exhaustion_refresh_pending
            and refresh_generation == _live_exhaustion_resume_generation
            and _active_session is session
            and _active_prediction_coordinator is coordinator
            and _active_start_request is request
        )

    def finish_without_candidates() -> None:
        message = _live_exhaustion_complete_message(session)
        shown = _clear_active_session_after_stop_with_options(
            show_retention=True,
            retention_message=message,
        )
        if not shown:
            _show_no_live_review_candidates(from_exhaustion=True)

    def rebuild_live_deck(bootstrap) -> None:
        """Rebuild only after Anki's finished-page request has settled."""

        global _live_exhaustion_rebuild_generation

        if not refresh_context_is_current():
            return
        if getattr(mw, "state", None) != "overview":
            stop_after_rebuild_navigation()
            return
        _record_live_review_trace(
            "live_deck_rebuild_started",
            phase="rebuilding_live_deck",
            session=session,
            candidate_count=len(bootstrap.candidates),
        )

        def task():
            if (
                not refresh_context_is_current()
                or _live_exhaustion_timeout_generation == refresh_generation
            ):
                return None
            return session.restart_with_candidates(
                mw.col,
                list(bootstrap.candidates),
                merge_undo_target=merge_undo_target,
                merge_undo=merge_undo_target is not None,
                candidate_universe_prepared=True,
            )

        def done(future) -> None:
            _release_live_exhaustion_rebuild(refresh_generation)
            try:
                exception = future.exception()
            except Exception as exc:
                failure(exc)
                return
            if exception is not None:
                failure(exception)
                return
            try:
                result = future.result()
            except Exception as exc:
                failure(exc)
                return
            if not refresh_context_is_current():
                return
            if _live_exhaustion_timeout_generation == refresh_generation:
                _stop_live_exhaustion_after_timeout(refresh_generation)
                return
            if getattr(mw, "state", None) != "overview":
                stop_after_rebuild_navigation()
                return
            if result is None or result.status != LiveBridgeStatus.REFILLED:
                finish_without_candidates()
                return
            try:
                resume_reviewer(bootstrap, result)
            except Exception as exc:
                failure(exc)

        taskman = getattr(mw, "taskman", None)
        run_in_background = getattr(taskman, "run_in_background", None)
        if run_in_background is None:
            failure(RuntimeError("Anki's background task manager is unavailable."))
            return
        try:
            _live_exhaustion_rebuild_generation = refresh_generation
            run_in_background(task, done, uses_collection=True)
        except Exception as exc:
            _release_live_exhaustion_rebuild(refresh_generation)
            failure(exc)

    def stop_after_rebuild_navigation() -> None:
        if not refresh_context_is_current():
            return
        diagnostic_details = _termination_diagnostic_details(
            session,
            reason_code="navigation_during_live_deck_rebuild",
        )
        _stop_live_review_with_reason(
            mw.col,
            message=_EXHAUSTION_NAVIGATION_INTERRUPTED_MESSAGE,
            diagnostic_details=diagnostic_details,
        )

    def resume_reviewer(bootstrap, result) -> None:
        global _active_prediction_coordinator, _active_session, _active_start_request
        if not refresh_context_is_current():
            return
        restart_request = request.for_exhaustion_restart()
        _record_live_review_trace(
            "live_deck_rebuilt",
            phase="waiting_for_question",
            session=session,
            selected_card_id=result.selected_card_id,
        )
        _expect_live_review_question()
        _move_to_live_review_state(reset_timebox=False)
        if _active_session is not session:
            return
        _active_session = session
        _active_prediction_coordinator = coordinator
        _active_start_request = restart_request
        _update_live_review_status_overlay()
        tooltip(
            "RWKV Live Session rechecked the deck. "
            f"{_potential_review_count_text(bootstrap, request.settings, coordinator)} "
            f"Next card: {result.selected_card_id}.",
            parent=mw,
        )

    def success(payload) -> None:
        bootstrap, reconciled = payload
        if not refresh_context_is_current():
            return
        _record_live_review_trace(
            "exhaustion_recheck_finished",
            phase="waiting_for_finished_page",
            session=session,
            candidate_count=len(bootstrap.candidates),
            reconciled=bool(reconciled),
        )
        if not reconciled:
            finish_without_candidates()
            return
        _defer_live_review_resume_until_overview_settles(
            lambda: rebuild_live_deck(bootstrap),
        )

    def failure(exception: Exception) -> None:
        if not refresh_context_is_current():
            return
        _record_live_review_trace(
            "exhaustion_recheck_failed",
            phase="terminating",
            session=session,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
        )
        _stop_live_review_session(mw.col, show_retention=False)
        show_web_warning(
            f"RWKV Live Session stopped because the exhaustion recheck failed: {exception}",
            title=_LIVE_SESSION_TITLE,
            parent=mw,
        )

    def cancel() -> None:
        if not refresh_context_is_current():
            return
        diagnostic_details = _termination_diagnostic_details(
            session,
            reason_code="exhaustion_recheck_cancelled",
        )
        _stop_live_review_with_reason(
            mw.col,
            message=_EXHAUSTION_CANCELLED_MESSAGE,
            diagnostic_details=diagnostic_details,
            quiet_if_no_summary=True,
        )

    run_with_progress_stages(
        parent=mw,
        title=_LIVE_SESSION_TITLE,
        label="Rechecking RWKV Live Session candidates...",
        stages=[
            ProgressStage(refresh_op, uses_collection=True),
            ProgressStage(reconcile_op, uses_collection=True),
        ],
        on_success=success,
        on_failure=failure,
        on_cancel=cancel,
    )


def _move_to_live_review_state(*, reset_timebox: bool) -> None:
    move_to_state = getattr(mw, "moveToState", None)
    if move_to_state is None:
        return
    _start_live_review_timebox(reset=reset_timebox)
    move_to_state("review")


def _start_live_review_timebox(*, reset: bool) -> None:
    col = getattr(mw, "col", None)
    if col is None:
        return
    start_timebox = getattr(col, "startTimebox", None)
    if start_timebox is None:
        return
    if not reset and _live_review_timebox_is_started(col):
        return
    with suppress(Exception):
        start_timebox()


def _live_review_timebox_is_started(col) -> bool:
    try:
        return col._startTime is not None and col._startReps is not None
    except AttributeError:
        return False


def _on_webview_will_set_content(web_content, context) -> None:
    if not _is_reviewer_context(context):
        return
    web_content.head += _live_review_status_css()
    web_content.body += _live_review_status_html()


def _is_reviewer_context(context) -> bool:
    with suppress(Exception):
        from aqt.reviewer import Reviewer

        return isinstance(context, Reviewer)
    reviewer = getattr(mw, "reviewer", None)
    return context is reviewer


def _live_review_status_text(
    session: LiveReviewBridgeSession | None,
) -> tuple[str, str]:
    if session is None:
        return "", "hidden"
    if not getattr(session, "active", False):
        return "", "hidden"
    if _show_remaining_card_count_enabled():
        remaining = _live_review_remaining_count(session)
        if remaining is not None:
            return f"RWKV Live\n{remaining:>4} left", "active"
    return "RWKV Live", "active"


def _live_review_expansion_status_text(
    session: LiveReviewBridgeSession | None,
) -> str:
    if session is None or not getattr(session, "active", False):
        return ""
    engine = getattr(session, "engine", None)
    try:
        expansion = float(getattr(engine, "minimum_retention_extra", 0.0))
    except (TypeError, ValueError):
        return ""
    if expansion <= 0:
        return ""
    percent = max(1, int(round(expansion * 100)))
    return f"+{percent}%"


def _show_remaining_card_count_enabled() -> bool:
    col = getattr(mw, "col", None)
    if col is None:
        return False
    try:
        return bool(col.get_preferences().reviewing.show_remaining_due_counts)
    except Exception:
        return False


def _live_review_remaining_count(session: LiveReviewBridgeSession) -> int | None:
    try:
        limit = int(session.review_limit)
    except (AttributeError, TypeError, ValueError):
        return None
    try:
        reviews_done = int(session.reviews_done)
    except (AttributeError, TypeError, ValueError):
        reviews_done = 0
    return min(9999, max(0, limit - reviews_done))


def _update_live_review_status_overlay() -> None:
    text, kind = _live_review_status_text(_active_session)
    expansion_text = _live_review_expansion_status_text(_active_session)
    _set_live_review_status_overlay(text, kind=kind, expansion_text=expansion_text)


def _hide_live_review_status_overlay() -> None:
    _set_live_review_status_overlay("", kind="hidden", expansion_text="")


def _set_live_review_status_overlay(
    text: str,
    *,
    kind: str,
    expansion_text: str = "",
) -> None:
    web = getattr(getattr(mw, "reviewer", None), "web", None)
    if web is None or not hasattr(web, "eval"):
        return
    payload = json.dumps(
        {
            "text": _live_review_status_display_text(text),
            "kind": kind,
            "expansion_text": _live_review_status_display_text(expansion_text),
        },
        ensure_ascii=True,
    )
    # The reviewer content hook loads the frontend once per document. Review
    # answers send only a tiny presentation snapshot, never the compiled bundle.
    js = (
        "if (window.RWKVFrontend) {"
        f"window.RWKVFrontend.setLiveReviewStatus({payload});"
        "}"
    )
    with suppress(Exception):
        web.eval(js)


def _live_review_status_display_text(text: str) -> str:
    return text


def _live_review_status_css() -> str:
    return f"""
<style>
#{_LIVE_STATUS_ELEMENT_ID} {{
  position: fixed;
  right: 12px;
  bottom: 12px;
  z-index: 2147483647;
  display: flex;
  align-items: flex-end;
  gap: 6px;
  pointer-events: none;
}}
#{_LIVE_STATUS_ELEMENT_ID}[hidden] {{
  display: none;
}}
#{_LIVE_STATUS_ELEMENT_ID} .rwkv-live-review-badge {{
  padding: 5px 8px;
  border-radius: 6px;
  border: 1px solid rgba(72, 138, 236, 0.45);
  background: rgba(28, 34, 43, 0.88);
  color: #f3f6fb;
  font: 12px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-variant-numeric: tabular-nums;
  white-space: pre;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.22);
}}
#{_LIVE_STATUS_ELEMENT_ID}[data-kind="active"] .rwkv-live-review-main {{
  border-color: rgba(69, 180, 109, 0.62);
  background: rgba(18, 67, 42, 0.92);
  color: #dff8e8;
}}
#{_LIVE_STATUS_ELEMENT_ID} .rwkv-live-review-expansion {{
  border-color: rgba(217, 139, 35, 0.72);
  background: rgba(92, 61, 18, 0.94);
  color: #fff1cc;
}}
</style>
"""


def _live_review_status_html() -> str:
    from ..svelte_ui import shared_frontend_script

    # Anki owns the surrounding reviewer document. The badge stays unmounted
    # until an active session sends its first status; ordinary reviews are inert.
    return (
        '<script data-rwkv-live-status="true">if (!window.RWKVFrontend) {'
        + shared_frontend_script()
        + "}</script>"
    )


def _set_live_review_redo_block(
    *,
    review_id: int,
    card_id: int | None = None,
    undo_counter: int | None = None,
) -> None:
    global _live_redo_block
    _live_redo_block = _LiveRedoBlock(
        review_id=int(review_id),
        card_id=None if card_id is None else int(card_id),
        undo_counter=None if undo_counter is None else int(undo_counter),
        verify_failure_count=0,
    )
    _disable_redo_action()


def _clear_live_review_redo_block() -> None:
    global _live_redo_block
    _live_redo_block = None


def _disable_redo_action() -> None:
    action = getattr(getattr(mw, "form", None), "actionRedo", None)
    if action is not None:
        with suppress(Exception):
            action.setEnabled(False)


def _live_review_review_row_exists(
    review_id: int,
    *,
    card_id: int | None = None,
) -> bool | None:
    if mw.col is None:
        return None
    try:
        if card_id is None:
            existing = mw.col.db.scalar(
                "select 1 from revlog where id = ?",
                int(review_id),
            )
        else:
            existing = mw.col.db.scalar(
                "select 1 from revlog where id = ? and cid = ?",
                int(review_id),
                int(card_id),
            )
    except Exception:
        return None
    return existing is not None


def _anki_redo_available() -> bool:
    if mw.col is None:
        return False
    try:
        return bool(getattr(mw.col.undo_status(), "redo", None))
    except Exception:
        return False


def _current_undo_step(col) -> int | None:
    if col is None:
        return None
    try:
        step = getattr(col.undo_status(), "last_step", None)
        return None if step is None else int(step)
    except Exception:
        return None


def _refresh_live_predictions_before_refill(
    session: LiveReviewBridgeSession,
    coordinator: LiveReviewPredictionCoordinator | None,
) -> LivePredictionStatus | None:
    if (
        coordinator is None
        or _active_session is not session
        or _active_prediction_coordinator is not coordinator
        or not getattr(session, "active", False)
        or not getattr(session, "engine", None)
    ):
        return None
    if coordinator.in_flight:
        # The in-flight refresh was started before this answer was processed.
        # Retire it so the answer path can refresh from the new RWKV state; if
        # the old task returns later, its engine token will be stale.
        coordinator.finish_failed_refresh()
    job = coordinator.begin_refresh()
    if job is None:
        return None
    try:
        result = coordinator.run_job(job)
    except Exception:
        coordinator.finish_failed_refresh()
        raise
    return coordinator.apply_result(result).status


def _coordinator_native_session_active(coordinator) -> bool:
    return bool(getattr(coordinator, "native_session_active", False))
