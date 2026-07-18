from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .live_retention_table_html import render_live_retention_table
from .live_review_history import (
    LiveReviewHistoryOverview,
    LiveReviewHistorySession,
    LiveReviewPeriod,
    records_in_live_review_period,
)
from .live_review_history_html import render_live_review_history_html
from .live_review_stats import (
    LiveRetentionRecord,
    LiveRetentionSummary,
    summarize_live_retention_records,
)
from .web_dialog_bridge import BridgePayloadError, WebDialogCommand
from .web_dialog_controller import BaseWebDialogController

ALL_DECK_FILTER = "__all__"
UNKNOWN_DECK_FILTER = "__unknown_deck__"
RECENT_SESSION_PAGE_SIZE = 20

SELECT_DECK_ACTION = "select-deck"
APPLY_SEARCH_ACTION = "apply-search"
SET_RECENT_PAGE_ACTION = "set-recent-page"
SET_ALL_REVIEWS_PERIOD_ACTION = "set-all-reviews-period"


class LiveReviewHistorySearchUnavailable(RuntimeError):
    """Raised by the GUI search adapter when no Anki collection is available."""


@dataclass(frozen=True, slots=True)
class _DerivedHistoryKey:
    source_revision: int
    deck_filter: str
    search_revision: int
    include_fsrs: bool
    recent_session_page_size: int


@dataclass(frozen=True, slots=True)
class _DerivedHistoryData:
    key: _DerivedHistoryKey
    sessions: tuple[LiveReviewHistorySession, ...]
    summary: LiveRetentionSummary
    matching_session_count: int
    recent_page_count: int
    retention_table_html: str


@dataclass(frozen=True, slots=True)
class _AllReviewsDisplayData:
    summary: LiveRetentionSummary
    matching_session_count: int
    retention_table_html: str


class LiveReviewHistoryController(BaseWebDialogController):
    """Qt-independent filtering and rendering state for Live Review History."""

    actions = frozenset(
        {
            SELECT_DECK_ACTION,
            APPLY_SEARCH_ACTION,
            SET_RECENT_PAGE_ACTION,
            SET_ALL_REVIEWS_PERIOD_ACTION,
        }
    )

    def __init__(
        self,
        *,
        overview: LiveReviewHistoryOverview,
        search_card_ids: Callable[[str], Iterable[int]],
        include_fsrs: bool = True,
        is_dark: bool = False,
        recent_session_page_size: int = RECENT_SESSION_PAGE_SIZE,
        next_day_at_seconds: int | None = None,
    ) -> None:
        if not callable(search_card_ids):
            raise TypeError("search_card_ids must be callable")
        if next_day_at_seconds is None:
            next_day_at_seconds = int(time.time())
        if isinstance(next_day_at_seconds, bool) or not isinstance(
            next_day_at_seconds,
            int,
        ):
            raise TypeError("next_day_at_seconds must be an integer")
        if next_day_at_seconds <= 0:
            raise ValueError("next_day_at_seconds must be positive")
        self._overview = overview
        self._search_card_ids_callback = search_card_ids
        self._include_fsrs = bool(include_fsrs)
        self.is_dark = bool(is_dark)
        self.recent_session_page_size = max(1, int(recent_session_page_size))
        self._deck_options = deck_filter_options(overview.sessions)
        self._deck_labels = dict(self._deck_options)
        self._selected_deck_filter = ALL_DECK_FILTER
        self._search_input = ""
        self._applied_search_query = ""
        self._matching_card_ids: frozenset[int] | None = None
        self._search_error = ""
        self._source_revision = 0
        self._search_revision = 0
        self._derived_cache: _DerivedHistoryData | None = None
        self._all_reviews_display_cache: dict[
            tuple[_DerivedHistoryKey, LiveReviewPeriod],
            _AllReviewsDisplayData,
        ] = {}
        self._next_day_at_seconds = next_day_at_seconds
        self._selected_review_period = LiveReviewPeriod.ALL
        self._review_period_focus = False
        self._recent_session_page = 0
        self._recent_sessions_expanded = False
        self._recent_page_focus: str | None = None
        self._restore_scroll_y: float | None = None
        self._rerender: Callable[[], Any] | None = None

    @property
    def overview(self) -> LiveReviewHistoryOverview:
        return self._overview

    @overview.setter
    def overview(self, overview: LiveReviewHistoryOverview) -> None:
        self.replace_overview(overview)

    @property
    def include_fsrs(self) -> bool:
        return self._include_fsrs

    @include_fsrs.setter
    def include_fsrs(self, include_fsrs: bool) -> None:
        resolved = bool(include_fsrs)
        if resolved == self._include_fsrs:
            return
        self._include_fsrs = resolved
        self._invalidate_derived_cache()

    @property
    def deck_options(self) -> tuple[tuple[str, str], ...]:
        return self._deck_options

    @property
    def selected_deck_filter(self) -> str:
        return self._selected_deck_filter

    @property
    def search_input(self) -> str:
        return self._search_input

    @property
    def applied_search_query(self) -> str:
        return self._applied_search_query

    @property
    def search_error(self) -> str:
        return self._search_error

    @property
    def selected_review_period(self) -> LiveReviewPeriod:
        return self._selected_review_period

    @property
    def recent_session_page(self) -> int:
        return self._recent_session_page

    @property
    def recent_sessions_expanded(self) -> bool:
        return self._recent_sessions_expanded

    def attach_rerender(self, rerender: Callable[[], Any]) -> None:
        if not callable(rerender):
            raise TypeError("Live Review History rerender callback must be callable")
        self._rerender = rerender

    def replace_overview(self, overview: LiveReviewHistoryOverview) -> None:
        """Replace the immutable source snapshot and invalidate derived history.

        The current dialog loads one snapshot, but keeping this transition
        explicit prevents a future live reload from reusing pagination data
        derived from the prior source.
        """

        self._overview = overview
        self._source_revision += 1
        self._deck_options = deck_filter_options(overview.sessions)
        self._deck_labels = dict(self._deck_options)
        valid_filters = {ALL_DECK_FILTER, *self._deck_labels}
        if self._selected_deck_filter not in valid_filters:
            self._selected_deck_filter = ALL_DECK_FILTER
        self._invalidate_derived_cache()
        self._reset_recent_sessions_view()

    def render_html(self, generation: int) -> str:
        derived = self._derived_history_data()
        display = self._all_reviews_display_data(derived)
        restore_scroll_y = self._restore_scroll_y
        rendered = render_live_review_history_html(
            display.summary,
            derived.sessions,
            total_session_count=len(self.overview.sessions),
            filtered_session_count=display.matching_session_count,
            recent_session_page_size=self.recent_session_page_size,
            recent_session_page_count=derived.recent_page_count,
            recent_session_page=self._recent_session_page,
            recent_sessions_expanded=self._recent_sessions_expanded,
            recent_page_focus=self._recent_page_focus,
            restore_scroll_y=restore_scroll_y,
            retention_table_html=display.retention_table_html,
            selected_review_period=self._selected_review_period,
            review_period_focus=self._review_period_focus,
            filter_description=self.filter_description(),
            deck_options=self._deck_options,
            selected_deck_filter=self._selected_deck_filter,
            search_query=self._search_input,
            include_fsrs=self.include_fsrs,
            is_dark=self.is_dark,
            generation=generation,
        )
        self._recent_page_focus = None
        self._review_period_focus = False
        self._restore_scroll_y = None
        return rendered

    def handle_command(self, command: WebDialogCommand) -> dict[str, bool]:
        if command.action == SELECT_DECK_ACTION:
            deck_filter, search_input = _filter_form_values(command.payload)
            self._select_deck(deck_filter)
            # The native form retained an unsubmitted search when its deck
            # selection rerendered the report. Keep that same distinction here.
            self._search_input = search_input
            self._reset_recent_sessions_view()
            self._request_rerender()
            return {"updated": True}

        if command.action == APPLY_SEARCH_ACTION:
            deck_filter, search_input = _filter_form_values(command.payload)
            self._select_deck(deck_filter)
            self._apply_search(search_input)
            self._reset_recent_sessions_view()
            self._request_rerender()
            return {"updated": True}

        if command.action == SET_RECENT_PAGE_ACTION:
            page, scroll_y = _recent_page_values(command.payload)
            page_count = self._derived_history_data().recent_page_count
            if page < 0 or page >= page_count:
                raise BridgePayloadError("Recent Sessions page is not available.")
            previous_page = self._recent_session_page
            self._recent_session_page = page
            self._recent_sessions_expanded = True
            self._recent_page_focus = "next" if page > previous_page else "previous"
            self._restore_scroll_y = scroll_y
            self._request_rerender()
            return {"updated": True}

        if command.action == SET_ALL_REVIEWS_PERIOD_ACTION:
            period, scroll_y = _review_period_values(command.payload)
            self._selected_review_period = period
            self._review_period_focus = True
            self._restore_scroll_y = scroll_y
            self._request_rerender()
            return {"updated": True}

        raise BridgePayloadError(f"Unhandled Live Review History action: {command.action}")

    def filtered_sessions(self) -> tuple[LiveReviewHistorySession, ...]:
        return self._derived_history_data().sessions

    def _filter_sessions(self) -> tuple[LiveReviewHistorySession, ...]:
        sessions: list[LiveReviewHistorySession] = []
        for session in self.overview.sessions:
            if not matches_deck_filter(
                self._selected_deck_filter,
                deck_filter_value(session),
            ):
                continue
            filtered_session = session_matching_card_ids(
                session,
                self._matching_card_ids,
            )
            if filtered_session is not None:
                sessions.append(filtered_session)
        return tuple(sessions)

    def _derived_history_data(self) -> _DerivedHistoryData:
        key = _DerivedHistoryKey(
            source_revision=self._source_revision,
            deck_filter=self._selected_deck_filter,
            search_revision=self._search_revision,
            include_fsrs=self._include_fsrs,
            recent_session_page_size=self.recent_session_page_size,
        )
        cached = self._derived_cache
        if cached is not None and cached.key == key:
            return cached

        sessions = self._filter_sessions()
        summary = summarize_live_retention_records(
            records_for_sessions(sessions),
            skipped_count=sum(session.skipped_count for session in sessions),
        )
        derived = _DerivedHistoryData(
            key=key,
            sessions=sessions,
            summary=summary,
            matching_session_count=len(sessions),
            recent_page_count=recent_session_page_count(
                len(sessions),
                self.recent_session_page_size,
            ),
            retention_table_html=render_live_retention_table(
                summary,
                include_fsrs=self._include_fsrs,
            ),
        )
        self._derived_cache = derived
        return derived

    def _all_reviews_display_data(
        self,
        derived: _DerivedHistoryData,
    ) -> _AllReviewsDisplayData:
        period = self._selected_review_period
        if period is LiveReviewPeriod.ALL:
            return _AllReviewsDisplayData(
                summary=derived.summary,
                matching_session_count=derived.matching_session_count,
                retention_table_html=derived.retention_table_html,
            )

        key = (derived.key, period)
        cached = self._all_reviews_display_cache.get(key)
        if cached is not None:
            return cached

        records: list[LiveRetentionRecord] = []
        matching_session_count = 0
        for session in derived.sessions:
            session_records = records_in_live_review_period(
                session.summary.records,
                period,
                next_day_at_seconds=self._next_day_at_seconds,
            )
            if not session_records:
                continue
            matching_session_count += 1
            records.extend(session_records)
        summary = summarize_live_retention_records(records)
        display = _AllReviewsDisplayData(
            summary=summary,
            matching_session_count=matching_session_count,
            retention_table_html=render_live_retention_table(
                summary,
                include_fsrs=self._include_fsrs,
            ),
        )
        self._all_reviews_display_cache[key] = display
        return display

    def filter_description(self) -> str:
        deck_label = (
            "All decks"
            if self._selected_deck_filter == ALL_DECK_FILTER
            else self._deck_labels.get(self._selected_deck_filter, "Unknown deck")
        )
        search_description = search_filter_label(self._applied_search_query)
        if self._search_error:
            search_description = f"{search_description} ({self._search_error})"
        return f"Live Session Selected Deck: {deck_label} · Anki Search: {search_description}"

    def _select_deck(self, deck_filter: str) -> None:
        valid_filters = {ALL_DECK_FILTER, *self._deck_labels}
        if deck_filter not in valid_filters:
            raise BridgePayloadError("Selected deck filter is not available.")
        if deck_filter != self._selected_deck_filter:
            self._selected_deck_filter = deck_filter
            self._invalidate_derived_cache()

    def _apply_search(self, search_input: str) -> None:
        query = search_input.strip()
        self._search_input = query
        self._applied_search_query = query
        self._matching_card_ids = None
        self._search_error = ""
        if query:
            try:
                self._matching_card_ids = frozenset(
                    int(card_id) for card_id in self._search_card_ids_callback(query)
                )
            except LiveReviewHistorySearchUnavailable as exc:
                self._matching_card_ids = frozenset()
                self._search_error = str(exc).strip() or "collection unavailable"
            except Exception as exc:  # Anki search parsers use several exception types.
                self._matching_card_ids = frozenset()
                self._search_error = search_error_text(exc)
        self._search_revision += 1
        self._invalidate_derived_cache()

    def _invalidate_derived_cache(self) -> None:
        self._derived_cache = None
        self._all_reviews_display_cache.clear()

    def _reset_recent_sessions_view(self) -> None:
        self._recent_session_page = 0
        self._recent_sessions_expanded = False
        self._recent_page_focus = None
        self._review_period_focus = False
        self._restore_scroll_y = None

    def _request_rerender(self) -> None:
        if self._rerender is None:
            raise RuntimeError("Live Review History controller is not attached to its dialog")
        self._rerender()


def session_matching_card_ids(
    session: LiveReviewHistorySession,
    card_ids: frozenset[int] | None,
) -> LiveReviewHistorySession | None:
    if card_ids is None:
        return session
    records = tuple(record for record in session.summary.records if int(record.card_id) in card_ids)
    if not records:
        return None
    return replace(
        session,
        skipped_count=0,
        summary=summarize_live_retention_records(records),
    )


def deck_filter_options(
    sessions: Iterable[LiveReviewHistorySession],
) -> tuple[tuple[str, str], ...]:
    labels: dict[str, str] = {}
    for session in sessions:
        value = deck_filter_value(session)
        labels.setdefault(value, deck_filter_label(session))
    return tuple(sorted(labels.items(), key=lambda item: item[1].casefold()))


def deck_filter_value(session: LiveReviewHistorySession) -> str:
    if session.source_deck_id is not None:
        return f"id:{int(session.source_deck_id)}"
    if session.source_deck_name:
        return f"name:{session.source_deck_name}"
    return UNKNOWN_DECK_FILTER


def deck_filter_label(session: LiveReviewHistorySession) -> str:
    if session.source_deck_name:
        return session.source_deck_name
    if session.source_deck_id is not None:
        return f"Deck {session.source_deck_id}"
    return "Unknown deck"


def matches_deck_filter(selected: str, candidate: str) -> bool:
    return selected in (ALL_DECK_FILTER, candidate)


def search_filter_label(query: str) -> str:
    return query if query else "None"


def search_error_text(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return f"invalid query: {message}"


def records_for_sessions(
    sessions: Iterable[LiveReviewHistorySession],
) -> tuple[LiveRetentionRecord, ...]:
    return tuple(record for session in sessions for record in session.summary.records)


def recent_session_page_count(session_count: int, page_size: int) -> int:
    count = max(0, int(session_count))
    size = max(1, int(page_size))
    return max(1, (count + size - 1) // size)


def _filter_form_values(payload: Mapping[str, Any]) -> tuple[str, str]:
    expected = {"deck_filter", "search_query"}
    missing = expected - set(payload)
    extra = set(payload) - expected
    if missing:
        raise BridgePayloadError(
            "Live Review History filter is missing: " + ", ".join(sorted(missing)) + "."
        )
    if extra:
        raise BridgePayloadError(
            "Live Review History filter contains unsupported fields: "
            + ", ".join(sorted(extra))
            + "."
        )
    deck_filter = payload["deck_filter"]
    search_query = payload["search_query"]
    if not isinstance(deck_filter, str) or not isinstance(search_query, str):
        raise BridgePayloadError("Live Review History filters must be text.")
    return deck_filter, search_query


def _recent_page_values(payload: Mapping[str, Any]) -> tuple[int, float]:
    if set(payload) != {"page", "scroll_y"}:
        raise BridgePayloadError(
            "Recent Sessions pagination requires a page number and window position."
        )
    page = payload["page"]
    if isinstance(page, bool) or not isinstance(page, int):
        raise BridgePayloadError("Recent Sessions page must be an integer.")
    return page, _scroll_y_value(payload["scroll_y"], label="Recent Sessions")


def _review_period_values(
    payload: Mapping[str, Any],
) -> tuple[LiveReviewPeriod, float]:
    if set(payload) != {"period", "scroll_y"}:
        raise BridgePayloadError(
            "All Reviews period selection requires a period and window position."
        )
    period_value = payload["period"]
    if not isinstance(period_value, str):
        raise BridgePayloadError("All Reviews period must be text.")
    try:
        period = LiveReviewPeriod(period_value)
    except ValueError as exc:
        raise BridgePayloadError("All Reviews period is not available.") from exc
    return period, _scroll_y_value(payload["scroll_y"], label="All Reviews")


def _scroll_y_value(value: Any, *, label: str) -> float:
    scroll_y = value
    if (
        isinstance(scroll_y, bool)
        or not isinstance(scroll_y, (int, float))
        or not math.isfinite(scroll_y)
        or scroll_y < 0
    ):
        raise BridgePayloadError(f"{label} window position must be non-negative.")
    return float(scroll_y)
