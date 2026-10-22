from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime

from .live_retention_payload import live_retention_table_data, metric_cells, retention_cells
from .live_retention_table_html import LOG_LOSS_IMPROVEMENT_HELP, RETENTION_IMPROVEMENT_HELP
from .live_review_history import (
    LIVE_REVIEW_SCHEDULING_LABELS,
    LiveReviewHistorySession,
    LiveReviewPeriod,
    LiveReviewScheduling,
)
from .live_review_stats import LiveRetentionSummary, LiveRetentionSummaryRow
from .modal_html import FieldOption, ModalButton, ModalDisclosure, ModalField
from .svelte_ui import render_svelte_document


def render_live_review_history_html(
    summary: LiveRetentionSummary,
    sessions: tuple[LiveReviewHistorySession, ...],
    *,
    total_session_count: int,
    filtered_session_count: int,
    recent_session_page_size: int,
    recent_session_page_count: int | None = None,
    recent_session_page: int = 0,
    recent_sessions_expanded: bool = False,
    recent_page_focus: str | None = None,
    restore_scroll_y: float | None = None,
    retention_table_data: dict | None = None,
    session_metric_tables: Mapping[int, dict] | None = None,
    session_metrics_message: str = "",
    selected_review_period: LiveReviewPeriod = LiveReviewPeriod.ALL,
    review_period_focus: bool = False,
    selected_scheduling: LiveReviewScheduling = LiveReviewScheduling.ALL,
    scheduling_focus: bool = False,
    filter_description: str,
    deck_options: tuple[tuple[str, str], ...] = (),
    selected_deck_filter: str = "__all__",
    search_query: str = "",
    include_fsrs: bool = True,
    is_dark: bool = False,
    generation: int = 1,
    log_loss: bool = False,
    active_session_key: str | None = None,
    comparison_focus: str | None = None,
) -> str:
    page_size = max(1, int(recent_session_page_size))
    calculated_page_count = max(1, (len(sessions) + page_size - 1) // page_size)
    page_count = (
        calculated_page_count if recent_session_page_count is None
        else max(1, int(recent_session_page_count))
    )
    page = min(max(0, int(recent_session_page)), page_count - 1)
    start = page * page_size
    visible = sessions[start : start + page_size]
    filter_fields = [
        asdict(field)
        for field in (
            ModalField(
                name="deck_filter",
                label="Live Session Selected Deck",
                kind="select",
                value=selected_deck_filter,
                tooltip="Show only sessions that started from this deck.",
                options=(FieldOption("__all__", "All decks"),)
                + tuple(FieldOption(value, label) for value, label in deck_options),
                change_action="select-deck",
                change_serialize_form=True,
            ),
            ModalField(
                name="search_query",
                label="Anki Search",
                kind="search",
                value=search_query,
                tooltip=(
                    "Use Anki search syntax to keep only saved reviews whose cards currently match."
                ),
                placeholder="Filter saved reviews by current Anki search",
            ),
        )
    ]
    scheduling_selector = asdict(ModalField(
        name="scheduling",
        label="Scheduling",
        kind="select",
        value=selected_scheduling.value,
        options=tuple(
            FieldOption(value.value, label)
            for value, label in LIVE_REVIEW_SCHEDULING_LABELS.items()
        ),
        change_action="set-scheduling",
        initial_focus=scheduling_focus,
        tooltip=(
            "Filter the summary and individual sessions. Randomly Scheduled includes draws "
            "from Random card %. RWKV-Scheduled includes ordinary retention-based draws, "
            "even with Random sort order."
        ),
    ))
    previous_disabled = page <= 0
    next_disabled = page >= page_count - 1
    focus_previous = recent_page_focus == "previous" and not previous_disabled
    focus_next = recent_page_focus == "next" and not next_disabled
    if recent_page_focus == "previous" and previous_disabled:
        focus_next = True
    elif recent_page_focus == "next" and next_disabled:
        focus_previous = True
    pagination = dict(
        page=page, count=page_count, range=_recent_range_label(
            first_index=start + 1 if visible else 0, visible_count=len(visible),
            matching_count=len(sessions),
        ), buttons=[
            asdict(ModalButton(
                "Previous", "set-recent-page", payload={"page": max(0, page - 1)},
                disabled=previous_disabled, button_id="rwkv-history-previous-page",
                initial_focus=focus_previous,
            )),
            asdict(ModalButton(
                "Next", "set-recent-page", payload={"page": min(page_count - 1, page + 1)},
                disabled=next_disabled, button_id="rwkv-history-next-page",
                initial_focus=focus_next,
            )),
        ],
    )
    table = retention_table_data
    if table is None:
        table = live_retention_table_data(
            summary, include_fsrs=include_fsrs, include_rmse_bins=True, include_auc=True,
        )
    return render_svelte_document(
        "live-history", dict(
            filterFields=filter_fields, schedulingField=scheduling_selector,
            applyButton=asdict(ModalButton(
                "Apply Search", "apply-search", variant="primary", submit=True,
                button_id="rwkv-history-apply-search",
                tooltip="Apply the current Anki search. Deck changes apply immediately.",
            )),
            reviewsMeta=_reviews_meta(
                summary, filtered_session_count=filtered_session_count,
                total_session_count=total_session_count,
            ),
            filterDescription=filter_description, table=table,
            logLoss=log_loss, includeFsrs=include_fsrs, restoreScrollY=restore_scroll_y,
            activeSessionKey=active_session_key, comparisonFocus=comparison_focus,
            periods=[dict(
                value=period.value, label=label, selected=period is selected_review_period,
                focus=period is selected_review_period and review_period_focus,
            ) for period, label in (
                (LiveReviewPeriod.WEEK, "1 week"), (LiveReviewPeriod.MONTH, "1 month"),
                (LiveReviewPeriod.THREE_MONTHS, "3 months"), (LiveReviewPeriod.YEAR, "1 year"),
                (LiveReviewPeriod.ALL, "all"),
            )],
            recentDisclosure=asdict(ModalDisclosure(
                button_id="rwkv-history-recent-disclosure", panel_id="rwkv-history-recent-panel",
                collapsed_label="Show individual sessions...",
                expanded_label="Hide individual sessions...", expanded=recent_sessions_expanded,
                button_classes=("rwkv-history-recent-disclosure",),
                panel_classes=("rwkv-history-recent-panel",),
            )),
            pagination=pagination,
            sessions=[_session_data(
                session, include_fsrs=include_fsrs, detail_index=start + index,
                retention_table_data=(session_metric_tables or {}).get(start + index),
                metrics_message=session_metrics_message,
            ) for index, session in enumerate(visible)],
            retentionHelp=RETENTION_IMPROVEMENT_HELP, logLossHelp=LOG_LOSS_IMPROVEMENT_HELP,
        ),
        title="Live Review History", generation=generation, is_dark=is_dark, width="wide",
        bootstrap={"backgroundSelectors": [".rwkv-modal-page", ".rwkv-dialog-footer"]},
        root_extra_classes="rwkv-history" + (" rwkv-live-log-loss" if log_loss else ""),
    )


def live_history_session_key(index: int) -> str:
    return f"rwkv-history-session-detail-{index}"


def _session_data(
    session: LiveReviewHistorySession, *, include_fsrs: bool, detail_index: int,
    retention_table_data: dict | None = None, metrics_message: str = "",
) -> dict:
    row = _total_row(session.summary)
    if row is None:
        row = LiveRetentionSummaryRow(
            label="Total + Same Day", review_count=0, predicted_retention=None,
            actual_retention=None, remembered_count=0,
        )
    ended = _format_time(session.ended_at_ms)
    deck = _session_deck_label(session)
    if retention_table_data is None:
        retention_table_data = live_retention_table_data(
            session.summary, include_fsrs=include_fsrs, include_rmse_bins=True,
            include_auc=True, rmse_message=metrics_message,
        )
    return dict(
        ended=ended, deck=deck, key=live_history_session_key(detail_index),
        detailLabel=f"View review details for {deck}, ended {ended}",
        retention=retention_cells(row, include_fsrs=include_fsrs, session=True),
        metrics=metric_cells(row.log_loss),
        table=retention_table_data,
    )


def _reviews_meta(
    summary: LiveRetentionSummary,
    *,
    filtered_session_count: int,
    total_session_count: int,
) -> str:
    if summary.review_count <= 0:
        return "No saved RWKV Live Session reviews match this filter."
    review_noun = "review" if summary.review_count == 1 else "reviews"
    session_noun = "session" if filtered_session_count == 1 else "sessions"
    total = max(0, int(total_session_count))
    shown = max(0, int(filtered_session_count))
    return (
        f"{summary.review_count} {review_noun} from {shown} matching {session_noun} "
        f"out of {total} saved sessions."
    )


def _recent_range_label(
    *,
    first_index: int,
    visible_count: int,
    matching_count: int,
) -> str:
    if matching_count <= 0:
        return "0 sessions"
    session_noun = "session" if matching_count == 1 else "sessions"
    if visible_count == 1:
        return f"{first_index} of {matching_count} {session_noun}"
    last_index = first_index + visible_count - 1
    return f"{first_index}–{last_index} of {matching_count} {session_noun}"


def _total_row(summary: LiveRetentionSummary) -> LiveRetentionSummaryRow | None:
    for row in summary.rows:
        if row.label == "Total + Same Day":
            return row
    return None


def _session_deck_label(session: LiveReviewHistorySession) -> str:
    if session.source_deck_name:
        return session.source_deck_name
    if session.source_deck_id is not None:
        return f"Deck {session.source_deck_id}"
    return "Unknown deck"


def _format_time(timestamp_ms: int) -> str:
    try:
        return datetime.fromtimestamp(int(timestamp_ms) / 1000).strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, TypeError, ValueError):
        return "Unknown"
