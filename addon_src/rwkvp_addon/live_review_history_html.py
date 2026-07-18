from __future__ import annotations

import html
import json
from datetime import datetime

from .live_retention_table_html import (
    format_live_retention_error_ratio,
    format_live_retention_fsrs_predicted,
    live_retention_delta_class,
    live_retention_error_ratio_class,
    render_live_retention_table,
)
from .live_review_history import LiveReviewHistorySession, LiveReviewPeriod
from .live_review_stats import (
    LiveRetentionSummary,
    LiveRetentionSummaryRow,
    format_live_retention_delta,
    format_live_retention_percent,
)
from .modal_html import (
    FieldOption,
    ModalButton,
    ModalDisclosure,
    ModalField,
    render_button,
    render_disclosure,
    render_field,
    render_modal_document,
)


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
    retention_table_html: str | None = None,
    selected_review_period: LiveReviewPeriod = LiveReviewPeriod.ALL,
    review_period_focus: bool = False,
    filter_description: str,
    deck_options: tuple[tuple[str, str], ...] = (),
    selected_deck_filter: str = "__all__",
    search_query: str = "",
    include_fsrs: bool = True,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    page_size = max(1, int(recent_session_page_size))
    calculated_page_count = max(1, (len(sessions) + page_size - 1) // page_size)
    recent_page_count = (
        calculated_page_count
        if recent_session_page_count is None
        else max(1, int(recent_session_page_count))
    )
    page = min(max(0, int(recent_session_page)), recent_page_count - 1)
    page_start = page * page_size
    visible_sessions = sessions[page_start : page_start + page_size]
    retention_table = (
        render_live_retention_table(summary, include_fsrs=include_fsrs)
        if retention_table_html is None
        else retention_table_html
    )
    rendered_sessions = tuple(
        _render_session_entry(
            session,
            include_fsrs=include_fsrs,
            detail_index=page_start + index,
        )
        for index, session in enumerate(visible_sessions)
    )
    session_rows = "\n".join(row for row, _template in rendered_sessions)
    session_detail_templates = "\n".join(
        template for _row, template in rendered_sessions
    )
    if not session_rows:
        session_colspan = 9 if include_fsrs else 6
        session_rows = (
            f'<tr><td colspan="{session_colspan}" class="empty">No matching sessions.</td></tr>'
        )
    fsrs_session_headers = _session_fsrs_headers() if include_fsrs else ""
    reviews_meta = _reviews_meta(
        summary,
        filtered_session_count=filtered_session_count,
        total_session_count=total_session_count,
    )
    recent_range = _recent_range_label(
        first_index=page_start + 1 if visible_sessions else 0,
        visible_count=len(visible_sessions),
        matching_count=len(sessions),
    )
    recent_pagination = _render_recent_pagination(
        page=page,
        page_count=recent_page_count,
        range_label=recent_range,
        focus_direction=recent_page_focus,
    )
    escaped_filter_description = html.escape(filter_description)
    filter_fields = "".join(
        render_field(field)
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
    )
    apply_button = render_button(
        ModalButton(
            "Apply Search",
            "apply-search",
            variant="primary",
            submit=True,
            button_id="rwkv-history-apply-search",
            tooltip="Apply the current Anki search. Deck changes apply immediately.",
        )
    )
    filter_form = f"""
<form class="rwkv-form rwkv-history-filter-form" id="rwkv-history-filters"
      data-rwkv-form-action="apply-search">
  <div class="rwkv-history-filter-row">
    {filter_fields}
    <div class="rwkv-history-filter-actions">{apply_button}</div>
  </div>
</form>
""".strip()
    filter_help = (
        "Choose which saved Live Sessions and reviewed cards contribute to the "
        "report. Filtering never changes or deletes history."
    )
    all_reviews_help = (
        "All matching saved Live Session reviews, grouped by review type and "
        "compared with the predictions recorded while reviewing."
    )
    review_period_selector = _render_review_period_selector(
        selected_review_period,
        focus_selected=review_period_focus,
    )
    recent_help = "Inspect the individual saved Live Sessions behind the All Reviews summary."
    recent_details = render_disclosure(
        ModalDisclosure(
            button_id="rwkv-history-recent-disclosure",
            panel_id="rwkv-history-recent-panel",
            collapsed_label="Show individual sessions...",
            expanded_label="Hide individual sessions...",
            expanded=recent_sessions_expanded,
            button_classes=("rwkv-history-recent-disclosure",),
            panel_classes=("rwkv-history-recent-panel",),
        ),
        f"""
          {recent_pagination}
          <div class="rwkv-table-wrap">
            <table class="rwkv-data-table">
              <caption class="rwkv-sr-only">Recent Live Session retention</caption>
              <thead>
                <tr>
                  <th scope="col" class="text">Ended</th>
                  <th scope="col" class="text">Deck</th>
                  <th scope="col" class="rwkv-number number count">Reviews</th>
                  <th scope="col" class="rwkv-number number total group-start">Actual</th>
                  <th scope="col" class="rwkv-number number total">RWKV Predicted</th>
                  <th scope="col" class="rwkv-number number total">RWKV Difference</th>
                  {fsrs_session_headers}
                </tr>
              </thead>
              <tbody>
                {session_rows}
              </tbody>
            </table>
          </div>
""".strip(),
    )
    body_html = f"""
    <section class="rwkv-section history-filters">
        <h2 class="rwkv-section-title rwkv-help-surface" tabindex="0"
            data-rwkv-tooltip="{html.escape(filter_help, quote=True)}">History Filters</h2>
        <div class="rwkv-card">{filter_form}</div>
    </section>
    <section class="rwkv-section all-reviews">
      <div class="rwkv-history-summary-heading">
        <h2 class="rwkv-section-title rwkv-help-surface" tabindex="0"
            data-rwkv-tooltip="{html.escape(all_reviews_help, quote=True)}">All Reviews</h2>
        <p class="rwkv-history-summary" aria-live="polite">{reviews_meta}
          Filters: {escaped_filter_description}</p>
      </div>
      <div class="rwkv-card">
        {review_period_selector}
        {retention_table}
      </div>
    </section>
    <section class="rwkv-section recent-sessions"
             aria-labelledby="rwkv-history-recent-title">
      <h2 class="rwkv-section-title rwkv-help-surface"
          id="rwkv-history-recent-title" tabindex="0"
          data-rwkv-tooltip="{html.escape(recent_help, quote=True)}">Recent Sessions Details</h2>
      <div class="rwkv-card">
        {recent_details}
      </div>
    </section>
""".strip()
    return render_modal_document(
        title="Live Review History",
        body_html=body_html,
        generation=generation,
        is_dark=bool(is_dark),
        width="wide",
        overlay_html=_render_session_details_overlay(session_detail_templates),
        head_html=_render_history_page_script(restore_scroll_y),
        root_extra_classes="rwkv-history",
    )


def _render_history_page_script(restore_scroll_y: float | None) -> str:
    restore_value = (
        "null"
        if restore_scroll_y is None
        else json.dumps(float(restore_scroll_y), allow_nan=False)
    )
    return f"""
<script>
(() => {{
  const restoreScrollY = {restore_value};
  const restoreClass = 'rwkv-history-scroll-restore-pending';
  window.RWKV_MODAL_PAYLOAD_PROVIDER = (action, payload={{}}) => {{
    if (action !== 'set-recent-page' && action !== 'set-all-reviews-period') {{
      return payload;
    }}
    return Object.assign({{}}, payload, {{
      scroll_y: Math.max(0, Number(window.scrollY) || 0),
    }});
  }};
  if (Number.isFinite(restoreScrollY)) {{
    document.documentElement.classList.add(restoreClass);
    document.addEventListener('DOMContentLoaded', () => {{
      try {{
        // Restore focus and scroll before the replacement document's first
        // visible paint. Removing the marker also prevents the shared runtime's
        // scheduled focus pass from changing the restored position afterward.
        const initialFocus = document.querySelector(
          '[data-rwkv-initial-focus]:not([disabled])'
        );
        if (initialFocus) {{
          try {{
            initialFocus.focus({{preventScroll: true}});
          }} catch (_error) {{
            initialFocus.focus();
          }}
          initialFocus.removeAttribute('data-rwkv-initial-focus');
        }}
        window.scrollTo(0, restoreScrollY);
      }} finally {{
        document.documentElement.classList.remove(restoreClass);
      }}
    }}, {{once: true}});
  }}
  document.addEventListener('DOMContentLoaded', () => {{
    const overlay = document.getElementById('rwkv-history-session-overlay');
    const content = overlay?.querySelector('[data-rwkv-history-session-content]');
    const close = overlay?.querySelector('[data-rwkv-history-session-close]');
    if (!overlay || !content || !close) {{
      return;
    }}

    const openSessionDetails = (trigger) => {{
      const templateId = trigger?.dataset.rwkvHistorySessionTemplate;
      const template = templateId ? document.getElementById(templateId) : null;
      if (!(template instanceof HTMLTemplateElement)) {{
        return;
      }}
      content.replaceChildren(template.content.cloneNode(true));
      window.RWKVModal?.showOverlay(overlay);
    }};

    document.addEventListener('click', (event) => {{
      const row = event.target.closest('[data-rwkv-history-session-row]');
      if (!row || overlay.contains(row)) {{
        return;
      }}
      openSessionDetails(
        row.querySelector('[data-rwkv-history-session-open]')
      );
    }});
    close.addEventListener('click', () => {{
      window.RWKVModal?.hideOverlay(overlay);
      content.replaceChildren();
    }});
  }}, {{once: true}});
}})();
</script>
""".strip()


def _render_review_period_selector(
    selected_period: LiveReviewPeriod,
    *,
    focus_selected: bool,
) -> str:
    options = (
        (LiveReviewPeriod.WEEK, "1 week"),
        (LiveReviewPeriod.MONTH, "1 month"),
        (LiveReviewPeriod.THREE_MONTHS, "3 months"),
        (LiveReviewPeriod.YEAR, "1 year"),
        (LiveReviewPeriod.ALL, "all"),
    )
    rendered: list[str] = []
    for period, label in options:
        selected = period is selected_period
        state_attributes = ""
        if selected:
            state_attributes += " checked"
            if focus_selected:
                state_attributes += " data-rwkv-initial-focus"
        rendered.append(
            f"""
<label class="rwkv-history-period-option">
  <input type="radio" name="period" value="{period.value}"
         data-rwkv-change-action="set-all-reviews-period"{state_attributes}>
  <span>{label}</span>
</label>
""".strip()
        )
    rendered_options = "\n".join(rendered)
    return f"""
<fieldset class="rwkv-history-period-selector">
  <legend class="rwkv-sr-only">Reviews included in All Reviews</legend>
  {rendered_options}
</fieldset>
""".strip()


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


def _render_recent_pagination(
    *,
    page: int,
    page_count: int,
    range_label: str,
    focus_direction: str | None,
) -> str:
    if page_count <= 1:
        return f"""
<nav class="rwkv-history-pagination rwkv-history-pagination--single-page"
     aria-label="Recent Sessions Details pages">
  <span class="rwkv-history-pagination__status" aria-live="polite">{range_label}</span>
</nav>
""".strip()
    previous_disabled = page <= 0
    next_disabled = page >= page_count - 1
    focus_previous = focus_direction == "previous" and not previous_disabled
    focus_next = focus_direction == "next" and not next_disabled
    if focus_direction == "previous" and previous_disabled:
        focus_next = True
    elif focus_direction == "next" and next_disabled:
        focus_previous = True
    previous_button = render_button(
        ModalButton(
            "Previous",
            "set-recent-page",
            variant="secondary",
            payload={"page": max(0, page - 1)},
            disabled=previous_disabled,
            button_id="rwkv-history-previous-page",
            initial_focus=focus_previous,
        )
    )
    next_button = render_button(
        ModalButton(
            "Next",
            "set-recent-page",
            variant="secondary",
            payload={"page": min(page_count - 1, page + 1)},
            disabled=next_disabled,
            button_id="rwkv-history-next-page",
            initial_focus=focus_next,
        )
    )
    return f"""
<nav class="rwkv-history-pagination" aria-label="Recent Sessions Details pages">
  {previous_button}
  <span class="rwkv-history-pagination__status" aria-live="polite">
    Page {page + 1} of {page_count}
    <span class="rwkv-history-pagination__separator" aria-hidden="true">·</span>
    {range_label}
  </span>
  {next_button}
</nav>
""".strip()


def _render_session_entry(
    session: LiveReviewHistorySession,
    *,
    include_fsrs: bool,
    detail_index: int,
) -> tuple[str, str]:
    row = _total_row(session.summary)
    if row is None:
        row = LiveRetentionSummaryRow(
            label="Total + Same Day",
            review_count=0,
            predicted_retention=None,
            actual_retention=None,
            remembered_count=0,
        )
    delta_class = live_retention_delta_class(
        row.actual_retention,
        row.predicted_retention,
    )
    fsrs_delta_class = live_retention_delta_class(
        row.fsrs_actual_retention,
        row.fsrs_predicted_retention,
    )
    ended_text = _format_time(session.ended_at_ms)
    deck_text = _session_deck_label(session)
    ended = html.escape(ended_text)
    deck = html.escape(deck_text)
    template_id = f"rwkv-history-session-detail-{max(0, int(detail_index))}"
    detail_label = html.escape(
        f"View review details for {deck_text}, ended {ended_text}",
        quote=True,
    )
    actual = format_live_retention_percent(row.actual_retention)
    predicted = format_live_retention_percent(row.predicted_retention)
    delta = format_live_retention_delta(row.actual_retention, row.predicted_retention)
    fsrs_predicted = format_live_retention_fsrs_predicted(row)
    fsrs_delta = format_live_retention_delta(
        row.fsrs_actual_retention,
        row.fsrs_predicted_retention,
    )
    comparison_class = live_retention_error_ratio_class(row)
    comparison = format_live_retention_error_ratio(row)
    fsrs_cells = (
        f"""
                    <td class="rwkv-number number total group-start">{fsrs_predicted}</td>
                    <td class="rwkv-number number {fsrs_delta_class}">{fsrs_delta}</td>
                    <td class="rwkv-number number {comparison_class}">{comparison}</td>
""".rstrip()
        if include_fsrs
        else ""
    )
    session_row = f"""
                <tr class="rwkv-history-session-row"
                    data-rwkv-history-session-row>
                    <td class="text">
                      <button class="rwkv-history-session-open" type="button"
                              aria-label="{detail_label}" aria-haspopup="dialog"
                              aria-controls="rwkv-history-session-overlay"
                              data-rwkv-history-session-open
                              data-rwkv-history-session-template="{template_id}">{ended}</button>
                    </td>
                    <td class="text">{deck}</td>
                    <td class="rwkv-number number count">{row.review_count}</td>
                    <td class="rwkv-number number total group-start">{actual}</td>
                    <td class="rwkv-number number total">{predicted}</td>
                    <td class="rwkv-number number {delta_class}">{delta}</td>
{fsrs_cells}
                </tr>
""".strip()
    detail_context = (
        '<p class="rwkv-history-session-overlay__context">'
        f"<strong>{deck}</strong>"
        '<span aria-hidden="true">·</span>'
        f"<span>Ended {ended}</span>"
        "</p>"
    )
    detail_table = render_live_retention_table(
        session.summary,
        include_fsrs=include_fsrs,
    )
    detail_template = f"""
<template id="{template_id}">
  {detail_context}
  {detail_table}
</template>
""".strip()
    return session_row, detail_template


def _render_session_details_overlay(detail_templates: str) -> str:
    return f"""
<div class="rwkv-modal-overlay rwkv-history-session-overlay"
     id="rwkv-history-session-overlay" hidden aria-hidden="true"
     role="dialog" aria-modal="true"
     aria-labelledby="rwkv-history-session-overlay-title"
     data-rwkv-overlay data-rwkv-overlay-kind="info">
  <section class="rwkv-overlay-panel rwkv-overlay-panel--info
                  rwkv-history-session-overlay__panel" tabindex="-1">
    <div class="rwkv-history-session-overlay__header">
      <h2 class="rwkv-overlay-title" id="rwkv-history-session-overlay-title">
        Live Session Review Details
      </h2>
      <button class="rwkv-icon-close" type="button"
              aria-label="Close Live Session review details"
              data-rwkv-history-session-close data-rwkv-overlay-cancel>&times;</button>
    </div>
    <div class="rwkv-history-session-overlay__content"
         data-rwkv-history-session-content></div>
  </section>
  {detail_templates}
</div>
""".strip()


def _session_fsrs_headers() -> str:
    return """
                    <th scope="col" class="rwkv-number number total group-start">FSRS Predicted</th>
                    <th scope="col" class="rwkv-number number total">FSRS Difference</th>
                    <th scope="col" class="rwkv-number number total">RWKV / FSRS</th>
""".rstrip()


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
