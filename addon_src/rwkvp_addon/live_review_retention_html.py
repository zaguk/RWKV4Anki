from __future__ import annotations

import html

from .live_retention_table_html import render_live_retention_table
from .live_review_stats import LiveRetentionSummary
from .modal_html import render_card, render_modal_document


def render_live_retention_html(
    summary: LiveRetentionSummary,
    *,
    message: str | None = None,
    include_fsrs: bool = True,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    note = (
        f'<p class="rwkv-callout rwkv-retention-note">{html.escape(message)}</p>'
        if message
        else ""
    )
    reviewed_count = summary.review_count + summary.skipped_count
    skipped = f" - Skipped: {summary.skipped_count}" if summary.skipped_count else ""
    retention_table = render_live_retention_table(
        summary,
        include_fsrs=include_fsrs,
    )
    body_html = note + render_card(
        f"{retention_table}"
        f'<p class="rwkv-retention-summary">Reviews done: {reviewed_count}{skipped}</p>'
    )
    return render_modal_document(
        title="Live Session Retention",
        intro="Compare observed recall with the predictions captured while reviewing.",
        body_html=body_html,
        generation=generation,
        is_dark=bool(is_dark),
        width="wide",
        root_extra_classes="rwkv-retention",
    )
