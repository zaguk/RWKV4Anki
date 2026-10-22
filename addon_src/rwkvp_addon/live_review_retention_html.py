from __future__ import annotations

from .live_retention_payload import live_retention_table_data
from .live_review_stats import LiveRetentionSummary
from .svelte_ui import render_svelte_document


def render_live_retention_html(
    summary: LiveRetentionSummary,
    *,
    message: str | None = None,
    diagnostic_details: str | None = None,
    include_fsrs: bool = True,
    is_dark: bool = False,
    generation: int = 1,
    log_loss: bool = False,
) -> str:
    return render_svelte_document(
        "live-retention", dict(
            table=live_retention_table_data(summary, include_fsrs=include_fsrs),
            message=message, diagnosticDetails=diagnostic_details,
            reviewedCount=summary.review_count + summary.skipped_count,
            skippedCount=summary.skipped_count, logLoss=log_loss,
        ),
        title="Live Session Retention", generation=generation, is_dark=is_dark, width="wide",
        intro="Compare observed recall with the predictions captured while reviewing.",
        bootstrap={"backgroundSelectors": [".rwkv-modal-page", ".rwkv-dialog-footer"]},
        root_extra_classes="rwkv-retention" + (" rwkv-live-log-loss" if log_loss else ""),
    )
