from __future__ import annotations

from ..live_review_retention_html import render_live_retention_html
from ..live_review_stats import LiveRetentionSummary
from ..web_dialog_controller import CloseOnlyReportController
from .web_dialog import WebDialogHost, widget_uses_dark_palette

_is_dark_widget = widget_uses_dark_palette


class LiveReviewRetentionDialog(WebDialogHost):
    def __init__(
        self,
        parent,
        summary: LiveRetentionSummary,
        *,
        message: str | None = None,
        include_fsrs: bool = True,
    ) -> None:
        self.summary = summary
        self.include_fsrs = bool(include_fsrs)
        is_dark = _is_dark_widget(parent)
        self._report_controller = CloseOnlyReportController(
            lambda generation: render_live_retention_html(
                self.summary,
                message=message,
                include_fsrs=self.include_fsrs,
                is_dark=is_dark,
                generation=generation,
            )
        )
        super().__init__(
            parent,
            title="RWKV Live Session Retention",
            controller=self._report_controller,
            size=(980, 360),
            web_minimum_height=250,
        )
        self.setMinimumSize(980, 360)
