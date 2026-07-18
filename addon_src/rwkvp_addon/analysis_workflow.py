from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from enum import Enum

from .modal_html import (
    ModalButton,
    ModalDisclosure,
    render_button,
    render_disclosure,
    render_notice,
)

__all__ = (
    "AnalysisControls",
    "AnalysisSubmissionDecision",
    "AnalysisWorkflowState",
    "render_analysis_controls",
    "render_analysis_help_heading",
    "render_analysis_results_heading",
    "render_analysis_status",
)

_UNSET_SOURCE = object()
_HTML_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_CSS_CLASS_PATTERN = re.compile(r"^-?[_A-Za-z]+[-_A-Za-z0-9]*$")


class AnalysisSubmissionDecision(str, Enum):
    """Work required after an analysis form submission."""

    CALCULATE_THEN_FILTER = "calculate-then-filter"
    FILTER_EXISTING = "filter-existing"


@dataclass
class AnalysisWorkflowState:
    """Shared lifecycle state for a calculate-once, refilter-many workflow."""

    has_result_set: bool = False
    source_expanded: bool = False
    source_inputs_changed: bool = False
    running: bool = False
    error_message: str = ""
    focus_target: str | None = None
    _result_source_key: object = field(default=_UNSET_SOURCE, init=False, repr=False)

    def record_source(self, source_key: object) -> bool:
        self.source_inputs_changed = bool(
            self.has_result_set
            and (
                self._result_source_key is _UNSET_SOURCE
                or source_key != self._result_source_key
            )
        )
        return self.source_inputs_changed

    def decide(self, source_key: object) -> AnalysisSubmissionDecision:
        self.record_source(source_key)
        if (
            not self.has_result_set
            or self.source_expanded
            or self.source_inputs_changed
        ):
            return AnalysisSubmissionDecision.CALCULATE_THEN_FILTER
        return AnalysisSubmissionDecision.FILTER_EXISTING

    def mark_source_inputs_changed(self) -> None:
        self.source_inputs_changed = self.has_result_set

    def set_source_expanded(self, expanded: bool) -> bool:
        self.source_expanded = bool(expanded)
        return self.source_expanded

    def begin_calculation(self) -> bool:
        if self.running:
            return False
        self.running = True
        self.error_message = ""
        self.focus_target = None
        self.source_expanded = False
        return True

    def finish_calculation(self, *, focus_target: str) -> bool:
        """Finish an unsuccessful/cancelled calculation and retain prior results."""

        self.running = False
        self.focus_target = focus_target
        self.source_expanded = self.has_result_set
        return self.has_result_set

    def accept_result(self, source_key: object, *, focus_target: str) -> None:
        self.has_result_set = True
        self._result_source_key = source_key
        self.source_inputs_changed = False
        self.running = False
        self.error_message = ""
        self.source_expanded = False
        self.focus_target = focus_target

    def set_error(
        self,
        message: str,
        *,
        focus_target: str | None = None,
        expand_source: bool = False,
    ) -> None:
        self.error_message = str(message)
        self.focus_target = focus_target
        if expand_source and self.has_result_set:
            self.source_expanded = True

    def clear_error(self, *, focus_target: str | None = None) -> None:
        self.error_message = ""
        if focus_target is not None:
            self.focus_target = focus_target


@dataclass(frozen=True)
class AnalysisControls:
    """Trusted composition inputs for the shared analysis control surface."""

    form_id: str
    source_panel_id: str
    source_disclosure_id: str
    source_disclosure_action: str
    source_disclosure_label: str
    source_disclosure_tooltip: str
    source_panel_html: str
    filter_fields_html: str
    calculate_button_id: str
    has_result_set: bool = False
    source_expanded: bool = False
    running: bool = False
    calculate_tooltip: str | None = None
    source_panel_classes: tuple[str, ...] = ()
    filter_row_classes: tuple[str, ...] = ()


def render_analysis_controls(controls: AnalysisControls) -> str:
    """Render source disclosure, filters, and the single Calculate action."""

    form_id = _html_id(controls.form_id, "analysis form ID")
    source_panel_id = _html_id(controls.source_panel_id, "analysis source panel ID")
    source_disclosure_id = _html_id(
        controls.source_disclosure_id,
        "analysis source disclosure ID",
    )
    calculate_button_id = _html_id(
        controls.calculate_button_id,
        "analysis Calculate button ID",
    )
    source_classes = _classes(
        ("rwkv-analysis-source-panel", *controls.source_panel_classes)
    )
    filter_classes = _classes(
        ("rwkv-analysis-filter-row", *controls.filter_row_classes)
    )
    if controls.has_result_set:
        source_controls = render_disclosure(
            ModalDisclosure(
                button_id=source_disclosure_id,
                panel_id=source_panel_id,
                collapsed_label=controls.source_disclosure_label,
                expanded_label=controls.source_disclosure_label,
                expanded=controls.source_expanded,
                disabled=controls.running,
                action=controls.source_disclosure_action,
                tooltip=controls.source_disclosure_tooltip,
                button_classes=("rwkv-analysis-source-disclosure",),
                panel_classes=source_classes,
            ),
            controls.source_panel_html,
        )
    else:
        source_controls = (
            f'<div class="{html.escape(" ".join(source_classes), quote=True)}" '
            f'id="{html.escape(source_panel_id, quote=True)}">'
            f"{controls.source_panel_html}</div>"
        )
    calculate_button = render_button(
        ModalButton(
            "Calculate",
            "calculate",
            variant="primary",
            disabled=controls.running,
            submit=True,
            button_id=calculate_button_id,
            tooltip=controls.calculate_tooltip,
        )
    )
    return f"""
<form class="rwkv-form rwkv-analysis-workflow" id="{html.escape(form_id, quote=True)}"
      data-rwkv-form-action="calculate">
  {source_controls}
  <div class="{html.escape(" ".join(filter_classes), quote=True)}">
    {controls.filter_fields_html}
    <div class="rwkv-analysis-actions">{calculate_button}</div>
  </div>
</form>
""".strip()


def render_analysis_help_heading(title: str, tooltip: str) -> str:
    return (
        '<h2 class="rwkv-section-title rwkv-help-surface" tabindex="0" '
        f'data-rwkv-tooltip="{html.escape(tooltip, quote=True)}">'
        f"{html.escape(title)}</h2>"
    )


def render_analysis_results_heading(title: str, summary_text: str) -> str:
    return (
        '<div class="rwkv-analysis-results-heading">'
        f'<h2 class="rwkv-section-title">{html.escape(title)}</h2>'
        '<p class="rwkv-analysis-summary" aria-live="polite">'
        f"{html.escape(summary_text)}</p></div>"
    )


def render_analysis_status(
    *,
    error_message: str,
    running: bool,
    running_message: str,
) -> str:
    if error_message:
        return render_notice(error_message, tone="error")
    if running:
        return render_notice(running_message, tone="info")
    return ""


def _html_id(value: str, label: str) -> str:
    text = str(value)
    if not _HTML_ID_PATTERN.fullmatch(text):
        raise ValueError(f"{label} must be a valid HTML ID")
    return text


def _classes(values: tuple[str, ...]) -> tuple[str, ...]:
    if any(not _CSS_CLASS_PATTERN.fullmatch(value) for value in values):
        raise ValueError("analysis modifier classes must be valid CSS class names")
    return values
