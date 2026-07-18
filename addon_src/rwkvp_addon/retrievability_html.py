from __future__ import annotations

from .analysis_workflow import (
    AnalysisControls,
    render_analysis_controls,
    render_analysis_help_heading,
    render_analysis_results_heading,
    render_analysis_status,
)
from .modal_html import (
    ModalField,
    render_field,
    render_modal_document,
    render_notice,
)
from .retrievability import RetrievabilitySummary
from .retrievability_graph import render_retrievability_graph_fragment
from .rwkv_modes import RetrievabilityMode


def render_retrievability_dialog_html(
    *,
    mode: RetrievabilityMode,
    target_time: str,
    initial_search: str,
    filter_search: str,
    latest_review_warning: str,
    summary: RetrievabilitySummary,
    summary_text: str,
    error_message: str = "",
    invalid_log_path: str = "",
    running: bool = False,
    focus_target: str | None = None,
    has_result_set: bool = False,
    prediction_editor_expanded: bool = False,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    target_time_help = "Estimate recall at this local date and time."
    if latest_review_warning:
        target_time_help += f" {latest_review_warning}"
    source_fields = "".join(
        render_field(field)
        for field in (
            ModalField(
                name="target_time",
                label="Prediction date and time",
                kind="datetime-local",
                value=target_time,
                tooltip=target_time_help,
                required=not has_result_set or prediction_editor_expanded,
                step=1,
                disabled=running,
                initial_focus=focus_target == "target_time",
            ),
            ModalField(
                name="initial_search",
                label="Cards to calculate",
                kind="search",
                value=initial_search,
                tooltip="An Anki search that selects the cards to predict.",
                placeholder="For example: deck:Japanese is:due",
                disabled=running,
                initial_focus=focus_target == "initial_search",
            ),
        )
    )
    filter_field = render_field(
        ModalField(
            name="filter_search",
            label="Display filter",
            kind="search",
            value=filter_search,
            tooltip=(
                "Narrow the current results without rerunning RWKV. When generating "
                "a new prediction set, this filter is applied afterward."
            ),
            placeholder="Optional Anki search",
            disabled=running,
            initial_focus=focus_target == "filter_search",
        )
    )
    source_panel_content = f'<div class="rwkv-form-grid">{source_fields}</div>'
    submission_form = render_analysis_controls(
        AnalysisControls(
            form_id="rwkv-retrievability-form",
            source_panel_id="rwkv-retrievability-source-panel",
            source_disclosure_id="rwkv-retrievability-source-disclosure",
            source_disclosure_action="toggle-prediction-editor",
            source_disclosure_label="Generate a new prediction set",
            source_disclosure_tooltip=(
                "Expand to change the prediction time or cards. The next Calculate "
                "will run RWKV again before applying the display filter."
            ),
            source_panel_html=source_panel_content,
            filter_fields_html=filter_field,
            calculate_button_id="rwkv-retrievability-calculate",
            calculate_tooltip=(
                "Generate predictions when needed, then apply the display filter."
            ),
            has_result_set=has_result_set,
            source_expanded=prediction_editor_expanded,
            running=running,
            source_panel_classes=("rwkv-retrievability-source-panel",),
            filter_row_classes=("rwkv-retrievability-filter-row",),
        )
    )
    status = render_analysis_status(
        error_message=error_message,
        running=running,
        running_message="Retrievability calculation is running.",
    )
    if invalid_log_path:
        status += render_notice(
            f"Non-finite predictions were recorded in {invalid_log_path}.",
            tone="warning",
        )
    prediction_help = (
        "Calculate creates predictions when none exist or when the prediction controls "
        "are expanded. Otherwise it only reapplies the display filter to the current "
        "results."
    )
    body_html = f"""
<section class="rwkv-section" id="rwkv-retrievability-prediction">
  {render_analysis_help_heading("Prediction", prediction_help)}
  <div class="rwkv-card rwkv-analysis-controls">
    {submission_form}
  </div>
</section>
<section class="rwkv-section" id="rwkv-retrievability-results">
  {render_analysis_results_heading("Retrievability Results", summary_text)}
  {status}
  <div class="rwkv-card rwkv-analysis-graph-region">
    {render_retrievability_graph_fragment(summary)}
  </div>
</section>
""".strip()
    return render_modal_document(
        title=mode.window_title,
        body_html=body_html,
        generation=generation,
        is_dark=is_dark,
        width="wide",
        root_extra_classes="rwkv-analysis-dialog rwkv-retrievability-dialog",
    )
