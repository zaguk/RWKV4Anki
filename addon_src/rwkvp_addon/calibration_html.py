from __future__ import annotations

from .analysis_workflow import (
    AnalysisControls,
    render_analysis_controls,
    render_analysis_help_heading,
    render_analysis_results_heading,
    render_analysis_status,
)
from .calibration import CalibrationSummary
from .calibration_graph import render_calibration_graph_fragment
from .modal_html import (
    FieldOption,
    ModalField,
    render_field,
    render_modal_document,
)
from .rwkv_modes import RetrievabilityMode, mode_spec


def render_calibration_dialog_html(
    *,
    mode: RetrievabilityMode,
    initial_search: str,
    filter_search: str,
    include_sameday: bool,
    review_interval_operator: str,
    review_interval_value: str,
    summary: CalibrationSummary,
    summary_text: str,
    error_message: str = "",
    running: bool = False,
    focus_target: str | None = None,
    has_result_set: bool = False,
    review_source_expanded: bool = False,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    spec = mode_spec(mode)
    title = f"{spec.evaluate_label} Calibration Graph"
    source_field = render_field(
        ModalField(
            name="initial_search",
            label="Reviews to evaluate",
            kind="search",
            value=initial_search,
            tooltip="An Anki search that selects which cards contribute reviews.",
            placeholder="For example: deck:Japanese",
            disabled=running,
            initial_focus=focus_target == "initial_search",
        )
    )
    filter_fields = "".join(
        render_field(field)
        for field in (
            ModalField(
                name="filter_search",
                label="Filter calculated reviews",
                kind="search",
                value=filter_search,
                tooltip=(
                    "Optionally narrow the displayed reviews with another Anki "
                    "search without recalculating predictions."
                ),
                placeholder="Optional Anki search",
                disabled=running,
                initial_focus=focus_target == "filter_search",
            ),
            ModalField(
                name="review_interval_operator",
                label="Review interval comparison",
                kind="select",
                value=review_interval_operator,
                tooltip="Compare the elapsed interval with the number of days below.",
                options=tuple(
                    FieldOption(operator, operator)
                    for operator in ("<", "<=", "=", ">=", ">")
                ),
                disabled=running,
            ),
            ModalField(
                name="review_interval_value",
                label="Review interval (days)",
                kind="text",
                value=review_interval_value,
                tooltip="Leave blank to include every review interval.",
                placeholder="Any",
                disabled=running,
                initial_focus=focus_target == "review_interval_value",
            ),
            ModalField(
                name="include_sameday",
                label="Include same-day reviews",
                kind="checkbox",
                checked=include_sameday,
                tooltip="Include reviews repeated on the same day in the chart.",
                disabled=running,
                change_action="set-same-day",
                change_serialize_form=True,
                initial_focus=focus_target == "include_sameday",
            ),
        )
    )
    source_panel_content = f'<div class="rwkv-form-grid">{source_field}</div>'
    submission_form = render_analysis_controls(
        AnalysisControls(
            form_id="rwkv-calibration-form",
            source_panel_id="rwkv-calibration-source-panel",
            source_disclosure_id="rwkv-calibration-source-disclosure",
            source_disclosure_action="toggle-review-source",
            source_disclosure_label="Evaluate a different set of reviews",
            source_disclosure_tooltip=(
                "Expand to choose a different review set. The next Calculate will "
                "rebuild predictions before applying the current filters."
            ),
            source_panel_html=source_panel_content,
            filter_fields_html=filter_fields,
            calculate_button_id="rwkv-calibration-calculate",
            has_result_set=has_result_set,
            source_expanded=review_source_expanded,
            running=running,
            source_panel_classes=("rwkv-calibration-source-panel",),
            filter_row_classes=("rwkv-calibration-filter-row",),
        )
    )
    status = render_analysis_status(
        error_message=error_message,
        running=running,
        running_message="Calibration calculation is running.",
    )
    controls_help = (
        "Calculate creates predictions when needed. After results exist, it normally "
        "applies the display filters without rerunning RWKV."
    )
    body_html = f"""
<section class="rwkv-section" id="rwkv-calibration-controls">
  {render_analysis_help_heading("Reviews and Filters", controls_help)}
  <div class="rwkv-card rwkv-analysis-controls">
    {submission_form}
  </div>
</section>
<section class="rwkv-section" id="rwkv-calibration-results">
  {render_analysis_results_heading("Calibration Results", summary_text)}
  {status}
  <div class="rwkv-card rwkv-analysis-graph-region">
    {render_calibration_graph_fragment(summary, title=title)}
  </div>
</section>
""".strip()
    return render_modal_document(
        title=title,
        body_html=body_html,
        generation=generation,
        is_dark=is_dark,
        width="wide",
        root_extra_classes="rwkv-analysis-dialog rwkv-calibration-dialog",
    )
