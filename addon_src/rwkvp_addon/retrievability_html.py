from __future__ import annotations

from dataclasses import asdict

from .modal_html import ModalDisclosure, ModalField
from .report_payload import chart_markup
from .retrievability import RetrievabilitySummary
from .retrievability_graph import render_retrievability_graph_fragment
from .rwkv_modes import RetrievabilityMode
from .svelte_ui import render_svelte_document


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
    source_fields = [
        asdict(field)
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
    ]
    filter_field = asdict(
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
    disclosure = asdict(
        ModalDisclosure(
            panel_id="rwkv-retrievability-source-panel",
            button_id="rwkv-retrievability-source-disclosure",
            action="toggle-prediction-editor",
            collapsed_label="Generate a new prediction set",
            expanded_label="Generate a new prediction set",
            tooltip=(
                "Expand to change the prediction time or cards. The next Calculate "
                "will run RWKV again before applying the display filter."
            ),
            expanded=prediction_editor_expanded,
            disabled=running,
            button_classes=("rwkv-analysis-source-disclosure",),
            panel_classes=("rwkv-analysis-source-panel", "rwkv-retrievability-source-panel"),
        )
    )
    notices = []
    if error_message:
        notices.append(dict(message=error_message, tone="error"))
    elif running:
        notices.append(dict(message="Retrievability calculation is running.", tone="info"))
    if invalid_log_path:
        notices.append(dict(
            message=f"Non-finite predictions were recorded in {invalid_log_path}.",
            tone="warning",
        ))
    prediction_help = (
        "Calculate creates predictions when none exist or when the prediction controls "
        "are expanded. Otherwise it only reapplies the display filter to the current "
        "results."
    )
    return render_svelte_document(
        "analysis", dict(
            title=mode.window_title, prefix="retrievability", controlsTitle="Prediction",
            controlsHelp=prediction_help, sourceFields=source_fields,
            filterGroups=[dict(fields=[filter_field])], disclosure=disclosure,
            hasResultSet=has_result_set, running=running,
            calculateTooltip="Generate predictions when needed, then apply the display filter.",
            resultsTitle="Retrievability Results", summaryText=summary_text, notices=notices,
            graph=chart_markup(render_retrievability_graph_fragment(summary)),
        ),
        title=mode.window_title,
        generation=generation,
        is_dark=is_dark,
        width="wide",
        root_extra_classes="rwkv-analysis-dialog rwkv-retrievability-dialog",
    )
