from __future__ import annotations

from dataclasses import asdict

from .calibration import CalibrationSummary
from .calibration_graph import render_calibration_graph_fragment
from .modal_html import (
    FieldOption,
    ModalDisclosure,
    ModalField,
)
from .model_generation import model_generation_label
from .report_payload import chart_markup
from .rwkv_modes import RetrievabilityMode, mode_spec
from .svelte_ui import render_svelte_document
from .um_plus import UmPlusSummary
from .um_plus_graph import render_um_plus_graph_fragment


def render_calibration_dialog_html(
    *,
    mode: RetrievabilityMode,
    initial_search: str,
    filter_search: str,
    include_sameday: bool,
    review_interval_operator: str,
    review_interval_value: str,
    summary: CalibrationSummary | UmPlusSummary,
    summary_text: str,
    show_small_buckets: bool = False,
    include_fsrs: bool = False,
    fsrs_summary: CalibrationSummary | None = None,
    unpaired_count: int = 0,
    active_label: str = "RWKV",
    comparison_generation: str | None = None,
    compatible_generations: tuple[str, ...] = (),
    comparison_loaded: bool = True,
    error_message: str = "",
    running: bool = False,
    focus_target: str | None = None,
    has_result_set: bool = False,
    review_source_expanded: bool = False,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    spec = mode_spec(mode)
    um_plus = isinstance(summary, UmPlusSummary)
    title = f"{spec.evaluate_label} {'UM+' if um_plus else 'Calibration Graph'}"
    comparison_label = (
        "FSRS"
        if comparison_generation is None
        else f"RWKV {model_generation_label(comparison_generation)}"
    )
    source_field = asdict(
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
    filter_fields = [
        asdict(field)
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
                    FieldOption(operator, operator) for operator in ("<", "<=", "=", ">=", ">")
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
                label="Same-day",
                kind="checkbox",
                checked=include_sameday,
                tooltip="Include reviews repeated on the same day in the chart.",
                disabled=running,
                change_action="set-same-day",
                change_serialize_form=True,
                initial_focus=focus_target == "include_sameday",
            ),
        )
    ]
    comparison_field = asdict(
        ModalField(
            name="comparison_generation",
            label="Comparison Target",
            kind="select",
            value=comparison_generation or "",
            options=(
                FieldOption("", "FSRS"),
                *(
                    FieldOption(value, f"RWKV {model_generation_label(value)}")
                    for value in compatible_generations
                ),
            ),
            tooltip=(
                "Choose FSRS or another generation supporting the active history and evaluation "
                "settings, as in Evaluate. Click Calculate to load a changed target."
                + (
                    " No other generation supports these settings."
                    if not compatible_generations
                    else ""
                )
            ),
            disabled=running,
            change_action="set-comparison",
            change_serialize_form=True,
            initial_focus=focus_target == "comparison_generation",
        )
    )
    small_buckets_fields = []
    if um_plus:
        small_buckets_fields.append(asdict(
            ModalField(
                name="show_small_buckets",
                label="Show small buckets",
                kind="checkbox",
                checked=show_small_buckets,
                tooltip=(
                    "Reveal buckets with fewer than 200 reviews as faint points and dotted lines. "
                    "UM+ scores always include every paired review; slopes use visible buckets."
                ),
                disabled=running,
                change_action="set-small-buckets",
                change_serialize_form=True,
                initial_focus=focus_target == "show_small_buckets",
            )
        ))
    filter_groups = [
        dict(cssClass="rwkv-comparison-review-filters", fields=filter_fields[:-1]),
        dict(cssClass="rwkv-comparison-filter-options",
             fields=[comparison_field, filter_fields[-1], *small_buckets_fields]),
    ]
    fsrs_field = asdict(
        ModalField(
            name="include_fsrs",
            label=f"Include {comparison_label}",
            kind="checkbox",
            checked=include_fsrs,
            tooltip=(
                "Compare with FSRS using each card's current preset parameters. "
                "Click Calculate to load the comparison. Both lines use the same "
                "answers with available predictions; this does not fit FSRS parameters."
                if comparison_generation is None
                else "Compare saved review-time predictions from the selected generation. "
                "Click Calculate to load the comparison. Both lines use the same answers "
                "with available predictions."
            ),
            disabled=running,
            change_action="set-fsrs",
            change_serialize_form=True,
            initial_focus=focus_target == "include_fsrs",
        )
    )
    source_fields = [source_field] if um_plus else [source_field, fsrs_field]
    disclosure = asdict(
        ModalDisclosure(
            panel_id="rwkv-calibration-source-panel",
            button_id="rwkv-calibration-source-disclosure",
            action="toggle-review-source",
            collapsed_label="Evaluate a different set of reviews",
            expanded_label="Evaluate a different set of reviews",
            tooltip=(
                "Expand to choose a different review set. The next Calculate will "
                "rebuild predictions before applying the current filters."
            ),
            expanded=review_source_expanded,
            disabled=running,
            button_classes=("rwkv-analysis-source-disclosure",),
            panel_classes=("rwkv-analysis-source-panel", "rwkv-calibration-source-panel"),
        )
    )
    notices = []
    if error_message:
        notices.append(dict(message=error_message, tone="error"))
    elif running:
        notices.append(dict(message=(
            "UM+ calculation is running." if um_plus else "Calibration calculation is running."
        ), tone="info"))
    controls_help = (
        "Calculate creates predictions when needed. After results exist, it normally "
        "applies the display filters without rerunning RWKV."
    )
    if um_plus:
        graph_html = render_um_plus_graph_fragment(
            summary,
            title=title,
            show_small_buckets=show_small_buckets,
            has_result_set=has_result_set and comparison_loaded,
            active_label=active_label,
            comparison_label=comparison_label,
            comparison_is_fsrs=comparison_generation is None,
        )
    else:
        graph_html = render_calibration_graph_fragment(
            summary,
            title=title,
            include_fsrs=include_fsrs,
            fsrs_summary=fsrs_summary,
            unpaired_count=unpaired_count,
            active_label=active_label,
            comparison_label=comparison_label,
            comparison_is_fsrs=comparison_generation is None,
        )
    results_title = "UM+ Results" if um_plus else "Calibration Results"
    return render_svelte_document(
        "analysis", dict(
            title=title, prefix="calibration", controlsTitle="Reviews and Filters",
            controlsHelp=controls_help, sourceFields=source_fields, sourceGrid=not um_plus,
            filterGroups=filter_groups, disclosure=disclosure,
            hasResultSet=has_result_set, running=running,
            resultsTitle=results_title, summaryText=summary_text, notices=notices,
            graph=chart_markup(graph_html),
        ),
        title=title,
        generation=generation,
        is_dark=is_dark,
        width="wide",
        root_extra_classes="rwkv-analysis-dialog rwkv-calibration-dialog",
    )
