from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING

from .modal_html import (
    FieldOption,
    ModalButton,
    ModalDisclosure,
    ModalField,
)
from .model_generation import model_generation_label
from .svelte_ui import render_svelte_document

if TYPE_CHECKING:
    from .evaluate_controller import EvaluationDisplayRow, EvaluationScopeSelection


def render_evaluate_dialog_html(
    *,
    title: str,
    rwkv_label: str,
    scope_selection: EvaluationScopeSelection,
    start_date: dt.date,
    end_date: dt.date,
    full_collection_start_date: dt.date,
    full_collection_end_date: dt.date,
    cheating_fsrs: bool,
    show_insufficient: bool,
    show_comparison: bool,
    comparison_available: bool,
    rows: Sequence[EvaluationDisplayRow],
    selected_scope_count: int,
    hidden_insufficient_count: int,
    running: bool = False,
    focus_target: str | None = None,
    advanced_options_expanded: bool = False,
    is_dark: bool = False,
    generation: int = 1,
    comparison_generation: str | None = None,
    comparison_generations: Sequence[str] = (),
    active_generation: str = "v1",
    comparison_button_label: str | None = None,
) -> str:
    comparison_label = (
        model_generation_label(comparison_generation) if comparison_generation else "FSRS"
    )
    active_label = model_generation_label(active_generation) if comparison_generation else "RWKV"
    scope_fields = [
        asdict(field)
        for field in (
            ModalField(
                name="include_collection",
                label="Collection",
                kind="checkbox",
                checked=scope_selection.include_collection,
                tooltip="Add one result row for your entire collection.",
                disabled=running,
                change_action="update-options",
                change_serialize_form=True,
                initial_focus=focus_target == "include_collection",
            ),
            ModalField(
                name="include_presets",
                label="Presets",
                kind="checkbox",
                checked=scope_selection.include_presets,
                tooltip=(
                    "Add one result row for each FSRS scheduling preset currently "
                    "assigned to a deck."
                ),
                disabled=running,
                change_action="update-options",
                change_serialize_form=True,
            ),
            ModalField(
                name="include_decks",
                label="Decks",
                kind="checkbox",
                checked=scope_selection.include_decks,
                tooltip="Add one result row for each normal deck.",
                disabled=running,
                change_action="update-options",
                change_serialize_form=True,
            ),
        )
    ]
    date_fields = [
        asdict(field)
        for field in (
            ModalField(
                name="start_date",
                label="RWKV reviews from",
                kind="date",
                value=start_date.isoformat(),
                tooltip=(
                    "First local calendar day included in RWKV metrics. The full "
                    f"collection starts on {full_collection_start_date.isoformat()}."
                ),
                required=True,
                disabled=running,
                change_action="update-options",
                change_serialize_form=True,
                initial_focus=focus_target == "start_date",
            ),
            ModalField(
                name="end_date",
                label="Through",
                kind="date",
                value=end_date.isoformat(),
                tooltip=(
                    "Last local calendar day included in RWKV metrics. The full "
                    f"collection ends on {full_collection_end_date.isoformat()}."
                ),
                required=True,
                disabled=running,
                change_action="update-options",
                change_serialize_form=True,
            ),
        )
    ]
    advanced_fields = [
        asdict(field)
        for field in (
            ModalField(
                name="cheating_fsrs",
                label="Cheating FSRS-6",
                kind="checkbox",
                checked=cheating_fsrs,
                tooltip=(
                    "Train fresh FSRS-6 parameters for each scope and evaluate them "
                    "on those same review items. Available when comparing with FSRS-6."
                ),
                disabled=running or not comparison_available or comparison_generation is not None,
                change_action="update-options",
                change_serialize_form=True,
            ),
            ModalField(
                name="show_insufficient",
                label="Show insufficient reviews",
                kind="checkbox",
                checked=show_insufficient,
                tooltip="Keep scopes that do not contain enough evaluable history.",
                disabled=running,
                change_action="update-options",
                change_serialize_form=True,
            ),
        )
    ]
    full_range_button = asdict(
        ModalButton(
            "Use Full Range",
            "use-full-range",
            variant="quiet",
            disabled=running,
            serialize_form=True,
            button_id="rwkv-evaluate-full-range",
            form_no_validate=True,
        )
    )
    date_range_help = (
        "Both RWKV generations are scored on the same eligible reviews within these dates."
        if comparison_generation else
        "Date limits apply only to RWKV. FSRS-6 comparison requires the full collection range."
    )
    swap_field = asdict(ModalField(
        name="comparison_generation",
        label="Model generation",
        kind="select",
        value=comparison_generation or "",
        options=(FieldOption("", "FSRS-6 (default)"), *(
            FieldOption(value, f"RWKV {model_generation_label(value)}")
            for value in comparison_generations
        )),
        tooltip=(
            f"Active: RWKV {model_generation_label(active_generation)}. Only generations "
            "supporting the same history and evaluation settings are listed."
            if comparison_generations else
            "No other generation supports the active history and evaluation settings."
        ),
        disabled=running or not comparison_generations,
        change_action="update-options",
        change_serialize_form=True,
    ))
    additional_options_help = "Less commonly needed evaluation and result-display controls."
    disclosure = asdict(ModalDisclosure(
        button_id="rwkv-evaluate-advanced-disclosure",
        panel_id="rwkv-evaluate-advanced-panel",
        collapsed_label="Show advanced options...",
        expanded_label="Hide advanced options...",
        expanded=advanced_options_expanded, disabled=running,
        action="toggle-advanced-options",
        button_classes=("rwkv-evaluation-disclosure",),
        panel_classes=("rwkv-evaluation-advanced-panel",),
    ))
    actions = [
        asdict(ModalButton(
            f"{rwkv_label} Only", "run-rwkv-only", variant="secondary",
            disabled=running, serialize_form=True, button_id="rwkv-evaluate-rwkv-only",
            initial_focus=focus_target == "run_rwkv_only",
        )),
        asdict(ModalButton(
            comparison_button_label or f"Compare FSRS-6 and {rwkv_label}", "run-comparison",
            variant="primary", disabled=running or not comparison_available,
            serialize_form=True, button_id="rwkv-evaluate-compare",
            initial_focus=focus_target == "run_comparison",
        )),
    ]
    results_help = (
        "Lower RMSE(bins) and LogLoss values are better. Improvement shows the active "
        f"model as a percentage of {comparison_label}. For RMSE(bins) and LogLoss, "
        "under 100% is better (green), while over 100% is worse (red). "
        "Higher AUC is better: it measures how well predicted recall "
        "distinguishes successes from lapses (0.5 means chance ranking, 1 means perfect "
        "ranking). AUC is rounded to six decimals, matching srs-benchmark. A dash means "
        "the scored reviews lack successes or lapses. "
        + (
            "For AUC Improvement, over 100% is better (green), while under 100% is "
            "worse (red). Equal scores are neutral. Missing AUC or a zero comparison "
            "AUC shows a dash for Improvement. Both generations use the same eligible "
            "review IDs and date range."
            if comparison_generation else
            "Anki does not expose FSRS AUC, so this column reports RWKV only."
        )
    )
    status = (
        "Evaluation is running." if running else
        f"The selected dates can be evaluated with {rwkv_label} Only. Use Full "
        "Range to enable the FSRS-6 comparison." if not comparison_available else ""
    )
    hidden_note = ""
    if hidden_insufficient_count:
        suffix = "scope" if hidden_insufficient_count == 1 else "scopes"
        hidden_note = f"{hidden_insufficient_count} {suffix} with insufficient reviews hidden."
    if selected_scope_count == 0:
        empty_message = "Select at least one evaluation scope."
    elif hidden_insufficient_count:
        empty_message = (
            "Every selected scope has insufficient reviews. Enable Show "
            "insufficient reviews to display them."
        )
    else:
        empty_message = "No evaluation scopes are available."
    return render_svelte_document(
        "evaluate", dict(
            title=title, rwkvLabel=rwkv_label, scopeFields=scope_fields,
            dateFields=date_fields, advancedFields=advanced_fields, swapField=swap_field,
            fullRangeButton=full_range_button, disclosure=disclosure, actions=actions,
            dateRangeHelp=date_range_help, additionalOptionsHelp=additional_options_help,
            resultsHelp=results_help, status=status, hiddenNote=hidden_note,
            emptyMessage=empty_message, showComparison=show_comparison,
            compareAuc=comparison_generation is not None,
            comparisonLabel=comparison_label, activeLabel=active_label,
            rows=[dict(asdict(row), sort_values=dict(
                asdict(row.sort_values), scope=row.sort_values.scope or row.scope.casefold(),
            )) for row in rows],
        ), title=title, generation=generation, is_dark=is_dark, width="wide",
        root_extra_classes="rwkv-evaluation-dialog",
    )
