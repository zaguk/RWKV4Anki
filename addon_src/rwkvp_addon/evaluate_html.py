from __future__ import annotations

import datetime as dt
import html
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .modal_html import (
    ModalButton,
    ModalDisclosure,
    ModalField,
    render_button,
    render_disclosure,
    render_field,
    render_modal_document,
    render_notice,
)

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
) -> str:
    scope_fields = "".join(
        render_field(field)
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
                tooltip="Add one result row for each FSRS scheduling preset.",
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
    )
    date_fields = "".join(
        render_field(field)
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
    )
    advanced_fields = "".join(
        render_field(field)
        for field in (
            ModalField(
                name="cheating_fsrs",
                label="Cheating FSRS-6",
                kind="checkbox",
                checked=cheating_fsrs,
                tooltip=(
                    "Train fresh FSRS-6 parameters for each scope and evaluate them "
                    "on those same review items."
                ),
                disabled=running or not comparison_available,
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
    )
    full_range_button = render_button(
        ModalButton(
            "Use Full Range",
            "use-full-range",
            variant="quiet",
            disabled=running,
            serialize_form=True,
            button_id="rwkv-evaluate-full-range",
        )
    )
    date_range_help = (
        "Date limits apply only to RWKV. FSRS-6 comparison requires the full collection range."
    )
    additional_options_help = "Less commonly needed evaluation and result-display controls."
    advanced_options = render_disclosure(
        ModalDisclosure(
            button_id="rwkv-evaluate-advanced-disclosure",
            panel_id="rwkv-evaluate-advanced-panel",
            collapsed_label="Show advanced options...",
            expanded_label="Hide advanced options...",
            expanded=advanced_options_expanded,
            disabled=running,
            action="toggle-advanced-options",
            button_classes=("rwkv-evaluation-disclosure",),
            panel_classes=("rwkv-evaluation-advanced-panel",),
        ),
        f"""
      <section class="rwkv-evaluation-advanced-subsection"
               aria-labelledby="rwkv-evaluate-date-title">
        <h3 class="rwkv-evaluation-option-title rwkv-evaluation-subsection-title rwkv-help-surface"
            tabindex="0"
            id="rwkv-evaluate-date-title"
            data-rwkv-tooltip="{html.escape(date_range_help, quote=True)}">
          RWKV Date Range</h3>
        <div class="rwkv-evaluation-date-grid">
          {date_fields}
          <div class="rwkv-evaluation-date-action">{full_range_button}</div>
        </div>
      </section>
      <section class="rwkv-evaluation-advanced-subsection"
               aria-labelledby="rwkv-evaluate-additional-title">
        <h3 class="rwkv-evaluation-option-title rwkv-evaluation-subsection-title rwkv-help-surface"
            tabindex="0"
            id="rwkv-evaluate-additional-title"
            data-rwkv-tooltip="{html.escape(additional_options_help, quote=True)}">
          Additional Evaluation Options</h3>
        <div class="rwkv-evaluation-advanced-grid">{advanced_fields}</div>
      </section>
""".strip(),
    )
    options_form = f"""
<form class="rwkv-form rwkv-evaluation-form" id="rwkv-evaluate-options"
      data-rwkv-form-action="update-options">
  <section class="rwkv-evaluation-option-group rwkv-evaluation-scope-options"
           aria-labelledby="rwkv-evaluate-group-results-title">
    <h3 class="rwkv-evaluation-option-title rwkv-help-surface" tabindex="0"
        id="rwkv-evaluate-group-results-title"
        data-rwkv-tooltip="Choose one or more ways to group the metric results.">
      Group results by</h3>
    <div class="rwkv-evaluation-scope-grid">{scope_fields}</div>
  </section>
  <section class="rwkv-evaluation-option-group rwkv-evaluation-advanced-options">
    {advanced_options}
  </section>
  {
        _options_actions(
            rwkv_label=rwkv_label,
            comparison_available=comparison_available,
            running=running,
            focus_target=focus_target,
        )
    }
</form>
""".strip()
    scope_help = (
        "Choose whether metrics are grouped for the whole collection, scheduling "
        "presets, decks, or any combination."
    )
    results_help = (
        f"Lower RMSE(bins) and LogLoss values are better. Improvement shows {rwkv_label} "
        "as a percentage of FSRS-6: under 100% is better (green), while over 100% "
        "is worse (red)."
    )
    status = ""
    if running:
        status = render_notice("Evaluation is running.", tone="info")
    elif not comparison_available:
        status = render_notice(
            f"The selected dates can be evaluated with {rwkv_label} Only. Use Full "
            "Range to enable the FSRS-6 comparison.",
            tone="info",
        )
    hidden_note = ""
    if hidden_insufficient_count:
        suffix = "scope" if hidden_insufficient_count == 1 else "scopes"
        hidden_note = render_notice(
            f"{hidden_insufficient_count} {suffix} with insufficient reviews hidden.",
            tone="info",
        )
    body_html = f"""
<section class="rwkv-section" id="rwkv-evaluate-scope">
  <h2 class="rwkv-section-title rwkv-help-surface" tabindex="0"
      data-rwkv-tooltip="{html.escape(scope_help, quote=True)}">Evaluation Scope</h2>
  <div class="rwkv-card">
    {options_form}
    {status}
  </div>
</section>
<section class="rwkv-section" id="rwkv-evaluate-results">
  <h2 class="rwkv-section-title rwkv-help-surface" tabindex="0"
      data-rwkv-tooltip="{html.escape(results_help, quote=True)}">Evaluation Results</h2>
  {hidden_note}
  <div class="rwkv-card rwkv-evaluation-results" aria-live="polite">
    {
        _render_results_table(
            rwkv_label=rwkv_label,
            rows=rows,
            show_comparison=show_comparison,
            selected_scope_count=selected_scope_count,
            hidden_insufficient_count=hidden_insufficient_count,
        )
    }
  </div>
</section>
""".strip()
    return render_modal_document(
        title=title,
        body_html=body_html,
        generation=generation,
        is_dark=is_dark,
        width="wide",
        root_extra_classes="rwkv-evaluation-dialog",
    )


def _options_actions(
    *,
    rwkv_label: str,
    comparison_available: bool,
    running: bool,
    focus_target: str | None,
) -> str:
    return (
        '<div class="rwkv-evaluation-actions">'
        + '<div class="rwkv-evaluation-actions__runs">'
        + render_button(
            ModalButton(
                f"{rwkv_label} Only",
                "run-rwkv-only",
                variant="secondary",
                disabled=running,
                serialize_form=True,
                button_id="rwkv-evaluate-rwkv-only",
                initial_focus=focus_target == "run_rwkv_only",
            )
        )
        + render_button(
            ModalButton(
                f"Compare FSRS-6 and {rwkv_label}",
                "run-comparison",
                variant="primary",
                disabled=running or not comparison_available,
                serialize_form=True,
                button_id="rwkv-evaluate-compare",
                initial_focus=focus_target == "run_comparison",
            )
        )
        + "</div></div>"
    )


def _render_results_table(
    *,
    rwkv_label: str,
    rows: Sequence[EvaluationDisplayRow],
    show_comparison: bool,
    selected_scope_count: int,
    hidden_insufficient_count: int,
) -> str:
    if show_comparison:
        headers = (
            '<th scope="col" class="rwkv-evaluation-scope-header">Scope</th>'
            '<th scope="col" class="rwkv-number rwkv-evaluation-reviews-header">'
            "Reviews Evaluated</th>"
            + _render_metric_header(
                "RMSE(bins)",
                rwkv_label=rwkv_label,
                show_comparison=True,
            )
            + '<th scope="col" class="rwkv-number rwkv-evaluation-improvement-header">'
            "Improvement</th>"
            + _render_metric_header(
                "LogLoss",
                rwkv_label=rwkv_label,
                show_comparison=True,
            )
            + '<th scope="col" class="rwkv-number rwkv-evaluation-improvement-header">'
            "Improvement</th>"
        )
        column_count = 6
    else:
        headers = (
            '<th scope="col" class="rwkv-evaluation-scope-header">Scope</th>'
            '<th scope="col" class="rwkv-number rwkv-evaluation-reviews-header">'
            "Reviews Evaluated</th>"
            + _render_metric_header(
                "RMSE(bins)",
                rwkv_label=rwkv_label,
                show_comparison=False,
            )
            + _render_metric_header(
                "LogLoss",
                rwkv_label=rwkv_label,
                show_comparison=False,
            )
        )
        column_count = 4
    if rows:
        body = "".join(_render_result_row(row, show_comparison=show_comparison) for row in rows)
    else:
        if selected_scope_count == 0:
            message = "Select at least one evaluation scope."
        elif hidden_insufficient_count:
            message = (
                "Every selected scope has insufficient reviews. Enable Show "
                "insufficient reviews to display them."
            )
        else:
            message = "No evaluation scopes are available."
        body = (
            f'<tr class="rwkv-table-empty"><td colspan="{column_count}">'
            f"{html.escape(message)}</td></tr>"
        )
    return f"""
<div class="rwkv-table-wrap" tabindex="0" role="region"
     aria-label="Evaluation results">
  <table class="rwkv-data-table rwkv-evaluation-table" id="rwkv-evaluation-table">
    <caption class="rwkv-sr-only">Evaluation results</caption>
    <thead><tr>{headers}</tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
""".strip()


def _render_metric_header(
    metric: str,
    *,
    rwkv_label: str,
    show_comparison: bool,
) -> str:
    if show_comparison:
        visible_sources = "FSRS | RWKV"
        accessible_sources = f"FSRS-6 and {rwkv_label}"
    else:
        visible_sources = "RWKV"
        accessible_sources = rwkv_label
    return (
        '<th scope="col" class="rwkv-number rwkv-evaluation-metric-header">'
        f'<span class="rwkv-evaluation-metric-header__name">{html.escape(metric)}</span>'
        '<span class="rwkv-evaluation-metric-header__sources" '
        f'aria-label="{html.escape(accessible_sources, quote=True)}">'
        f"{html.escape(visible_sources)}</span></th>"
    )


def _render_result_row(
    row: EvaluationDisplayRow,
    *,
    show_comparison: bool,
) -> str:
    cells = [
        f'<th scope="row" class="rwkv-evaluation-scope-cell">{html.escape(row.scope)}</th>',
        '<td class="rwkv-number rwkv-evaluation-reviews-cell">'
        f"{html.escape(row.reviews_evaluated)}</td>",
    ]
    if show_comparison:
        cells.extend(
            (
                _render_metric_pair_cell(row.fsrs_rmse, row.rwkv_rmse),
                _render_improvement_cell(
                    row.rmse_improvement,
                    row.rmse_improvement_state,
                ),
                _render_metric_pair_cell(row.fsrs_logloss, row.rwkv_logloss),
                _render_improvement_cell(
                    row.logloss_improvement,
                    row.logloss_improvement_state,
                ),
            )
        )
    else:
        cells.extend(
            (
                '<td class="rwkv-number rwkv-evaluation-metric-value">'
                f"{html.escape(row.rwkv_rmse)}</td>",
                '<td class="rwkv-number rwkv-evaluation-metric-value">'
                f"{html.escape(row.rwkv_logloss)}</td>",
            )
        )
    return "<tr>" + "".join(cells) + "</tr>"


def _render_metric_pair_cell(fsrs_value: str, rwkv_value: str) -> str:
    if not fsrs_value and not rwkv_value:
        value = ""
    else:
        value = f"{fsrs_value or '—'} | {rwkv_value or '—'}"
    return f'<td class="rwkv-number rwkv-evaluation-metric-pair">{html.escape(value)}</td>'


def _render_improvement_cell(value: str, state: str) -> str:
    classes = ["rwkv-number", "rwkv-evaluation-improvement-cell"]
    if state in {"better", "worse"}:
        classes.append(f"rwkv-evaluation-cell--{state}")
    return f'<td class="{html.escape(" ".join(classes), quote=True)}">{html.escape(value)}</td>'
