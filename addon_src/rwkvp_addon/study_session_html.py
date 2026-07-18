from __future__ import annotations

import html
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .anki_api import DeckRetention
from .modal_html import (
    FieldOption,
    ModalButton,
    ModalButtonGroup,
    ModalDisclosure,
    ModalField,
    render_button,
    render_disclosure,
    render_field,
    render_footer,
    render_modal_document,
    render_notice,
)
from .search_filters import candidate_filter_count_label

if TYPE_CHECKING:
    from .study_session_controller import StudySessionDraft, StudySessionFormSpec

_FORM_ID = "rwkv-study-session-form"


def render_study_session_dialog_html(
    *,
    spec: StudySessionFormSpec,
    source_name: str,
    retentions: Sequence[DeckRetention],
    order_options: Sequence[tuple[int, str]],
    draft: StudySessionDraft,
    adaptive_available: bool,
    filter_count: int | None,
    running: bool = False,
    focus_target: str | None = None,
    retention_editor_expanded: bool = False,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    retention_help = (
        "Set each deck's recall target. Cards reviewed today can use a separate target."
    )
    count_notice = (
        render_notice(candidate_filter_count_label(filter_count), tone="success")
        if filter_count is not None
        else ""
    )
    body_html = f"""
<form class="rwkv-study-session-form" id="{_FORM_ID}"
      data-rwkv-form-action="{html.escape(spec.primary_action, quote=True)}">
  <section class="rwkv-section" id="rwkv-study-setup">
    <h2 class="rwkv-section-title rwkv-help-surface" tabindex="0"
        data-rwkv-tooltip="{html.escape(spec.size_section_intro, quote=True)}">
      {html.escape(spec.size_section_title)}
    </h2>
    <div class="rwkv-card">
      <div class="rwkv-study-source-grid">
        <div class="rwkv-study-source-deck">
          <span class="rwkv-field__label">Source deck</span>
          <strong class="rwkv-study-source-value rwkv-help-surface" tabindex="0"
                  data-rwkv-tooltip="{html.escape(source_name, quote=True)}">
            {html.escape(source_name)}
          </strong>
        </div>
        {
        render_field(
            ModalField(
                name="search_filter",
                label="Additional Anki search",
                kind="search",
                value=draft.search_filter,
                tooltip=(
                    "This is combined with the source deck and active-card rules. "
                    "For example: is:due or prop:ivl>20."
                ),
                placeholder="Optional Anki search, e.g. is:due or prop:ivl>20",
                disabled=running,
                enter_action="check-search",
                initial_focus=focus_target == "search_filter",
            )
        )
    }
        <div class="rwkv-study-search-action">
          {
        render_button(
            ModalButton(
                "Check",
                "check-search",
                variant="secondary",
                submit=True,
                form_no_validate=True,
                disabled=running,
                tooltip="Count cards matched by the source deck and Anki search.",
            )
        )
    }
        </div>
      </div>
      <div class="rwkv-study-search-result" aria-live="polite">{count_notice}</div>
      <div class="rwkv-study-selection-grid">
        {
        render_field(
            ModalField(
                name="minimum",
                label=spec.minimum_label,
                kind="number",
                value=draft.minimum,
                tooltip=spec.minimum_description,
                minimum=0,
                maximum=spec.maximum_value,
                step=1,
                required=True,
                disabled=running,
            )
        )
    }
        {
        render_field(
            ModalField(
                name="maximum",
                label=spec.maximum_label,
                kind="number",
                value=draft.maximum,
                tooltip=spec.maximum_description,
                minimum=1,
                maximum=spec.maximum_value,
                step=1,
                required=True,
                disabled=running,
            )
        )
    }
        {
        render_field(
            ModalField(
                name="order_index",
                label="Sort order",
                kind="select",
                value=draft.order_index,
                tooltip="Controls how selected cards are placed in the study queue.",
                options=tuple(
                    FieldOption(str(value), label) for value, label in order_options
                ),
                disabled=running,
            )
        )
    }
      </div>
    </div>
  </section>

  <section class="rwkv-section" id="rwkv-study-retention">
    <h2 class="rwkv-section-title rwkv-help-surface" tabindex="0"
        data-rwkv-tooltip="{html.escape(retention_help, quote=True)}">
      Desired Retention
    </h2>
    <div class="rwkv-card">
      {
        _render_retention_editor(
            retentions=retentions,
            draft=draft,
            running=running,
            focus_target=focus_target,
            expanded=retention_editor_expanded,
        )
    }
      {_render_adaptive_editor(draft=draft, running=running) if adaptive_available else ""}
    </div>
  </section>
</form>
""".strip()
    footer = render_footer(
        (
            ModalButton(
                "Cancel",
                "dialog-close",
                variant="secondary",
                payload={"outcome": "reject"},
                disabled=running,
            ),
            *_render_start_actions(
                spec=spec,
                running=running,
                primary_has_focus=focus_target == "primary_action",
            ),
        ),
        leading_buttons=(
            ModalButton(
                "Restore Defaults",
                "restore-defaults",
                variant="quiet",
                disabled=running,
                tooltip="Forget the saved settings for this source deck.",
            ),
        ),
        sticky=True,
    )
    root_classes = "rwkv-study-session-dialog"
    if retention_editor_expanded:
        root_classes += " rwkv-retention-editor-expanded"
    return render_modal_document(
        title=spec.title,
        intro=spec.intro,
        body_html=body_html,
        footer_html=footer,
        generation=generation,
        is_dark=is_dark,
        width="standard",
        root_extra_classes=root_classes,
    )


def _render_start_actions(
    *,
    spec: StudySessionFormSpec,
    running: bool,
    primary_has_focus: bool,
) -> tuple[ModalButton | ModalButtonGroup, ...]:
    primary = ModalButton(
        spec.primary_label,
        spec.primary_action,
        variant="primary",
        disabled=running,
        submit=True,
        form_id=_FORM_ID,
        form_no_validate=True,
        initial_focus=primary_has_focus,
    )
    same_day = ModalButton(
        "☀" if spec.split_same_day_action else spec.same_day_label,
        spec.same_day_action,
        variant="primary",
        disabled=running,
        submit=True,
        form_id=_FORM_ID,
        form_no_validate=True,
        tooltip=spec.same_day_description,
        aria_label=spec.same_day_label if spec.split_same_day_action else None,
    )
    if not spec.split_same_day_action:
        return same_day, primary
    return (
        ModalButtonGroup(
            (same_day, primary),
            label="Live Session start options",
            group_id="rwkv-live-session-start",
            extra_classes=("rwkv-button-group--joined", "rwkv-study-start-split"),
        ),
    )


def _render_retention_editor(
    *,
    retentions: Sequence[DeckRetention],
    draft: StudySessionDraft,
    running: bool,
    focus_target: str | None,
    expanded: bool,
) -> str:
    rows = []
    base_depth = min(
        (_deck_path_depth(retention.name) for retention in retentions),
        default=0,
    )
    for index, retention in enumerate(retentions):
        desired_name = f"desired_retention[{index}]"
        same_day_name = f"same_day_desired_retention[{index}]"
        deck_depth = min(10, max(0, _deck_path_depth(retention.name) - base_depth))
        deck_label = _deck_leaf_name(retention.name)
        deck_path_label = _render_deck_path_label(
            label=deck_label,
            full_path=retention.name,
        )
        same_day_value = draft.same_day_desired_retentions[index]
        desired_input = _render_percent_input(
            input_id=f"rwkv-retention-{index}",
            name=desired_name,
            value=draft.desired_retentions[index],
            label=f"Desired retention for {retention.name}",
            disabled=running,
            readonly=draft.override_enabled and not running,
            initial_focus=focus_target == "retention_table" and index == 0,
        )
        same_day_input = _render_percent_input(
            input_id=f"rwkv-same-day-retention-{index}",
            name=same_day_name,
            value=same_day_value,
            label=f"Same-day desired retention for {retention.name}",
            disabled=running,
            readonly=draft.same_day_override_enabled and not running,
        )
        rows.append(
            '<div class="rwkv-retention-grid__row" role="row">'
            f'<div class="rwkv-retention-grid__deck" role="rowheader" '
            f'style="--rwkv-deck-depth: {deck_depth}">{deck_path_label}</div>'
            f'<div class="rwkv-retention-grid__cell" role="cell">{desired_input}</div>'
            f'<div class="rwkv-retention-grid__cell" role="cell">{same_day_input}</div>'
            "</div>"
        )
    if not rows:
        rows.append(
            '<div class="rwkv-retention-grid__row" role="row">'
            '<div class="rwkv-study-empty-row" role="cell">No decks found.</div></div>'
        )
    grid_classes = "rwkv-retention-grid"
    if draft.override_enabled:
        grid_classes += " rwkv-retention-grid--dr-overridden"
    if draft.same_day_override_enabled:
        grid_classes += " rwkv-retention-grid--same-day-overridden"
    override_editor = _render_override_pair(
        name="override",
        label="Override DR",
        checked=draft.override_enabled,
        value=draft.override_value,
        tooltip="Use one desired-retention percentage for every deck.",
        running=running,
    )
    same_day_override_editor = _render_override_pair(
        name="same_day_override",
        label="Override Same Day DR",
        checked=draft.same_day_override_enabled,
        value=draft.same_day_override_value,
        tooltip="Use one same-day desired-retention percentage for every deck.",
        running=running,
    )
    retention_table = render_disclosure(
        ModalDisclosure(
            button_id="rwkv-retention-disclosure",
            panel_id="rwkv-retention-table",
            collapsed_label="View and edit saved retention targets for individual decks",
            expanded_label="View and edit saved retention targets for individual decks",
            expanded=expanded,
            disabled=running,
            action="toggle-retention-editor",
            expanded_root_class="rwkv-retention-editor-expanded",
            panel_classes=("rwkv-table-wrap", "rwkv-study-retention-table-wrap"),
            panel_role="table",
            panel_label="Desired retention by deck",
            panel_tabindex=0,
        ),
        f"""
    <div class="rwkv-retention-grid__header-viewport" role="rowgroup">
      <div class="rwkv-retention-grid__header" role="row">
        <div role="columnheader">Deck</div>
        <div role="columnheader">DR</div>
        <div role="columnheader">Same-day DR</div>
      </div>
    </div>
    <div class="rwkv-retention-grid__body {grid_classes}"
         id="rwkv-retention-grid-body" role="rowgroup">
      {"".join(rows)}
    </div>
""".strip(),
    )
    return f"""
<div class="rwkv-retention-overrides" aria-label="Retention overrides">
  {override_editor}
  {same_day_override_editor}
</div>
<div class="rwkv-retention-editor">
  {retention_table}
</div>
""".strip()


def _deck_path_depth(name: str) -> int:
    return str(name).count("::")


def _deck_leaf_name(name: str) -> str:
    full_name = str(name)
    leaf = full_name.rsplit("::", 1)[-1]
    return leaf or full_name


def _render_deck_path_label(*, label: str, full_path: str) -> str:
    escaped_path = html.escape(full_path, quote=True)
    return (
        f'<span class="rwkv-deck-path rwkv-help-surface" tabindex="0" '
        f'aria-label="{escaped_path}" data-rwkv-tooltip="{escaped_path}">'
        f"{html.escape(label)}</span>"
    )


def _render_override_pair(
    *,
    name: str,
    label: str,
    checked: bool,
    value: str,
    tooltip: str,
    running: bool,
) -> str:
    value_name = f"{name}_value"
    checkbox = render_field(
        ModalField(
            name=f"{name}_enabled",
            label=label,
            kind="checkbox",
            checked=checked,
            tooltip=tooltip,
            disabled=running,
            change_action="update-form",
            change_serialize_form=True,
            enable_fields=(value_name,),
        )
    )
    value_input = _render_percent_input(
        input_id=f"rwkv-field-{value_name}",
        name=value_name,
        value=value,
        label=f"{label} percentage",
        disabled=running or not checked,
        tooltip=tooltip,
    )
    return f'<div class="rwkv-retention-override-pair">{checkbox}{value_input}</div>'


def _render_percent_input(
    *,
    input_id: str,
    name: str,
    value: str,
    label: str,
    disabled: bool,
    readonly: bool = False,
    initial_focus: bool = False,
    tooltip: str | None = None,
) -> str:
    attributes = [
        f'id="{html.escape(input_id, quote=True)}"',
        f'name="{html.escape(name, quote=True)}"',
        'type="number"',
        'min="1"',
        'max="99"',
        'step="0.1"',
        f'value="{html.escape(value, quote=True)}"',
        "required",
        'aria-required="true"',
    ]
    if tooltip:
        attributes.append(f'aria-describedby="{html.escape(input_id, quote=True)}-description"')
    if disabled:
        attributes.extend(("disabled", 'aria-disabled="true"'))
    if readonly:
        attributes.extend(("readonly", 'aria-disabled="true"', 'data-rwkv-overridden="true"'))
    if initial_focus:
        attributes.append("data-rwkv-initial-focus")
    description = (
        f'<span class="rwkv-sr-only" '
        f'id="{html.escape(input_id, quote=True)}-description">'
        f"{html.escape(tooltip)}</span>"
        if tooltip
        else ""
    )
    return (
        f'<label class="rwkv-sr-only" for="{html.escape(input_id, quote=True)}">'
        f"{html.escape(label)} (percent)</label>"
        f"{description}"
        '<div class="rwkv-percent-control">'
        f'<input class="rwkv-table-input rwkv-percent-control__input" {" ".join(attributes)}>'
        '<span class="rwkv-percent-control__suffix" aria-hidden="true">%</span>'
        "</div>"
    )


def _render_adaptive_editor(*, draft: StudySessionDraft, running: bool) -> str:
    parameters_disabled = running or not draft.adaptive_enabled
    open_attribute = " open" if draft.adaptive_enabled else ""
    return f"""
<details class="rwkv-adaptive-retention"{open_attribute}>
  <summary>Adaptive Desired Retention <span>Optional</span></summary>
  <div class="rwkv-adaptive-retention__body">
  <p class="rwkv-subtle">Calculate each non-same-day card's DR from its latest
    RWKV curve stability and FSRS difficulty.</p>
  {
        render_field(
            ModalField(
                name="adaptive_enabled",
                label="Use Adaptive DR",
                kind="checkbox",
                checked=draft.adaptive_enabled,
                disabled=running,
                change_action="update-form",
                change_serialize_form=True,
                enable_fields=(
                    "adaptive_flat",
                    "adaptive_s_multi",
                    "adaptive_d_multi",
                ),
            )
        )
    }
  <div class="rwkv-adaptive-retention__parameters">
    {
        render_field(
            ModalField(
                name="adaptive_flat",
                label="flat",
                kind="number",
                value=draft.adaptive_flat,
                minimum=-10,
                maximum=10,
                step=0.01,
                required=True,
                disabled=parameters_disabled,
            )
        )
    }
    {
        render_field(
            ModalField(
                name="adaptive_s_multi",
                label="s_multi",
                kind="number",
                value=draft.adaptive_s_multi,
                minimum=-10,
                maximum=10,
                step=0.01,
                required=True,
                disabled=parameters_disabled,
            )
        )
    }
    {
        render_field(
            ModalField(
                name="adaptive_d_multi",
                label="d_multi",
                kind="number",
                value=draft.adaptive_d_multi,
                minimum=-10,
                maximum=10,
                step=0.01,
                required=True,
                disabled=parameters_disabled,
            )
        )
    }
  </div>
  </div>
</details>
""".strip()
