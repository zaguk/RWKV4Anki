from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .modal_style import modal_root_classes, shared_style_tag
from .web_dialog_bridge import WEB_DIALOG_BRIDGE_PREFIX, require_web_dialog_action

__all__ = (
    "FieldOption",
    "ModalButton",
    "ModalButtonGroup",
    "ModalDisclosure",
    "ModalField",
    "ModalTab",
    "ProgressState",
    "document_with_popup_control_mode",
    "render_badge",
    "render_button",
    "render_button_group",
    "render_card",
    "render_close_footer",
    "render_disclosure",
    "render_field",
    "render_footer",
    "render_footnotes",
    "render_message_overlay_template",
    "render_modal_document",
    "render_modal_shell",
    "render_number_input",
    "render_notice",
    "popup_control_mode_script",
    "render_prompt_overlay_template",
    "render_progress_region",
    "render_progress_overlay",
    "render_status_state",
    "render_tab_list",
    "shared_modal_script",
    "shared_modal_script_tag",
)

_COMPONENT_SCRIPT_PATH = Path(__file__).with_name("modal_components.js")
_FIELD_KINDS = frozenset(
    {
        "text",
        "search",
        "number",
        "date",
        "time",
        "datetime-local",
        "select",
        "checkbox",
        "switch",
        "textarea",
    }
)
_BUTTON_VARIANTS = frozenset({"primary", "secondary", "quiet", "destructive"})
_NOTICE_TONES = frozenset({"info", "success", "warning", "error"})
_STATUS_STATES = frozenset({"empty", "loading", "disabled", "error", "success"})
_BADGE_TONES = frozenset({"neutral", "accent", "success", "warning", "error"})
_IDENTIFIER_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")
_HTML_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_CSS_CLASS_PATTERN = re.compile(r"^-?[_A-Za-z]+[-_A-Za-z0-9]*$")
_DISCLOSURE_PANEL_ROLES = frozenset({"region", "table"})


@dataclass(frozen=True)
class ModalButton:
    label: str
    action: str | None
    variant: Literal["primary", "secondary", "quiet", "destructive"] = "secondary"
    payload: Mapping[str, Any] | None = None
    disabled: bool = False
    button_id: str | None = None
    initial_focus: bool = False
    overlay_cancel: bool = False
    submit: bool = False
    serialize_form: bool = False
    form_id: str | None = None
    tooltip: str | None = None
    form_no_validate: bool = False
    aria_label: str | None = None


@dataclass(frozen=True)
class ModalButtonGroup:
    """A labelled set of buttons that should remain visually associated."""

    buttons: tuple[ModalButton, ...]
    label: str
    group_id: str | None = None
    extra_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModalDisclosure:
    """State and accessible relationships for one expandable panel."""

    button_id: str
    panel_id: str
    collapsed_label: str
    expanded_label: str | None = None
    expanded: bool = False
    disabled: bool = False
    action: str | None = None
    tooltip: str | None = None
    expanded_root_class: str | None = None
    button_classes: tuple[str, ...] = ()
    panel_classes: tuple[str, ...] = ()
    panel_role: Literal["region", "table"] = "region"
    panel_label: str | None = None
    panel_tabindex: Literal[-1, 0] | None = None


@dataclass(frozen=True)
class ModalTab:
    key: str
    label: str
    panel_id: str
    selected: bool = False
    disabled: bool = False


@dataclass(frozen=True)
class FieldOption:
    value: str
    label: str
    disabled: bool = False


@dataclass(frozen=True)
class ModalField:
    name: str
    label: str
    kind: Literal[
        "text",
        "search",
        "number",
        "date",
        "time",
        "datetime-local",
        "select",
        "checkbox",
        "switch",
        "textarea",
    ] = "text"
    value: str | int | float = ""
    description: str | None = None
    tooltip: str | None = None
    options: Sequence[FieldOption] = ()
    checked: bool = False
    disabled: bool = False
    required: bool = False
    minimum: int | float | str | None = None
    maximum: int | float | str | None = None
    step: int | float | str | None = None
    placeholder: str | None = None
    rows: int = 4
    change_action: str | None = None
    change_serialize_form: bool = False
    enter_action: str | None = None
    enable_fields: Sequence[str] = ()
    initial_focus: bool = False


@dataclass(frozen=True)
class ProgressState:
    title: str
    label: str
    current: int | float | None = None
    total: int | float | None = None
    cancellable: bool = False
    cancel_pending: bool = False
    visible: bool = True


@lru_cache(maxsize=1)
def shared_modal_script() -> str:
    """Return the packaged interaction layer used by all web dialog pages."""

    return _COMPONENT_SCRIPT_PATH.read_text(encoding="utf-8")


def shared_modal_script_tag() -> str:
    return f'<script data-rwkv-modal-components="true">{shared_modal_script()}</script>'


def popup_control_mode_script(mode: str) -> str:
    """Set the shared popup-control mode before the interaction layer loads."""

    normalized = str(mode)
    if normalized not in {"in-page", "native"}:
        raise ValueError(f"unsupported popup control mode: {mode!r}")
    return (
        '<script data-rwkv-popup-control-mode="true">'
        f"window.RWKV_MODAL_POPUP_CONTROL_MODE = {_json_for_script(normalized)};"
        "</script>"
    )


def document_with_popup_control_mode(document_html: str, *, mode: str) -> str:
    """Apply a host-selected popup mode to an already rendered modal document."""

    return f"{popup_control_mode_script(mode)}\n{document_html}"


def render_modal_document(
    *,
    title: str,
    body_html: str,
    generation: int,
    intro: str | None = None,
    footer_html: str = "",
    overlay_html: str = "",
    is_dark: bool = False,
    width: Literal["compact", "standard", "wide"] = "standard",
    head_html: str = "",
    root_extra_classes: str | None = None,
    escape_action: str = "dialog-close",
    escape_payload: Mapping[str, Any] | None = None,
    enter_action: str | None = None,
    enter_payload: Mapping[str, Any] | None = None,
) -> str:
    """Compose a complete, accessible page for :class:`WebDialogHost`.

    ``body_html``, ``footer_html``, and ``overlay_html`` are trusted markup from
    the render helpers in this module or a workflow renderer. User and
    collection strings passed to those helpers are escaped at their boundary.
    """

    if width not in {"compact", "standard", "wide"}:
        raise ValueError(f"unsupported modal width: {width}")
    title_id = "rwkv-dialog-title"
    page_class = "rwkv-modal-page"
    if width != "standard":
        page_class += f" rwkv-modal-page--{width}"
    intro_markup = (
        f'<p class="rwkv-page-intro" id="rwkv-dialog-intro">{html.escape(intro)}</p>'
        if intro
        else ""
    )
    page_html = f"""
  <div class="{page_class}">
    <header class="rwkv-page-header">
      <h1 class="rwkv-page-title" id="{title_id}">{html.escape(title)}</h1>
      {intro_markup}
    </header>
    <main class="rwkv-modal-content">{body_html}</main>
  </div>
""".strip()
    return render_modal_shell(
        page_html=page_html,
        generation=generation,
        title_id=title_id,
        described_by_id="rwkv-dialog-intro" if intro else None,
        footer_html=footer_html,
        overlay_html=overlay_html,
        head_html=head_html,
        is_dark=is_dark,
        root_extra_classes=root_extra_classes,
        escape_action=escape_action,
        escape_payload=escape_payload,
        enter_action=enter_action,
        enter_payload=enter_payload,
    )


def render_modal_shell(
    *,
    page_html: str,
    generation: int,
    title_id: str,
    footer_html: str = "",
    overlay_html: str = "",
    head_html: str = "",
    described_by_id: str | None = None,
    is_dark: bool = False,
    root_extra_classes: str | None = None,
    escape_action: str = "dialog-close",
    escape_payload: Mapping[str, Any] | None = None,
    enter_action: str | None = None,
    enter_payload: Mapping[str, Any] | None = None,
) -> str:
    """Compose the shared bridge/lifecycle shell around custom trusted page markup.

    Workflow renderers with specialized page structure (for example Behavior
    Lab) can use this lower-level helper without duplicating the generation,
    keyboard, announcer, theme, or component-script contract.
    """

    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ValueError("generation must be a positive integer")
    if 'id="rwkv-progress-overlay"' not in page_html + overlay_html:
        overlay_html = overlay_html + render_progress_overlay(
            ProgressState(
                title="Working",
                label="Preparing operation",
                cancellable=True,
                visible=False,
            ),
            cancel_action="progress-cancel",
        )
    if 'id="rwkv-message-overlay"' not in overlay_html:
        overlay_html = overlay_html + render_message_overlay_template()
    bootstrap = _json_for_script(
        {
            "bridgePrefix": WEB_DIALOG_BRIDGE_PREFIX,
            "generation": generation,
        }
    )
    root_classes = modal_root_classes(is_dark=is_dark, extra=root_extra_classes or "")
    attributes = [
        f'class="{root_classes}"',
        'role="document"',
        f'aria-labelledby="{html.escape(title_id, quote=True)}"',
        f'data-rwkv-generation="{generation}"',
        (
            'data-rwkv-escape-action="'
            + html.escape(require_web_dialog_action(escape_action), quote=True)
            + '"'
        ),
        (
            'data-rwkv-escape-payload="'
            + html.escape(
                json.dumps(
                    dict(escape_payload or {"outcome": "reject"}),
                    separators=(",", ":"),
                ),
                quote=True,
            )
            + '"'
        ),
    ]
    if described_by_id:
        attributes.append(f'aria-describedby="{html.escape(described_by_id, quote=True)}"')
    if enter_action is not None:
        attributes.extend(
            (
                'data-rwkv-enter-action="'
                + html.escape(require_web_dialog_action(enter_action), quote=True)
                + '"',
                'data-rwkv-enter-payload="'
                + html.escape(
                    json.dumps(dict(enter_payload or {}), separators=(",", ":")),
                    quote=True,
                )
                + '"',
            )
        )
    return f"""
{shared_style_tag()}
{head_html}
<div {" ".join(attributes)}>
  {page_html}
  {footer_html}
  <div class="rwkv-sr-only" id="rwkv-dialog-announcer" aria-live="polite" aria-atomic="true"></div>
  {overlay_html}
</div>
<script>window.RWKV_MODAL_BOOTSTRAP = {bootstrap};</script>
{shared_modal_script_tag()}
""".strip()


def render_card(
    content_html: str,
    *,
    title: str | None = None,
    aria_label: str | None = None,
    extra_classes: Sequence[str] = (),
) -> str:
    """Render a simple canonical card around trusted add-on-owned markup."""

    title_markup = (
        f'<h3 class="rwkv-card-title">{html.escape(title)}</h3>' if title is not None else ""
    )
    label_attribute = (
        f' aria-label="{html.escape(aria_label, quote=True)}"'
        if aria_label is not None
        else ""
    )
    classes = _validated_classes(("rwkv-card", *extra_classes), "card class")
    return (
        f'<section class="{html.escape(" ".join(classes), quote=True)}"{label_attribute}>'
        f"{title_markup}{content_html}</section>"
    )


def render_notice(
    message: str,
    *,
    tone: Literal["info", "success", "warning", "error"] = "info",
) -> str:
    _require_member(tone, _NOTICE_TONES, "notice tone")
    role = "alert" if tone == "error" else "status"
    return (
        f'<div class="rwkv-notice rwkv-notice--{tone}" role="{role}">{html.escape(message)}</div>'
    )


def render_tab_list(
    tabs: Sequence[ModalTab],
    *,
    label: str,
    extra_classes: Sequence[str] = (),
) -> str:
    """Render one accessible tab list with a single roving tab stop."""

    if not tabs:
        raise ValueError("tab lists require at least one tab")
    if len({tab.key for tab in tabs}) != len(tabs):
        raise ValueError("tab keys must be unique")
    if len({tab.panel_id for tab in tabs}) != len(tabs):
        raise ValueError("tab panel IDs must be unique")
    if any(tab.selected and tab.disabled for tab in tabs):
        raise ValueError("a disabled tab cannot be selected")
    selected = [tab for tab in tabs if tab.selected]
    if len(selected) > 1:
        raise ValueError("tab lists may select only one tab")
    selected_key = (
        selected[0].key if selected else next((tab.key for tab in tabs if not tab.disabled), None)
    )
    rendered = []
    for tab in tabs:
        tab_id = f"rwkv-tab-{_identifier(tab.key)}"
        is_selected = tab.key == selected_key
        disabled = ' disabled aria-disabled="true"' if tab.disabled else ""
        rendered.append(
            f'<button class="rwkv-tab" type="button" role="tab" id="{tab_id}" '
            f'aria-controls="{html.escape(tab.panel_id, quote=True)}" '
            f'aria-selected="{str(is_selected).lower()}" '
            f'tabindex="{0 if is_selected else -1}"{disabled}>'
            f"{html.escape(tab.label)}</button>"
        )
    classes = _validated_classes(("rwkv-tabs", *extra_classes), "tab list class")
    return (
        f'<div class="{html.escape(" ".join(classes), quote=True)}" role="tablist" '
        f'aria-label="{html.escape(label, quote=True)}">'
        f"{''.join(rendered)}</div>"
    )


def render_button(button: ModalButton) -> str:
    _require_member(button.variant, _BUTTON_VARIANTS, "button variant")
    attributes = [
        f'class="rwkv-button rwkv-button--{button.variant}"',
        f'type="{"submit" if button.submit else "button"}"',
    ]
    if button.action is not None:
        action = require_web_dialog_action(button.action)
        attributes.append(f'data-rwkv-action="{html.escape(action, quote=True)}"')
    elif button.payload is not None or button.serialize_form:
        raise ValueError("payload and form serialization require a bridge action")
    if button.payload is not None:
        attributes.append(
            'data-rwkv-payload="'
            + html.escape(json.dumps(dict(button.payload), separators=(",", ":")), quote=True)
            + '"'
        )
    if button.button_id:
        attributes.append(f'id="{html.escape(button.button_id, quote=True)}"')
    if button.form_id:
        attributes.append(f'form="{html.escape(button.form_id, quote=True)}"')
    if button.tooltip:
        attributes.append(_tooltip_data_attribute(button.tooltip))
    if button.aria_label:
        attributes.append(f'aria-label="{html.escape(button.aria_label, quote=True)}"')
    if button.form_no_validate:
        if not button.submit:
            raise ValueError("form_no_validate requires a submit button")
        attributes.append("formnovalidate")
    if button.disabled:
        attributes.extend(("disabled", 'aria-disabled="true"'))
    if button.initial_focus:
        attributes.append("data-rwkv-initial-focus")
    if button.overlay_cancel:
        attributes.append("data-rwkv-overlay-cancel")
    if button.serialize_form:
        attributes.append("data-rwkv-serialize-form")
    return f"<button {' '.join(attributes)}>{html.escape(button.label)}</button>"


def render_button_group(group: ModalButtonGroup) -> str:
    if not group.buttons:
        raise ValueError("button groups require at least one button")
    if not str(group.label).strip():
        raise ValueError("button groups require an accessible label")
    classes = _validated_classes(
        ("rwkv-button-group", *group.extra_classes),
        "button group class",
    )
    attributes = [
        f'class="{html.escape(" ".join(classes), quote=True)}"',
        'role="group"',
        f'aria-label="{html.escape(group.label, quote=True)}"',
    ]
    if group.group_id is not None:
        group_id = _require_html_id(group.group_id, "button group id")
        attributes.append(f'id="{html.escape(group_id, quote=True)}"')
    rendered = "".join(render_button(button) for button in group.buttons)
    return f"<div {' '.join(attributes)}>{rendered}</div>"


def render_footer(
    buttons: Sequence[ModalButton | ModalButtonGroup],
    *,
    leading_buttons: Sequence[ModalButton | ModalButtonGroup] = (),
    sticky: bool = False,
    label: str = "Dialog actions",
    footer_id: str | None = None,
    hidden: bool = False,
) -> str:
    classes = "rwkv-dialog-footer"
    if sticky:
        classes += " rwkv-dialog-footer--sticky"
    attributes = [f'class="{classes}"', f'aria-label="{html.escape(label, quote=True)}"']
    if footer_id is not None:
        attributes.append(f'id="{html.escape(footer_id, quote=True)}"')
    if hidden:
        attributes.extend(("hidden", 'aria-hidden="true"'))
    leading = "".join(_render_footer_action(action) for action in leading_buttons)
    rendered = "".join(_render_footer_action(action) for action in buttons)
    return (
        f"<footer {' '.join(attributes)}>"
        f'<div class="rwkv-dialog-footer__leading">{leading}</div>'
        f'<div class="rwkv-dialog-footer__actions">{rendered}</div></footer>'
    )


def _render_footer_action(action: ModalButton | ModalButtonGroup) -> str:
    if isinstance(action, ModalButton):
        return render_button(action)
    if isinstance(action, ModalButtonGroup):
        return render_button_group(action)
    raise TypeError(f"unsupported footer action: {type(action).__name__}")


def render_close_footer(
    *,
    label: str = "Close",
    outcome: Literal["accept", "reject"] = "accept",
    disabled: bool = False,
    sticky: bool = True,
) -> str:
    """Render the standard action footer for a close-only report window."""

    return render_footer(
        (
            ModalButton(
                label,
                "dialog-close",
                variant="primary",
                payload={"outcome": outcome},
                disabled=disabled,
                button_id="rwkv-dialog-close",
            ),
        ),
        sticky=sticky,
    )


def render_disclosure(disclosure: ModalDisclosure, panel_html: str) -> str:
    """Render one canonical disclosure button and its controlled panel.

    ``panel_html`` is trusted add-on-owned markup. Labels, tooltip text, IDs,
    actions, and CSS classes are validated or escaped at this boundary.
    """

    button_id = _require_html_id(disclosure.button_id, "disclosure button ID")
    panel_id = _require_html_id(disclosure.panel_id, "disclosure panel ID")
    if button_id == panel_id:
        raise ValueError("disclosure button and panel IDs must differ")
    _require_member(disclosure.panel_role, _DISCLOSURE_PANEL_ROLES, "disclosure panel role")
    if disclosure.panel_tabindex not in {None, -1, 0}:
        raise ValueError("disclosure panel tabindex must be -1, 0, or None")

    button_classes = _validated_classes(
        ("rwkv-inline-disclosure", *disclosure.button_classes),
        "disclosure button class",
    )
    panel_classes = _validated_classes(
        ("rwkv-disclosure-panel", *disclosure.panel_classes),
        "disclosure panel class",
    )
    expanded_label = disclosure.expanded_label or disclosure.collapsed_label
    visible_label = expanded_label if disclosure.expanded else disclosure.collapsed_label
    expanded_text = str(bool(disclosure.expanded)).lower()
    button_attributes = [
        f'class="{html.escape(" ".join(button_classes), quote=True)}"',
        f'id="{html.escape(button_id, quote=True)}"',
        'type="button"',
        f'aria-expanded="{expanded_text}"',
        f'aria-controls="{html.escape(panel_id, quote=True)}"',
        "data-rwkv-disclosure",
    ]
    if disclosure.action is not None:
        button_attributes.append(
            'data-rwkv-disclosure-action="'
            + html.escape(require_web_dialog_action(disclosure.action), quote=True)
            + '"'
        )
    if disclosure.expanded_root_class is not None:
        root_class = _require_css_class(
            disclosure.expanded_root_class,
            "disclosure expanded root class",
        )
        button_attributes.append(
            f'data-rwkv-expanded-root-class="{html.escape(root_class, quote=True)}"'
        )
    if disclosure.tooltip:
        button_attributes.append(_tooltip_data_attribute(disclosure.tooltip))
    if disclosure.disabled:
        button_attributes.extend(("disabled", 'aria-disabled="true"'))

    panel_attributes = [
        f'class="{html.escape(" ".join(panel_classes), quote=True)}"',
        f'id="{html.escape(panel_id, quote=True)}"',
        f'role="{disclosure.panel_role}"',
    ]
    if disclosure.panel_label is not None:
        panel_attributes.append(
            f'aria-label="{html.escape(disclosure.panel_label, quote=True)}"'
        )
    else:
        panel_attributes.append(f'aria-labelledby="{html.escape(button_id, quote=True)}"')
    if disclosure.panel_tabindex is not None:
        panel_attributes.append(f'tabindex="{disclosure.panel_tabindex}"')
    if not disclosure.expanded:
        panel_attributes.append("hidden")

    label_markup = (
        '<span class="rwkv-inline-disclosure__label" data-rwkv-disclosure-label '
        f'data-rwkv-collapsed-label="{html.escape(disclosure.collapsed_label, quote=True)}" '
        f'data-rwkv-expanded-label="{html.escape(expanded_label, quote=True)}">'
        f"{html.escape(visible_label)}</span>"
    )
    return (
        f'<button {" ".join(button_attributes)}>{label_markup}'
        '<span class="rwkv-inline-disclosure__indicator" aria-hidden="true">▸</span>'
        "</button>"
        f'<div {" ".join(panel_attributes)}>{panel_html}</div>'
    )


def render_field(field: ModalField) -> str:
    _require_member(field.kind, _FIELD_KINDS, "field kind")
    field_id = _field_id(field.name)
    description_id = f"{field_id}-description"
    description = (
        f'<div class="rwkv-field__description" id="{description_id}">'
        f"{html.escape(field.description)}</div>"
        if field.description
        else (
            f'<span class="rwkv-sr-only" id="{description_id}">'
            f"{html.escape(field.tooltip or '')}</span>"
            if field.tooltip
            else ""
        )
    )
    is_choice = field.kind in {"checkbox", "switch"}
    common = _field_common_attributes(field, description_id=description_id)
    tooltip = f" {_tooltip_data_attribute(field.tooltip)}" if field.tooltip else ""
    if is_choice:
        control_class = "rwkv-switch" if field.kind == "switch" else "rwkv-checkbox"
        checked = " checked" if field.checked else ""
        label_tooltip = tooltip
        control = (
            f'<label class="{control_class}" for="{field_id}">'
            f'<input id="{field_id}" name="{html.escape(field.name, quote=True)}" '
            f'type="checkbox"{checked}{common}>'
            + (
                '<span class="rwkv-switch__track" aria-hidden="true"></span>'
                if field.kind == "switch"
                else ""
            )
            + f'<span class="{control_class}__label"{label_tooltip}>'
            f"{html.escape(field.label)}</span></label>"
        )
        return f'<div class="rwkv-field rwkv-field--choice">{control}{description}</div>'

    label = (
        f'<label class="rwkv-field__label" for="{field_id}"{tooltip}>'
        f"{html.escape(field.label)}</label>"
    )
    if field.kind == "select":
        selected_value = str(field.value)
        options = []
        for option in field.options:
            selected = " selected" if option.value == selected_value else ""
            disabled = " disabled" if option.disabled else ""
            options.append(
                f'<option value="{html.escape(option.value, quote=True)}"{selected}{disabled}>'
                f"{html.escape(option.label)}</option>"
            )
        control = (
            f'<select class="rwkv-field__control" id="{field_id}" '
            f'name="{html.escape(field.name, quote=True)}"{common}>{"".join(options)}</select>'
        )
    elif field.kind == "textarea":
        rows = max(1, int(field.rows))
        control = (
            f'<textarea class="rwkv-field__control" id="{field_id}" '
            f'name="{html.escape(field.name, quote=True)}" rows="{rows}"{common}>'
            f"{html.escape(str(field.value))}</textarea>"
        )
    elif field.kind == "number":
        control = render_number_input(field)
    else:
        extra = (
            _numeric_attributes(field)
            if field.kind in {"date", "time", "datetime-local"}
            else ""
        )
        control = (
            f'<input class="rwkv-field__control" id="{field_id}" '
            f'name="{html.escape(field.name, quote=True)}" type="{field.kind}" '
            f'value="{html.escape(str(field.value), quote=True)}"{extra}{common}>'
        )
    return f'<div class="rwkv-field">{label}{control}{description}</div>'


def render_number_input(
    field: ModalField,
    *,
    control_id: str | None = None,
    data_attributes: Mapping[str, str] | None = None,
) -> str:
    """Render the canonical numeric input used across RWKV WebViews."""

    if field.kind != "number":
        raise ValueError("render_number_input requires a number field")
    field_id = _field_id(field.name) if control_id is None else _require_html_id(
        control_id, "field ID"
    )
    description_id = f"{field_id}-description"
    common = _field_common_attributes(field, description_id=description_id)
    data_markup = ""
    for name, value in (data_attributes or {}).items():
        if not re.fullmatch(r"data-[a-z][a-z0-9-]*", name):
            raise ValueError(f"invalid numeric input data attribute: {name!r}")
        data_markup += f' {name}="{html.escape(str(value), quote=True)}"'
    return (
        f'<input class="rwkv-field__control" id="{html.escape(field_id, quote=True)}" '
        f'name="{html.escape(field.name, quote=True)}" type="number" '
        f'value="{html.escape(str(field.value), quote=True)}"'
        f"{_numeric_attributes(field)}{common}{data_markup}>"
    )


def render_status_state(
    *,
    title: str,
    message: str,
    state: Literal["empty", "loading", "disabled", "error", "success"],
) -> str:
    _require_member(state, _STATUS_STATES, "status state")
    role = "alert" if state == "error" else "status"
    busy = ' aria-busy="true"' if state == "loading" else ""
    disabled = ' aria-disabled="true"' if state == "disabled" else ""
    spinner = (
        '<span class="rwkv-state__spinner" aria-hidden="true"></span>' if state == "loading" else ""
    )
    return (
        f'<section class="rwkv-state rwkv-state--{state}" role="{role}"{busy}{disabled}>'
        f'{spinner}<div><h3 class="rwkv-state__title">{html.escape(title)}</h3>'
        f'<p class="rwkv-state__message">{html.escape(message)}</p></div></section>'
    )


def render_badge(
    label: str,
    *,
    tone: Literal["neutral", "accent", "success", "warning", "error"] = "neutral",
) -> str:
    _require_member(tone, _BADGE_TONES, "badge tone")
    return f'<span class="rwkv-badge rwkv-badge--{tone}">{html.escape(label)}</span>'


def render_footnotes(notes: Sequence[str]) -> str:
    if not notes:
        return ""
    items = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    return f'<ul class="rwkv-footer-notes">{items}</ul>'


def render_progress_overlay(
    state: ProgressState,
    *,
    overlay_id: str = "rwkv-progress-overlay",
    cancel_action: str = "cancel-operation",
) -> str:
    hidden = "" if state.visible else " hidden"
    progress_body = _render_progress_body(state, cancel_action=cancel_action)
    return f"""
<div class="rwkv-modal-overlay" id="{html.escape(overlay_id, quote=True)}"{hidden}
     aria-hidden="{str(not state.visible).lower()}"
     role="dialog" aria-modal="true" aria-labelledby="{html.escape(overlay_id, quote=True)}-title"
     data-rwkv-overlay data-rwkv-overlay-kind="progress">
  <section class="rwkv-overlay-panel" tabindex="-1">
    <h2 class="rwkv-overlay-title" id="{html.escape(overlay_id, quote=True)}-title"
        data-rwkv-progress-title>
      {html.escape(state.title)}
    </h2>
    {progress_body}
  </section>
</div>
""".strip()


def render_progress_region(
    state: ProgressState,
    *,
    region_id: str = "rwkv-progress-overlay",
    cancel_action: str = "progress-cancel",
) -> str:
    """Render progress as the primary content of a standalone workflow page."""

    hidden = "" if state.visible else " hidden"
    progress_body = _render_progress_body(state, cancel_action=cancel_action)
    return f"""
<section class="rwkv-progress-surface" id="{html.escape(region_id, quote=True)}"{hidden}
         aria-hidden="{str(not state.visible).lower()}" aria-busy="true"
         role="status" aria-labelledby="rwkv-dialog-title"
         data-rwkv-inline-progress>
  {progress_body}
</section>
""".strip()


def _render_progress_body(state: ProgressState, *, cancel_action: str) -> str:
    total = None if state.total is None else max(0.0, float(state.total))
    current = None if state.current is None else max(0.0, float(state.current))
    determinate = total is not None and total > 0 and current is not None
    if determinate:
        bounded_current = min(current, total)
        percent = 100.0 * bounded_current / total
        progress_attributes = (
            f'aria-valuemin="0" aria-valuemax="{_format_number(total)}" '
            f'aria-valuenow="{_format_number(bounded_current)}"'
        )
        bar_style = f' style="width: {percent:.3f}%"'
        value_text = f"{_format_number(bounded_current)} of {_format_number(total)}"
    else:
        progress_attributes = ""
        bar_style = ""
        value_text = "Working"
    cancel_markup = ""
    if state.cancellable:
        label = "Cancelling…" if state.cancel_pending else "Cancel"
        cancel_markup = render_button(
            ModalButton(
                label,
                cancel_action,
                variant="secondary",
                disabled=state.cancel_pending,
                overlay_cancel=True,
                initial_focus=state.visible,
            )
        )
    return f"""
<p class="rwkv-progress-label" data-rwkv-progress-label aria-live="polite">
  {html.escape(state.label)}
</p>
<div class="rwkv-progress" role="progressbar" {progress_attributes}
     aria-label="{html.escape(state.label, quote=True)}">
  <div class="rwkv-progress__bar" data-rwkv-progress-bar{bar_style}></div>
</div>
<div class="rwkv-progress-meta">
  <span class="rwkv-progress-value" data-rwkv-progress-value>{html.escape(value_text)}</span>
  <span class="rwkv-progress-eta" data-rwkv-progress-eta>ETA unknown</span>
</div>
<div class="rwkv-overlay-actions">{cancel_markup}</div>
""".strip()


def render_message_overlay_template(
    *,
    overlay_id: str = "rwkv-message-overlay",
    response_action: str = "message-response",
) -> str:
    """Render the hidden runtime target shared by alerts and confirmations."""

    action = require_web_dialog_action(response_action)
    buttons = "".join(
        (
            f'<button class="rwkv-button rwkv-button--secondary" type="button" '
            f'data-rwkv-action="{html.escape(action, quote=True)}" '
            f'data-rwkv-message-button="{index}" hidden aria-hidden="true"></button>'
        )
        for index in range(3)
    )
    escaped_id = html.escape(overlay_id, quote=True)
    checkbox = (
        '<label class="rwkv-checkbox rwkv-message-checkbox" '
        'data-rwkv-message-checkbox-container hidden aria-hidden="true">'
        '<input type="checkbox" data-rwkv-message-checkbox '
        'data-rwkv-change-action="message-checkbox-change">'
        '<span class="rwkv-checkbox__label" data-rwkv-message-checkbox-label></span>'
        "</label>"
    )
    return f"""
<div class="rwkv-modal-overlay rwkv-message-overlay" id="{escaped_id}" hidden
     aria-hidden="true" role="alertdialog" aria-modal="true"
     aria-labelledby="{escaped_id}-title" aria-describedby="{escaped_id}-message"
     data-rwkv-overlay data-rwkv-overlay-kind="info">
  <section class="rwkv-overlay-panel rwkv-overlay-panel--info" tabindex="-1"
           data-rwkv-message-panel>
    <h2 class="rwkv-overlay-title" id="{escaped_id}-title" data-rwkv-message-title></h2>
    <div id="{escaped_id}-message" class="rwkv-message-text"
         data-rwkv-message-text></div>
    <pre class="rwkv-alert-details" tabindex="0" data-rwkv-message-details
         hidden aria-hidden="true"></pre>
    {checkbox}
    <div class="rwkv-overlay-actions" data-rwkv-message-actions>{buttons}</div>
  </section>
</div>
""".strip()


def render_prompt_overlay_template(
    *,
    overlay_id: str = "rwkv-prompt-overlay",
) -> str:
    """Render the shared editable prompt target configured by ``RWKVPrompt``.

    The runtime switches between the single-line and multiline controls, keeps
    validation failures inside the overlay, and resolves the caller's promise
    only after the user confirms a valid value or cancels.
    """

    escaped_id = html.escape(overlay_id, quote=True)
    return f"""
<div class="rwkv-modal-overlay rwkv-prompt-overlay" id="{escaped_id}" hidden
     aria-hidden="true" role="dialog" aria-modal="true"
     aria-labelledby="{escaped_id}-title" aria-describedby="{escaped_id}-message"
     data-rwkv-overlay data-rwkv-overlay-kind="info">
  <section class="rwkv-overlay-panel rwkv-overlay-panel--info" tabindex="-1">
    <form class="rwkv-prompt-form" data-rwkv-prompt-form novalidate>
      <h2 class="rwkv-overlay-title" id="{escaped_id}-title"
          data-rwkv-prompt-title>Enter a value</h2>
      <p class="rwkv-prompt-message" id="{escaped_id}-message"
         data-rwkv-prompt-message></p>
      <div class="rwkv-field rwkv-prompt-field">
        <span class="rwkv-field__label" id="{escaped_id}-label"
              data-rwkv-prompt-label>Value</span>
        <input class="rwkv-field__control" id="{escaped_id}-input" type="text"
               autocomplete="off" aria-labelledby="{escaped_id}-label"
               data-rwkv-prompt-input>
        <textarea class="rwkv-field__control" id="{escaped_id}-textarea" rows="10"
                  aria-labelledby="{escaped_id}-label" data-rwkv-prompt-textarea
                  hidden aria-hidden="true"></textarea>
      </div>
      <div class="rwkv-notice rwkv-notice--error rwkv-prompt-error" role="alert"
           data-rwkv-prompt-error hidden aria-hidden="true"></div>
      <div class="rwkv-overlay-actions">
        <button class="rwkv-button rwkv-button--secondary" type="button"
                data-rwkv-prompt-cancel data-rwkv-overlay-cancel>Cancel</button>
        <button class="rwkv-button rwkv-button--primary" type="submit"
                data-rwkv-prompt-confirm>Continue</button>
      </div>
    </form>
  </section>
</div>
""".strip()


def _field_common_attributes(
    field: ModalField,
    *,
    description_id: str,
) -> str:
    attributes = []
    if field.description or field.tooltip:
        attributes.append(f'aria-describedby="{description_id}"')
    if field.disabled:
        attributes.extend(("disabled", 'aria-disabled="true"'))
    if field.required:
        attributes.extend(("required", 'aria-required="true"'))
    if field.placeholder is not None:
        attributes.append(f'placeholder="{html.escape(field.placeholder, quote=True)}"')
    if field.change_action is not None:
        attributes.append(
            'data-rwkv-change-action="'
            + html.escape(require_web_dialog_action(field.change_action), quote=True)
            + '"'
        )
        if field.change_serialize_form:
            attributes.append("data-rwkv-change-serialize-form")
    elif field.change_serialize_form:
        raise ValueError("change_serialize_form requires change_action")
    if field.enter_action is not None:
        attributes.append(
            'data-rwkv-enter-action="'
            + html.escape(require_web_dialog_action(field.enter_action), quote=True)
            + '"'
        )
    if field.enable_fields:
        if field.kind not in {"checkbox", "switch"}:
            raise ValueError("enable_fields requires a checkbox or switch")
        target_ids = ",".join(_field_id(name) for name in field.enable_fields)
        attributes.append(f'data-rwkv-enable-targets="{html.escape(target_ids, quote=True)}"')
    if field.initial_focus:
        attributes.extend(("data-rwkv-initial-focus", "autofocus"))
    return (" " + " ".join(attributes)) if attributes else ""


def _tooltip_data_attribute(tooltip: str) -> str:
    return f'data-rwkv-tooltip="{html.escape(tooltip, quote=True)}"'


def _numeric_attributes(field: ModalField) -> str:
    attributes = []
    if field.minimum is not None:
        attributes.append(f'min="{html.escape(str(field.minimum), quote=True)}"')
    if field.maximum is not None:
        attributes.append(f'max="{html.escape(str(field.maximum), quote=True)}"')
    if field.step is not None:
        attributes.append(f'step="{html.escape(str(field.step), quote=True)}"')
    return (" " + " ".join(attributes)) if attributes else ""


def _field_id(name: str) -> str:
    return f"rwkv-field-{_identifier(name)}"


def _identifier(value: str) -> str:
    return _IDENTIFIER_PATTERN.sub("-", str(value)).strip("-") or "item"


def _require_html_id(value: str, label: str) -> str:
    candidate = str(value)
    if not _HTML_ID_PATTERN.fullmatch(candidate):
        raise ValueError(f"invalid {label}: {value!r}")
    return candidate


def _require_css_class(value: str, label: str) -> str:
    candidate = str(value)
    if not _CSS_CLASS_PATTERN.fullmatch(candidate):
        raise ValueError(f"invalid {label}: {value!r}")
    return candidate


def _validated_classes(values: Sequence[str], label: str) -> tuple[str, ...]:
    classes: list[str] = []
    for value in values:
        candidate = _require_css_class(value, label)
        if candidate not in classes:
            classes.append(candidate)
    return tuple(classes)


def _format_number(value: float) -> str:
    return f"{value:g}"


def _require_member(value: str, allowed: frozenset[str], label: str) -> None:
    if value not in allowed:
        raise ValueError(f"unsupported {label}: {value}")


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
