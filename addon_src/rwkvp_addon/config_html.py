from __future__ import annotations

# The inline CSS/JavaScript is intentionally kept readable as browser source.
# ruff: noqa: E501
import html
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any

from .addon_config import (
    CALCULATE_FORGETTING_CURVES_CONFIG_KEY,
    CARD_INFO_FORGETTING_CURVE_GRAPH_CONFIG_KEY,
    CURVE_RESCHEDULING_CONFIG_KEY,
    webview_popup_control_mode,
)
from .config_options import CONFIG_OPTIONS, CONFIG_SECTIONS, ConfigOption
from .glossary import GlossaryRenderer
from .modal_html import (
    ModalButton,
    ModalField,
    ModalTab,
    ProgressState,
    popup_control_mode_script,
    render_button,
    render_footer,
    render_message_overlay_template,
    render_number_input,
    render_progress_overlay,
    render_tab_list,
    shared_modal_script_tag,
)
from .modal_style import shared_style_tag
from .review_type_normalization import parse_cutoff_datetime
from .setup_faq_html import render_setup_faq_help_button
from .setup_wizard_html import (
    SETUP_WIZARD_CSS,
    SETUP_WIZARD_SCRIPT,
    render_setup_launcher,
    render_setup_overlay,
)

BRIDGE_PREFIX = "rwkvConfig:"


_PAGE_INTROS = {
    "General": (
        "Choose which RWKV features to enable and how much work they perform. Speed "
        "tests use your collection and saved checkpoint, so the results reflect this computer."
    ),
    "Advanced": (
        "These settings control how RWKV builds and saves its model state, plus less "
        "common Live Session behavior. Defaults are appropriate for most collections."
    ),
    "Experimental": (
        "These features have not been tested as carefully as the rest of RWKV4Anki. "
        "Use them at your own risk, especially options that modify scheduling data."
    ),
}

_GROUP_INTROS = {
    ("General", "Performance"): (
        "These settings determine how quickly RWKV processes history and predicts cards. "
        "They do not change the underlying model."
    ),
    ("General", "Features"): (
        "Choose which RWKV workflows and Card Info fields are available in Anki."
    ),
    ("Advanced", "RWKV State Settings"): (
        "RWKV state is the model's memory of your review history. These settings control "
        "what that state includes and which model reads it."
    ),
    ("Advanced", "Checkpoint Settings"): (
        "A checkpoint saves RWKV state to disk. When RWKV loads it, reviews added since "
        "the last save are processed automatically."
    ),
    ("Advanced", "Live Session Settings"): (
        "These settings control fallback behavior and how RWKV interprets reviews "
        "created through filtered scheduling."
    ),
    ("Advanced", "Appearance"): (
        "These settings control how RWKV windows are displayed. Changes apply when a "
        "window is next opened."
    ),
    ("Experimental", "Functionality"): (
        "Optional workflows for investigating RWKV or changing how cards are scheduled."
    ),
    ("Experimental", "Performance"): (
        "Leave these overrides at 0 unless you are diagnosing performance. Batch size is "
        "the number of cards processed together; larger values can be faster, slower, or "
        "use more memory."
    ),
}

_SUBSECTION_INTROS = {
    ("General", "Performance", "State Building and Predictions"): (
        "State building turns your review history into saved RWKV state. Predictions use "
        "that state to estimate how likely you are to remember each card."
    ),
    ("General", "Performance", "Live Session Performance"): (
        "After each answer, a Live Session updates RWKV and checks cards before choosing "
        "the next one. The speed test shows whether your selected card count will cause "
        "a noticeable delay."
    ),
    ("General", "Features", "RWKV Immediate"): (
        "RWKV Immediate estimates how likely you are to recall a card right now. Its "
        "master switch hides or shows all Immediate workflows throughout Anki."
    ),
    ("General", "Features", "RWKV Forgetting Curve"): (
        "RWKV Forgetting Curve estimates how that recall probability changes as time "
        "passes after a review."
    ),
}

_SPEED_BUTTON_LABELS = {
    "state-building": "Compare Modes",
    "predictions": "Compare Modes",
    "curves": "Compare On vs. Off",
    "live-predictions": "Test This Size",
}

_COMPARISON_BUTTON_LABELS = {
    "models": "Compare Models",
    "deleted-reviews": "Compare On vs. Off",
}


def render_initial_setup_html(
    *,
    generation: int = 1,
    popup_control_mode: str = "native",
) -> str:
    """Render first-run Guided Setup without mounting the Settings interface."""

    bootstrap = _script_json(
        {
            "bridgePrefix": BRIDGE_PREFIX,
            "bridgeMode": "flat",
            "bridgeExpectsReply": False,
            "generation": int(generation),
            "rootSelector": "#rwkv-guided-setup-root",
            "backgroundSelectors": ("#rwkv-setup-overlay",),
        }
    )
    return f"""
{shared_style_tag()}
<style>
    html, body {{ height: 100%; overflow: hidden; }}
    body {{ background: var(--rwkv-canvas); }}
    .rwkv-guided-setup-host {{
        height: 100%; min-height: 0; overflow: hidden;
    }}
    .rwkv-guided-setup-host .setup-overlay {{
        background: var(--rwkv-canvas); backdrop-filter: none;
    }}
    .rwkv-guided-setup-host .setup-dialog {{
        height: calc(100vh - 44px);
    }}
    {SETUP_WIZARD_CSS}
</style>
<div class="rwkv-modal-shell rwkv-guided-setup-host" id="rwkv-guided-setup-root"
     role="document" aria-label="RWKV Guided Setup"
     data-rwkv-generation="{int(generation)}">
  {render_message_overlay_template()}
  {render_setup_overlay(initially_open=True, show_close_button=False)}
  <div class="rwkv-sr-only" id="rwkv-dialog-announcer"
       aria-live="polite" aria-atomic="true"></div>
</div>
{popup_control_mode_script(popup_control_mode)}
<script>window.RWKV_MODAL_BOOTSTRAP = {bootstrap};</script>
{shared_modal_script_tag()}
<script>
(() => {{
    function values() {{ return {{}}; }}
    function send(action, extra={{}}) {{
        window.RWKVModal.send(action, extra).catch(() => {{}});
    }}
    {SETUP_WIZARD_SCRIPT}
}})();
</script>
""".strip()


def visible_config_options(*, predict_gpu_available: bool) -> tuple[ConfigOption, ...]:
    return tuple(
        option
        for option in CONFIG_OPTIONS
        if not option.requires_gpu or bool(predict_gpu_available)
    )


def option_path_name(path: tuple[str, ...]) -> str:
    return "/".join(path)


def render_config_html(
    config: Mapping[str, Any],
    *,
    choices: Mapping[tuple[str, ...], Sequence[str]],
    predict_gpu_available: bool,
    process_gpu_available: bool,
    checkpoint_usable: bool,
    initial_setup: bool = False,
    apply_enabled: bool = False,
    generation: int = 1,
) -> str:
    popup_control_mode = webview_popup_control_mode(config)
    if initial_setup:
        return render_initial_setup_html(
            generation=generation,
            popup_control_mode=popup_control_mode,
        )

    options = visible_config_options(predict_gpu_available=predict_gpu_available)
    capability_note = _capability_note(
        predict_gpu_available=predict_gpu_available,
        process_gpu_available=process_gpu_available,
    )
    pages = "\n".join(
        _render_page(
            section,
            config=config,
            options=options,
            choices=choices,
            checkpoint_usable=checkpoint_usable,
            capability_note=capability_note if section == "General" else "",
        )
        for section in CONFIG_SECTIONS
    )
    tabs = render_tab_list(
        tuple(
            ModalTab(
                key=f"settings-{section.casefold()}",
                label=section,
                panel_id=f"settings-page-{section}",
                selected=section == "General",
            )
            for section in CONFIG_SECTIONS
        ),
        label="RWKV Settings sections",
        extra_classes=("rwkv-settings-tabs",),
    )
    bootstrap = _script_json(
        {
            "bridgePrefix": BRIDGE_PREFIX,
            "bridgeMode": "flat",
            "bridgeExpectsReply": False,
            "generation": int(generation),
            "rootSelector": "#rwkv-settings-root",
            "backgroundSelectors": (
                ".settings-shell > .rwkv-settings-tabs",
                ".settings-shell > .settings-pages",
                ".settings-shell > .rwkv-dialog-footer",
                "#rwkv-setup-overlay",
            ),
        }
    )
    curve_path = _script_json(option_path_name((CALCULATE_FORGETTING_CURVES_CONFIG_KEY,)))
    graph_path = _script_json(option_path_name((CARD_INFO_FORGETTING_CURVE_GRAPH_CONFIG_KEY,)))
    reschedule_path = _script_json(option_path_name((CURVE_RESCHEDULING_CONFIG_KEY,)))
    settings_footer = _render_settings_footer(apply_enabled=apply_enabled)
    apply_enabled_script = "true" if apply_enabled else "false"
    progress_overlay = render_progress_overlay(
        ProgressState(
            title="Working",
            label="Preparing operation",
            cancellable=True,
            visible=False,
        ),
        cancel_action="progress-cancel",
    )
    message_overlay = render_message_overlay_template()
    return f"""
{shared_style_tag()}
<style>
    .rwkv-settings-tabs {{ position: sticky; top: 0; z-index: 5; flex: 0 0 auto; gap: 4px; padding: 12px 18px 0; background: var(--rwkv-canvas); }}
    .rwkv-settings-tabs .rwkv-tab {{ padding: 9px 16px 10px; }}
    .page {{ display: block; max-width: 980px; margin: 0 auto; padding: 22px 24px 40px; }}
    .page[hidden] {{ display: none; }}
    .settings-page-heading {{ align-items: center; display: flex; gap: 14px; justify-content: space-between; }}
    .settings-page-heading .page-title {{ margin: 0; }}
    .settings-page-heading > button.setup-faq-help-button {{
        align-self: flex-start; border: 1px solid var(--rwkv-border); border-radius: 50%;
        box-shadow: none;
        flex: 0 0 32px; font-size: 15px; font-weight: 750; height: 32px; min-height: 32px;
        margin: 0; padding: 0; width: 32px;
    }}
    .settings-page-heading > button.setup-faq-help-button:hover:not(:disabled) {{
        border-color: var(--rwkv-accent-border); border-radius: 50%; box-shadow: none;
        color: var(--rwkv-accent);
    }}
    .capability-note {{ margin-top: 12px; border-left: 3px solid var(--rwkv-accent); padding: 7px 10px; background: rgba(75,123,236,.08); border-radius: 3px; }}
    .major-group {{ margin-top: 28px; }}
    .major-group > h2 {{ font-size: 18px; margin: 0 0 4px; }}
    .settings-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; margin-top: 13px; align-items: start; }}
    .settings-grid--stacked {{ grid-template-columns: minmax(0, 1fr); }}
    .settings-card h3 {{ font-size: 15px; margin: 0 0 3px; }}
    .subsection-intro {{ margin-bottom: 10px; font-size: 12px; }}
    .curve-disabled-note {{ display: none; margin: 10px 0 4px; border-radius: 6px; padding: 8px 10px; color: var(--fg-subtle, #626872); background: rgba(127,127,127,.12); }}
    .settings-card.curves-disabled .curve-disabled-note {{ display: block; }}
    .setting {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(170px, auto); gap: 10px 18px; align-items: center; padding: 13px 0; border-top: 1px solid var(--rwkv-border); }}
    .setting:first-of-type {{ border-top: 0; }}
    .setting.conditional-hidden {{ display: none; }}
    .setting.disabled {{ opacity: .52; }}
    .setting-title {{ font-weight: 650; }}
    .setting-description {{ color: var(--rwkv-subtle); font-size: 12px; margin-top: 3px; }}
    .restart-badge, .rebuild-badge {{ display: inline-block; color: #8a5a00; background: rgba(230,165,0,.16); border-radius: 99px; font-size: 10px; font-weight: 700; margin-left: 6px; padding: 2px 7px; vertical-align: 1px; }}
    .control {{ display: flex; justify-content: flex-end; align-items: center; gap: 7px; min-width: 170px; }}
    select, input[type=date], .datetime-time {{ width: 180px; max-width: 100%; color: inherit; background: var(--rwkv-control); border: 1px solid var(--rwkv-border); border-radius: 6px; padding: 7px 9px; }}
    .rwkv-settings-number-field {{ flex: 0 1 180px; width: 180px; }}
    .rwkv-settings-number-field > input[type=number] {{ appearance: textfield; }}
    .rwkv-settings-number-field > input[type=number]::-webkit-inner-spin-button,
    .rwkv-settings-number-field > input[type=number]::-webkit-outer-spin-button {{
        appearance: none;
        margin: 0;
    }}
    .setting-datetime {{ grid-template-columns: minmax(0, 1fr); }}
    .setting-datetime .control {{ justify-content: flex-start; }}
    .datetime-editor {{ display: flex; align-items: flex-start; flex-wrap: wrap; gap: 7px; width: 100%; }}
    .datetime-part {{ align-content: start; display: grid; gap: 3px; grid-template-rows: 12px 34px; }}
    .datetime-part-label {{ color: var(--fg-subtle, #626872); font-size: 10px; font-weight: 650; letter-spacing: .02em; line-height: 12px; }}
    .datetime-date, .datetime-time, .datetime-period,
    .datetime-part > .rwkv-popup-select, .datetime-part > .rwkv-popup-temporal {{ height: 34px; min-height: 34px; }}
    .datetime-date {{ width: 155px; }}
    .datetime-time {{ width: 96px; font-variant-numeric: tabular-nums; }}
    .datetime-period {{ width: 72px; }}
    .datetime-date[aria-invalid="true"], .datetime-time[aria-invalid="true"] {{ border-color: #c64747; box-shadow: 0 0 0 1px rgba(198,71,71,.18); }}
    .datetime-error {{ flex-basis: 100%; color: #b23b3b; font-size: 11px; }}
    .unit {{ color: var(--fg-subtle, #626872); font-size: 12px; }}
    @media (max-width: 620px) {{
        .settings-grid {{ grid-template-columns: 1fr; }}
        .setting {{ grid-template-columns: 1fr; }} .control {{ justify-content: flex-start; flex-wrap: wrap; }}
    }}
    .nightMode .settings-shell, .night_mode .settings-shell {{ color: #e7e9ec; }}
    .nightMode .restart-badge, .night_mode .restart-badge, .nightMode .rebuild-badge, .night_mode .rebuild-badge {{ color: #ffd77b; }}
    html, body {{ height: 100%; overflow: hidden; }}
    .rwkv-settings-host {{
        height: 100%; min-height: 0; overflow: hidden;
    }}
    .settings-shell {{
        display: flex; flex-direction: column; height: 100%; min-height: 0;
        overflow: hidden;
    }}
    .settings-pages {{
        flex: 1 1 0; min-height: 0; overflow-y: auto;
        overscroll-behavior: contain;
    }}
    #rwkv-settings-footer {{ flex: 0 0 auto; z-index: 6; }}
    {SETUP_WIZARD_CSS}
</style>
<div class="rwkv-modal-shell rwkv-settings-host" id="rwkv-settings-root"
     role="document" aria-label="RWKV Settings" data-rwkv-generation="{int(generation)}"
     data-rwkv-escape-action="cancel" data-rwkv-enter-action="ok"
     data-rwkv-enter-from-inputs="true">
  <div class="settings-shell">
    {tabs}
    <div class="settings-pages">{pages}</div>
    {settings_footer}
    {progress_overlay}
  </div>
  <!-- Keep runtime messages outside the inert Settings shell so Guided Setup can
       display and dismiss errors without exposing the underlying settings page. -->
  {message_overlay}
  {render_setup_overlay()}
  <div class="rwkv-sr-only" id="rwkv-dialog-announcer"
       aria-live="polite" aria-atomic="true"></div>
</div>
{popup_control_mode_script(popup_control_mode)}
<script>window.RWKV_MODAL_BOOTSTRAP = {bootstrap};</script>
{shared_modal_script_tag()}
<script>
(() => {{
    const curvePath = {curve_path};
    const graphPath = {graph_path};
    const reschedulePath = {reschedule_path};
    const settingsFooter = document.getElementById('rwkv-settings-footer');
    let settingsApplyEnabled = {apply_enabled_script};
    const controls = () => Array.from(document.querySelectorAll('[data-config-path]'));
    function valueFor(path) {{
        const el = document.querySelector(`[data-config-path="${{CSS.escape(path)}}"]`);
        if (!el) return null;
        if (el.type === 'checkbox') return el.checked;
        if (el.type === 'number') return el.value === '' ? null : Number(el.value);
        return el.value;
    }}
    function values() {{
        const result = {{}};
        controls().forEach((el) => {{ result[el.dataset.configPath] = valueFor(el.dataset.configPath); }});
        return result;
    }}
    function send(action, extra={{}}) {{
        window.RWKVModal.send(action, extra).catch(() => {{}});
    }}
    const actionsWithSettingsValues = new Set([
        'change', 'speed-test', 'comparison', 'setup',
        'restore-defaults', 'apply', 'ok', 'cancel',
    ]);
    window.RWKV_MODAL_PAYLOAD_PROVIDER = (action, payload={{}}) => {{
        if (!actionsWithSettingsValues.has(action) || Object.hasOwn(payload, 'values')) {{
            return payload;
        }}
        return Object.assign({{}}, payload, {{values: values()}});
    }};
    function parseTypedTime(value) {{
        const match = /^\\s*(\\d{{1,2}})(?::([0-5]?\\d))?(?::([0-5]?\\d))?\\s*$/.exec(value);
        if (!match) return null;
        const hour = Number(match[1]);
        if (hour < 1 || hour > 12) return null;
        return {{hour: hour, minute: Number(match[2] || 0), second: Number(match[3] || 0)}};
    }}
    function twoDigits(value) {{
        return String(value).padStart(2, '0');
    }}
    function updateDateTimeEditor(editor, {{commit=false, normalize=false}}={{}}) {{
        const stored = editor.querySelector('[data-datetime-value]');
        const date = editor.querySelector('[data-datetime-date]');
        const time = editor.querySelector('[data-datetime-time]');
        const period = editor.querySelector('[data-datetime-period]');
        const error = editor.querySelector('[data-datetime-error]');
        const parsed = parseTypedTime(time.value);
        const dateIsValid = Boolean(date.value) && date.validity.valid;
        const valid = dateIsValid && parsed !== null;
        date.setAttribute('aria-invalid', String(!dateIsValid));
        time.setAttribute('aria-invalid', String(parsed === null));
        error.hidden = valid;
        if (!commit) return;
        if (valid) {{
            if (normalize) {{
                time.value = `${{parsed.hour}}:${{twoDigits(parsed.minute)}}:${{twoDigits(parsed.second)}}`;
            }}
            const hour = (parsed.hour % 12) + (period.value === 'PM' ? 12 : 0);
            stored.value = `${{date.value}}T${{twoDigits(hour)}}:${{twoDigits(parsed.minute)}}:${{twoDigits(parsed.second)}}`;
        }} else {{
            // Send an invalid value once the edit is committed so Python's
            // existing validation prevents an accidental save of the old date.
            stored.value = `${{date.value}}T${{time.value.trim()}} ${{period.value}}`;
        }}
        stored.dispatchEvent(new Event('input', {{bubbles: true}}));
    }}
    function initializeDateTimeEditors() {{
        document.querySelectorAll('[data-datetime-editor]').forEach((editor) => {{
            const date = editor.querySelector('[data-datetime-date]');
            const time = editor.querySelector('[data-datetime-time]');
            const period = editor.querySelector('[data-datetime-period]');
            date.addEventListener('input', () => updateDateTimeEditor(editor));
            time.addEventListener('input', () => updateDateTimeEditor(editor));
            date.addEventListener('change', () => updateDateTimeEditor(editor, {{commit: true}}));
            time.addEventListener('change', () => updateDateTimeEditor(editor, {{commit: true, normalize: true}}));
            period.addEventListener('change', () => updateDateTimeEditor(editor, {{commit: true}}));
            time.addEventListener('keydown', (event) => {{
                if (event.key === 'Enter') {{
                    event.preventDefault();
                    time.blur();
                }}
            }});
            updateDateTimeEditor(editor);
        }});
    }}
    function refreshDependencies() {{
        const curves = Boolean(valueFor(curvePath));
        const graph = Boolean(valueFor(graphPath));
        const rescheduling = Boolean(valueFor(reschedulePath));
        function setDependencyDisabled(row, dependencyDisabled) {{
            row.querySelectorAll('input, select, button').forEach((el) => {{
                const unavailable = Boolean(el.closest('.rwkv-disabled-control-help'));
                el.disabled = Boolean(dependencyDisabled) || unavailable;
            }});
        }}
        document.querySelectorAll('.requires-curves').forEach((row) => {{
            row.classList.toggle('disabled', !curves);
            setDependencyDisabled(row, !curves);
        }});
        document.querySelectorAll('.curve-feature-card').forEach((card) => {{
            card.classList.toggle('curves-disabled', !curves);
        }});
        document.querySelectorAll('[data-parent-path]').forEach((row) => {{
            const needsCurves = row.classList.contains('requires-curves');
            const parent = document.querySelector(
                `[data-config-path="${{CSS.escape(row.dataset.parentPath)}}"]`,
            );
            const enabled = Boolean(valueFor(row.dataset.parentPath))
                && !Boolean(parent && parent.disabled)
                && (!needsCurves || curves);
            row.classList.toggle('disabled', !enabled);
            setDependencyDisabled(row, !enabled);
        }});
        document.querySelectorAll('[data-visible-path]').forEach((row) => {{
            const visible = curves && Boolean(valueFor(row.dataset.visiblePath));
            row.classList.toggle('conditional-hidden', !visible);
        }});
    }}
    controls().forEach((control) => control.addEventListener('input', () => {{
        refreshDependencies();
        send('change', {{values: values()}});
    }}));
    initializeDateTimeEditors();
    function setSettingsButtonDisabled(button, disabled) {{
        if (!button) return;
        button.disabled = Boolean(disabled);
        button.setAttribute('aria-disabled', String(Boolean(disabled)));
    }}
    function setSettingsApplyEnabled(enabled) {{
        settingsApplyEnabled = Boolean(enabled);
        const button = document.getElementById('rwkv-settings-apply');
        if (button && !settingsFooter.hidden) {{
            setSettingsButtonDisabled(button, !settingsApplyEnabled);
        }}
    }}
    function setSettingsFooterActive(active) {{
        if (!settingsFooter) return;
        const enabled = Boolean(active);
        settingsFooter.hidden = !enabled;
        settingsFooter.setAttribute('aria-hidden', String(!enabled));
        settingsFooter.querySelectorAll('button').forEach((button) => {{
            setSettingsButtonDisabled(
                button,
                !enabled || (
                    button.id === 'rwkv-settings-apply' && !settingsApplyEnabled
                ),
            );
        }});
    }}
    window.rwkvSetSettingsApplyEnabled = setSettingsApplyEnabled;
    window.rwkvSetSettingsFooterActive = setSettingsFooterActive;
    window.rwkvSetSpeedTestEnabled = (name, enabled) => {{
        const button = document.getElementById(`speed-${{name}}`);
        if (button) button.disabled = !enabled;
    }};
    window.rwkvSetComparisonEnabled = (name, enabled) => {{
        const button = document.getElementById(`comparison-${{name}}`);
        if (button) button.disabled = !enabled;
    }};
    {SETUP_WIZARD_SCRIPT}
    refreshDependencies();
}})();
</script>
""".strip()


def config_from_web_values(
    base: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    options: Sequence[ConfigOption],
    choices: Mapping[tuple[str, ...], Sequence[str]],
) -> dict[str, Any]:
    updated = deepcopy(dict(base))
    for option in options:
        name = option_path_name(option.key_path)
        if name not in values or values[name] is None:
            continue
        value = _coerce_value(option, values[name], choices.get(option.key_path, ()))
        if option.inverted:
            value = not bool(value)
        _set_path(updated, option.key_path, value)
    return updated


def merge_config_option_values(
    base: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    options: Sequence[ConfigOption],
) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for option in options:
        value = _get_path(source, option.key_path)
        if value is not None:
            _set_path(merged, option.key_path, deepcopy(value))
    return merged


def sanitize_choice_values(
    config: Mapping[str, Any],
    *,
    options: Sequence[ConfigOption],
    choices: Mapping[tuple[str, ...], Sequence[str]],
) -> dict[str, Any]:
    sanitized = deepcopy(dict(config))
    for option in options:
        allowed = tuple(str(item) for item in choices.get(option.key_path, ()))
        if option.value_type != "choice" or not allowed:
            continue
        current = str(_get_path(sanitized, option.key_path) or "")
        if current not in allowed:
            _set_path(sanitized, option.key_path, allowed[0])
    return sanitized


def _render_settings_footer(*, apply_enabled: bool) -> str:
    return render_footer(
        (
            ModalButton(
                "Cancel",
                "cancel",
                variant="quiet",
                button_id="rwkv-settings-cancel",
            ),
            ModalButton(
                "Apply",
                "apply",
                variant="secondary",
                disabled=not apply_enabled,
                button_id="rwkv-settings-apply",
            ),
            ModalButton(
                "OK",
                "ok",
                variant="primary",
                button_id="rwkv-settings-ok",
            ),
        ),
        leading_buttons=(
            ModalButton(
                "Restore Defaults",
                "restore-defaults",
                variant="quiet",
                button_id="rwkv-settings-restore-defaults",
            ),
        ),
        label="RWKV Settings actions",
        footer_id="rwkv-settings-footer",
    )


def _render_page(
    section: str,
    *,
    config: Mapping[str, Any],
    options: Sequence[ConfigOption],
    choices: Mapping[tuple[str, ...], Sequence[str]],
    checkpoint_usable: bool,
    capability_note: str,
) -> str:
    glossary = GlossaryRenderer(section)
    page_intro = glossary.render(_PAGE_INTROS[section])
    capability_html = glossary.render(capability_note) if capability_note else ""
    page_options = tuple(option for option in options if option.section == section)
    groups = tuple(dict.fromkeys(option.group for option in page_options))
    rendered_groups = "\n".join(
        _render_group(
            section,
            group,
            config=config,
            options=page_options,
            choices=choices,
            checkpoint_usable=checkpoint_usable,
            glossary=glossary,
        )
        for group in groups
    )
    hidden = "" if section == "General" else " hidden"
    help_button = render_setup_faq_help_button() if section == "General" else ""
    return f"""
<main class="page rwkv-tab-panel" role="tabpanel"
      id="settings-page-{html.escape(section, quote=True)}"
      aria-labelledby="rwkv-tab-settings-{html.escape(section.casefold(), quote=True)}"{hidden}>
    <div class="settings-page-heading">
      <h1 class="page-title">{html.escape(section)}</h1>
      {help_button}
    </div>
    <p class="page-intro">{page_intro}</p>
    {f'<div class="capability-note">{capability_html}</div>' if capability_html else ""}
    {render_setup_launcher() if section == "General" else ""}
    {rendered_groups}
</main>
""".strip()


def _render_group(
    section: str,
    group: str,
    *,
    config: Mapping[str, Any],
    options: Sequence[ConfigOption],
    choices: Mapping[tuple[str, ...], Sequence[str]],
    checkpoint_usable: bool,
    glossary: GlossaryRenderer,
) -> str:
    group_options = tuple(option for option in options if option.group == group)
    subsections = tuple(dict.fromkeys(option.subsection for option in group_options))
    heading = glossary.render(group)
    intro = _GROUP_INTROS.get((section, group), "")
    intro_html = glossary.render(intro)
    cards = "\n".join(
        _render_card(
            section,
            group,
            subsection,
            config=config,
            options=tuple(option for option in group_options if option.subsection == subsection),
            choices=choices,
            checkpoint_usable=checkpoint_usable,
            glossary=glossary,
        )
        for subsection in subsections
    )
    grid_class = "settings-grid settings-grid--stacked" if section == "General" else "settings-grid"
    return f"""
<section class="major-group">
    <h2>{heading}</h2>
    <p class="group-intro">{intro_html}</p>
    <div class="{grid_class}">{cards}</div>
</section>
""".strip()


def _render_card(
    section: str,
    group: str,
    subsection: str,
    *,
    config: Mapping[str, Any],
    options: Sequence[ConfigOption],
    choices: Mapping[tuple[str, ...], Sequence[str]],
    checkpoint_usable: bool,
    glossary: GlossaryRenderer,
) -> str:
    title = subsection or group
    intro = _SUBSECTION_INTROS.get((section, group, subsection), "")
    curve_card = subsection == "RWKV Forgetting Curve"
    heading = f"<h3>{glossary.render(title)}</h3>" if subsection else ""
    intro_html = f'<p class="subsection-intro">{glossary.render(intro)}</p>' if intro else ""
    curve_note = (
        '<div class="curve-disabled-note">'
        + glossary.render(
            "Enable Calculate Forgetting Curves under Performance to use these features."
        )
        + "</div>"
        if curve_card
        else ""
    )
    rows = "\n".join(
        _render_option(
            option,
            config=config,
            choices=choices.get(option.key_path, ()),
            checkpoint_usable=checkpoint_usable,
            glossary=glossary,
        )
        for option in options
    )
    card_class = "settings-card curve-feature-card" if curve_card else "settings-card"
    return f'<section class="{card_class}">{heading}{intro_html}{curve_note}{rows}</section>'


def _render_option(
    option: ConfigOption,
    *,
    config: Mapping[str, Any],
    choices: Sequence[str],
    checkpoint_usable: bool,
    glossary: GlossaryRenderer,
) -> str:
    path_name = option_path_name(option.key_path)
    value = _get_path(config, option.key_path)
    display_value = not bool(value) if option.inverted else value
    classes = ["setting"]
    if option.value_type == "datetime":
        classes.append("setting-datetime")
    attributes = [f'data-option-row="{html.escape(path_name, quote=True)}"']
    if option.requires_curves:
        classes.append("requires-curves")
    if option.parent_key_path:
        attributes.append(
            f'data-parent-path="{html.escape(option_path_name(option.parent_key_path), quote=True)}"'
        )
    if option.visible_when_key_path:
        attributes.append(
            f'data-visible-path="{html.escape(option_path_name(option.visible_when_key_path), quote=True)}"'
        )
    restart = (
        '<span class="restart-badge" tabindex="0" '
        f'data-rwkv-tooltip="{html.escape(option.restart_tooltip, quote=True)}">'
        f"{html.escape(option.restart_badge)}</span>"
        if option.restart_required
        else ""
    )
    rebuild = (
        '<span class="rebuild-badge" tabindex="0" '
        'data-rwkv-tooltip="Changing this setting may require rebuilding '
        'the RWKV checkpoint">Rebuild</span>'
        if option.checkpoint_rebuild_required
        else ""
    )
    rendered_label = glossary.render(option.label)
    rendered_description = glossary.render(option.tooltip)
    control_id = html.escape(_config_control_id(path_name), quote=True)
    title = (
        f'<div class="setting-title">{rendered_label}{restart}{rebuild}</div>'
        if option.value_type == "datetime"
        else (
            f'<label class="setting-title" for="{control_id}">'
            f"{rendered_label}{restart}{rebuild}</label>"
        )
    )
    control = _render_control(
        option,
        path_name=path_name,
        value=display_value,
        choices=choices,
    )
    if option.speed_test:
        reason = (
            "Run this benchmark using the current checkpoint."
            if checkpoint_usable
            else "Unavailable: speed tests require an idle, usable checkpoint. Use "
            "RWKV > Manage Checkpoint to initialize or rebuild it, or finish the RWKV "
            "operation currently using it."
        )
        speed_button = render_button(
            ModalButton(
                _SPEED_BUTTON_LABELS[option.speed_test],
                "speed-test",
                variant="secondary",
                payload={"test": option.speed_test},
                disabled=not checkpoint_usable,
                button_id=f"speed-{option.speed_test}",
                tooltip=reason if checkpoint_usable else None,
            )
        )
        if not checkpoint_usable:
            escaped_reason = html.escape(reason, quote=True)
            escaped_label = html.escape(_SPEED_BUTTON_LABELS[option.speed_test], quote=True)
            speed_button = (
                '<span class="rwkv-disabled-control-help rwkv-help-surface" tabindex="0" '
                f'aria-label="{escaped_label}. {escaped_reason}" '
                f'data-rwkv-tooltip="{escaped_reason}">{speed_button}</span>'
            )
        control += speed_button
    if option.comparison:
        control += render_button(
            ModalButton(
                _COMPARISON_BUTTON_LABELS[option.comparison],
                "comparison",
                variant="secondary",
                payload={"comparison": option.comparison},
                button_id=f"comparison-{option.comparison}",
                tooltip="Build disposable states and compare RWKV Immediate accuracy.",
            )
        )
    return f"""
<div class="{" ".join(classes)}" {" ".join(attributes)}>
    <div>
        {title}
        <div class="setting-description">{rendered_description}</div>
    </div>
    <div class="control">{control}</div>
</div>
""".strip()


def _render_control(
    option: ConfigOption,
    *,
    path_name: str,
    value: Any,
    choices: Sequence[str],
) -> str:
    escaped_path = html.escape(path_name, quote=True)
    control_id = html.escape(_config_control_id(path_name), quote=True)
    if option.value_type == "bool":
        checked = " checked" if bool(value) else ""
        return (
            f'<label class="rwkv-switch" for="{control_id}">'
            f'<input id="{control_id}" type="checkbox" data-config-path="{escaped_path}" '
            f'aria-label="{html.escape(option.label, quote=True)}"{checked}>'
            '<span class="rwkv-switch__track" aria-hidden="true"></span>'
            "</label>"
        )
    if option.value_type == "choice":
        selected_value = str(value or "")
        rendered = "".join(
            f'<option value="{html.escape(str(item), quote=True)}"'
            f"{' selected' if str(item) == selected_value else ''}>"
            f"{html.escape(_choice_label(str(item)))}</option>"
            for item in choices
        )
        return f'<select id="{control_id}" data-config-path="{escaped_path}">{rendered}</select>'
    if option.value_type == "int":
        suffix = (
            '<span class="unit">cards</span>' if option.speed_test == "live-predictions" else ""
        )
        number_input = render_number_input(
            ModalField(
                name=path_name,
                label=option.label,
                kind="number",
                value=int(value or 0),
                minimum=None if option.minimum is None else int(option.minimum),
                maximum=None if option.maximum is None else int(option.maximum),
                step=1,
                required=True,
            ),
            control_id=_config_control_id(path_name),
            data_attributes={"data-config-path": path_name},
        )
        return f'<div class="rwkv-field rwkv-settings-number-field">{number_input}</div>{suffix}'
    if option.value_type == "datetime":
        date_value, time_value, period = _datetime_control_parts(value)
        error_id = control_id + "-error"
        return (
            '<div class="datetime-editor" data-datetime-editor>'
            f'<input id="{control_id}" type="hidden" data-config-path="{escaped_path}" '
            f'data-datetime-value value="{html.escape(str(value or ""), quote=True)}">'
            '<label class="datetime-part">'
            '<span class="datetime-part-label">Date</span>'
            f'<input class="datetime-date" type="date" data-datetime-date '
            f'aria-describedby="{error_id}" value="{html.escape(date_value, quote=True)}">'
            "</label>"
            '<label class="datetime-part">'
            '<span class="datetime-part-label" tabindex="0" '
            'data-rwkv-tooltip="Type an hour, or a time such as 9:30 or 9:30:15">Time</span>'
            f'<input class="datetime-time" type="text" data-datetime-time '
            f'aria-describedby="{error_id}" autocomplete="off" spellcheck="false" '
            f'placeholder="h:mm:ss" '
            f'value="{html.escape(time_value, quote=True)}">'
            "</label>"
            '<label class="datetime-part">'
            '<span class="datetime-part-label">AM / PM</span>'
            '<select class="datetime-period" data-datetime-period>'
            f'<option value="AM"{" selected" if period == "AM" else ""}>AM</option>'
            f'<option value="PM"{" selected" if period == "PM" else ""}>PM</option>'
            "</select></label>"
            f'<span id="{error_id}" class="datetime-error" data-datetime-error role="alert" hidden>'
            "Enter a date and a time from 1:00:00 to 12:59:59.</span>"
            "</div>"
        )
    return (
        f'<input id="{control_id}" type="text" data-config-path="{escaped_path}" '
        f'value="{html.escape(str(value or ""), quote=True)}">'
    )


def _config_control_id(path_name: str) -> str:
    return "config-" + path_name.replace("/", "-")


def _coerce_value(
    option: ConfigOption,
    value: Any,
    choices: Sequence[str],
) -> Any:
    if option.value_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if option.value_type == "choice":
        parsed = str(value)
        allowed = tuple(str(item) for item in choices)
        if parsed not in allowed:
            raise ValueError(f"Unsupported value for {option.label}: {parsed!r}")
        return parsed
    if option.value_type == "int":
        if isinstance(value, bool):
            raise ValueError(f"{option.label} must be a whole number.")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{option.label} must be a whole number.") from exc
        if option.minimum is not None:
            parsed = max(parsed, int(option.minimum))
        if option.maximum is not None:
            parsed = min(parsed, int(option.maximum))
        return parsed
    if option.value_type == "datetime":
        parsed = str(value).strip()
        parse_cutoff_datetime(parsed)
        return parsed
    return str(value)


def _datetime_control_parts(value: Any) -> tuple[str, str, str]:
    """Return editable local date, 12-hour time, and period fields."""

    try:
        parsed = datetime.fromisoformat(str(value or "").strip())
    except ValueError:
        return "", "", "AM"
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    hour = parsed.hour % 12 or 12
    period = "PM" if parsed.hour >= 12 else "AM"
    return parsed.strftime("%Y-%m-%d"), f"{hour}:{parsed:%M:%S}", period


def _get_path(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _set_path(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _choice_label(value: str) -> str:
    return "GPU" if value.lower() == "gpu" else value.replace("_", " ").title()


def _capability_note(*, predict_gpu_available: bool, process_gpu_available: bool) -> str:
    if predict_gpu_available and process_gpu_available:
        return "Compatible GPU acceleration is available for State Building and Predictions."
    if predict_gpu_available:
        return "GPU acceleration is available for Predictions, but not State Building."
    if process_gpu_available:
        return "GPU acceleration is available for State Building, but not Predictions."
    return "No compatible GPU acceleration was detected. RWKV will use CPU Fast."


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
