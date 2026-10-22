from __future__ import annotations

# Presentation data stays in Python; Svelte owns the Settings DOM and events.
# ruff: noqa: E501
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any

from .addon_config import (
    CALCULATE_FORGETTING_CURVES_CONFIG_KEY,
    LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_GENERATION_CONFIG_KEY,
    PREDICT_MANY_MODE_CONFIG_KEY,
    configured_model_generation,
    webview_popup_control_mode,
)
from .config_options import CONFIG_OPTIONS, CONFIG_SECTIONS, ConfigOption
from .model_generation import MODEL_GENERATIONS, V5O_MODEL_GENERATION, model_generation_label
from .review_type_normalization import parse_cutoff_datetime
from .rwkv_performance_modes import prediction_mode_label
from .settings_view import settings_pages
from .setup_faq_html import setup_faq_items
from .svelte_ui import render_svelte_document

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
    "auto-live-predictions": "Auto",
}

_COMPARISON_BUTTON_LABELS = {
    "generations": "Compare Generations",
    "models": "Compare Models",
    "deleted-reviews": "Compare On vs. Off",
    "state-precision": "Compare FP16 vs FP32",
    "gpu-math": "Compare Predictions",
}


def visible_config_options(*, predict_gpu_available: bool) -> tuple[ConfigOption, ...]:
    return tuple(
        option
        for option in CONFIG_OPTIONS
        if not option.requires_gpu or bool(predict_gpu_available)
    )


def option_path_name(path: tuple[str, ...]) -> str:
    return "/".join(path)


def _model_choices_by_generation(
    choices: Mapping[tuple[str, ...], Sequence[str]],
    configured: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    fallback = tuple(str(value) for value in choices.get((MODEL_CONFIG_KEY,), ()))
    return {
        generation: tuple(
            str(value)
            for value in (
                configured.get(generation, fallback) if configured is not None else fallback
            )
        )
        for generation in MODEL_GENERATIONS
    }


def build_settings_payload(
    config: Mapping[str, Any],
    *,
    choices: Mapping[tuple[str, ...], Sequence[str]],
    predict_gpu_available: bool,
    process_gpu_available: bool,
    checkpoint_usable: bool,
    initial_setup: bool = False,
    apply_enabled: bool = False,
    model_choices_by_generation: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    generation_model_choices = _model_choices_by_generation(choices, model_choices_by_generation)
    rendered_choices = dict(choices)
    rendered_choices[(MODEL_CONFIG_KEY,)] = generation_model_choices[
        configured_model_generation(config)
    ]
    pages = (
        []
        if initial_setup
        else settings_pages(
            config,
            options=visible_config_options(predict_gpu_available=predict_gpu_available),
            sections=CONFIG_SECTIONS,
            choices=rendered_choices,
            page_intros=_PAGE_INTROS,
            group_intros=_GROUP_INTROS,
            subsection_intros=_SUBSECTION_INTROS,
            capability_note=_capability_note(
                predict_gpu_available=predict_gpu_available,
                process_gpu_available=process_gpu_available,
            ),
            choice_label=_choice_label,
            datetime_parts=_datetime_control_parts,
            model_key=MODEL_CONFIG_KEY,
        )
    )
    return {
        "pages": pages,
        "initialSetup": bool(initial_setup),
        "applyEnabled": bool(apply_enabled),
        "checkpointUsable": bool(checkpoint_usable),
        "modelChoices": generation_model_choices,
        "speedLabels": _SPEED_BUTTON_LABELS,
        "comparisonLabels": _COMPARISON_BUTTON_LABELS,
        "keys": {
            "curves": CALCULATE_FORGETTING_CURVES_CONFIG_KEY,
            "model": MODEL_CONFIG_KEY,
            "generation": MODEL_GENERATION_CONFIG_KEY,
            "predictionMode": PREDICT_MANY_MODE_CONFIG_KEY,
            "predictionRefreshLimit": LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY,
            "v5": V5O_MODEL_GENERATION,
        },
        "faq": setup_faq_items(),
    }


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
    model_choices_by_generation: Mapping[str, Sequence[str]] | None = None,
) -> str:
    data = build_settings_payload(
        config,
        choices=choices,
        predict_gpu_available=predict_gpu_available,
        process_gpu_available=process_gpu_available,
        checkpoint_usable=checkpoint_usable,
        initial_setup=initial_setup,
        apply_enabled=apply_enabled,
        model_choices_by_generation=model_choices_by_generation,
    )
    return render_svelte_document(
        "settings",
        {"data": data},
        title="RWKV Guided Setup" if initial_setup else "RWKV Settings",
        generation=generation,
        layout="custom",
        title_id=None,
        root_attributes={"data-rwkv-enter-from-inputs": "true"} if not initial_setup else None,
        root_extra_classes="rwkv-guided-setup-host" if initial_setup else "rwkv-settings-host",
        escape_action="cancel",
        enter_action=None if initial_setup else "ok",
        bootstrap={
            "bridgePrefix": BRIDGE_PREFIX,
            "bridgeMode": "flat",
            "bridgeExpectsReply": False,
            "popupControlMode": webview_popup_control_mode(config),
            "backgroundSelectors": (
                ("#rwkv-setup-overlay",)
                if initial_setup
                else (
                    ".settings-shell > .rwkv-settings-tabs",
                    ".settings-shell > .settings-pages",
                    ".settings-shell > .rwkv-dialog-footer",
                    "#rwkv-setup-overlay",
                )
            ),
        },
    )


def config_from_web_values(
    base: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    options: Sequence[ConfigOption],
    choices: Mapping[tuple[str, ...], Sequence[str]],
) -> dict[str, Any]:
    updated = deepcopy(dict(base))
    for option in options:
        if option.value_type == "action":
            continue
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
        if option.value_type == "action":
            continue
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
    if value.casefold() in MODEL_GENERATIONS:
        return model_generation_label(value)
    return prediction_mode_label(value)


def _capability_note(*, predict_gpu_available: bool, process_gpu_available: bool) -> str:
    if predict_gpu_available and process_gpu_available:
        return "Compatible GPU acceleration is available for State Building and Predictions."
    if predict_gpu_available:
        return "GPU acceleration is available for Predictions, but not State Building."
    if process_gpu_available:
        return "GPU acceleration is available for State Building, but not Predictions."
    return "No compatible GPU acceleration was detected. RWKV will use CPU Fast."
