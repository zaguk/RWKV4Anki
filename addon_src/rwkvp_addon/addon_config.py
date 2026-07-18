from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .constants import (
    ADDON_ROOT,
    CHECKPOINT_SAVE_INTERVAL,
    DEFAULT_MODEL_ID,
    PREDICT_BATCH_SIZE,
    VENDOR_ROOT,
)
from .live_review_engine import (
    DEFAULT_HOT_PREDICT_LIMIT,
    DEFAULT_PREDICTION_REFRESH_LIMIT,
    DEFAULT_QUIET_REFRESH_ATTEMPTS,
    DEFAULT_SAME_DAY_REENTRY_DELAY_REVIEWS,
)
from .metrics import RWKVPredictionMode
from .prediction_cache import (
    PER_REVIEW_CACHE_SPEC,
    PREDICT_AHEAD_CACHE_SPEC,
    PredictionCacheSpec,
)
from .review_load_policy import DEFAULT_MINIMUM_RETENTION_STEP
from .rwkv_performance_modes import (
    DEFAULT_PREDICT_MANY_MODE,
    DEFAULT_PROCESS_MANY_MODE,
    PREDICT_MANY_FAST_MODE,
    PREDICT_MANY_GPU_MODE,
    PREDICT_MANY_ORACLE_MODE,
    normalize_predict_many_mode,
    normalize_process_many_mode,
)

MODEL_CONFIG_KEY = "model"
PREDICT_MANY_MODE_CONFIG_KEY = "predict_many_mode"
# Kept only to migrate configurations written before per-mode automatic
# batching was introduced.
PREDICT_MANY_BATCH_SIZE_CONFIG_KEY = "predict_many_batch_size"
PREDICT_MANY_BATCH_SIZES_CONFIG_KEY = "predict_many_batch_sizes"
PROCESS_MANY_MODE_CONFIG_KEY = "process_many_mode"
ENABLE_RWKV_IMMEDIATE_CONFIG_KEY = "enable_rwkv_immediate"
CALCULATE_FORGETTING_CURVES_CONFIG_KEY = "calculate_forgetting_curves"
SAVE_CHECKPOINT_EVERY_REVIEWS_CONFIG_KEY = "save_checkpoint_every_reviews"
SHOW_CHECKPOINT_REBUILD_CONFIRMATION_CONFIG_KEY = (
    "show_checkpoint_rebuild_confirmation"
)
WINDOWS_DISPLAY_BUG_PATCH_CONFIG_KEY = "windows_display_bug_patch"
EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY = "exclude_deleted_card_revlogs"
CARD_INFO_RETRIEVABILITY_CONFIG_KEY = "enable_card_info_retrievability"
CARD_INFO_RETRIEVABILITY_AUTO_REFRESH_CONFIG_KEY = (
    "auto_refresh_card_info_retrievability"
)
CARD_INFO_INTERVALS_CONFIG_KEY = "enable_card_info_rwkv_intervals"
CARD_INFO_FORGETTING_CURVE_GRAPH_CONFIG_KEY = "enable_card_info_rwkv_forgetting_curve_graph"
CARD_INFO_FORGETTING_CURVE_GRAPH_LOWER_BOUND_CONFIG_KEY = (
    "card_info_rwkv_forgetting_curve_graph_lower_bound"
)
_LEGACY_LIVE_REVIEW_READY_QUEUE_SIZE_CONFIG_KEY = "live_review_ready_queue_size"
LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY = "live_review_prediction_refresh_limit"
LIVE_REVIEW_SAMEDAY_PREDICTION_LIMIT_CONFIG_KEY = "live_review_sameday_prediction_limit"
LIVE_REVIEW_QUIET_REFRESH_ATTEMPTS_CONFIG_KEY = "live_review_quiet_refresh_attempts"
LIVE_REVIEW_SAMEDAY_REENTRY_DELAY_REVIEWS_CONFIG_KEY = (
    "live_review_sameday_reentry_delay_reviews"
)
MINIMUM_REVIEW_WIDENING_EXTRA_PERCENT_CONFIG_KEY = "minimum_review_widening_extra_percent"
CURVE_RESCHEDULING_CONFIG_KEY = "experimental_enable_curve_rescheduling"
EXPERIMENTAL_SHORT_TERM_RESCHEDULING_CONFIG_KEY = "experimental_enable_short_term_rescheduling"
ACTIVE_REVIEW_PROTOTYPE_CONFIG_KEY = "experimental_enable_active_review_prototype"
ADAPTIVE_DESIRED_RETENTION_CONFIG_KEY = "experimental_enable_adaptive_desired_retention"
EXPERIMENTAL_BEHAVIOR_LAB_CONFIG_KEY = "experimental_enable_behavior_lab"
DEFAULT_MINIMUM_REVIEW_WIDENING_EXTRA_PERCENT = max(
    1,
    int(round(DEFAULT_MINIMUM_RETENTION_STEP * 100)),
)


def default_addon_config() -> dict[str, Any]:
    path = ADDON_ROOT / "config.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        loaded = {
            MODEL_CONFIG_KEY: DEFAULT_MODEL_ID,
            SAVE_CHECKPOINT_EVERY_REVIEWS_CONFIG_KEY: CHECKPOINT_SAVE_INTERVAL,
            SHOW_CHECKPOINT_REBUILD_CONFIRMATION_CONFIG_KEY: True,
            PREDICT_MANY_MODE_CONFIG_KEY: DEFAULT_PREDICT_MANY_MODE,
            PREDICT_MANY_BATCH_SIZES_CONFIG_KEY: _default_predict_many_batch_sizes(),
            PROCESS_MANY_MODE_CONFIG_KEY: DEFAULT_PROCESS_MANY_MODE,
            ENABLE_RWKV_IMMEDIATE_CONFIG_KEY: True,
            CALCULATE_FORGETTING_CURVES_CONFIG_KEY: True,
            EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY: True,
            CARD_INFO_RETRIEVABILITY_CONFIG_KEY: True,
            CARD_INFO_RETRIEVABILITY_AUTO_REFRESH_CONFIG_KEY: False,
            CARD_INFO_INTERVALS_CONFIG_KEY: True,
            CARD_INFO_FORGETTING_CURVE_GRAPH_CONFIG_KEY: False,
            CARD_INFO_FORGETTING_CURVE_GRAPH_LOWER_BOUND_CONFIG_KEY: 60,
            LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY: (DEFAULT_PREDICTION_REFRESH_LIMIT),
            LIVE_REVIEW_SAMEDAY_PREDICTION_LIMIT_CONFIG_KEY: DEFAULT_HOT_PREDICT_LIMIT,
            LIVE_REVIEW_QUIET_REFRESH_ATTEMPTS_CONFIG_KEY: DEFAULT_QUIET_REFRESH_ATTEMPTS,
            LIVE_REVIEW_SAMEDAY_REENTRY_DELAY_REVIEWS_CONFIG_KEY: (
                DEFAULT_SAME_DAY_REENTRY_DELAY_REVIEWS
            ),
            MINIMUM_REVIEW_WIDENING_EXTRA_PERCENT_CONFIG_KEY: (
                DEFAULT_MINIMUM_REVIEW_WIDENING_EXTRA_PERCENT
            ),
            ACTIVE_REVIEW_PROTOTYPE_CONFIG_KEY: True,
            ADAPTIVE_DESIRED_RETENTION_CONFIG_KEY: False,
            EXPERIMENTAL_BEHAVIOR_LAB_CONFIG_KEY: False,
            CURVE_RESCHEDULING_CONFIG_KEY: False,
            EXPERIMENTAL_SHORT_TERM_RESCHEDULING_CONFIG_KEY: False,
        }
    loaded.setdefault(
        WINDOWS_DISPLAY_BUG_PATCH_CONFIG_KEY,
        default_windows_display_bug_patch_enabled(),
    )
    return loaded


def normalized_addon_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source_config = dict(config or {})
    normalized = default_addon_config()
    _deep_update(normalized, source_config)
    normalized[MODEL_CONFIG_KEY] = configured_model_id(normalized)
    normalized[PREDICT_MANY_MODE_CONFIG_KEY] = predict_many_mode(normalized)
    batch_config_source = (
        source_config
        if PREDICT_MANY_BATCH_SIZES_CONFIG_KEY in source_config
        or PREDICT_MANY_BATCH_SIZE_CONFIG_KEY in source_config
        else normalized
    )
    normalized[PREDICT_MANY_BATCH_SIZES_CONFIG_KEY] = predict_many_batch_sizes(batch_config_source)
    normalized.pop(PREDICT_MANY_BATCH_SIZE_CONFIG_KEY, None)
    normalized[PROCESS_MANY_MODE_CONFIG_KEY] = process_many_mode(normalized)
    normalized[SAVE_CHECKPOINT_EVERY_REVIEWS_CONFIG_KEY] = checkpoint_save_interval(normalized)
    normalized[SHOW_CHECKPOINT_REBUILD_CONFIRMATION_CONFIG_KEY] = (
        show_checkpoint_rebuild_confirmation(normalized)
    )
    normalized[WINDOWS_DISPLAY_BUG_PATCH_CONFIG_KEY] = (
        windows_display_bug_patch_enabled(normalized)
    )
    normalized[CARD_INFO_FORGETTING_CURVE_GRAPH_LOWER_BOUND_CONFIG_KEY] = (
        card_info_forgetting_curve_graph_lower_bound_percent(normalized)
    )
    normalized[LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY] = (
        live_review_prediction_refresh_limit(normalized)
    )
    normalized.pop(_LEGACY_LIVE_REVIEW_READY_QUEUE_SIZE_CONFIG_KEY, None)
    # These settings were removed from the UI and runtime. FSRS comparison is
    # always recorded for new Live Sessions, and the unused JSONL diagnostics
    # writer was retired. Do not preserve stale values as unknown config keys.
    normalized.pop("enable_live_review_fsrs_comparison", None)
    normalized.pop("experimental_enable_active_review_diagnostics", None)
    normalized[LIVE_REVIEW_SAMEDAY_PREDICTION_LIMIT_CONFIG_KEY] = (
        live_review_sameday_prediction_limit(normalized)
    )
    normalized[LIVE_REVIEW_QUIET_REFRESH_ATTEMPTS_CONFIG_KEY] = (
        live_review_quiet_refresh_attempts(normalized)
    )
    normalized[LIVE_REVIEW_SAMEDAY_REENTRY_DELAY_REVIEWS_CONFIG_KEY] = (
        live_review_sameday_reentry_delay_reviews(normalized)
    )
    normalized[MINIMUM_REVIEW_WIDENING_EXTRA_PERCENT_CONFIG_KEY] = (
        minimum_review_widening_extra_percent(normalized)
    )
    # Drop the retired daily/review-batch prediction experiment from existing
    # user configurations instead of preserving it as an unknown nested key.
    normalized.pop("rwkv_prediction_caches", None)
    return normalized


def available_model_ids() -> tuple[str, ...]:
    pretrained_dir = VENDOR_ROOT / "rwkv_srs" / "pretrained"
    try:
        model_ids = tuple(sorted(path.stem for path in pretrained_dir.glob("*.safetensors")))
    except OSError:
        model_ids = ()
    if model_ids:
        return model_ids
    return (DEFAULT_MODEL_ID,)


def configured_model_id(config: Mapping[str, Any] | None) -> str:
    if not config:
        return DEFAULT_MODEL_ID
    value = str(config.get(MODEL_CONFIG_KEY) or DEFAULT_MODEL_ID)
    models = available_model_ids()
    if value in models:
        return value
    if DEFAULT_MODEL_ID in models:
        return DEFAULT_MODEL_ID
    return models[0]


def addon_config_for_mw(mw) -> dict[str, Any]:
    addon_manager = getattr(mw, "addonManager", None)
    if addon_manager is None:
        return {}
    addon = addon_manager.addonFromModule(__name__)
    return normalized_addon_config(addon_manager.getConfig(addon) or {})


def write_addon_config_for_mw(mw, config: Mapping[str, Any]) -> None:
    addon_manager = getattr(mw, "addonManager", None)
    if addon_manager is None:
        return
    addon = addon_manager.addonFromModule(__name__)
    normalized = normalized_addon_config(config)
    write_config = getattr(addon_manager, "writeConfig", None)
    if callable(write_config):
        write_config(addon, normalized)
        return

    addon_meta = getattr(addon_manager, "addonMeta", None)
    write_addon_meta = getattr(addon_manager, "writeAddonMeta", None)
    if callable(addon_meta) and callable(write_addon_meta):
        meta = addon_meta(addon)
        meta["config"] = normalized
        write_addon_meta(addon, meta)


def enabled_prediction_cache_specs(
    config: Mapping[str, Any] | None,
) -> tuple[PredictionCacheSpec, ...]:
    if calculate_forgetting_curves(config):
        return (PER_REVIEW_CACHE_SPEC, PREDICT_AHEAD_CACHE_SPEC)
    return (PER_REVIEW_CACHE_SPEC,)


def enabled_rwkv_prediction_modes(
    config: Mapping[str, Any] | None,
) -> tuple[RWKVPredictionMode, ...]:
    if calculate_forgetting_curves(config):
        return (RWKVPredictionMode.PER_REVIEW, RWKVPredictionMode.PREDICT_AHEAD)
    return (RWKVPredictionMode.PER_REVIEW,)


def enabled_immediate_rwkv_prediction_modes(
    config: Mapping[str, Any] | None,
) -> tuple[RWKVPredictionMode, ...]:
    if not rwkv_immediate_enabled(config):
        return ()
    return tuple(
        mode
        for mode in enabled_rwkv_prediction_modes(config)
        if mode != RWKVPredictionMode.PREDICT_AHEAD
    )


def enabled_forgetting_curve_rwkv_prediction_modes(
    config: Mapping[str, Any] | None,
) -> tuple[RWKVPredictionMode, ...]:
    if not calculate_forgetting_curves(config):
        return ()
    return (RWKVPredictionMode.PREDICT_AHEAD,)


def calculate_forgetting_curves(config: Mapping[str, Any] | None) -> bool:
    if not config:
        return True
    return bool(config.get(CALCULATE_FORGETTING_CURVES_CONFIG_KEY, True))


def rwkv_immediate_enabled(config: Mapping[str, Any] | None) -> bool:
    if not config:
        return True
    return bool(config.get(ENABLE_RWKV_IMMEDIATE_CONFIG_KEY, True))


def predict_many_batch_sizes(config: Mapping[str, Any] | None) -> dict[str, int]:
    sizes = _default_predict_many_batch_sizes()
    if not config:
        return sizes

    configured = config.get(PREDICT_MANY_BATCH_SIZES_CONFIG_KEY)
    if isinstance(configured, Mapping):
        for mode in sizes:
            value = configured.get(mode)
            if mode == PREDICT_MANY_FAST_MODE and mode not in configured:
                value = configured.get("lightning")
            sizes[mode] = _batch_size_override_value(value)
        return sizes

    # The former packaged default was 192 for every route. Treat that exact
    # legacy value as Automatic so Fast and GPU receive their optimized
    # upstream defaults. Preserve a genuinely customized legacy value for the
    # mode that was selected when it was written.
    legacy = _batch_size_override_value(config.get(PREDICT_MANY_BATCH_SIZE_CONFIG_KEY))
    if legacy and legacy != PREDICT_BATCH_SIZE:
        sizes[predict_many_mode(config)] = legacy
    return sizes


def predict_many_batch_size(
    config: Mapping[str, Any] | None,
    mode: object | None = None,
) -> int | None:
    selected_mode = predict_many_mode(config) if mode is None else normalize_predict_many_mode(mode)
    value = predict_many_batch_sizes(config)[selected_mode]
    return value if value > 0 else None


def predict_many_mode(config: Mapping[str, Any] | None) -> str:
    value = None if not config else config.get(PREDICT_MANY_MODE_CONFIG_KEY)
    return normalize_predict_many_mode(value)


def process_many_mode(config: Mapping[str, Any] | None) -> str:
    value = None if not config else config.get(PROCESS_MANY_MODE_CONFIG_KEY)
    return normalize_process_many_mode(value)


def checkpoint_save_interval(config: Mapping[str, Any] | None) -> int:
    if not config:
        return CHECKPOINT_SAVE_INTERVAL
    value = config.get(SAVE_CHECKPOINT_EVERY_REVIEWS_CONFIG_KEY, CHECKPOINT_SAVE_INTERVAL)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return CHECKPOINT_SAVE_INTERVAL
    return parsed if parsed > 0 else CHECKPOINT_SAVE_INTERVAL


def show_checkpoint_rebuild_confirmation(config: Mapping[str, Any] | None) -> bool:
    if not config:
        return True
    return bool(config.get(SHOW_CHECKPOINT_REBUILD_CONFIRMATION_CONFIG_KEY, True))


def default_windows_display_bug_patch_enabled(platform: str | None = None) -> bool:
    """Return the platform-aware default for the WebView popup workaround."""

    current_platform = sys.platform if platform is None else str(platform)
    return current_platform.startswith("win")


def windows_display_bug_patch_enabled(
    config: Mapping[str, Any] | None,
    *,
    platform: str | None = None,
) -> bool:
    """Return whether RWKV should replace native popup controls in WebViews."""

    if config is not None and WINDOWS_DISPLAY_BUG_PATCH_CONFIG_KEY in config:
        return bool(config[WINDOWS_DISPLAY_BUG_PATCH_CONFIG_KEY])
    return default_windows_display_bug_patch_enabled(platform)


def webview_popup_control_mode(
    config: Mapping[str, Any] | None,
    *,
    platform: str | None = None,
) -> str:
    """Translate the user-facing workaround toggle into the shared UI mode."""

    return (
        "in-page"
        if windows_display_bug_patch_enabled(config, platform=platform)
        else "native"
    )


def exclude_deleted_card_revlogs(config: Mapping[str, Any] | None) -> bool:
    if not config:
        return True
    return bool(config.get(EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY, True))


def card_info_retrievability_enabled(config: Mapping[str, Any] | None) -> bool:
    if not rwkv_immediate_enabled(config):
        return False
    if not config:
        return True
    return bool(config.get(CARD_INFO_RETRIEVABILITY_CONFIG_KEY, True))


def card_info_retrievability_auto_refresh_enabled(
    config: Mapping[str, Any] | None,
) -> bool:
    if not card_info_retrievability_enabled(config):
        return False
    if not config:
        return False
    return bool(config.get(CARD_INFO_RETRIEVABILITY_AUTO_REFRESH_CONFIG_KEY, False))


def card_info_rwkv_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether Card Info has any enabled RWKV fields."""

    return card_info_retrievability_enabled(config) or calculate_forgetting_curves(config)


def card_info_intervals_enabled(config: Mapping[str, Any] | None) -> bool:
    if not calculate_forgetting_curves(config):
        return False
    if not config:
        return True
    return bool(config.get(CARD_INFO_INTERVALS_CONFIG_KEY, True))


def card_info_forgetting_curve_graph_enabled(config: Mapping[str, Any] | None) -> bool:
    if not calculate_forgetting_curves(config):
        return False
    if not config:
        return False
    return bool(config.get(CARD_INFO_FORGETTING_CURVE_GRAPH_CONFIG_KEY, False))


def card_info_forgetting_curve_graph_lower_bound_percent(
    config: Mapping[str, Any] | None,
) -> int:
    return _integer_config(
        config,
        CARD_INFO_FORGETTING_CURVE_GRAPH_LOWER_BOUND_CONFIG_KEY,
        60,
        minimum=0,
        maximum=99,
    )


def live_review_prediction_refresh_limit(config: Mapping[str, Any] | None) -> int:
    return _integer_config(
        config,
        LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY,
        DEFAULT_PREDICTION_REFRESH_LIMIT,
        minimum=1,
    )


def live_review_sameday_prediction_limit(config: Mapping[str, Any] | None) -> int:
    return _integer_config(
        config,
        LIVE_REVIEW_SAMEDAY_PREDICTION_LIMIT_CONFIG_KEY,
        DEFAULT_HOT_PREDICT_LIMIT,
        minimum=0,
    )


def live_review_quiet_refresh_attempts(config: Mapping[str, Any] | None) -> int:
    return _integer_config(
        config,
        LIVE_REVIEW_QUIET_REFRESH_ATTEMPTS_CONFIG_KEY,
        DEFAULT_QUIET_REFRESH_ATTEMPTS,
        minimum=0,
        maximum=100,
    )


def live_review_sameday_reentry_delay_reviews(
    config: Mapping[str, Any] | None,
) -> int:
    return _integer_config(
        config,
        LIVE_REVIEW_SAMEDAY_REENTRY_DELAY_REVIEWS_CONFIG_KEY,
        DEFAULT_SAME_DAY_REENTRY_DELAY_REVIEWS,
        minimum=0,
    )


def minimum_review_widening_extra_percent(config: Mapping[str, Any] | None) -> int:
    return _integer_config(
        config,
        MINIMUM_REVIEW_WIDENING_EXTRA_PERCENT_CONFIG_KEY,
        DEFAULT_MINIMUM_REVIEW_WIDENING_EXTRA_PERCENT,
        minimum=1,
        maximum=100,
    )


def minimum_review_widening_extra(config: Mapping[str, Any] | None) -> float:
    return minimum_review_widening_extra_percent(config) / 100.0


def curve_rescheduling_enabled(config: Mapping[str, Any] | None) -> bool:
    if not calculate_forgetting_curves(config):
        return False
    if not config:
        return False
    return bool(config.get(CURVE_RESCHEDULING_CONFIG_KEY, False))


def experimental_short_term_rescheduling_enabled(
    config: Mapping[str, Any] | None,
) -> bool:
    if not calculate_forgetting_curves(config):
        return False
    if not config:
        return False
    return bool(config.get(EXPERIMENTAL_SHORT_TERM_RESCHEDULING_CONFIG_KEY, False))


def active_review_prototype_enabled(config: Mapping[str, Any] | None) -> bool:
    if not rwkv_immediate_enabled(config):
        return False
    if not config:
        return True
    return bool(config.get(ACTIVE_REVIEW_PROTOTYPE_CONFIG_KEY, True))


def adaptive_desired_retention_enabled(config: Mapping[str, Any] | None) -> bool:
    if not calculate_forgetting_curves(config):
        return False
    if not config:
        return False
    return bool(config.get(ADAPTIVE_DESIRED_RETENTION_CONFIG_KEY, False))


def behavior_lab_enabled(config: Mapping[str, Any] | None) -> bool:
    if not config:
        return False
    return bool(config.get(EXPERIMENTAL_BEHAVIOR_LAB_CONFIG_KEY, False))


def _default_predict_many_batch_sizes() -> dict[str, int]:
    return {
        PREDICT_MANY_ORACLE_MODE: 0,
        PREDICT_MANY_FAST_MODE: 0,
        PREDICT_MANY_GPU_MODE: 0,
    }


def _batch_size_override_value(value: object) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _integer_config(
    config: Mapping[str, Any] | None,
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if not config:
        return int(default)
    value = config.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    if parsed < int(minimum):
        return int(default)
    if maximum is not None and parsed > int(maximum):
        return int(default)
    return parsed


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], dict(value))
        else:
            target[key] = deepcopy(value)
