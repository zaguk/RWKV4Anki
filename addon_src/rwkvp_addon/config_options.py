from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .addon_config import (
    ACTIVE_REVIEW_PROTOTYPE_CONFIG_KEY,
    ADAPTIVE_DESIRED_RETENTION_CONFIG_KEY,
    CALCULATE_FORGETTING_CURVES_CONFIG_KEY,
    CARD_INFO_FORGETTING_CURVE_GRAPH_CONFIG_KEY,
    CARD_INFO_FORGETTING_CURVE_GRAPH_LOWER_BOUND_CONFIG_KEY,
    CARD_INFO_INTERVALS_CONFIG_KEY,
    CARD_INFO_RETRIEVABILITY_AUTO_REFRESH_CONFIG_KEY,
    CARD_INFO_RETRIEVABILITY_CONFIG_KEY,
    CURVE_RESCHEDULING_CONFIG_KEY,
    ENABLE_RWKV_IMMEDIATE_CONFIG_KEY,
    EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY,
    EXPERIMENTAL_BEHAVIOR_LAB_CONFIG_KEY,
    EXPERIMENTAL_SHORT_TERM_RESCHEDULING_CONFIG_KEY,
    LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY,
    LIVE_REVIEW_QUIET_REFRESH_ATTEMPTS_CONFIG_KEY,
    LIVE_REVIEW_SAMEDAY_PREDICTION_LIMIT_CONFIG_KEY,
    LIVE_REVIEW_SAMEDAY_REENTRY_DELAY_REVIEWS_CONFIG_KEY,
    MINIMUM_REVIEW_WIDENING_EXTRA_PERCENT_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    PREDICT_MANY_BATCH_SIZES_CONFIG_KEY,
    PREDICT_MANY_MODE_CONFIG_KEY,
    PROCESS_MANY_MODE_CONFIG_KEY,
    SAVE_CHECKPOINT_EVERY_REVIEWS_CONFIG_KEY,
    SHOW_CHECKPOINT_REBUILD_CONFIRMATION_CONFIG_KEY,
    WINDOWS_DISPLAY_BUG_PATCH_CONFIG_KEY,
)
from .constants import CHECKPOINT_SAVE_INTERVAL, PREDICT_BATCH_SIZE
from .live_review_engine import DEFAULT_QUIET_REFRESH_ATTEMPTS
from .review_load_policy import DEFAULT_MINIMUM_RETENTION_STEP
from .review_type_normalization import (
    FILTERED_REVIEW_NORMALIZATION_CUTOFF_CONFIG_KEY,
    FILTERED_REVIEW_NORMALIZATION_ENABLED_CONFIG_KEY,
)
from .rwkv_performance_modes import (
    PREDICT_MANY_FAST_MODE,
    PREDICT_MANY_GPU_MODE,
    PREDICT_MANY_MODES,
    PREDICT_MANY_ORACLE_MODE,
    PROCESS_MANY_MODES,
)

CONFIG_SECTIONS = ("General", "Advanced", "Experimental")


@dataclass(frozen=True)
class ConfigOption:
    section: str
    key_path: tuple[str, ...]
    label: str
    value_type: str
    tooltip: str = ""
    minimum: int | None = None
    maximum: int | None = None
    requires_gpu: bool = False
    requires_curves: bool = False
    group: str = ""
    subsection: str = ""
    inverted: bool = False
    restart_required: bool = False
    restart_badge: str = "May need restart"
    restart_tooltip: str = "Some changes take effect after restarting Anki"
    parent_key_path: tuple[str, ...] | None = None
    visible_when_key_path: tuple[str, ...] | None = None
    speed_test: str | None = None
    comparison: str | None = None
    profile_scoped: bool = False
    checkpoint_rebuild_required: bool = False


CONFIG_OPTIONS = (
    ConfigOption(
        "General",
        (PROCESS_MANY_MODE_CONFIG_KEY,),
        "State Building Mode",
        "choice",
        "Chooses how RWKV processes review history when building or updating a "
        "checkpoint. Fast is optimized for CPU; GPU can be faster for large histories. "
        "Fast and GPU can be switched without restarting Anki. Individual Live Session "
        "answers still use Fast.",
        group="Performance",
        subsection="State Building and Predictions",
        speed_test="state-building",
    ),
    ConfigOption(
        "General",
        (PREDICT_MANY_MODE_CONFIG_KEY,),
        "Prediction Mode",
        "choice",
        "Chooses how RWKV scores large groups of cards for Live Sessions, reports, and "
        "filtered decks. Fast uses CPU; GPU can be faster for large groups; Oracle is "
        "the reference mode.",
        group="Performance",
        subsection="State Building and Predictions",
        speed_test="predictions",
    ),
    ConfigOption(
        "General",
        (CALCULATE_FORGETTING_CURVES_CONFIG_KEY,),
        "Calculate Forgetting Curves",
        "bool",
        "Builds the additional state required by RWKV Forgetting Curve features. Turning "
        "it off can speed up state building, but disables Forgetting Curve evaluation "
        "and filtered decks, curve details in Card Info, Adaptive Desired Retention, and "
        "curve-based rescheduling. Immediate predictions and Live Sessions still work. "
        "If you turn it on again, rebuild the checkpoint.",
        group="Performance",
        subsection="State Building and Predictions",
        speed_test="curves",
    ),
    ConfigOption(
        "General",
        (LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY,),
        "Cards Checked Between Reviews",
        "int",
        "After each Live Session answer, RWKV updates predictions for up to this many "
        "cards before choosing the next card. Higher values consider more cards but can "
        "delay the next card.",
        minimum=1,
        maximum=99999,
        group="Performance",
        subsection="Live Session Performance",
        speed_test="live-predictions",
        parent_key_path=(ENABLE_RWKV_IMMEDIATE_CONFIG_KEY,),
    ),
    ConfigOption(
        "General",
        (LIVE_REVIEW_SAMEDAY_PREDICTION_LIMIT_CONFIG_KEY,),
        "Same-Day Cards Always Checked",
        "int",
        "Up to this many same-day cards are always included because their recall "
        "probability can change quickly. They count toward Cards Checked Between Reviews. "
        "Set 0 to disable this special handling.",
        minimum=0,
        maximum=99999,
        group="Performance",
        subsection="Live Session Performance",
        parent_key_path=(ENABLE_RWKV_IMMEDIATE_CONFIG_KEY,),
    ),
    ConfigOption(
        "General",
        (ENABLE_RWKV_IMMEDIATE_CONFIG_KEY,),
        "Enable RWKV Immediate",
        "bool",
        "Shows RWKV Immediate tools throughout Anki, including Live Session, "
        "evaluation and retrievability reports, Immediate filtered decks, "
        "and Immediate Retrievability in Card Info. Turn it off to hide all of those "
        "features without disabling RWKV Forgetting Curve.",
        group="Features",
        subsection="RWKV Immediate",
        restart_required=True,
        restart_tooltip=(
            "Enabling this also requires a restart when it enables Live Session "
            "hooks that were disabled when Anki started. Disabling it does not "
            "require a restart."
        ),
    ),
    ConfigOption(
        "General",
        (ACTIVE_REVIEW_PROTOTYPE_CONFIG_KEY,),
        "Enable RWKV Live Session",
        "bool",
        "Adds Live Session to RWKV's deck tools. It updates RWKV after every answer and "
        "chooses the next card immediately before showing it.",
        group="Features",
        subsection="RWKV Immediate",
        restart_required=True,
        restart_badge="Restart to enable",
        restart_tooltip=(
            "Enabling Live Session requires a restart when its review hooks were "
            "disabled at Anki startup. Disabling it does not require a restart."
        ),
        parent_key_path=(ENABLE_RWKV_IMMEDIATE_CONFIG_KEY,),
    ),
    ConfigOption(
        "General",
        (CARD_INFO_RETRIEVABILITY_CONFIG_KEY,),
        "Show Immediate Retrievability in Card Info",
        "bool",
        "Shows RWKV's estimated chance that you would recall the card now. It is "
        "available during a Live Session or after loading RWKV data from the Browse "
        "window. Forgetting Curve Card Info is controlled separately by Calculate "
        "Forgetting Curves.",
        group="Features",
        subsection="RWKV Immediate",
        parent_key_path=(ENABLE_RWKV_IMMEDIATE_CONFIG_KEY,),
    ),
    ConfigOption(
        "General",
        (CARD_INFO_INTERVALS_CONFIG_KEY,),
        "Show Stability and Interval in Card Info",
        "bool",
        "Shows estimated memory stability and the review interval for the card's desired "
        "retention.",
        requires_curves=True,
        group="Features",
        subsection="RWKV Forgetting Curve",
    ),
    ConfigOption(
        "General",
        (CARD_INFO_FORGETTING_CURVE_GRAPH_CONFIG_KEY,),
        "Show Forgetting Curve Graph in Card Info",
        "bool",
        "Draws the card's predicted RWKV Forgetting Curve at the top of Card Info.",
        requires_curves=True,
        group="Features",
        subsection="RWKV Forgetting Curve",
    ),
    ConfigOption(
        "General",
        (CARD_INFO_FORGETTING_CURVE_GRAPH_LOWER_BOUND_CONFIG_KEY,),
        "Forgetting Curve Graph Lower Bound (%)",
        "int",
        "The lowest retrievability displayed on the graph. A higher value gives more "
        "detail near your desired retention.",
        minimum=0,
        maximum=99,
        requires_curves=True,
        group="Features",
        subsection="RWKV Forgetting Curve",
        visible_when_key_path=(CARD_INFO_FORGETTING_CURVE_GRAPH_CONFIG_KEY,),
    ),
    ConfigOption(
        "Advanced",
        (MODEL_CONFIG_KEY,),
        "Underlying Model",
        "choice",
        "Chooses the bundled RWKV model. Changing it requires rebuilding the checkpoint. "
        "Most users should leave this unchanged.",
        group="RWKV State Settings",
        comparison="models",
    ),
    ConfigOption(
        "Advanced",
        (EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY,),
        "Include History from Deleted Cards",
        "bool",
        "Includes reviews from cards you have deleted. This matches RWKV's training data "
        "and may slightly improve predictions, but adds state-building work. It is off "
        "by default; current cards remain the main source of checkpoint and RAM usage.",
        group="RWKV State Settings",
        inverted=True,
        checkpoint_rebuild_required=True,
        comparison="deleted-reviews",
    ),
    ConfigOption(
        "Advanced",
        (SAVE_CHECKPOINT_EVERY_REVIEWS_CONFIG_KEY,),
        "Checkpoint Save Interval (reviews)",
        "int",
        "Saves a new checkpoint after this many reviews. Reviews since the last save are "
        "processed automatically when RWKV loads, so a larger value does not lose review "
        "data. Saving too often adds file-writing work. Use RWKV > Manage Checkpoint > "
        f"Update Checkpoint to save immediately. Default: {CHECKPOINT_SAVE_INTERVAL:,} reviews.",
        minimum=1,
        maximum=100000,
        group="Checkpoint Settings",
    ),
    ConfigOption(
        "Advanced",
        (SHOW_CHECKPOINT_REBUILD_CONFIRMATION_CONFIG_KEY,),
        "Show “Ready to Rebuild” Confirmation",
        "bool",
        "Before rebuilding, shows the estimated checkpoint size and processing time "
        "and asks you to continue. You can also turn it off from that confirmation.",
        group="Checkpoint Settings",
    ),
    ConfigOption(
        "Advanced",
        (MINIMUM_REVIEW_WIDENING_EXTRA_PERCENT_CONFIG_KEY,),
        "Desired Retention Expansion Step (%)",
        "int",
        "When a Live Session or filtered deck cannot find the requested minimum number "
        "of cards at the chosen desired retention, each full refresh broadens the allowed "
        "retrievability range by this many percentage points. "
        f"The default is {int(round(DEFAULT_MINIMUM_RETENTION_STEP * 100))}%.",
        minimum=1,
        maximum=100,
        group="Live Session Settings",
    ),
    ConfigOption(
        "Advanced",
        (LIVE_REVIEW_QUIET_REFRESH_ATTEMPTS_CONFIG_KEY,),
        "Quiet Refreshes Before Full Refresh",
        "int",
        "Before showing full-refresh progress, Live Session can silently check this many "
        "additional groups of cards. Each attempt checks up to Cards Checked Between "
        "Reviews. Set 0 to show full progress immediately. "
        f"Default: {DEFAULT_QUIET_REFRESH_ATTEMPTS}.",
        minimum=0,
        maximum=100,
        group="Live Session Settings",
    ),
    ConfigOption(
        "Advanced",
        (LIVE_REVIEW_SAMEDAY_REENTRY_DELAY_REVIEWS_CONFIG_KEY,),
        "Same-Day Repeat Delay (reviews)",
        "int",
        "Controls how many other reviews must be completed after a Live Session answer "
        "before the same card can be selected again. Default: 2. Set 0 to allow an "
        "immediate repeat; this is not recommended with the current version of RWKV.",
        minimum=0,
        group="Live Session Settings",
    ),
    ConfigOption(
        "Advanced",
        (FILTERED_REVIEW_NORMALIZATION_ENABLED_CONFIG_KEY,),
        "Interpret Filtered Reviews for RWKV",
        "bool",
        "Live Session schedules cards through an Anki filtered deck, so cards reviewed "
        "early are saved as Filtered even when they behave like Review, Learning, or "
        "Relearning steps. RWKV-P was trained mostly on those more specific review "
        "types, so the generic Filtered label can reduce prediction accuracy. This "
        "setting infers the intended type before sending a review to RWKV; it never "
        "edits Anki's history. Recommended: keep this enabled. Because Anki does not "
        "identify which filtered deck created a review, interpretation also applies to "
        "manual filtered-deck reviews after the start time.",
        group="Live Session Settings",
        profile_scoped=True,
        checkpoint_rebuild_required=True,
    ),
    ConfigOption(
        "Advanced",
        (FILTERED_REVIEW_NORMALIZATION_CUTOFF_CONFIG_KEY,),
        "Interpret Filtered Reviews Starting",
        "datetime",
        "Only Filtered reviews on or after this local date and time are interpreted. "
        "Earlier reviews remain Filtered for RWKV. The default is when this Anki "
        "profile first opened RWKV4Anki with the feature available.",
        group="Live Session Settings",
        parent_key_path=(FILTERED_REVIEW_NORMALIZATION_ENABLED_CONFIG_KEY,),
        profile_scoped=True,
        checkpoint_rebuild_required=True,
    ),
    ConfigOption(
        "Advanced",
        (WINDOWS_DISPLAY_BUG_PATCH_CONFIG_KEY,),
        "Windows Display Bug Patch",
        "bool",
        "Uses in-window dropdowns and date pickers to prevent an expanding blank area "
        "seen on some Windows systems. It is enabled by default on Windows and usually "
        "unnecessary elsewhere. Changes apply to newly opened RWKV windows.",
        group="Appearance",
    ),
    ConfigOption(
        "Experimental",
        (EXPERIMENTAL_BEHAVIOR_LAB_CONFIG_KEY,),
        "Enable RWKV Behavior Lab",
        "bool",
        "Shows the Behavior Lab simulator in the RWKV menus in Anki's main window and "
        "Browse window. It can "
        "compare how hypothetical review sequences affect a card and related cards.",
        group="Functionality",
    ),
    ConfigOption(
        "Experimental",
        (CARD_INFO_RETRIEVABILITY_AUTO_REFRESH_CONFIG_KEY,),
        "Refresh Immediate Retrievability Every 5 Seconds",
        "bool",
        "Recalculates the displayed Immediate Retrievability every five seconds while "
        "Card Info is open. Each update performs a one-card CPU prediction and a small "
        "collection lookup, so slower computers may briefly pause during a refresh. It "
        "reuses RWKV state already loaded by Live Session or Browse and never loads state "
        "by itself.",
        group="Functionality",
        parent_key_path=(CARD_INFO_RETRIEVABILITY_CONFIG_KEY,),
    ),
    ConfigOption(
        "Experimental",
        (ADAPTIVE_DESIRED_RETENTION_CONFIG_KEY,),
        "Enable Adaptive Desired Retention",
        "bool",
        "Adds Adaptive Desired Retention controls to Live Session and filtered-deck "
        "dialogs. It varies desired retention using RWKV stability and FSRS difficulty, "
        "but RWKV4Anki does not calculate suitable parameters for you. This differs from "
        "RWKV's training conditions, so RWKV Immediate may behave differently than expected.",
        requires_curves=True,
        group="Functionality",
    ),
    ConfigOption(
        "Experimental",
        (CURVE_RESCHEDULING_CONFIG_KEY,),
        "Enable RWKV Forgetting Curve Rescheduling",
        "bool",
        "Allows manual rescheduling from RWKV Forgetting Curve predictions. This feature "
        "has not been thoroughly tested and modifies card scheduling data.",
        requires_curves=True,
        group="Functionality",
    ),
    ConfigOption(
        "Experimental",
        (EXPERIMENTAL_SHORT_TERM_RESCHEDULING_CONFIG_KEY,),
        "Enable Same-Day Rescheduling",
        "bool",
        "Allows Forgetting Curve rescheduling to choose minutes or hours when the curve "
        "predicts an interval shorter than one day.",
        requires_curves=True,
        group="Functionality",
        parent_key_path=(CURVE_RESCHEDULING_CONFIG_KEY,),
    ),
    ConfigOption(
        "Experimental",
        (PREDICT_MANY_BATCH_SIZES_CONFIG_KEY, PREDICT_MANY_ORACLE_MODE),
        "Oracle Prediction Batch Size",
        "int",
        f"Zero lets RWKV-SRS choose automatically (normally {PREDICT_BATCH_SIZE}).",
        minimum=0,
        maximum=65536,
        group="Performance",
    ),
    ConfigOption(
        "Experimental",
        (PREDICT_MANY_BATCH_SIZES_CONFIG_KEY, PREDICT_MANY_FAST_MODE),
        "Fast Prediction Batch Size",
        "int",
        "Zero lets RWKV-SRS choose its tuned Fast-mode batch size automatically.",
        minimum=0,
        maximum=65536,
        group="Performance",
    ),
    ConfigOption(
        "Experimental",
        (PREDICT_MANY_BATCH_SIZES_CONFIG_KEY, PREDICT_MANY_GPU_MODE),
        "GPU Prediction Batch Size",
        "int",
        "Zero lets RWKV-SRS choose a GPU-appropriate batch automatically. Higher values "
        "may use more GPU memory.",
        minimum=0,
        maximum=65536,
        requires_gpu=True,
        group="Performance",
    ),
)


CONFIG_CHOICE_VALUES = {
    (PREDICT_MANY_MODE_CONFIG_KEY,): PREDICT_MANY_MODES,
    (PROCESS_MANY_MODE_CONFIG_KEY,): PROCESS_MANY_MODES,
}


@dataclass(frozen=True)
class RestartRequirementContext:
    """Process state that determines whether a changed setting needs restart."""

    live_review_hooks_installed: bool


def config_option_by_path(path: tuple[str, ...]) -> ConfigOption:
    for option in CONFIG_OPTIONS:
        if option.key_path == path:
            return option
    raise KeyError(path)


def restart_required_option_labels(
    current: dict,
    updated: dict,
    *,
    context: RestartRequirementContext | None = None,
) -> tuple[str, ...]:
    runtime = context or RestartRequirementContext(
        live_review_hooks_installed=_effective_live_review_enabled(current),
    )
    labels: list[str] = []

    live_was_enabled = _effective_live_review_enabled(current)
    live_is_enabled = _effective_live_review_enabled(updated)
    if not live_was_enabled and live_is_enabled and not runtime.live_review_hooks_installed:
        active_path = (ACTIVE_REVIEW_PROTOTYPE_CONFIG_KEY,)
        immediate_path = (ENABLE_RWKV_IMMEDIATE_CONFIG_KEY,)
        if _path_value(current, active_path) != _path_value(updated, active_path):
            labels.append(config_option_by_path(active_path).label)
        else:
            labels.append(config_option_by_path(immediate_path).label)

    return tuple(labels)


def _effective_live_review_enabled(config: Mapping[str, object]) -> bool:
    immediate = config.get(ENABLE_RWKV_IMMEDIATE_CONFIG_KEY, True)
    live_review = config.get(ACTIVE_REVIEW_PROTOTYPE_CONFIG_KEY, True)
    return bool(immediate) and bool(live_review)


def checkpoint_rebuild_required_option_labels(
    current: Mapping[str, object],
    updated: Mapping[str, object],
) -> tuple[str, ...]:
    labels: list[str] = []
    for option in CONFIG_OPTIONS:
        if not option.checkpoint_rebuild_required:
            continue
        if _path_value(current, option.key_path) != _path_value(updated, option.key_path):
            labels.append(option.label)
    return tuple(labels)


def _path_value(config: Mapping[str, object], path: tuple[str, ...]):
    value: object = config
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value
