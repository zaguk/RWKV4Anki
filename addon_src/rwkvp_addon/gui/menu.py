from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter

from aqt import gui_hooks, mw
from aqt.qt import QAction, QMenu
from aqt.utils import qconnect, tooltip

from ..addon_config import (
    SHOW_CHECKPOINT_REBUILD_CONFIRMATION_CONFIG_KEY,
    active_review_prototype_enabled,
    addon_config_for_mw,
    behavior_lab_enabled,
    calculate_forgetting_curves,
    configured_model_id,
    curve_rescheduling_enabled,
    process_many_mode,
    rwkv_immediate_enabled,
    show_checkpoint_rebuild_confirmation,
    write_addon_config_for_mw,
)
from ..anki_api import (
    active_card_search_for_deck,
    current_deck_id,
    is_filtered_deck,
    is_fsrs_enabled,
    profile_name,
)
from ..checkpoint_manager import InconsistentCheckpointError
from ..checkpoint_storage import (
    BENCHMARK_PROCESS_MANY_SPEED_TOLERANCE,
    UNMEASURED_PROCESS_MANY_REVIEWS_PER_MINUTE,
    RustCheckpointStorageEstimate,
    estimate_checkpoint_processing_time_from_benchmark,
    estimate_rust_checkpoint_processing_time,
    estimate_rust_checkpoint_storage,
    format_processing_time_range,
    format_storage_bytes,
)
from ..dataset_export import (
    initialize_or_update_checkpoint_from_load,
    load_review_data_for_checkpoint,
)
from ..initial_setup import initial_setup_seen_for_mw
from ..process_speed_cache import (
    CHECKPOINT_SPEED_CPU_MEASUREMENT,
    CHECKPOINT_SPEED_MATCHING_MEASUREMENT,
    CHECKPOINT_SPEED_UNMEASURED,
    CHECKPOINT_SPEED_WITHOUT_CURVES,
    CheckpointBuildSpeedEstimate,
    cache_completed_checkpoint_build,
    checkpoint_build_speed_estimate,
    process_many_speed_cache_path,
)
from ..runtime import manager_for_mw, reset_runtime, store_for_mw
from ..rwkv_modes import (
    RETRIEVABILITY_MODES,
    RetrievabilityMode,
    enabled_prediction_modes_for_retrievability_mode,
    mode_spec,
)
from ..windows_native_cache import (
    prune_windows_native_cache_at_startup,
    schedule_windows_native_cache_removal_after_exit,
)
from ._standalone_overlay import create_standalone_overlay_host
from .browser_card_info import (
    add_browser_card_info_menu,
    close_browser_card_info_runtime,
    refresh_browser_card_info_menu_state,
)
from .card_info import handle_card_info_rwkv_message, inject_card_info_rwkv_rows
from .checkpoint_failure import (
    handle_checkpoint_failure,
    require_checkpoint_for_use,
    set_checkpoint_rebuild_handler,
)
from .common import ProgressStage, run_with_progress_stages, show_fsrs_disabled
from .web_message import (
    WebMessageButton,
    ask_web_choice,
    ask_web_confirmation,
    show_web_warning,
)

_rwkv_menu: QMenu | None = None
_checkpoint_action: QAction | None = None
_checkpoint_rebuild_action: QAction | None = None
_checkpoint_manage_action: QAction | None = None
_checkpoint_integrity_action: QAction | None = None
_live_review_history_action: QAction | None = None
_behavior_lab_action: QAction | None = None
_fsrs_disabled_action: QAction | None = None
_rwkv_normal_menu_actions: list[QAction] = []
_mode_evaluate_menu_actions: dict[RetrievabilityMode, QAction] = {}
_mode_evaluate_actions: dict[RetrievabilityMode, QAction] = {}
_mode_calibration_actions: dict[RetrievabilityMode, QAction] = {}
_mode_retrievability_actions: dict[RetrievabilityMode, QAction] = {}
_mode_reschedule_actions: dict[RetrievabilityMode, QAction] = {}
_active_review_prototype_actions: list[QAction] = []
_mode_section_separator: QAction | None = None
_curve_section_separator: QAction | None = None

_RWKV_MENU_TITLE = "RWKV"
_RWKV_MENU_HEALTHY_TITLE = "🟢 RWKV"
_RWKV_MENU_MISSING_TITLE = "🟠 RWKV"
_RWKV_MENU_REBUILD_TITLE = "🔴 RWKV"
_INITIAL_SETUP_DELAY_MS = 1_500
_initial_setup_pending_profile: str | None = None
_initial_setup_offered_profile: str | None = None


@dataclass(frozen=True)
class _CheckpointBuildPlan:
    review_load: object
    storage_estimate: RustCheckpointStorageEstimate | None


def _add_top_level_mode_actions(menu: QMenu, mode: RetrievabilityMode) -> None:
    retrievability_action = QAction(_retrievability_action_label(mode), mw)
    qconnect(
        retrievability_action.triggered,
        lambda _checked=False, mode=mode: show_retrievability_dialog(mode=mode),
    )
    menu.addAction(retrievability_action)

    evaluate_menu = QMenu(_evaluate_menu_label(mode), menu)
    menu.addMenu(evaluate_menu)

    evaluate_action = QAction("Evaluate...", mw)
    qconnect(
        evaluate_action.triggered,
        lambda _checked=False, mode=mode: show_evaluate_dialog(mode),
    )
    evaluate_menu.addAction(evaluate_action)

    calibration_action = QAction("Calibration Graph...", mw)
    qconnect(
        calibration_action.triggered,
        lambda _checked=False, mode=mode: show_calibration_dialog(mode=mode),
    )
    evaluate_menu.addAction(calibration_action)

    if mode == RetrievabilityMode.FORGETTING_CURVE:
        reschedule_action = QAction("Reschedule Cards", mw)
        qconnect(
            reschedule_action.triggered,
            lambda _checked=False: show_curve_reschedule_all_cards(),
        )
        menu.addAction(reschedule_action)
        _mode_reschedule_actions[mode] = reschedule_action

    _mode_evaluate_menu_actions[mode] = evaluate_menu.menuAction()
    _mode_evaluate_actions[mode] = evaluate_action
    _mode_calibration_actions[mode] = calibration_action
    _mode_retrievability_actions[mode] = retrievability_action


def setup_menu() -> None:
    global _rwkv_menu, _checkpoint_action, _checkpoint_rebuild_action
    global _checkpoint_manage_action, _checkpoint_integrity_action
    global _live_review_history_action, _behavior_lab_action
    global _fsrs_disabled_action, _rwkv_normal_menu_actions
    global _mode_section_separator, _curve_section_separator
    if _rwkv_menu is not None:
        return

    _rwkv_menu = QMenu(_RWKV_MENU_TITLE, mw)
    _install_top_level_rwkv_menu(_rwkv_menu)
    qconnect(_rwkv_menu.aboutToShow, _refresh_menu_and_schedule_initial_setup)
    _register_addon_config_action()
    set_checkpoint_rebuild_handler(run_checkpoint_rebuild_action)

    checkpoint_manage_menu = QMenu("Manage Checkpoint", _rwkv_menu)
    _checkpoint_manage_action = checkpoint_manage_menu.menuAction()
    _rwkv_menu.addMenu(checkpoint_manage_menu)

    _checkpoint_integrity_action = QAction("Check History Consistency", mw)
    qconnect(
        _checkpoint_integrity_action.triggered,
        lambda _checked=False: run_checkpoint_integrity_action(),
    )
    checkpoint_manage_menu.addAction(_checkpoint_integrity_action)

    _checkpoint_action = QAction("Update Checkpoint", mw)
    qconnect(_checkpoint_action.triggered, run_checkpoint_action)
    checkpoint_manage_menu.addAction(_checkpoint_action)

    _checkpoint_rebuild_action = QAction("Rebuild Checkpoint", mw)
    qconnect(
        _checkpoint_rebuild_action.triggered,
        lambda _checked=False: run_manual_checkpoint_rebuild_action(),
    )
    checkpoint_manage_menu.addAction(_checkpoint_rebuild_action)

    _live_review_history_action = QAction("Live Review History...", mw)
    qconnect(
        _live_review_history_action.triggered,
        lambda _checked=False: show_live_review_history(),
    )
    _rwkv_menu.addAction(_live_review_history_action)

    _behavior_lab_action = QAction("Behavior Lab...", mw)
    qconnect(
        _behavior_lab_action.triggered,
        lambda _checked=False: show_behavior_lab_dialog(),
    )
    _rwkv_menu.addAction(_behavior_lab_action)

    _mode_section_separator = _rwkv_menu.addSeparator()
    for mode in RETRIEVABILITY_MODES:
        if mode != RETRIEVABILITY_MODES[0]:
            separator = _rwkv_menu.addSeparator()
            if mode == RetrievabilityMode.FORGETTING_CURVE:
                _curve_section_separator = separator
        _add_top_level_mode_actions(_rwkv_menu, mode)

    _rwkv_normal_menu_actions = list(_menu_actions(_rwkv_menu))
    _fsrs_disabled_action = QAction("Enable FSRS to use RWKV", mw)
    _fsrs_disabled_action.setEnabled(False)
    _set_action_visible(_fsrs_disabled_action, False)
    _rwkv_menu.addAction(_fsrs_disabled_action)

    _rwkv_menu.addSeparator()
    settings_action = QAction("Settings", mw)
    qconnect(
        settings_action.triggered,
        lambda _checked=False: _show_addon_config_dialog(),
    )
    _rwkv_menu.addAction(settings_action)

    gui_hooks.deck_browser_will_show_options_menu.append(add_deck_options_action)
    if hasattr(gui_hooks, "browser_menus_did_init"):
        gui_hooks.browser_menus_did_init.append(add_browser_card_info_menu)
        # This hook is lightweight and checks the setting when each Browser
        # menu is created, so curve rescheduling can be toggled without restart.
        gui_hooks.browser_menus_did_init.append(add_browser_cards_action)
    # Card Info and Browser visibility is configuration-driven at call time.
    # Register these lightweight hooks unconditionally so enabling curve-only
    # Card Info does not unexpectedly require restarting Anki.
    if hasattr(gui_hooks, "webview_did_inject_style_into_page"):
        gui_hooks.webview_did_inject_style_into_page.append(inject_card_info_rwkv_rows)
    if hasattr(gui_hooks, "webview_did_receive_js_message"):
        gui_hooks.webview_did_receive_js_message.append(handle_card_info_rwkv_message)
    if _active_review_prototype_enabled():
        from .live_review_bridge import install_live_review_bridge_hooks

        install_live_review_bridge_hooks()
    with suppress(Exception):
        prune_windows_native_cache_at_startup()
    _register_addon_lifecycle_hooks()
    gui_hooks.profile_did_open.append(refresh_menu_state)
    gui_hooks.profile_did_open.append(_schedule_initial_setup)
    gui_hooks.profile_will_close.append(_on_profile_will_close)
    refresh_menu_state()
    # Add-ons normally register profile_did_open before Anki opens a profile,
    # but reloads and newer startup sequences can initialize us afterward.
    # The scheduler is idempotent, so eagerly checking closes that race.
    _schedule_initial_setup()


def _install_top_level_rwkv_menu(menu: QMenu) -> None:
    menubar = getattr(mw.form, "menubar", None)
    if menubar is None:
        mw.form.menuTools.addMenu(menu)
        return
    before_action = None
    help_menu = getattr(mw.form, "menuHelp", None)
    menu_action = getattr(help_menu, "menuAction", None)
    if callable(menu_action):
        before_action = menu_action()
    if before_action is not None and hasattr(menubar, "insertMenu"):
        menubar.insertMenu(before_action, menu)
        return
    menubar.addMenu(menu)


def _register_addon_config_action() -> None:
    addon_manager = getattr(mw, "addonManager", None)
    set_config_action = getattr(addon_manager, "setConfigAction", None)
    if not callable(set_config_action):
        return
    set_config_action(__name__, _show_addon_config_dialog)


def _register_addon_lifecycle_hooks() -> None:
    if hasattr(gui_hooks, "addons_dialog_will_delete_addons"):
        gui_hooks.addons_dialog_will_delete_addons.append(_on_addons_dialog_will_delete_addons)
    if hasattr(gui_hooks, "addon_manager_will_install_addon"):
        gui_hooks.addon_manager_will_install_addon.append(_on_addon_manager_will_install_addon)


def _on_addons_dialog_will_delete_addons(_dialog, addon_ids: list[str]) -> None:
    if _installed_addon_id() not in {str(addon_id) for addon_id in addon_ids}:
        return
    _release_all_runtime_state()
    with suppress(Exception):
        schedule_windows_native_cache_removal_after_exit()


def _on_addon_manager_will_install_addon(_manager, addon_id: str) -> None:
    if str(addon_id) == _installed_addon_id():
        _release_all_runtime_state()


def _on_profile_will_close() -> None:
    global _initial_setup_offered_profile, _initial_setup_pending_profile
    _initial_setup_pending_profile = None
    _initial_setup_offered_profile = None
    _release_all_runtime_state()


def _schedule_initial_setup() -> None:
    """Open Guided Setup once, after profile startup and any progress dialog settle."""

    global _initial_setup_offered_profile, _initial_setup_pending_profile
    if mw.col is None:
        return
    opened_profile = profile_name(mw)
    if (
        _initial_setup_pending_profile == opened_profile
        or _initial_setup_offered_profile == opened_profile
    ):
        return
    try:
        if initial_setup_seen_for_mw(mw):
            return
    except Exception:
        # A read-only or unavailable profile data folder should not make Anki
        # startup fail. Setup will be offered again on a later profile open.
        return
    _initial_setup_pending_profile = opened_profile

    def show_if_still_current() -> None:
        global _initial_setup_offered_profile, _initial_setup_pending_profile
        if _initial_setup_pending_profile == opened_profile:
            _initial_setup_pending_profile = None
        if mw.col is None or profile_name(mw) != opened_profile:
            return
        try:
            if initial_setup_seen_for_mw(mw):
                return
        except Exception:
            return
        from .config_dialog import show_initial_setup_dialog

        try:
            show_initial_setup_dialog()
        except Exception:
            # Do not break profile startup. Opening the RWKV menu or reopening
            # the profile will retry while the seen marker remains absent.
            return
        _initial_setup_offered_profile = opened_profile

    mw.progress.single_shot(
        _INITIAL_SETUP_DELAY_MS,
        show_if_still_current,
        True,
    )


def _refresh_menu_and_schedule_initial_setup() -> None:
    refresh_menu_state()
    _schedule_initial_setup()


def _release_all_runtime_state() -> None:
    close_browser_card_info_runtime(asynchronously=False)
    reset_runtime()


def _installed_addon_id() -> str:
    addon_manager = getattr(mw, "addonManager", None)
    resolver = getattr(addon_manager, "addonFromModule", None)
    if not callable(resolver):
        resolver = getattr(addon_manager, "addon_from_module", None)
    return str(resolver(__name__)) if callable(resolver) else __name__.split(".", 1)[0]


def _show_addon_config_dialog() -> None:
    from .config_dialog import show_config_dialog

    show_config_dialog()


def _retrievability_action_label(mode: RetrievabilityMode) -> str:
    if mode == RetrievabilityMode.FORGETTING_CURVE:
        return "Forgetting Curve Retrievability..."
    return "Immediate Retrievability..."


def show_behavior_lab_dialog() -> None:
    if not _behavior_lab_enabled():
        return
    from .behavior_lab_dialog import show_behavior_lab

    show_behavior_lab(parent=mw)


def _evaluate_menu_label(mode: RetrievabilityMode) -> str:
    if mode == RetrievabilityMode.FORGETTING_CURVE:
        return "Evaluate Forgetting Curve"
    return "Evaluate Immediate"


def refresh_menu_state() -> None:
    actions = [
        _checkpoint_action,
        _checkpoint_rebuild_action,
        _checkpoint_manage_action,
        _checkpoint_integrity_action,
        _live_review_history_action,
        _behavior_lab_action,
        _fsrs_disabled_action,
        *_mode_evaluate_menu_actions.values(),
        *_mode_evaluate_actions.values(),
        *_mode_calibration_actions.values(),
        *_mode_retrievability_actions.values(),
        *_mode_reschedule_actions.values(),
        *_active_review_prototype_actions,
    ]
    if (
        any(action is None for action in actions)
        or len(_mode_evaluate_menu_actions) != len(RETRIEVABILITY_MODES)
        or len(_mode_evaluate_actions) != len(RETRIEVABILITY_MODES)
        or len(_mode_calibration_actions) != len(RETRIEVABILITY_MODES)
        or len(_mode_retrievability_actions) != len(RETRIEVABILITY_MODES)
    ):
        return

    refresh_browser_card_info_menu_state()

    fsrs = is_fsrs_enabled(mw.col) if mw.col else False
    if not fsrs:
        _set_fsrs_disabled_menu_visible(True)
        _update_rwkv_menu_status("fsrs_disabled")
        return

    _set_fsrs_disabled_menu_visible(False)
    behavior_lab_available = _behavior_lab_enabled()
    _set_action_visible(_behavior_lab_action, behavior_lab_available)
    immediate_available = _rwkv_immediate_enabled()
    curves_available = _forgetting_curve_features_enabled()
    _set_retrievability_feature_visibility(
        immediate=immediate_available,
        curves=curves_available,
    )

    manager = manager_for_mw(mw)
    status = manager.status()
    _update_rwkv_menu_status(status)
    if status == "missing":
        _checkpoint_action.setText("Initialize Checkpoint")
    elif status == "partial":
        _checkpoint_action.setText("Resume Checkpoint")
    else:
        _checkpoint_action.setText("Update Checkpoint")

    ready = fsrs and status in {"valid", "invalid", "partial", "stale_cache", "legacy"}
    _checkpoint_manage_action.setEnabled(fsrs)
    _set_action_visible(
        _checkpoint_action,
        status not in {"invalid", "stale_cache", "legacy"},
    )
    _checkpoint_rebuild_action.setEnabled(fsrs and manager.has_checkpoint)
    _checkpoint_integrity_action.setEnabled(ready and status != "legacy")
    _live_review_history_action.setEnabled(fsrs)
    _behavior_lab_action.setEnabled(
        behavior_lab_available and ready and status not in {"legacy", "invalid", "stale_cache"}
    )
    for mode in RETRIEVABILITY_MODES:
        mode_ready = ready and _retrievability_mode_enabled(mode)
        for action in (
            _mode_evaluate_menu_actions[mode],
            _mode_evaluate_actions[mode],
            _mode_calibration_actions[mode],
            _mode_retrievability_actions[mode],
        ):
            action.setEnabled(mode_ready)
    for action in _mode_reschedule_actions.values():
        rescheduling_enabled = curves_available and _curve_rescheduling_enabled()
        _set_action_visible(action, rescheduling_enabled)
        action.setEnabled(ready and rescheduling_enabled)
    for action in _active_review_prototype_actions:
        action.setEnabled(ready and _active_review_prototype_available())


def _update_rwkv_menu_status(status: str) -> None:
    if _rwkv_menu is None:
        return
    title = _rwkv_menu_title_for_status(status)
    set_title = getattr(_rwkv_menu, "setTitle", None)
    if callable(set_title):
        set_title(title)
    elif hasattr(_rwkv_menu, "title"):
        _rwkv_menu.title = title
    menu_action = _rwkv_menu.menuAction()
    menu_action.setText(title)
    set_tooltip = getattr(menu_action, "setToolTip", None)
    if callable(set_tooltip):
        set_tooltip(_rwkv_menu_tooltip_for_status(status))


def _rwkv_menu_title_for_status(status: str) -> str:
    if status in {"invalid", "stale_cache", "legacy"}:
        return _RWKV_MENU_REBUILD_TITLE
    if status == "missing":
        return _RWKV_MENU_MISSING_TITLE
    if status in {"valid", "partial"}:
        return _RWKV_MENU_HEALTHY_TITLE
    return _RWKV_MENU_TITLE


def _rwkv_menu_tooltip_for_status(status: str) -> str:
    if status == "fsrs_disabled":
        return "Enable FSRS to use RWKV."
    if status == "invalid":
        return "RWKV checkpoint is inconsistent; rebuild required."
    if status == "stale_cache":
        return "RWKV checkpoint cache is stale; rebuild required."
    if status == "legacy":
        return "RWKV checkpoint format is obsolete; rebuild required."
    if status == "missing":
        return "RWKV checkpoint is not initialized."
    return "RWKV checkpoint is ready."


def _set_fsrs_disabled_menu_visible(visible: bool) -> None:
    if _fsrs_disabled_action is not None:
        _set_action_visible(_fsrs_disabled_action, visible)
    for action in _rwkv_normal_menu_actions:
        _set_action_visible(action, not visible)


def _set_retrievability_feature_visibility(*, immediate: bool, curves: bool) -> None:
    if _mode_section_separator is not None:
        _set_action_visible(_mode_section_separator, immediate or curves)

    immediate_mode = RetrievabilityMode.IMMEDIATE
    immediate_actions = (
        _mode_evaluate_menu_actions.get(immediate_mode),
        _mode_evaluate_actions.get(immediate_mode),
        _mode_calibration_actions.get(immediate_mode),
        _mode_retrievability_actions.get(immediate_mode),
    )
    for action in immediate_actions:
        if action is not None:
            _set_action_visible(action, immediate)

    if _curve_section_separator is not None:
        _set_action_visible(_curve_section_separator, immediate and curves)
    curve_mode = RetrievabilityMode.FORGETTING_CURVE
    actions = (
        _mode_evaluate_menu_actions.get(curve_mode),
        _mode_evaluate_actions.get(curve_mode),
        _mode_calibration_actions.get(curve_mode),
        _mode_retrievability_actions.get(curve_mode),
    )
    for action in actions:
        if action is not None:
            _set_action_visible(action, curves)


def _set_action_visible(action: QAction, visible: bool) -> None:
    set_visible = getattr(action, "setVisible", None)
    if callable(set_visible):
        set_visible(bool(visible))


def _menu_actions(menu: QMenu) -> list[QAction]:
    actions = getattr(menu, "actions", None)
    if callable(actions):
        return list(actions())
    if actions is not None:
        return list(actions)
    return []


def run_checkpoint_action() -> None:
    _run_checkpoint_action(parent=mw)


def run_checkpoint_rebuild_action(parent=None) -> None:
    _run_checkpoint_action(parent=parent or mw, force_rebuild=True, confirm_rebuild=False)


def run_manual_checkpoint_rebuild_action() -> None:
    if not is_fsrs_enabled(mw.col):
        show_fsrs_disabled(mw)
        return
    store = store_for_mw(mw)
    manager = manager_for_mw(mw)
    if not manager.has_checkpoint:
        show_web_warning("Initialize an RWKV checkpoint first.", title="RWKV", parent=mw)
        return
    _run_checkpoint_build_workflow(
        parent=mw,
        store=store,
        manager=manager,
        force_rebuild=True,
        confirm_rebuild=False,
        cancellation_message="RWKV checkpoint rebuild cancelled.",
    )


def run_checkpoint_integrity_action() -> None:
    if not is_fsrs_enabled(mw.col):
        show_fsrs_disabled(mw)
        return
    store = store_for_mw(mw)
    manager = manager_for_mw(mw)
    if not manager.has_checkpoint:
        show_web_warning("Initialize an RWKV checkpoint first.", title="RWKV", parent=mw)
        return

    def export_op(col, progress, _previous):
        return load_review_data_for_checkpoint(
            col,
            store,
            manager,
            progress,
            force_export=True,
        )

    def integrity_op(_col, progress, previous):
        result = manager.check_integrity(previous.review_data.rows, progress)
        manager.remember_review_data(
            previous.review_data,
            latest_collection_review_id=getattr(previous, "latest_review_id", None),
        )
        return result

    def success(result) -> None:
        refresh_menu_state()
        status = manager.status()
        if status == "stale_cache":
            message = (
                "RWKV checkpoint review history verified; processed "
                f"{result.processed_review_count} reviews. Cached evaluation data "
                "still needs rebuild."
            )
        elif status == "invalid":
            message = (
                "RWKV checkpoint review history verified; processed "
                f"{result.processed_review_count} reviews. The checkpoint is still "
                "invalid for the current configuration; rebuild it before using "
                "predictions."
            )
        else:
            message = (
                "RWKV checkpoint integrity verified; processed "
                f"{result.processed_review_count} reviews."
            )
        tooltip(
            message,
            parent=mw,
        )

    def failure(exc: Exception) -> None:
        if isinstance(exc, InconsistentCheckpointError):
            refresh_menu_state()
            ask_web_confirmation(
                parent=mw,
                title="RWKV",
                message=(
                    "RWKV checkpoint history check failed. Rebuild it from the full "
                    "review history now? This can take a long time."
                ),
                confirm_label="Rebuild Checkpoint",
                destructive=True,
                on_result=lambda confirmed: (
                    run_checkpoint_rebuild_action(mw) if confirmed else None
                ),
            )
            return
        handle_checkpoint_failure(
            exc,
            run_checkpoint_integrity_action,
            parent=mw,
            on_decline=refresh_menu_state,
        )

    run_with_progress_stages(
        parent=mw,
        title="RWKV Checkpoint Integrity",
        label="Preparing full review history",
        stages=[
            ProgressStage(export_op, uses_collection=True),
            ProgressStage(integrity_op, uses_collection=False),
        ],
        on_success=success,
        on_failure=failure,
    )


def _run_checkpoint_action(
    *,
    parent,
    force_rebuild: bool | None = None,
    confirm_rebuild: bool = True,
) -> None:
    if not is_fsrs_enabled(mw.col):
        show_fsrs_disabled(parent)
        return
    store = store_for_mw(mw)
    manager = manager_for_mw(mw)
    status = manager.status()
    rebuild = (
        bool(force_rebuild)
        if force_rebuild is not None
        else status in {"missing", "invalid", "stale_cache", "legacy"}
    )
    if confirm_rebuild and status in {"invalid", "stale_cache"}:
        reason = (
            "The current RWKV checkpoint cache is missing data needed by this add-on version."
            if status == "stale_cache"
            else "The current RWKV checkpoint is marked inconsistent."
        )
        message_html = f"<p>{reason} Rebuild it from the full review history?</p>"
        continue_label = (
            "Continue with Invalid Checkpoint"
            if status == "invalid"
            else "Continue with Stale Cache"
        )
        ask_web_choice(
            parent=parent,
            title="RWKV Checkpoint",
            message=f"{reason} Rebuild it from the full review history?",
            trusted_message_html=message_html,
            choices=(
                WebMessageButton("cancel", continue_label, "destructive"),
                WebMessageButton("rebuild", "Rebuild Checkpoint", "primary"),
            ),
            initial_outcome="rebuild",
            on_result=lambda outcome: (
                _run_checkpoint_action(
                    parent=parent,
                    force_rebuild=force_rebuild,
                    confirm_rebuild=False,
                )
                if outcome == "rebuild"
                else None
            ),
        )
        return

    if rebuild:
        _run_checkpoint_build_workflow(
            parent=parent,
            store=store,
            manager=manager,
            force_rebuild=force_rebuild,
            confirm_rebuild=confirm_rebuild,
            cancellation_message=(
                "RWKV checkpoint initialization cancelled."
                if status == "missing"
                else "RWKV checkpoint rebuild cancelled."
            ),
        )
        return

    def export_op(col, progress, _previous):
        return load_review_data_for_checkpoint(
            col,
            store,
            manager,
            progress,
            force_export=rebuild,
        )

    def checkpoint_op(_col, progress, previous):
        return _run_checkpoint_operation(
            manager,
            store,
            previous,
            progress,
            rebuild=rebuild,
            force_save=True,
        )

    def success(result) -> None:
        refresh_menu_state()
        save_note = (
            " Writing checkpoint/cache to disk in the background."
            if manager.save_in_progress
            else ""
        )
        tooltip(
            f"RWKV checkpoint {result.status}; processed "
            f"{result.processed_review_count} reviews.{save_note}",
            parent=parent,
        )

    run_with_progress_stages(
        parent=parent,
        title="RWKV Checkpoint",
        label="Preparing review history",
        stages=[
            ProgressStage(export_op, uses_collection=True),
            ProgressStage(checkpoint_op, uses_collection=False),
        ],
        on_success=success,
        on_failure=lambda exc: handle_checkpoint_failure(
            exc,
            lambda: _run_checkpoint_action(
                parent=parent,
                force_rebuild=force_rebuild,
                confirm_rebuild=confirm_rebuild,
            ),
            parent=parent,
            on_decline=refresh_menu_state,
        ),
    )


def _run_checkpoint_build_workflow(
    *,
    parent,
    store,
    manager,
    force_rebuild: bool | None,
    confirm_rebuild: bool,
    cancellation_message: str,
) -> None:
    rebuild = bool(manager.has_checkpoint)
    show_build_estimate = not rebuild or show_checkpoint_rebuild_confirmation(
        addon_config_for_mw(mw)
    )

    def retry() -> None:
        _run_checkpoint_action(
            parent=parent,
            force_rebuild=force_rebuild,
            confirm_rebuild=confirm_rebuild,
        )

    try:
        workflow = create_standalone_overlay_host(
            parent=parent,
            title="RWKV Checkpoint",
            status_title=("Preparing Checkpoint Rebuild" if rebuild else "Preparing Checkpoint"),
            status_message="Preparing Anki review history",
            status_state="loading",
            size=(620, 360),
            intro=None,
            inline_progress=True,
        )
    except Exception as exc:
        handle_checkpoint_failure(
            exc,
            retry,
            parent=parent,
            on_decline=refresh_menu_state,
        )
        return

    def cancel_workflow() -> None:
        workflow.reject()

    def cancel_confirmation() -> None:
        workflow.reject()
        tooltip(cancellation_message, parent=parent)

    def fail_workflow(exc: Exception) -> None:
        workflow.reject()
        handle_checkpoint_failure(
            exc,
            retry,
            parent=parent,
            on_decline=refresh_menu_state,
        )

    def export_op(col, progress, _previous):
        return load_review_data_for_checkpoint(
            col,
            store,
            manager,
            progress,
            force_export=True,
        )

    def size_op(_col, progress, review_load):
        progress.check_cancelled()
        progress.update(0, 1, "Calculating expected RWKV checkpoint size")
        expected_storage = getattr(manager, "expected_checkpoint_storage", None)
        storage_estimate = (
            expected_storage(review_load.review_data.rows)
            if callable(expected_storage)
            else estimate_rust_checkpoint_storage(review_load.review_data.rows)
        )
        progress.check_cancelled()
        progress.update(1, 1, "Calculated expected RWKV checkpoint size")
        return _CheckpointBuildPlan(
            review_load=review_load,
            storage_estimate=storage_estimate,
        )

    def start_build(plan: _CheckpointBuildPlan) -> None:
        review_load = plan.review_load

        def checkpoint_op(_col, progress, _previous):
            return _run_checkpoint_operation(
                manager,
                store,
                review_load,
                progress,
                rebuild=True,
                force_save=True,
            )

        def success(result) -> None:
            workflow.accept()
            refresh_menu_state()
            save_note = (
                " Writing checkpoint/cache to disk in the background."
                if manager.save_in_progress
                else ""
            )
            tooltip(
                f"RWKV checkpoint {result.status}; processed "
                f"{result.processed_review_count} reviews.{save_note}",
                parent=parent,
            )

        run_with_progress_stages(
            parent=workflow,
            title="RWKV Checkpoint",
            label="Building RWKV checkpoint",
            stages=[ProgressStage(checkpoint_op, uses_collection=False)],
            on_success=success,
            on_failure=fail_workflow,
            on_cancel=cancel_workflow,
            cancelled_tooltip=cancellation_message,
            cancelled_tooltip_parent=parent,
        )

    def history_loaded(plan: _CheckpointBuildPlan) -> None:
        if not show_build_estimate:
            start_build(plan)
            return
        _confirm_checkpoint_build_estimate(
            plan.review_load.review_data,
            workflow,
            processing_speed=_selected_checkpoint_build_speed(store, manager),
            rebuild=rebuild,
            storage_estimate=plan.storage_estimate,
            on_show_rebuild_confirmation_changed=(
                _write_show_checkpoint_rebuild_confirmation if rebuild else None
            ),
            on_result=lambda confirmed: start_build(plan) if confirmed else cancel_confirmation(),
        )

    stages = [ProgressStage(export_op, uses_collection=True)]
    if show_build_estimate:
        stages.append(ProgressStage(size_op, uses_collection=False))

    def preparation_complete(payload) -> None:
        plan = (
            payload
            if isinstance(payload, _CheckpointBuildPlan)
            else _CheckpointBuildPlan(review_load=payload, storage_estimate=None)
        )
        history_loaded(plan)

    run_with_progress_stages(
        parent=workflow,
        title="RWKV Checkpoint",
        label="Preparing review history",
        stages=stages,
        on_success=preparation_complete,
        on_failure=fail_workflow,
        on_cancel=cancel_workflow,
        cancelled_tooltip=cancellation_message,
        cancelled_tooltip_parent=parent,
    )


def _run_checkpoint_operation(
    manager,
    store,
    review_load,
    progress,
    *,
    rebuild: bool,
    force_save: bool,
):
    started = perf_counter() if rebuild else None
    readiness = initialize_or_update_checkpoint_from_load(
        manager,
        review_load,
        progress,
        rebuild=rebuild,
        force_save=force_save,
    )
    result = readiness.checkpoint_result
    if started is not None:
        _cache_completed_checkpoint_build_speed(
            store,
            manager,
            review_load,
            result,
            duration_seconds=perf_counter() - started,
        )
    return result


def _cache_completed_checkpoint_build_speed(
    store,
    manager,
    review_load,
    result,
    *,
    duration_seconds: float,
) -> None:
    rows = getattr(getattr(review_load, "review_data", None), "rows", None)
    cache_dir = getattr(store, "cache_dir", None)
    model_id = getattr(manager, "model_id", None)
    mode = getattr(manager, "process_many_mode", None)
    return_curves = getattr(manager, "calculate_curves", None)
    if (
        rows is None
        or cache_dir is None
        or model_id is None
        or mode is None
        or return_curves is None
        or getattr(result, "status", None) != "valid"
    ):
        return
    review_count = len(rows)
    try:
        processed_review_count = int(result.processed_review_count)
    except (AttributeError, TypeError, ValueError):
        return
    if processed_review_count != review_count:
        return
    # This is a disposable estimate cache. A read-only cache must never turn a
    # successful checkpoint build into a user-visible failure.
    with suppress(OSError, TypeError, ValueError):
        cache_completed_checkpoint_build(
            process_many_speed_cache_path(cache_dir),
            model_id=str(model_id),
            return_curves=bool(return_curves),
            mode=str(mode),
            review_count=review_count,
            duration_seconds=duration_seconds,
        )


def _confirm_checkpoint_build_estimate(
    review_data,
    parent,
    *,
    processing_speed: CheckpointBuildSpeedEstimate | None = None,
    rebuild: bool = False,
    storage_estimate: RustCheckpointStorageEstimate | None = None,
    on_show_rebuild_confirmation_changed: Callable[[bool], None] | None = None,
    on_result: Callable[[bool], None],
) -> None:
    estimate = storage_estimate or estimate_rust_checkpoint_storage(review_data.rows)
    message = _checkpoint_build_estimate_html(
        estimate,
        review_count=len(review_data.rows),
        processing_speed=processing_speed or _unmeasured_checkpoint_build_speed(),
    )
    ask_web_confirmation(
        parent=parent,
        title="Ready to Rebuild" if rebuild else "Ready to Build",
        message="Review the estimate, then continue or cancel.",
        trusted_message_html=message,
        confirm_label="Start Rebuild" if rebuild else "Build Checkpoint",
        destructive=False,
        checkbox_label=("Show this confirmation before future rebuilds" if rebuild else None),
        checkbox_checked=True,
        on_checkbox_changed=(on_show_rebuild_confirmation_changed if rebuild else None),
        on_result=on_result,
    )


def _write_show_checkpoint_rebuild_confirmation(enabled: bool) -> None:
    config = addon_config_for_mw(mw)
    enabled = bool(enabled)
    if show_checkpoint_rebuild_confirmation(config) == enabled:
        return
    config[SHOW_CHECKPOINT_REBUILD_CONFIRMATION_CONFIG_KEY] = enabled
    write_addon_config_for_mw(mw, config)


def _checkpoint_build_estimate_html(
    estimate,
    *,
    review_count: int,
    processing_speed: CheckpointBuildSpeedEstimate,
) -> str:
    checkpoint_size = format_storage_bytes(estimate.estimated_checkpoint_bytes)
    processing_time = _checkpoint_processing_time_text(
        review_count,
        processing_speed=processing_speed,
    )
    processing_note = _checkpoint_processing_speed_note(processing_speed)
    return (
        '<div class="rwkv-checkpoint-estimate">'
        '<div class="rwkv-checkpoint-estimate__row">'
        "<span>Estimated Checkpoint Size:</span>"
        f"<b>{checkpoint_size}</b>"
        "</div>"
        '<div class="rwkv-checkpoint-estimate__row">'
        "<span>Estimated Processing Time:</span>"
        f"<b>{processing_time}</b>"
        "</div>"
        f"{processing_note}"
        '<p class="rwkv-checkpoint-estimate__note">'
        "The checkpoint uses approximately this much disk space and roughly the "
        "same amount of RAM while loaded."
        "</p>"
        "</div>"
    )


def _checkpoint_processing_time_text(
    review_count: int,
    *,
    processing_speed: CheckpointBuildSpeedEstimate,
) -> str:
    if processing_speed.basis == CHECKPOINT_SPEED_UNMEASURED:
        estimate = estimate_rust_checkpoint_processing_time(
            review_count,
            reviews_per_minute=int(processing_speed.reviews_per_minute),
        )
    else:
        estimate = estimate_checkpoint_processing_time_from_benchmark(
            review_count,
            processing_speed.reviews_per_minute,
            speed_tolerance=BENCHMARK_PROCESS_MANY_SPEED_TOLERANCE,
        )
    return format_processing_time_range(estimate)


def _checkpoint_processing_speed_note(speed: CheckpointBuildSpeedEstimate) -> str:
    if speed.basis == CHECKPOINT_SPEED_MATCHING_MEASUREMENT:
        return ""
    if speed.basis == CHECKPOINT_SPEED_CPU_MEASUREMENT:
        return (
            '<p class="rwkv-checkpoint-estimate__note">No GPU speed measurement is '
            "available for this build, so the estimate uses measured CPU Fast "
            "speed instead.</p>"
        )
    if speed.basis == CHECKPOINT_SPEED_WITHOUT_CURVES:
        measurement = speed.measurement
        mode_label = (
            "GPU"
            if measurement is not None and measurement.mode == "gpu"
            else "CPU Fast"
        )
        return (
            f'<p class="rwkv-checkpoint-estimate__note">No {mode_label} speed '
            "measurement with Forgetting Curves is available. This estimate uses "
            f"measured {mode_label} speed without curves, reduced by 25%.</p>"
        )
    if speed.basis == CHECKPOINT_SPEED_UNMEASURED:
        return (
            '<p class="rwkv-checkpoint-estimate__note"><strong>No State Building '
            "speed measurement is available.</strong> This conservative estimate "
            f"assumes <strong>{speed.reviews_per_minute:,.0f} reviews/minute</strong>; "
            "it is not based on a test of this computer. For a measured estimate, "
            "<strong>run Compare Modes under RWKV Settings → General → State "
            "Building Mode.</strong></p>"
        )
    raise ValueError(f"Unsupported checkpoint speed-estimate basis: {speed.basis!r}")


def _unmeasured_checkpoint_build_speed() -> CheckpointBuildSpeedEstimate:
    return CheckpointBuildSpeedEstimate(
        reviews_per_minute=float(UNMEASURED_PROCESS_MANY_REVIEWS_PER_MINUTE),
        basis=CHECKPOINT_SPEED_UNMEASURED,
    )


def _selected_checkpoint_build_speed(store, manager) -> CheckpointBuildSpeedEstimate:
    cache_dir = getattr(store, "cache_dir", None)
    if cache_dir is None:
        return _unmeasured_checkpoint_build_speed()
    config = addon_config_for_mw(mw)
    model_id = getattr(manager, "model_id", None)
    if model_id is None:
        model_id = configured_model_id(config)
    selected_mode = getattr(manager, "process_many_mode", None)
    if selected_mode is None:
        selected_mode = process_many_mode(config)
    return_curves = getattr(manager, "calculate_curves", None)
    if return_curves is None:
        return_curves = calculate_forgetting_curves(config)
    return checkpoint_build_speed_estimate(
        process_many_speed_cache_path(cache_dir),
        model_id=str(model_id),
        return_curves=bool(return_curves),
        mode=str(selected_mode),
    )


def show_evaluate_dialog(mode: RetrievabilityMode = RetrievabilityMode.IMMEDIATE) -> None:
    if not _require_retrievability_mode_enabled(mode, mw):
        return
    if not _require_ready():
        return
    from .evaluate_dialog import EvaluateDialog

    config = addon_config_for_mw(mw)
    spec = mode_spec(mode)
    prediction_modes = enabled_prediction_modes_for_retrievability_mode(mode, config)
    if not prediction_modes:
        return
    EvaluateDialog(
        mw,
        rwkv_mode=prediction_modes[0],
        title=spec.evaluate_title,
        rwkv_label=spec.evaluate_label,
    ).show()


def show_calibration_dialog(
    deck_id: int | None = None,
    *,
    mode: RetrievabilityMode = RetrievabilityMode.IMMEDIATE,
) -> None:
    if not _require_retrievability_mode_enabled(mode, mw):
        return
    if not _require_ready(deck_id):
        return
    from .calibration_dialog import CalibrationDialog

    initial_search = active_card_search_for_deck(mw.col, deck_id) if deck_id is not None else None
    CalibrationDialog(mw, initial_search=initial_search, mode=mode).show()


def show_retrievability_dialog(
    deck_id: int | None = None,
    *,
    mode: RetrievabilityMode = RetrievabilityMode.IMMEDIATE,
) -> None:
    if not _require_retrievability_mode_enabled(mode, mw):
        return
    if not _require_ready(deck_id):
        return
    from .retrievability_dialog import RetrievabilityDialog

    initial_search = active_card_search_for_deck(mw.col, deck_id) if deck_id is not None else None
    RetrievabilityDialog(mw, initial_search=initial_search, mode=mode).show()


def add_deck_options_action(menu: QMenu, deck_id: int) -> None:
    normal_deck = not is_filtered_deck(mw.col, deck_id)
    action_enabled = (
        normal_deck and is_fsrs_enabled(mw.col, deck_id) and manager_for_mw(mw).has_checkpoint
    )

    if _active_review_prototype_available():
        _add_deck_action(
            menu,
            "RWKV Live Session...",
            action_enabled,
            lambda _checked=False, deck_id=deck_id: show_active_review_prototype(deck_id),
        )

    for mode in RETRIEVABILITY_MODES:
        if not _retrievability_mode_enabled(mode):
            continue
        _add_deck_mode_menu(menu, deck_id, mode, action_enabled)


def _add_deck_mode_menu(
    parent_menu: QMenu,
    deck_id: int,
    mode: RetrievabilityMode,
    action_enabled: bool,
) -> None:
    mode_menu = QMenu(mode_spec(mode).deck_menu_title, parent_menu)
    parent_menu.addMenu(mode_menu)
    mode_menu.menuAction().setEnabled(action_enabled)

    _add_deck_action(
        mode_menu,
        "Generate Filtered Deck...",
        action_enabled,
        lambda _checked=False, deck_id=deck_id, mode=mode: show_filtered_deck_dialog(
            deck_id,
            mode=mode,
        ),
    )
    _add_deck_action(
        mode_menu,
        "Average Retrievability...",
        action_enabled,
        lambda _checked=False, deck_id=deck_id, mode=mode: show_retrievability_dialog(
            deck_id,
            mode=mode,
        ),
    )
    if mode == RetrievabilityMode.FORGETTING_CURVE:
        if not _curve_rescheduling_enabled():
            return
        _add_deck_action(
            mode_menu,
            "Reschedule Cards",
            action_enabled,
            lambda _checked=False, deck_id=deck_id: show_curve_reschedule_deck(deck_id),
        )


def _add_deck_action(
    menu: QMenu,
    text: str,
    enabled: bool,
    callback,
) -> QAction:
    action = QAction(text, menu)
    action.setEnabled(enabled)
    qconnect(action.triggered, callback)
    menu.addAction(action)
    return action


def show_filtered_deck_dialog(
    deck_id: int,
    *,
    mode: RetrievabilityMode = RetrievabilityMode.IMMEDIATE,
) -> None:
    if not _require_retrievability_mode_enabled(mode, mw):
        return
    if is_filtered_deck(mw.col, deck_id):
        spec = mode_spec(mode)
        show_web_warning(
            "Generated filtered decks can only be created from normal decks.",
            title=spec.warning_title,
            parent=mw,
        )
        return
    if not _require_ready(deck_id):
        return
    from .filtered_deck_dialog import FilteredDeckDialog

    FilteredDeckDialog(mw, deck_id, mode=mode).show()


def show_curve_reschedule_all_cards() -> None:
    if not _require_curve_rescheduling_enabled(mw):
        return
    if not _require_ready():
        return
    from .curve_reschedule import show_reschedule_all_cards

    show_reschedule_all_cards()


def show_curve_reschedule_deck(deck_id: int) -> None:
    if not _require_curve_rescheduling_enabled(mw):
        return
    if is_filtered_deck(mw.col, deck_id):
        show_web_warning(
            "RWKV Forgetting Curve rescheduling can only be started from a normal deck.",
            title=mode_spec(RetrievabilityMode.FORGETTING_CURVE).warning_title,
            parent=mw,
        )
        return
    if not _require_ready(deck_id):
        return
    from .curve_reschedule import show_reschedule_deck

    show_reschedule_deck(deck_id)


def add_browser_cards_action(browser) -> None:
    if not _curve_rescheduling_enabled():
        return
    action = QAction("RWKV Forgetting Curve: Reschedule Selected Cards", browser)
    action.setEnabled(is_fsrs_enabled(mw.col) and manager_for_mw(mw).has_checkpoint)
    qconnect(
        action.triggered,
        lambda _checked=False, browser=browser: show_curve_reschedule_browser_cards(browser),
    )
    if hasattr(browser.form.menu_Cards, "addSeparator"):
        browser.form.menu_Cards.addSeparator()
    browser.form.menu_Cards.addAction(action)


def show_curve_reschedule_browser_cards(browser) -> None:
    if not _require_curve_rescheduling_enabled(browser):
        return
    if not _require_ready():
        return
    from .curve_reschedule import show_reschedule_selected_browser_cards

    show_reschedule_selected_browser_cards(browser)


def show_active_review_prototype(deck_id: int | None = None) -> None:
    if not _active_review_prototype_enabled():
        return
    if not _live_review_bridge_hooks_installed():
        show_web_warning(
            "Restart Anki to finish enabling RWKV Live Session.",
            title="RWKV Live Session",
            parent=mw,
        )
        return
    if not _require_ready(deck_id):
        return
    from .live_review_dialog import LiveReviewDialog

    LiveReviewDialog(mw, current_deck_id(mw.col) if deck_id is None else deck_id).show()


def show_live_review_history() -> None:
    try:
        from .live_review_history_dialog import LiveReviewHistoryDialog

        dialog = LiveReviewHistoryDialog(mw, store_for_mw(mw))
        dialog.show()
    except Exception as exc:
        show_web_warning(
            f"Unable to open RWKV Live Review History: {exc}",
            title="RWKV",
            parent=mw,
        )


def _require_ready(deck_id: int | None = None) -> bool:
    if not is_fsrs_enabled(mw.col, deck_id):
        show_fsrs_disabled(mw)
        return False
    return require_checkpoint_for_use(mw, manager=manager_for_mw(mw))


def _curve_rescheduling_enabled() -> bool:
    return curve_rescheduling_enabled(addon_config_for_mw(mw))


def _active_review_prototype_enabled() -> bool:
    return active_review_prototype_enabled(addon_config_for_mw(mw))


def _live_review_bridge_hooks_installed() -> bool:
    live_bridge = sys.modules.get(f"{__package__}.live_review_bridge")
    check = getattr(live_bridge, "live_review_bridge_hooks_installed", None)
    return bool(check and check())


def _active_review_prototype_available() -> bool:
    return _active_review_prototype_enabled() and _live_review_bridge_hooks_installed()


def _behavior_lab_enabled() -> bool:
    return behavior_lab_enabled(addon_config_for_mw(mw))


def _forgetting_curve_features_enabled() -> bool:
    return calculate_forgetting_curves(addon_config_for_mw(mw))


def _rwkv_immediate_enabled() -> bool:
    return rwkv_immediate_enabled(addon_config_for_mw(mw))


def _retrievability_mode_enabled(mode: RetrievabilityMode) -> bool:
    if mode == RetrievabilityMode.FORGETTING_CURVE:
        return _forgetting_curve_features_enabled()
    return _rwkv_immediate_enabled()


def _require_retrievability_mode_enabled(mode: RetrievabilityMode, parent) -> bool:
    if _retrievability_mode_enabled(mode):
        return True
    message = (
        "Calculate Forgetting Curves is disabled in RWKV Settings."
        if mode == RetrievabilityMode.FORGETTING_CURVE
        else "RWKV Immediate is disabled in RWKV Settings."
    )
    show_web_warning(
        message,
        title=mode_spec(mode).warning_title,
        parent=parent,
    )
    return False


def _require_curve_rescheduling_enabled(parent) -> bool:
    if _curve_rescheduling_enabled():
        return True
    show_web_warning(
        "RWKV Forgetting Curve rescheduling is disabled in the add-on config.",
        title=mode_spec(RetrievabilityMode.FORGETTING_CURVE).warning_title,
        parent=parent,
    )
    return False
