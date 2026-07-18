from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from aqt import mw
from aqt.qt import QAction, QMenu
from aqt.utils import qconnect, tooltip

from ..addon_config import (
    addon_config_for_mw,
    behavior_lab_enabled,
    card_info_rwkv_enabled,
)
from ..anki_api import find_cards, is_fsrs_enabled
from ..browser_card_info_runtime import (
    BrowserCardInfoRuntime,
    BrowserCardInfoRuntimeOwner,
)
from ..dataset_export import (
    load_review_data_for_checkpoint,
    open_checkpoint_runtime_from_load,
)
from ..review_rows import checkpoint_scope_cards_for_card_ids
from ..runtime import manager_for_mw, store_for_mw
from .checkpoint_failure import handle_checkpoint_failure, require_checkpoint_for_use
from .common import ProgressStage, run_with_progress_stages, show_fsrs_disabled
from .web_message import show_web_warning

BrowserLoadScope = Literal["all", "selection"]

_BROWSER_KEY_ATTRIBUTE = "_rwkvp_card_info_runtime_key"
_BROWSER_RUNTIME_OWNER = BrowserCardInfoRuntimeOwner()
_BROWSER_MENU_REFRESHERS: dict[object, Callable[[], None]] = {}


@dataclass(frozen=True)
class _BrowserRuntimeLoadResult:
    runtime: BrowserCardInfoRuntime
    card_count: int


class _EmptyBrowserSearchError(RuntimeError):
    pass


def add_browser_card_info_menu(browser) -> None:
    """Add the adaptive Browser RWKV menu.

    The menu is installed even when every Browser feature is currently disabled.
    Its actions remain hidden until configuration enables Card Info or Behavior
    Lab.  This lets non-restart-scoped curve changes take effect in Browser
    windows that are already open.
    """

    menu = QMenu("RWKV", browser)
    load_all_action = QAction("Load All", browser)
    load_selection_action = QAction("Load Selection", browser)
    load_all_action.setToolTip(
        "Load the complete RWKV checkpoint state for Card Info until this Browser closes."
    )
    load_selection_action.setToolTip(
        "Load RWKV state for all cards in the Browser's current search results."
    )
    qconnect(
        load_all_action.triggered,
        lambda _checked=False, browser=browser: load_browser_card_info_runtime(
            browser,
            scope="all",
        ),
    )
    qconnect(
        load_selection_action.triggered,
        lambda _checked=False, browser=browser: load_browser_card_info_runtime(
            browser,
            scope="selection",
        ),
    )
    menu.addAction(load_all_action)
    menu.addAction(load_selection_action)
    behavior_lab_separator = menu.addSeparator()
    behavior_lab_action = _add_behavior_lab_action(menu, browser)
    browser_key = _browser_key(browser)

    def refresh_actions() -> None:
        config = addon_config_for_mw(mw)
        config_enabled = card_info_rwkv_enabled(config)
        behavior_lab_visible = behavior_lab_enabled(config)
        menu.menuAction().setVisible(config_enabled or behavior_lab_visible)
        load_all_action.setVisible(config_enabled)
        load_selection_action.setVisible(config_enabled)
        behavior_lab_separator.setVisible(config_enabled and behavior_lab_visible)
        behavior_lab_action.setVisible(behavior_lab_visible)
        manager = manager_for_mw(mw)
        checkpoint_status = manager.status()
        card_info_enabled = bool(
            config_enabled
            and mw.col
            and is_fsrs_enabled(mw.col)
            and manager.has_checkpoint
            and checkpoint_status != "legacy"
        )
        behavior_lab_action_enabled = bool(
            behavior_lab_visible
            and mw.col
            and is_fsrs_enabled(mw.col)
            and manager.has_checkpoint
            and checkpoint_status not in {"legacy", "invalid", "stale_cache"}
        )
        load_all_action.setEnabled(card_info_enabled)
        load_selection_action.setEnabled(card_info_enabled)
        behavior_lab_action.setEnabled(behavior_lab_action_enabled)

    _BROWSER_MENU_REFRESHERS[browser_key] = refresh_actions
    qconnect(menu.aboutToShow, refresh_actions)
    refresh_actions()
    _append_browser_menu(browser, menu)

    destroyed = getattr(browser, "destroyed", None)
    if destroyed is not None:
        def browser_destroyed(*_args, browser_key=browser_key) -> None:
            _BROWSER_MENU_REFRESHERS.pop(browser_key, None)
            close_browser_card_info_runtime(
                browser_key,
                asynchronously=True,
            )

        qconnect(
            destroyed,
            browser_destroyed,
        )
def _add_behavior_lab_action(menu: QMenu, browser) -> QAction:
    action = QAction("Open Behavior Lab with Selection...", browser)
    action.setToolTip(
        "Open a disposable RWKV simulation using the selected Browser cards."
    )
    qconnect(
        action.triggered,
        lambda _checked=False, browser=browser: _show_browser_behavior_lab(browser),
    )
    menu.addAction(action)
    return action


def _show_browser_behavior_lab(browser) -> None:
    if not behavior_lab_enabled(addon_config_for_mw(mw)):
        return
    from .behavior_lab_dialog import show_behavior_lab

    selected_cards = getattr(browser, "selected_cards", None)
    try:
        card_ids = tuple(int(card_id) for card_id in selected_cards())
    except Exception:
        card_ids = ()
    show_behavior_lab(parent=browser, initial_card_ids=card_ids)


def refresh_browser_card_info_menu_state() -> None:
    """Refresh menus already attached to open Browser windows."""

    for browser_key, refresher in tuple(_BROWSER_MENU_REFRESHERS.items()):
        try:
            refresher()
        except RuntimeError:
            # Qt may delete a Browser/menu before its destroyed callback reaches
            # Python. Drop only that stale GUI record.
            _BROWSER_MENU_REFRESHERS.pop(browser_key, None)


def load_browser_card_info_runtime(browser, *, scope: BrowserLoadScope) -> None:
    if mw.col is None:
        return
    if not card_info_rwkv_enabled(addon_config_for_mw(mw)):
        show_web_warning(
            "RWKV Card Info features are disabled in RWKV Settings.",
            title="RWKV Card Info",
            parent=browser,
        )
        return
    if not is_fsrs_enabled(mw.col):
        show_fsrs_disabled(browser)
        return

    manager = manager_for_mw(mw)
    if not require_checkpoint_for_use(browser, manager=manager):
        return

    browser_key = _browser_key(browser)
    search = _active_browser_search(browser) if scope == "selection" else None
    token, previous_runtime = _BROWSER_RUNTIME_OWNER.begin_load(browser_key)
    _close_runtime(previous_runtime, asynchronously=True)
    store = store_for_mw(mw)

    def load_op(col, progress, _previous):
        card_ids: tuple[int, ...] | None = None
        if scope == "selection":
            progress.update(0, 1, "Finding cards in the current Browser search")
            card_ids = tuple(find_cards(col, search or ""))
            if not card_ids:
                raise _EmptyBrowserSearchError(
                    "No cards match the Browser search captured by Load Selection."
                )
            progress.update(1, 1, f"Found {len(card_ids):,} cards")

        review_load = load_review_data_for_checkpoint(
            col,
            store,
            manager,
            progress,
            allow_incremental=True,
        )
        scope_cards = (
            None
            if card_ids is None
            else checkpoint_scope_cards_for_card_ids(
                card_ids,
                review_load.review_data,
            )
        )
        _readiness, lease = open_checkpoint_runtime_from_load(
            manager,
            review_load,
            progress,
            scope_cards=scope_cards,
        )
        runtime = BrowserCardInfoRuntime(
            lease,
            visible_card_ids=None if card_ids is None else frozenset(card_ids),
        )
        try:
            progress.check_cancelled()
        except BaseException:
            runtime.close()
            raise
        return _BrowserRuntimeLoadResult(
            runtime=runtime,
            card_count=(len(review_load.review_data.cards) if card_ids is None else len(card_ids)),
        )

    def success(result: _BrowserRuntimeLoadResult) -> None:
        if not _BROWSER_RUNTIME_OWNER.complete_load(token, result.runtime):
            _close_runtime(result.runtime, asynchronously=True)
            return
        scope_label = "complete checkpoint" if scope == "all" else "search selection"
        tooltip(
            f"Loaded RWKV Card Info state for {result.card_count:,} cards "
            f"from the {scope_label}.",
            parent=browser,
        )
        _refresh_open_browser_card_info(browser)

    def failure(exception: Exception) -> None:
        _BROWSER_RUNTIME_OWNER.cancel_load(token)
        if isinstance(exception, _EmptyBrowserSearchError):
            show_web_warning(str(exception), title="RWKV Card Info", parent=browser)
            return
        handle_checkpoint_failure(
            exception,
            lambda: load_browser_card_info_runtime(browser, scope=scope),
            parent=browser,
        )

    def cancelled() -> None:
        _BROWSER_RUNTIME_OWNER.cancel_load(token)

    scope_label = "complete state" if scope == "all" else "search-result state"
    run_with_progress_stages(
        parent=browser,
        title="RWKV Card Info",
        label=f"Loading RWKV {scope_label}...",
        stages=[ProgressStage(load_op, uses_collection=True)],
        on_success=success,
        on_failure=failure,
        on_cancel=cancelled,
    )


def active_browser_card_info_runtime():
    return _BROWSER_RUNTIME_OWNER.active_runtime()


def close_browser_card_info_runtime(
    browser_key: object | None = None,
    *,
    asynchronously: bool,
) -> None:
    runtime = (
        _BROWSER_RUNTIME_OWNER.close_all()
        if browser_key is None
        else _BROWSER_RUNTIME_OWNER.close_browser(browser_key)
    )
    _close_runtime(runtime, asynchronously=asynchronously)


def _close_runtime(runtime, *, asynchronously: bool) -> None:
    if runtime is None:
        return

    def close() -> None:
        with suppress(Exception):
            runtime.close()

    taskman = getattr(mw, "taskman", None)
    run_in_background = getattr(taskman, "run_in_background", None)
    if asynchronously and callable(run_in_background):
        run_in_background(close, lambda _future: None, uses_collection=False)
    else:
        close()


def _browser_key(browser) -> object:
    key = getattr(browser, _BROWSER_KEY_ATTRIBUTE, None)
    if key is None:
        key = object()
        setattr(browser, _BROWSER_KEY_ATTRIBUTE, key)
    return key


def _active_browser_search(browser) -> str:
    active_search = getattr(browser, "_lastSearchTxt", None)
    if active_search is not None:
        return str(active_search)
    current_search = getattr(browser, "current_search", None)
    return str(current_search()) if callable(current_search) else ""


def _append_browser_menu(browser, menu: QMenu) -> None:
    menubar = getattr(browser.form, "menubar", None)
    if menubar is not None:
        menubar.addMenu(menu)


def _refresh_open_browser_card_info(browser) -> None:
    refresh = getattr(browser, "_update_card_info", None)
    if callable(refresh):
        with suppress(Exception):
            refresh()
