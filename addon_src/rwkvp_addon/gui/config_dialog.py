from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any

from aqt import mw
from aqt.qt import QDialog, Qt, QVBoxLayout
from aqt.utils import qconnect
from aqt.webview import AnkiWebView

from ..addon_config import (
    CALCULATE_FORGETTING_CURVES_CONFIG_KEY,
    EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY,
    LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    PREDICT_MANY_MODE_CONFIG_KEY,
    PROCESS_MANY_MODE_CONFIG_KEY,
    available_model_ids,
    default_addon_config,
    normalized_addon_config,
    predict_many_batch_size,
    write_addon_config_for_mw,
)
from ..anki_api import profile_name
from ..config_html import (
    BRIDGE_PREFIX,
    config_from_web_values,
    merge_config_option_values,
    render_config_html,
    sanitize_choice_values,
    visible_config_options,
)
from ..config_options import (
    CONFIG_CHOICE_VALUES,
    ConfigOption,
    RestartRequirementContext,
    checkpoint_rebuild_required_option_labels,
    restart_required_option_labels,
)
from ..initial_setup import mark_initial_setup_seen_for_mw
from ..profile_store import ProfileStore
from ..review_type_normalization import (
    FilteredReviewNormalizationPolicy,
    filtered_review_normalization_policy_for_store,
    filtered_review_policy_from_config_values,
    profile_config_values_for_filtered_review_policy,
    strip_filtered_review_profile_config_values,
    write_filtered_review_normalization_policy,
)
from ..rwkv_performance_modes import (
    available_predict_many_modes,
    available_process_many_modes,
)
from ..speed_test import speed_test_checkpoint_is_usable
from ..vendor_bootstrap import rwkv_gpu_available
from .web_message import (
    WebMessageOwner,
    WebMessageSession,
    WebMessageSpec,
    show_web_info,
    show_web_warning,
)
from .web_progress import WebProgressOwner, WebProgressSession

_STANDALONE_SETUP_DIALOG_SIZE = (804, 720)


class _WebActionButton:
    """ButtonLike adapter for progress actions launched by the settings webview."""

    def __init__(
        self,
        dialog: RWKVConfigDialog,
        name: str,
        *,
        action: str,
        enabled: bool,
    ) -> None:
        self.dialog = dialog
        self.name = str(name)
        self.action = str(action)
        self._enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self._enabled and not self.dialog._cleaned_up

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if self.dialog._cleaned_up:
            return
        setter = (
            "rwkvSetSpeedTestEnabled" if self.action == "speed-test" else "rwkvSetComparisonEnabled"
        )
        script = (
            f"window.{setter} && "
            f"window.{setter}({json.dumps(self.name)}, "
            f"{str(self._enabled).lower()});"
        )
        with suppress(RuntimeError):
            self.dialog.web.eval(script)


class RWKVConfigDialog(QDialog):
    def __init__(self, parent=None, *, initial_setup: bool = False) -> None:
        dialog_parent = parent or mw
        super().__init__(dialog_parent)
        # The Qt parent may be Anki's non-modal Add-ons window. Runtime and
        # configuration APIs must always resolve through the main window.
        self._mw = mw
        self._cleaned_up = False
        self._standalone_setup_host = bool(initial_setup)
        self._initial_setup_requested = bool(initial_setup)
        self._predict_gpu_available = rwkv_gpu_available("predict")
        self._process_gpu_available = rwkv_gpu_available("process")
        self._visible_options = visible_config_options(
            predict_gpu_available=self._predict_gpu_available
        )
        self._choices = _choices_for_options(
            self._visible_options,
            predict_gpu_available=self._predict_gpu_available,
            process_gpu_available=self._process_gpu_available,
        )
        self._speed_test_checkpoint_usable = _speed_test_checkpoint_available(self._mw)
        self._profile_store = ProfileStore.for_profile(profile_name(self._mw))
        self._filtered_review_policy = filtered_review_normalization_policy_for_store(
            self._profile_store
        )
        initial = normalized_addon_config(_read_config(self._mw))
        initial.update(
            profile_config_values_for_filtered_review_policy(self._filtered_review_policy)
        )
        self._config = sanitize_choice_values(
            initial,
            options=self._visible_options,
            choices=self._choices,
        )
        self._draft = dict(self._config)
        self._apply_enabled = False
        self._last_apply_notice: str | None = None
        self._render_generation = 0
        self._profile_progress_hook_registered = False

        self.setWindowTitle("RWKV Guided Setup" if self._standalone_setup_host else "RWKV Settings")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.web = AnkiWebView(self, title=self.windowTitle())
        self.web.requiresCol = False
        self.web.set_bridge_command(self._on_bridge_command, self)
        self.web.setMinimumHeight(580)
        layout.addWidget(self.web, 1)

        self._progress_owner = WebProgressOwner(
            eval_js=self.web.eval,
            generation=lambda: self._render_generation,
            is_closed=lambda: self._cleaned_up,
        )
        self._message_owner = WebMessageOwner(
            eval_js=self.web.eval,
            generation=lambda: self._render_generation,
            is_closed=lambda: self._cleaned_up,
            can_start=lambda: not self._progress_owner.active,
        )

        self._speed_test_buttons = {
            name: _WebActionButton(
                self,
                name,
                action="speed-test",
                enabled=self._speed_test_checkpoint_usable,
            )
            for name in (
                "state-building",
                "predictions",
                "curves",
                "live-predictions",
            )
        }
        self._comparison_buttons = {
            name: _WebActionButton(
                self,
                name,
                action="comparison",
                enabled=True,
            )
            for name in ("models", "deleted-reviews")
        }
        from .setup_wizard_controller import SetupWizardController

        self._setup_wizard = SetupWizardController(
            self,
            predict_gpu_available=self._predict_gpu_available,
            process_gpu_available=self._process_gpu_available,
            model_ids=tuple(self._choices.get((MODEL_CONFIG_KEY,), ())),
        )
        parent_destroyed = getattr(dialog_parent, "destroyed", None)
        if parent_destroyed is not None:
            with suppress(AttributeError, RuntimeError, TypeError):
                qconnect(parent_destroyed, self._on_progress_parent_destroyed)
        self._register_profile_progress_hook()
        self._render()
        if self._initial_setup_requested:
            # Anki queues web.eval() until its web channel reports domDone. Open
            # from Python instead of racing the asynchronously-created `pycmd`
            # bridge from an inline zero-delay JavaScript timer.
            self._open_initial_setup()
        if self._standalone_setup_host:
            self.setFixedSize(*_STANDALONE_SETUP_DIALOG_SIZE)
        else:
            self.resize(940, 720)

    def _render(self) -> None:
        if self._progress_owner.active:
            raise RuntimeError("cannot rerender RWKV Settings while progress is active")
        self._render_generation += 1
        self.web.stdHtml(
            render_config_html(
                self._draft,
                choices=self._choices,
                predict_gpu_available=self._predict_gpu_available,
                process_gpu_available=self._process_gpu_available,
                checkpoint_usable=self._speed_test_checkpoint_usable,
                initial_setup=self._standalone_setup_host,
                apply_enabled=self._apply_enabled,
                generation=self._render_generation,
            ),
            context=self,
        )
        self._message_owner.document_rerendered(self._render_generation)

    def _on_bridge_command(self, command: str):
        if not command.startswith(BRIDGE_PREFIX):
            return None
        if self._cleaned_up:
            return None
        action = ""
        action_name = ""
        try:
            payload = json.loads(command.removeprefix(BRIDGE_PREFIX))
            if not isinstance(payload, dict):
                raise ValueError("The RWKV Settings command must be an object.")
            action = str(payload.get("action", ""))
            if action not in {
                "change",
                "speed-test",
                "comparison",
                "setup",
                "restore-defaults",
                "apply",
                "ok",
                "cancel",
                "progress-cancel",
                "message-response",
                "message-checkbox-change",
            }:
                raise ValueError(f"Unknown RWKV Settings action: {action!r}.")
            # The embedded setup overlay is a modal transaction. Ignore stale
            # or keyboard-triggered commands from the underlying settings page
            # until the wizard has either kept or discarded its draft.
            if self._progress_owner.active and action != "progress-cancel":
                return None
            if self._message_owner.active and action not in {
                "message-response",
                "message-checkbox-change",
            }:
                return None
            if self._setup_wizard.active and action not in {
                "setup",
                "message-response",
                "message-checkbox-change",
            }:
                return None
            setup_event = str(payload.get("event", "")) if action == "setup" else ""
            # Webview clicks can already be queued when a terminal wizard event
            # rerenders the page. A duplicate finish/keep event must not replay
            # the stale underlying form values over the accepted wizard draft.
            if action == "setup" and not self._setup_wizard.active and setup_event != "open":
                return None
            values = payload.get("values", {})
            if not isinstance(values, Mapping):
                raise ValueError("RWKV Settings values must be an object.")
            draft_actions = {"change", "speed-test", "comparison", "apply", "ok"}
            should_merge_values = action in draft_actions or (
                action == "setup" and setup_event == "open" and not self._setup_wizard.active
            )
            if should_merge_values:
                self._draft = normalized_addon_config(
                    config_from_web_values(
                        self._draft,
                        values,
                        options=self._visible_options,
                        choices=self._choices,
                    )
                )
                self._set_apply_enabled(self._draft != self._config)
            if action == "speed-test":
                action_name = str(payload.get("test", ""))
                self._run_speed_test(action_name)
            elif action == "comparison":
                action_name = str(payload.get("comparison", ""))
                self._run_comparison(action_name)
            elif action == "restore-defaults":
                self.restore_defaults()
            elif action == "apply":
                if self._apply_button_enabled():
                    self.apply()
            elif action == "ok":
                self.accept()
            elif action == "cancel":
                self.reject()
            elif action == "progress-cancel":
                token = payload.get("token")
                if isinstance(token, bool) or not isinstance(token, int):
                    raise ValueError("Progress cancellation requires an operation token.")
                self._progress_owner.request_cancel(token)
            elif action == "message-response":
                token = payload.get("token")
                outcome = payload.get("outcome")
                if isinstance(token, bool) or not isinstance(token, int):
                    raise ValueError("Message response requires an operation token.")
                if not isinstance(outcome, str):
                    raise ValueError("Message response requires an outcome.")
                self._message_owner.respond(token, outcome)
            elif action == "message-checkbox-change":
                token = payload.get("token")
                checked = payload.get("checked")
                if isinstance(token, bool) or not isinstance(token, int):
                    raise ValueError("Message checkbox change requires an operation token.")
                if not isinstance(checked, bool):
                    raise ValueError("Message checkbox change requires a checked state.")
                self._message_owner.update_checkbox(token, checked)
            elif action == "setup":
                initial_launch = (
                    setup_event == "open"
                    and str(payload.get("value", "")).strip().lower() == "initial"
                )
                if initial_launch:
                    self._initial_setup_requested = False
                self._setup_wizard.handle(
                    setup_event,
                    "initial" if initial_launch else payload.get("value"),
                )
        except Exception as exc:
            # Lazy GUI imports can observe old modules still resident in Anki's
            # Python process immediately after an add-on update.  Never let an
            # action error escape the webview bridge: Anki's global exception
            # dialog can otherwise leave a window-modal workflow owning input.
            self._restore_web_action(action, action_name)
            # The only commands accepted while an overlay is active are that
            # overlay's own control commands. A malformed/stale control must not
            # try to stack a second message over progress or another message.
            if self._progress_owner.active or self._message_owner.active:
                return None
            show_web_warning(
                _settings_action_error_message(exc),
                title=self.windowTitle(),
                parent=self,
            )
        return None

    def _restore_web_action(self, action: str, name: str) -> None:
        buttons = (
            self._speed_test_buttons
            if action == "speed-test"
            else self._comparison_buttons
            if action == "comparison"
            else None
        )
        if buttons is None:
            return
        button = buttons.get(name)
        if button is not None:
            button.setEnabled(True)

    def _run_speed_test(self, name: str) -> None:
        button = self._speed_test_buttons.get(name)
        if button is None or not button.isEnabled():
            return
        from .speed_test_dialog import (
            show_curve_calculation_speed_test,
            show_live_prediction_speed_test,
            show_predict_many_speed_test,
            show_process_many_speed_test,
        )

        if name == "state-building":
            show_process_many_speed_test(
                parent=self,
                button=button,
                gpu_available=self._process_gpu_available,
                calculate_curves=bool(self._draft[CALCULATE_FORGETTING_CURVES_CONFIG_KEY]),
            )
            return
        if name == "predictions":
            modes = available_predict_many_modes(
                gpu_available=self._predict_gpu_available,
            )
            show_predict_many_speed_test(
                parent=self,
                button=button,
                batch_sizes={mode: predict_many_batch_size(self._draft, mode) for mode in modes},
                gpu_available=self._predict_gpu_available,
            )
            return
        if name == "curves":
            show_curve_calculation_speed_test(
                parent=self,
                button=button,
                mode=str(self._draft[PROCESS_MANY_MODE_CONFIG_KEY]),
            )
            return
        if name == "live-predictions":
            mode = str(self._draft[PREDICT_MANY_MODE_CONFIG_KEY])
            show_live_prediction_speed_test(
                parent=self,
                button=button,
                mode=mode,
                card_count=int(self._draft[LIVE_REVIEW_PREDICTION_REFRESH_LIMIT_CONFIG_KEY]),
                batch_size=predict_many_batch_size(self._draft, mode),
            )

    def _run_comparison(self, name: str) -> None:
        button = self._comparison_buttons.get(name)
        if button is None or not button.isEnabled():
            return
        from .state_comparison_dialog import (
            show_deleted_reviews_comparison,
            show_models_comparison,
        )

        include_deleted = not bool(self._draft[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY])
        if name == "models":
            show_models_comparison(
                parent=self,
                button=button,
                model_ids=tuple(self._choices.get((MODEL_CONFIG_KEY,), ())),
                current_model_id=str(self._draft[MODEL_CONFIG_KEY]),
                include_deleted_reviews=include_deleted,
            )
            return
        if name == "deleted-reviews":
            show_deleted_reviews_comparison(
                parent=self,
                button=button,
                model_id=str(self._draft[MODEL_CONFIG_KEY]),
                current_includes_deleted_reviews=include_deleted,
            )

    def apply(self, *, notify_restart: bool = True) -> bool:
        self._last_apply_notice = None
        # The dialog is modeless, so merge only its known option values onto the
        # latest persisted config instead of overwriting another writer's keys.
        persisted_policy = filtered_review_normalization_policy_for_store(self._profile_store)
        persisted = normalized_addon_config(_read_config(self._mw))
        persisted.update(profile_config_values_for_filtered_review_policy(persisted_policy))
        updated = normalized_addon_config(
            merge_config_option_values(
                persisted,
                self._draft,
                options=self._visible_options,
            )
        )
        updated_policy = filtered_review_policy_from_config_values(
            updated,
            fallback=persisted_policy,
        )
        if _model_change_is_blocked(self._mw, persisted, updated):
            show_web_warning(
                "Stop the active RWKV operation or Live Session before changing "
                "the RWKV model. Your settings have not been changed.",
                title="RWKV Settings",
                parent=self,
            )
            return False
        if _curve_calculation_change_is_blocked(self._mw, persisted, updated):
            show_web_warning(
                "Stop the active Browser load, RWKV operation, or Live Session "
                "before changing curve calculation. Your settings have not been changed.",
                title="RWKV Settings",
                parent=self,
            )
            return False
        if _manager_setting_change_is_blocked(self._mw, persisted, updated):
            show_web_warning(
                "Stop the active RWKV operation, Live Session, or checkpoint save "
                "before changing State Building Mode or deleted-card history. "
                "Your settings have not been changed.",
                title="RWKV Settings",
                parent=self,
            )
            return False
        if _filtered_review_policy_change_is_blocked(
            self._mw,
            persisted_policy,
            updated_policy,
        ):
            show_web_warning(
                "Stop the active Browser load, RWKV operation, Live Session, or "
                "checkpoint save before changing Filtered-review interpretation. "
                "Your settings have not been changed.",
                title="RWKV Settings",
                parent=self,
            )
            return False
        restart_labels = restart_required_option_labels(
            persisted,
            updated,
            context=self._restart_requirement_context(),
        )
        rebuild_labels = checkpoint_rebuild_required_option_labels(persisted, updated)
        policy_changed = updated_policy != persisted_policy
        checkpoint_exists = _checkpoint_exists(self._mw)
        if policy_changed:
            write_filtered_review_normalization_policy(
                self._profile_store,
                updated_policy,
            )
        write_addon_config_for_mw(
            self._mw,
            strip_filtered_review_profile_config_values(updated),
        )
        if policy_changed:
            from ..runtime import reset_runtime

            reset_runtime()
        updated.update(profile_config_values_for_filtered_review_policy(updated_policy))
        self._filtered_review_policy = updated_policy
        self._config = updated
        self._draft = dict(updated)
        self._set_apply_enabled(False)
        _refresh_menu_state()
        notices: list[str] = []
        if restart_labels:
            rendered = "\n".join(f"• {label}" for label in restart_labels)
            notices.append("Restart Anki for these changes to take effect:\n\n" + rendered)
        if rebuild_labels and checkpoint_exists:
            rendered = "\n".join(f"• {label}" for label in rebuild_labels)
            notices.append(
                "Rebuild the RWKV checkpoint before using it with these changes:\n\n"
                + rendered
                + "\n\nUse RWKV > Manage Checkpoint > Rebuild Checkpoint."
            )
        if notices and notify_restart:
            self._last_apply_notice = "\n\n".join(notices)
            show_web_info(
                self._last_apply_notice,
                title="RWKV Settings",
                parent=self,
            )
        elif notices:
            self._last_apply_notice = "\n\n".join(notices)
        return True

    def accept(self) -> None:
        if self._progress_owner.active:
            self._progress_owner.request_active_cancel()
            return
        if self._message_owner.active:
            self._message_owner.request_escape()
            return
        if self._setup_wizard.active:
            self._setup_wizard.request_exit()
            return
        if self._initial_setup_requested:
            if not self._record_initial_setup_seen():
                return
            self._initial_setup_requested = False
        post_close_notice = None
        if self._apply_button_enabled():
            if not self.apply(notify_restart=False):
                return
            post_close_notice = self._last_apply_notice
        self._cleanup_webview()
        super().accept()
        if post_close_notice:
            self._mw.progress.single_shot(
                50,
                lambda: show_web_info(
                    post_close_notice,
                    title="RWKV Settings",
                    parent=self._mw,
                ),
                True,
            )

    def reject(self) -> None:
        if self._progress_owner.active:
            self._progress_owner.request_active_cancel()
            return
        if self._message_owner.active:
            self._message_owner.request_escape()
            return
        if self._setup_wizard.active:
            self._setup_wizard.request_exit()
            return
        if self._initial_setup_requested:
            if not self._record_initial_setup_seen():
                return
            self._initial_setup_requested = False
        self._cleanup_webview()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._progress_owner.active:
            self._progress_owner.request_active_cancel()
            ignore = getattr(event, "ignore", None)
            if callable(ignore):
                ignore()
            return
        if self._message_owner.active:
            self._message_owner.request_escape()
            ignore = getattr(event, "ignore", None)
            if callable(ignore):
                ignore()
            return
        super().closeEvent(event)

    def start_web_progress(
        self,
        *,
        title: str,
        label: str,
        schedule_on_main: Callable[[Callable[[], None]], None],
        on_cancel: Callable[[], None],
        on_finished: Callable[[], None] | None = None,
    ) -> WebProgressSession:
        if self._message_owner.active:
            raise RuntimeError("cannot show progress while a message is active")
        return self._progress_owner.start(
            title=title,
            label=label,
            schedule_on_main=schedule_on_main,
            on_cancel=on_cancel,
            on_finished=on_finished,
        )

    def start_web_message(
        self,
        spec: WebMessageSpec,
        *,
        on_result: Callable[[str], None],
        on_checkbox_changed: Callable[[bool], None] | None = None,
    ) -> WebMessageSession:
        return self._message_owner.start(
            spec,
            on_result=on_result,
            on_checkbox_changed=on_checkbox_changed,
        )

    def restore_defaults(self) -> None:
        defaults = normalized_addon_config(default_addon_config())
        defaults.update(
            profile_config_values_for_filtered_review_policy(
                FilteredReviewNormalizationPolicy(
                    enabled=True,
                    cutoff_review_id=self._filtered_review_policy.cutoff_review_id,
                )
            )
        )
        self._draft = sanitize_choice_values(
            defaults,
            options=self._visible_options,
            choices=self._choices,
        )
        self._set_apply_enabled(self._draft != self._config)
        self._render()

    def _cleanup_webview(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._progress_owner.shutdown()
        self._message_owner.shutdown()
        self._remove_profile_progress_hook()
        self._setup_wizard.shutdown()
        self.web.cleanup()

    def _register_profile_progress_hook(self) -> None:
        try:
            from aqt import gui_hooks
        except (AttributeError, ImportError):
            return
        hook = getattr(gui_hooks, "profile_will_close", None)
        if hook is None:
            return
        hook.append(self._on_profile_will_close)
        self._profile_progress_hook_registered = True

    def _remove_profile_progress_hook(self) -> None:
        if not self._profile_progress_hook_registered:
            return
        self._profile_progress_hook_registered = False
        try:
            from aqt import gui_hooks
        except (AttributeError, ImportError):
            return
        hook = getattr(gui_hooks, "profile_will_close", None)
        if hook is not None:
            with suppress(ValueError):
                hook.remove(self._on_profile_will_close)

    def _on_profile_will_close(self) -> None:
        self._cleanup_webview()
        super().reject()

    def _on_progress_parent_destroyed(self, *_args: object) -> None:
        self._cleanup_webview()

    def _accept_setup_config(
        self,
        setup_config: Mapping[str, Any],
        *,
        render: bool = True,
    ) -> None:
        """Merge one completed wizard transaction into the unsaved settings draft."""

        self._draft = normalized_addon_config(
            merge_config_option_values(
                self._draft,
                setup_config,
                options=self._visible_options,
            )
        )
        self._draft = sanitize_choice_values(
            self._draft,
            options=self._visible_options,
            choices=self._choices,
        )
        self._set_apply_enabled(self._draft != self._config)
        if render:
            self._render()

    def _save_initial_setup_config(self, setup_config: Mapping[str, Any]) -> bool:
        """Persist a completed first-run transaction before offering state build."""

        self._accept_setup_config(setup_config, render=False)
        return self.apply(notify_restart=False)

    def _restart_requirement_context(self) -> RestartRequirementContext:
        return _current_restart_requirement_context()

    def _checkpoint_exists(self) -> bool:
        return _checkpoint_exists(self._mw)

    def _record_initial_setup_seen(self) -> bool:
        try:
            mark_initial_setup_seen_for_mw(self._mw)
        except Exception as exc:
            show_web_warning(
                "RWKV could not record that Guided Setup was shown. The setup window "
                "may appear again the next time this profile opens.\n\n"
                f"{exc}",
                title="RWKV Setup",
                parent=self,
            )
            return False
        return True

    def _finish_initial_setup_window(self, *, build_state: bool) -> None:
        """Close first-run Guided Setup and optionally launch checkpoint construction."""

        self._cleanup_webview()
        super().accept()
        if not build_state:
            return

        def build() -> None:
            from .menu import run_checkpoint_action

            run_checkpoint_action()

        self._mw.progress.single_shot(0, build, True)

    def _open_initial_setup(self) -> None:
        """Open the one-time wizard through Anki's queued webview lifecycle."""

        self._initial_setup_requested = False
        self.setWindowTitle("RWKV Guided Setup")
        self._setup_wizard.open(initial_launch=True)

    def _set_setup_active(self, active: bool) -> None:
        """Keep the outer settings controls from bypassing the wizard transaction."""

        if self._standalone_setup_host:
            return
        script = (
            "window.rwkvSetSettingsFooterActive && "
            f"window.rwkvSetSettingsFooterActive({str(not active).lower()});"
        )
        with suppress(RuntimeError):
            self.web.eval(script)

    def _set_apply_enabled(self, enabled: bool) -> None:
        self._apply_enabled = bool(enabled)
        if self._cleaned_up:
            return
        script = (
            "window.rwkvSetSettingsApplyEnabled && "
            f"window.rwkvSetSettingsApplyEnabled({str(self._apply_enabled).lower()});"
        )
        with suppress(RuntimeError):
            self.web.eval(script)

    def _apply_button_enabled(self) -> bool:
        return self._apply_enabled


_open_config_dialogs: set[RWKVConfigDialog] = set()
_open_initial_setup_dialogs: set[RWKVConfigDialog] = set()


def show_config_dialog(*, initial_setup: bool = False) -> RWKVConfigDialog:
    if initial_setup:
        return show_initial_setup_dialog()
    if _open_config_dialogs:
        return _activate_dialog(next(iter(_open_config_dialogs)))
    return _show_dialog(
        parent=_active_dialog_parent(),
        initial_setup=False,
        registry=_open_config_dialogs,
    )


def show_initial_setup_dialog() -> RWKVConfigDialog:
    """Show first-run Guided Setup without opening or mounting RWKV Settings."""

    if _open_initial_setup_dialogs:
        return _activate_dialog(next(iter(_open_initial_setup_dialogs)))
    return _show_dialog(
        parent=mw,
        initial_setup=True,
        registry=_open_initial_setup_dialogs,
    )


def _show_dialog(
    *,
    parent,
    initial_setup: bool,
    registry: set[RWKVConfigDialog],
) -> RWKVConfigDialog:
    dialog = RWKVConfigDialog(parent, initial_setup=initial_setup)
    registry.add(dialog)

    def finished(_result: int) -> None:
        registry.discard(dialog)
        dialog._cleanup_webview()
        with suppress(RuntimeError):
            dialog.setWindowModality(Qt.WindowModality.NonModal)
        with suppress(RuntimeError):
            dialog.deleteLater()

    qconnect(dialog.finished, finished)
    # Match Anki's own Add-ons ConfigEditor: do not enter a nested QDialog
    # event loop from AddonsDialog.onConfig().
    dialog.show()
    return dialog


def _activate_dialog(dialog: RWKVConfigDialog) -> RWKVConfigDialog:
    with suppress(RuntimeError):
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
    return dialog


def _active_dialog_parent():
    try:
        active = mw.app.activeWindow()
    except Exception:
        active = None
    return active or mw


def _read_config(parent) -> dict[str, Any]:
    addon_manager = getattr(parent, "addonManager", None)
    if addon_manager is None:
        return {}
    addon = addon_manager.addonFromModule(__name__)
    return dict(addon_manager.getConfig(addon) or {})


def _choices_for_options(
    options: Sequence[ConfigOption],
    *,
    predict_gpu_available: bool,
    process_gpu_available: bool,
) -> dict[tuple[str, ...], tuple[str, ...]]:
    choices: dict[tuple[str, ...], tuple[str, ...]] = {}
    for option in options:
        if option.value_type != "choice":
            continue
        if option.key_path == ("model",):
            value = available_model_ids()
        elif option.key_path == (PREDICT_MANY_MODE_CONFIG_KEY,):
            value = available_predict_many_modes(gpu_available=predict_gpu_available)
        elif option.key_path == (PROCESS_MANY_MODE_CONFIG_KEY,):
            value = available_process_many_modes(gpu_available=process_gpu_available)
        else:
            value = CONFIG_CHOICE_VALUES.get(option.key_path, ())
        choices[option.key_path] = tuple(value)
    return choices


def _refresh_menu_state() -> None:
    try:
        from .menu import refresh_menu_state
    except ImportError:
        return
    refresh_menu_state()


def _speed_test_checkpoint_available(parent) -> bool:
    try:
        from ..runtime import manager_for_mw

        return speed_test_checkpoint_is_usable(manager_for_mw(parent))
    except Exception:
        return False


def _settings_action_error_message(exc: Exception) -> str:
    if isinstance(exc, ImportError):
        return (
            "RWKV Settings could not load all of the code needed for this action. "
            "This can happen when the add-on is replaced while Anki still has the "
            "previous version loaded. Restart Anki and try again. If the problem "
            "continues after a restart, reinstall the latest add-on package.\n\n"
            f"Technical details: {exc}"
        )
    return str(exc) or exc.__class__.__name__


def _model_change_is_blocked(
    parent,
    current: Mapping[str, Any],
    updated: Mapping[str, Any],
) -> bool:
    if str(current.get("model") or "") == str(updated.get("model") or ""):
        return False
    try:
        from ..runtime import manager_for_mw

        manager = manager_for_mw(parent)
        return bool(getattr(manager, "runtime_scope_active", False))
    except Exception:
        return False


def _curve_calculation_change_is_blocked(
    parent,
    current: Mapping[str, Any],
    updated: Mapping[str, Any],
) -> bool:
    current_value = bool(current.get(CALCULATE_FORGETTING_CURVES_CONFIG_KEY, True))
    updated_value = bool(updated.get(CALCULATE_FORGETTING_CURVES_CONFIG_KEY, True))
    if current_value == updated_value:
        return False
    try:
        from ..runtime import manager_for_mw

        manager = manager_for_mw(parent)
        return bool(
            getattr(manager, "runtime_scope_active", False)
            or getattr(manager, "save_in_progress", False)
        )
    except Exception:
        return True


def _manager_setting_change_is_blocked(
    parent,
    current: Mapping[str, Any],
    updated: Mapping[str, Any],
) -> bool:
    process_changed = current.get(PROCESS_MANY_MODE_CONFIG_KEY) != updated.get(
        PROCESS_MANY_MODE_CONFIG_KEY
    )
    deleted_history_changed = current.get(EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY) != updated.get(
        EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY
    )
    if not process_changed and not deleted_history_changed:
        return False
    try:
        from ..runtime import manager_for_mw

        manager = manager_for_mw(parent)
        return bool(
            getattr(manager, "runtime_scope_active", False)
            or getattr(manager, "save_in_progress", False)
        )
    except Exception:
        return True


def _filtered_review_policy_change_is_blocked(
    parent,
    current: FilteredReviewNormalizationPolicy,
    updated: FilteredReviewNormalizationPolicy,
) -> bool:
    if current == updated:
        return False
    try:
        from ..runtime import manager_for_mw

        manager = manager_for_mw(parent)
        return bool(
            getattr(manager, "runtime_scope_active", False)
            or getattr(manager, "save_in_progress", False)
        )
    except Exception:
        return True


def _checkpoint_exists(parent) -> bool:
    try:
        from ..runtime import manager_for_mw

        return bool(manager_for_mw(parent).has_checkpoint)
    except Exception:
        return False


def _current_restart_requirement_context() -> RestartRequirementContext:
    live_bridge = sys.modules.get(f"{__package__}.live_review_bridge")
    hooks_installed = getattr(live_bridge, "live_review_bridge_hooks_installed", None)

    return RestartRequirementContext(
        live_review_hooks_installed=bool(hooks_installed and hooks_installed()),
    )
