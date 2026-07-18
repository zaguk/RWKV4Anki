from __future__ import annotations

from collections.abc import Callable

from aqt import mw

from ..checkpoint_manager import (
    CheckpointCacheBindingError,
    InconsistentCheckpointError,
    LegacyCheckpointError,
    MissingCheckpointError,
    StaleCheckpointDataError,
)
from ..runtime import manager_for_mw
from .web_message import (
    WebMessageButton,
    ask_web_choice,
    ask_web_confirmation,
    show_web_warning,
)

_checkpoint_rebuild_handler: Callable[[object], None] | None = None


def set_checkpoint_rebuild_handler(handler: Callable[[object], None] | None) -> None:
    global _checkpoint_rebuild_handler
    _checkpoint_rebuild_handler = handler


def handle_checkpoint_failure(
    exc: Exception,
    retry: Callable[[], None],
    *,
    parent,
    on_decline: Callable[[], None] | None = None,
) -> None:
    if isinstance(exc, LegacyCheckpointError):
        ask_web_confirmation(
            parent=parent,
            title="RWKV",
            message=(
                "This RWKV checkpoint uses the obsolete binary-v1 format and cannot "
                "be used by this add-on. Rebuild it from Anki's review history now?"
            ),
            confirm_label="Rebuild Checkpoint",
            destructive=True,
            on_result=lambda rebuild: _resolve_rebuild_choice(
                rebuild,
                parent=parent,
                on_decline=on_decline,
            ),
        )
        return
    if isinstance(exc, CheckpointCacheBindingError):
        ask_web_confirmation(
            parent=parent,
            title="RWKV",
            message=(
                "The RWKV evaluation cache no longer matches its checkpoint and "
                "cannot be used safely. Rebuild the checkpoint and cache now from "
                "the full review history?"
            ),
            confirm_label="Rebuild Checkpoint",
            destructive=True,
            on_result=lambda rebuild: _resolve_rebuild_choice(
                rebuild,
                parent=parent,
                on_decline=on_decline,
            ),
        )
        return
    if isinstance(exc, InconsistentCheckpointError):
        message = (
            "The RWKV checkpoint cache is missing data expected by this add-on "
            "version. Continuing may make predictions less accurate until you rebuild."
            if isinstance(exc, StaleCheckpointDataError)
            else "The RWKV checkpoint is inconsistent with current review history. "
            "Continuing may make predictions less accurate."
        )
        ask_web_choice(
            parent=parent,
            title="RWKV",
            message=(
                f"{message}\n\nRebuilding from the full review history can take a long time."
            ),
            choices=(
                WebMessageButton("cancel", "Cancel", "quiet"),
                WebMessageButton("rebuild", "Rebuild Checkpoint", "primary"),
                WebMessageButton("continue", "Continue Anyway", "destructive"),
            ),
            on_result=lambda outcome: _resolve_inconsistent_choice(
                outcome,
                retry=retry,
                parent=parent,
                on_decline=on_decline,
            ),
        )
        return
    if isinstance(exc, MissingCheckpointError):
        show_web_warning("Initialize an RWKV checkpoint first.", title="RWKV", parent=parent)
        return
    show_web_warning(str(exc), title="RWKV", parent=parent)


def _resolve_rebuild_choice(
    rebuild: bool,
    *,
    parent,
    on_decline: Callable[[], None] | None,
) -> None:
    if rebuild and _checkpoint_rebuild_handler is not None:
        _checkpoint_rebuild_handler(parent)
    elif not rebuild and on_decline is not None:
        on_decline()


def _resolve_inconsistent_choice(
    outcome: str,
    *,
    retry: Callable[[], None],
    parent,
    on_decline: Callable[[], None] | None,
) -> None:
    if outcome == "continue":
        manager_for_mw(mw).acknowledge_inconsistent_checkpoint()
        retry()
    elif outcome == "rebuild" and _checkpoint_rebuild_handler is not None:
        _checkpoint_rebuild_handler(parent)
    elif on_decline is not None:
        on_decline()


def require_checkpoint_for_use(parent, *, manager=None) -> bool:
    manager = manager_for_mw(mw) if manager is None else manager
    if not manager.has_checkpoint:
        handle_checkpoint_failure(
            MissingCheckpointError("Initialize an RWKV checkpoint first."),
            lambda: None,
            parent=parent,
        )
        return False
    if manager.status() == "legacy":
        handle_checkpoint_failure(
            LegacyCheckpointError("The obsolete RWKV binary-v1 checkpoint must be rebuilt."),
            lambda: None,
            parent=parent,
        )
        return False
    return True
