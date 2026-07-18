from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

WEB_DIALOG_BRIDGE_PREFIX = "rwkvWebDialog:"
MAX_BRIDGE_COMMAND_LENGTH = 256 * 1024

_ACTION_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_MAX_REQUEST_ID_LENGTH = 128


class BridgePayloadError(ValueError):
    """A controller rejected a syntactically valid command payload."""

    def __init__(self, message: str, *, code: str = "invalid-payload") -> None:
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class WebDialogCommand:
    """Validated command passed across the renderer/controller boundary."""

    generation: int
    action: str
    payload: Mapping[str, Any]
    request_id: str | None = None


class WebDialogBridge:
    """Validate and dispatch the one JSON bridge used by a web dialog.

    Malformed, unknown, disabled, stale, and post-close commands are rejected
    before workflow code sees them. Controller payload validation can raise
    :class:`BridgePayloadError`; unexpected controller exceptions deliberately
    propagate so programming errors are not silently converted into UI state.
    """

    def __init__(
        self,
        *,
        generation: int,
        allowed_actions: Collection[str],
        handler: Callable[[WebDialogCommand], Any],
        is_action_enabled: Callable[[str], bool] | None = None,
    ) -> None:
        self._generation = _require_generation(generation)
        self._allowed_actions = frozenset(
            require_web_dialog_action(action) for action in allowed_actions
        )
        self._handler = handler
        self._is_action_enabled = is_action_enabled or (lambda _action: True)
        self._active = True

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active(self) -> bool:
        return self._active

    def advance_generation(self) -> int:
        """Invalidate commands from the old document and return the new token."""

        self._generation += 1
        return self._generation

    def invalidate(self) -> None:
        self._active = False

    def dispatch(self, raw_command: str) -> dict[str, Any]:
        parsed = _parse_command(raw_command)
        if isinstance(parsed, dict):
            return parsed
        command = parsed
        if not self._active:
            return _error_reply(
                "closed",
                "This dialog is already closed.",
                request_id=command.request_id,
            )
        if command.generation != self._generation:
            return _error_reply(
                "stale-command",
                "This command came from an older version of the dialog.",
                request_id=command.request_id,
            )
        if command.action not in self._allowed_actions:
            return _error_reply(
                "unknown-action",
                f"Unknown dialog action: {command.action}",
                request_id=command.request_id,
            )
        if not bool(self._is_action_enabled(command.action)):
            return _error_reply(
                "disabled-action",
                f"Dialog action is currently disabled: {command.action}",
                request_id=command.request_id,
            )
        try:
            result = self._handler(command)
        except BridgePayloadError as error:
            return _error_reply(
                error.code,
                str(error),
                request_id=command.request_id,
            )
        if not _is_json_value(result):
            raise TypeError("web dialog command handlers must return JSON-compatible values")
        return {
            "ok": True,
            "requestId": command.request_id,
            "result": result,
        }


def _parse_command(raw_command: str) -> WebDialogCommand | dict[str, Any]:
    if not isinstance(raw_command, str):
        return _error_reply("invalid-command", "Dialog commands must be strings.")
    if len(raw_command) > MAX_BRIDGE_COMMAND_LENGTH:
        return _error_reply("invalid-command", "Dialog command is too large.")
    if not raw_command.startswith(WEB_DIALOG_BRIDGE_PREFIX):
        return _error_reply("invalid-command", "Unrecognized dialog command prefix.")
    encoded = raw_command[len(WEB_DIALOG_BRIDGE_PREFIX) :]
    try:
        value = json.loads(encoded)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error_reply("invalid-command", "Dialog command is not valid JSON.")
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return _error_reply("invalid-command", "Dialog command must be a JSON object.")

    request_id = value.get("requestId")
    if request_id is not None and (
        not isinstance(request_id, str) or len(request_id) > _MAX_REQUEST_ID_LENGTH
    ):
        return _error_reply("invalid-command", "Dialog requestId must be a short string.")
    if set(value) - {"generation", "action", "payload", "requestId"}:
        return _error_reply(
            "invalid-command",
            "Dialog command contains unsupported fields.",
            request_id=request_id,
        )

    generation = value.get("generation")
    action = value.get("action")
    payload = value.get("payload", {})
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        return _error_reply(
            "invalid-command",
            "Dialog generation must be a positive integer.",
            request_id=request_id,
        )
    if not isinstance(action, str) or _ACTION_PATTERN.fullmatch(action) is None:
        return _error_reply(
            "invalid-command",
            "Dialog action has an invalid name.",
            request_id=request_id,
        )
    if not isinstance(payload, dict) or not _is_json_value(payload):
        return _error_reply(
            "invalid-command",
            "Dialog payload must be a JSON-compatible object.",
            request_id=request_id,
        )
    return WebDialogCommand(
        generation=generation,
        action=action,
        payload=payload,
        request_id=request_id,
    )


def _require_generation(generation: int) -> int:
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ValueError("generation must be a positive integer")
    return generation


def require_web_dialog_action(action: str) -> str:
    """Return a canonical bridge action or reject an unsafe/unsupported name."""

    if not isinstance(action, str) or _ACTION_PATTERN.fullmatch(action) is None:
        raise ValueError(f"invalid web dialog action name: {action!r}")
    return action


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _error_reply(
    code: str,
    message: str,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "requestId": request_id,
        "error": {"code": str(code), "message": str(message)},
    }
