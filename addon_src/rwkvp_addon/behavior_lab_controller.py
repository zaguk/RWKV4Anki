from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .behavior_lab import (
    BehaviorLabExperiment,
    BehaviorLabResult,
    behavior_lab_template,
)
from .behavior_lab_html import (
    render_behavior_lab_editor,
    render_behavior_lab_results,
)
from .web_dialog_bridge import BridgePayloadError, WebDialogCommand
from .web_dialog_controller import BaseWebDialogController, CloseReason


class BehaviorLabWebController(BaseWebDialogController):
    """Typed, Qt-independent state boundary for the Behavior Lab WebView."""

    actions = frozenset({"run", "template", "save", "delete", "edit", "import", "rerun"})

    def __init__(
        self,
        *,
        experiment: BehaviorLabExperiment,
        experiment_store: Any,
        on_run_requested: Callable[[], None],
        on_delete_requested: Callable[[str], None],
        on_error: Callable[[str], None],
        is_dark: bool = False,
    ) -> None:
        for callback, label in (
            (on_run_requested, "run-request callback"),
            (on_delete_requested, "delete-request callback"),
            (on_error, "error callback"),
        ):
            if not callable(callback):
                raise TypeError(f"Behavior Lab {label} must be callable")
        self.experiment = experiment
        self._experiment_store = experiment_store
        self._on_run_requested = on_run_requested
        self._on_delete_requested = on_delete_requested
        self._on_error = on_error
        self.is_dark = bool(is_dark)
        self._rerender: Callable[[], Any] | None = None
        self._message = ""
        self._result: BehaviorLabResult | None = None
        self.running = False

    @property
    def result(self) -> BehaviorLabResult | None:
        return self._result

    def attach_rerender(self, rerender: Callable[[], Any]) -> None:
        if not callable(rerender):
            raise TypeError("Behavior Lab rerender callback must be callable")
        self._rerender = rerender

    def render_html(self, generation: int) -> str:
        if self._result is not None:
            return render_behavior_lab_results(
                self._result,
                generation=generation,
                is_dark=self.is_dark,
            )
        message = self._message
        try:
            saved = self._experiment_store.experiments()
        except (OSError, ValueError) as exc:
            saved = {}
            if not message:
                message = f"Saved experiments could not be read: {exc}"
        return render_behavior_lab_editor(
            self.experiment,
            saved_experiments=saved,
            message=message,
            generation=generation,
            is_dark=self.is_dark,
        )

    def handle_command(self, command: WebDialogCommand) -> dict[str, bool]:
        try:
            if command.action == "run":
                _require_payload_keys(command.payload, required={"experiment"})
                self.experiment = BehaviorLabExperiment.from_dict(
                    _require_mapping(command.payload["experiment"], "experiment")
                )
                self._on_run_requested()
                return {"updated": True}

            if command.action == "template":
                _require_payload_keys(
                    command.payload,
                    required={
                        "template",
                        "focal_card_id",
                        "selection_card_ids",
                        "delay_seconds",
                        "duration_seconds",
                        "context_count",
                    },
                )
                selection = command.payload["selection_card_ids"]
                if not isinstance(selection, list):
                    raise BridgePayloadError("selection_card_ids must be a list.")
                self.experiment = behavior_lab_template(
                    str(command.payload["template"]),
                    focal_card_id=_require_int(
                        command.payload["focal_card_id"],
                        "focal_card_id",
                    ),
                    selection_card_ids=(
                        _require_int(value, "selection_card_ids") for value in selection
                    ),
                    delay_seconds=_require_number(
                        command.payload["delay_seconds"],
                        "delay_seconds",
                    ),
                    duration_seconds=_require_number(
                        command.payload["duration_seconds"],
                        "duration_seconds",
                    ),
                    context_count=_require_int(
                        command.payload["context_count"],
                        "context_count",
                    ),
                )
                self._message = "Template applied. Review the timeline, then run it."
                self._result = None
                self._request_rerender()
                return {"updated": True}

            if command.action == "save":
                _require_payload_keys(command.payload, required={"name", "experiment"})
                self.experiment = BehaviorLabExperiment.from_dict(
                    _require_mapping(command.payload["experiment"], "experiment")
                )
                name = _require_string(command.payload["name"], "name")
                self._experiment_store.save(name, self.experiment)
                self._message = f"Saved experiment “{name}”."
                self._result = None
                self._request_rerender()
                return {"updated": True}

            if command.action == "import":
                _require_payload_keys(command.payload, required={"experiment"})
                self.experiment = BehaviorLabExperiment.from_dict(
                    _require_mapping(command.payload["experiment"], "experiment")
                )
                self._message = "Imported experiment. Review the timeline, then run it."
                self._result = None
                self._request_rerender()
                return {"updated": True}

            if command.action == "delete":
                _require_payload_keys(command.payload, required={"name"})
                name = _require_string(command.payload["name"], "name")
                self._on_delete_requested(name)
                return {"updated": False}

            if command.action == "edit":
                _require_payload_keys(command.payload)
                self._result = None
                self._message = ""
                self._request_rerender()
                return {"updated": True}

            if command.action == "rerun":
                _require_payload_keys(command.payload)
                self._on_run_requested()
                return {"updated": True}
        except BridgePayloadError:
            raise
        except (KeyError, TypeError, ValueError, OSError) as exc:
            self._on_error(str(exc))
            return {"updated": False}
        raise BridgePayloadError(f"Unhandled Behavior Lab action: {command.action}")

    def delete_saved_experiment(self, name: str) -> bool:
        """Apply a host-confirmed deletion and refresh the editor state."""

        try:
            deleted = self._experiment_store.delete(str(name))
        except (OSError, ValueError) as exc:
            self._on_error(str(exc))
            return False
        self._message = (
            f"Deleted experiment “{name}”." if deleted else "Nothing was deleted."
        )
        self._result = None
        self._request_rerender()
        return bool(deleted)

    def show_result(self, result: BehaviorLabResult) -> None:
        self._result = result
        self._message = ""
        self._request_rerender()

    def can_close(self, reason: CloseReason) -> bool:
        del reason
        return not self.running

    def _request_rerender(self) -> None:
        if self._rerender is None:
            raise RuntimeError("Behavior Lab controller is not attached to its dialog")
        self._rerender()


def _require_payload_keys(
    payload: Mapping[str, Any],
    *,
    required: set[str] | frozenset[str] = frozenset(),
) -> None:
    missing = required - set(payload)
    extra = set(payload) - required
    if missing:
        raise BridgePayloadError(
            "Behavior Lab command is missing: " + ", ".join(sorted(missing)) + "."
        )
    if extra:
        raise BridgePayloadError(
            "Behavior Lab command contains unsupported fields: "
            + ", ".join(sorted(extra))
            + "."
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgePayloadError(f"{label} must be an object.")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise BridgePayloadError(f"{label} must be text.")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BridgePayloadError(f"{label} must be an integer.")
    return int(value)


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BridgePayloadError(f"{label} must be a number.")
    return float(value)
