from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import Any

from .analysis_workflow import AnalysisSubmissionDecision, AnalysisWorkflowState
from .retrievability import (
    CardPrediction,
    RetrievabilitySummary,
    card_id_search,
    card_ids_for_retrievability_bin,
    filter_predictions,
    prediction_time_is_allowed,
    summarize_retrievability,
)
from .retrievability_html import render_retrievability_dialog_html
from .rwkv_modes import RetrievabilityMode
from .web_dialog_bridge import BridgePayloadError, WebDialogCommand
from .web_dialog_controller import BaseWebDialogController, CloseReason

CALCULATE_ACTION = "calculate"
OPEN_BUCKET_ACTION = "open-bucket"
TOGGLE_PREDICTION_EDITOR_ACTION = "toggle-prediction-editor"


class RetrievabilityController(BaseWebDialogController):
    """Qt-independent form, filtering, and graph state for Retrievability."""

    actions = frozenset(
        {
            CALCULATE_ACTION,
            OPEN_BUCKET_ACTION,
            TOGGLE_PREDICTION_EDITOR_ACTION,
        }
    )

    def __init__(
        self,
        *,
        mode: RetrievabilityMode,
        initial_search: str,
        target_timestamp_seconds: float,
        latest_review_timestamp_seconds: float | None,
        fallback_search: str,
        search_card_ids: Callable[[str], Iterable[int]],
        on_calculate_requested: Callable[[str, float], None],
        on_open_bucket_requested: Callable[[str], None],
        on_invalid_predictions: Callable[
            [tuple[CardPrediction, ...], float, str, str], object | None
        ]
        | None = None,
        is_dark: bool = False,
        bin_size: float = 0.05,
    ) -> None:
        if not callable(search_card_ids):
            raise TypeError("search_card_ids must be callable")
        if not callable(on_calculate_requested):
            raise TypeError("on_calculate_requested must be callable")
        if not callable(on_open_bucket_requested):
            raise TypeError("on_open_bucket_requested must be callable")
        if on_invalid_predictions is not None and not callable(on_invalid_predictions):
            raise TypeError("on_invalid_predictions must be callable")
        self.mode = mode
        self.initial_search = str(initial_search)
        self.filter_search = ""
        self.target_timestamp_seconds = float(target_timestamp_seconds)
        self.latest_review_timestamp_seconds = (
            None
            if latest_review_timestamp_seconds is None
            else float(latest_review_timestamp_seconds)
        )
        self.fallback_search = str(fallback_search)
        self.bin_size = float(bin_size)
        self.predictions: list[CardPrediction] = []
        self.displayed_predictions: list[CardPrediction] = []
        self.analysis_state = AnalysisWorkflowState()
        self._result_target_timestamp_seconds: float | None = None
        self._summary = summarize_retrievability((), bin_size=self.bin_size)
        self.invalid_log_path = ""
        self.is_dark = bool(is_dark)
        self._search_card_ids = search_card_ids
        self._on_calculate_requested = on_calculate_requested
        self._on_open_bucket_requested = on_open_bucket_requested
        self._on_invalid_predictions = on_invalid_predictions
        self._rerender: Callable[[], Any] | None = None

    @property
    def summary(self) -> RetrievabilitySummary:
        return self._summary

    @property
    def has_result_set(self) -> bool:
        return self.analysis_state.has_result_set

    @property
    def prediction_editor_expanded(self) -> bool:
        return self.analysis_state.source_expanded

    @prediction_editor_expanded.setter
    def prediction_editor_expanded(self, expanded: bool) -> None:
        self.analysis_state.set_source_expanded(expanded)

    @property
    def source_inputs_changed(self) -> bool:
        return self.analysis_state.source_inputs_changed

    @property
    def error_message(self) -> str:
        return self.analysis_state.error_message

    @property
    def running(self) -> bool:
        return self.analysis_state.running

    @property
    def focus_target(self) -> str | None:
        return self.analysis_state.focus_target

    @property
    def summary_text(self) -> str:
        summary = self.summary
        average = "n/a" if summary.average is None else f"{summary.average:.4f}"
        text = f"Cards: {summary.count}    Average R: {average}"
        if summary.skipped_count:
            text += f"    Skipped non-finite: {summary.skipped_count}"
            if self.invalid_log_path:
                text += f"    Log: {self.invalid_log_path}"
        return text

    @property
    def latest_review_warning(self) -> str:
        if self.latest_review_timestamp_seconds is None:
            return ""
        return (
            "Prediction time must be at or after the most recent review: "
            f"{format_local_timestamp(self.latest_review_timestamp_seconds)}."
        )

    def attach_rerender(self, rerender: Callable[[], Any]) -> None:
        if not callable(rerender):
            raise TypeError("Retrievability rerender callback must be callable")
        self._rerender = rerender

    def render_html(self, generation: int) -> str:
        return render_retrievability_dialog_html(
            mode=self.mode,
            target_time=format_datetime_local(self.target_timestamp_seconds),
            initial_search=self.initial_search,
            filter_search=self.filter_search,
            latest_review_warning=self.latest_review_warning,
            summary=self.summary,
            summary_text=self.summary_text,
            error_message=self.error_message,
            invalid_log_path=self.invalid_log_path,
            running=self.running,
            focus_target=self.focus_target,
            has_result_set=self.has_result_set,
            prediction_editor_expanded=self.prediction_editor_expanded,
            is_dark=self.is_dark,
            generation=generation,
        )

    def handle_command(self, command: WebDialogCommand) -> dict[str, bool]:
        if command.action == CALCULATE_ACTION:
            initial_search, target_time, filter_search = _submission_form_values(command.payload)
            try:
                target_timestamp_seconds = parse_datetime_local(target_time)
            except ValueError as exc:
                self.initial_search = initial_search
                self.filter_search = filter_search.strip()
                self.analysis_state.mark_source_inputs_changed()
                self.analysis_state.set_error(
                    str(exc),
                    focus_target="target_time",
                    expand_source=True,
                )
                self._request_rerender()
                return {"updated": False}
            self.initial_search = initial_search
            self.filter_search = filter_search.strip()
            effective_search = initial_search.strip() or self.fallback_search
            decision = self.analysis_state.decide(
                _retrievability_source_key(
                    effective_search,
                    target_timestamp_seconds,
                )
            )
            self.analysis_state.clear_error()
            if decision is AnalysisSubmissionDecision.CALCULATE_THEN_FILTER:
                self.target_timestamp_seconds = target_timestamp_seconds
                self.analysis_state.focus_target = "initial_search"
                self._on_calculate_requested(
                    effective_search,
                    target_timestamp_seconds,
                )
                return {"updated": True}
            result_target_timestamp = self._result_target_timestamp_seconds
            assert result_target_timestamp is not None
            self.target_timestamp_seconds = result_target_timestamp
            self.analysis_state.focus_target = "filter_search"
            updated = self._apply_display_filter()
            self._request_rerender()
            return {"updated": updated}

        if command.action == TOGGLE_PREDICTION_EDITOR_ACTION:
            expanded = self.analysis_state.set_source_expanded(
                _expanded_value(command.payload)
            )
            return {"expanded": expanded}

        if command.action == OPEN_BUCKET_ACTION:
            bucket_index = _bucket_index(command.payload)
            card_ids = card_ids_for_retrievability_bin(
                self.displayed_predictions,
                bucket_index,
                bin_size=self.bin_size,
            )
            search = card_id_search(card_ids)
            if not search:
                self.analysis_state.set_error(
                    "No cards in this retrievability bucket."
                )
                self._request_rerender()
                return {"updated": False}
            self.analysis_state.clear_error()
            self._on_open_bucket_requested(search)
            return {"updated": True}

        raise BridgePayloadError(f"Unhandled Retrievability action: {command.action}")

    def is_action_enabled(self, action: str) -> bool:
        return action in self.actions and not self.running

    def can_close(self, reason: CloseReason) -> bool:
        del reason
        return not self.running

    def update_latest_review_timestamp(self, timestamp_seconds: float | None) -> None:
        if timestamp_seconds is None:
            return
        timestamp = float(timestamp_seconds)
        if (
            self.latest_review_timestamp_seconds is None
            or timestamp > self.latest_review_timestamp_seconds
        ):
            self.latest_review_timestamp_seconds = timestamp

    def prediction_time_is_allowed(
        self,
        timestamp_seconds: float | None = None,
    ) -> bool:
        return prediction_time_is_allowed(
            (
                self.target_timestamp_seconds
                if timestamp_seconds is None
                else float(timestamp_seconds)
            ),
            self.latest_review_timestamp_seconds,
        )

    def show_error(self, message: str, *, focus_target: str | None = None) -> None:
        self.analysis_state.set_error(
            message,
            focus_target=focus_target,
            expand_source=focus_target in {"initial_search", "target_time"},
        )
        self._request_rerender()

    def begin_calculation(self) -> bool:
        if not self.analysis_state.begin_calculation():
            return False
        self.invalid_log_path = ""
        self._request_rerender()
        return True

    def finish_calculation(self, *, focus_target: str = "initial_search") -> None:
        if self.analysis_state.finish_calculation(focus_target=focus_target):
            self._apply_display_filter()
        self._request_rerender()

    def set_predictions(self, predictions: Iterable[CardPrediction]) -> None:
        self.predictions = list(predictions)
        self._result_target_timestamp_seconds = self.target_timestamp_seconds
        self.analysis_state.accept_result(
            _retrievability_source_key(
                self.initial_search.strip() or self.fallback_search,
                self.target_timestamp_seconds,
            ),
            focus_target="filter_search",
        )
        self._apply_display_filter(replace_on_error=True)
        self._request_rerender()

    def _apply_display_filter(self, *, replace_on_error: bool = False) -> bool:
        if not self.has_result_set:
            self.analysis_state.set_error("Run Calculate first.")
            return False
        if not self.filter_search:
            self._set_displayed_predictions(self.predictions)
            self.analysis_state.clear_error()
            return True
        try:
            allowed = self._search_card_ids(self.filter_search)
        except Exception as exc:  # Anki search parsers use several exception types.
            if replace_on_error:
                self._set_displayed_predictions(self.predictions)
            self.analysis_state.set_error(
                str(exc).strip() or exc.__class__.__name__,
                focus_target="filter_search",
            )
            return False
        self._set_displayed_predictions(filter_predictions(self.predictions, allowed))
        self.analysis_state.clear_error()
        return True

    def _set_displayed_predictions(
        self,
        predictions: Iterable[CardPrediction],
    ) -> None:
        self.displayed_predictions = list(predictions)
        self._summary = summarize_retrievability(
            self.displayed_predictions,
            bin_size=self.bin_size,
        )
        summary = self._summary
        self.invalid_log_path = ""
        if summary.invalid_predictions and self._on_invalid_predictions is not None:
            path = self._on_invalid_predictions(
                summary.invalid_predictions,
                self.target_timestamp_seconds,
                self.initial_search.strip(),
                self.filter_search.strip(),
            )
            if path is not None:
                self.invalid_log_path = str(path)

    def _request_rerender(self) -> None:
        if self._rerender is None:
            raise RuntimeError("Retrievability controller is not attached to its dialog")
        self._rerender()


def format_datetime_local(timestamp_seconds: float) -> str:
    return datetime.fromtimestamp(float(timestamp_seconds)).isoformat(timespec="seconds")


def parse_datetime_local(value: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Prediction date and time is required.")
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError("Prediction date and time is invalid.") from exc
    return float(parsed.timestamp())


def format_local_timestamp(timestamp_seconds: float) -> str:
    return datetime.fromtimestamp(float(timestamp_seconds)).strftime("%Y-%m-%d %H:%M:%S")


def _retrievability_source_key(
    effective_search: str,
    target_timestamp_seconds: float,
) -> tuple[str, str]:
    return str(effective_search), format_datetime_local(target_timestamp_seconds)


def _submission_form_values(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    _require_payload_keys(
        payload,
        expected={"filter_search", "initial_search", "target_time"},
        label="Retrievability submission",
    )
    initial_search = payload["initial_search"]
    target_time = payload["target_time"]
    filter_search = payload["filter_search"]
    if not all(isinstance(value, str) for value in (initial_search, target_time, filter_search)):
        raise BridgePayloadError("Retrievability submission values must be text.")
    return initial_search, target_time, filter_search


def _expanded_value(payload: Mapping[str, Any]) -> bool:
    _require_payload_keys(
        payload,
        expected={"expanded"},
        label="Prediction-editor state",
    )
    expanded = payload["expanded"]
    if not isinstance(expanded, bool):
        raise BridgePayloadError("Prediction-editor expanded state must be true or false.")
    return expanded


def _bucket_index(payload: Mapping[str, Any]) -> int:
    _require_payload_keys(
        payload,
        expected={"bucket_index"},
        label="Retrievability bucket",
    )
    value = payload["bucket_index"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BridgePayloadError("Retrievability bucket index must be a non-negative integer.")
    return int(value)


def _require_payload_keys(
    payload: Mapping[str, Any],
    *,
    expected: set[str],
    label: str,
) -> None:
    missing = expected - set(payload)
    extra = set(payload) - expected
    if missing:
        raise BridgePayloadError(f"{label} is missing: " + ", ".join(sorted(missing)) + ".")
    if extra:
        raise BridgePayloadError(
            f"{label} contains unsupported fields: " + ", ".join(sorted(extra)) + "."
        )
