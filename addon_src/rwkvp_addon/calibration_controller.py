from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .analysis_workflow import AnalysisSubmissionDecision, AnalysisWorkflowState
from .calibration import (
    CalibrationSummary,
    ReviewIntervalFilter,
    filter_rows_by_review_interval,
    summarize_calibration_rows,
)
from .calibration_html import render_calibration_dialog_html
from .rwkv_modes import RetrievabilityMode
from .web_dialog_bridge import BridgePayloadError, WebDialogCommand
from .web_dialog_controller import BaseWebDialogController, CloseReason

REVIEW_INTERVAL_OPERATORS = ("<", "<=", "=", ">=", ">")

CALCULATE_ACTION = "calculate"
SET_SAME_DAY_ACTION = "set-same-day"
TOGGLE_REVIEW_SOURCE_ACTION = "toggle-review-source"


class CalibrationController(BaseWebDialogController):
    """Qt-independent form, filtering, and graph state for Calibration."""

    actions = frozenset(
        {
            CALCULATE_ACTION,
            SET_SAME_DAY_ACTION,
            TOGGLE_REVIEW_SOURCE_ACTION,
        }
    )

    def __init__(
        self,
        *,
        mode: RetrievabilityMode,
        initial_search: str,
        fallback_search: str,
        search_card_ids: Callable[[str], Iterable[int]],
        on_calculate_requested: Callable[[str], None],
        is_dark: bool = False,
        bin_count: int = 20,
    ) -> None:
        if not callable(search_card_ids):
            raise TypeError("search_card_ids must be callable")
        if not callable(on_calculate_requested):
            raise TypeError("on_calculate_requested must be callable")
        self.mode = mode
        self.initial_search = str(initial_search)
        self.fallback_search = str(fallback_search)
        self.filter_search = ""
        self.include_sameday = False
        self.review_interval_operator = "<="
        self.review_interval_value = ""
        self.bin_count = int(bin_count)
        self.prediction_rows: list[dict] = []
        self.displayed_rows: list[dict] = []
        self.analysis_state = AnalysisWorkflowState()
        self._summary = summarize_calibration_rows(
            (),
            bin_count=self.bin_count,
            include_sameday=self.include_sameday,
        )
        self.is_dark = bool(is_dark)
        self._search_card_ids = search_card_ids
        self._on_calculate_requested = on_calculate_requested
        self._rerender: Callable[[], Any] | None = None

    @property
    def summary(self) -> CalibrationSummary:
        return self._summary

    @property
    def has_result_set(self) -> bool:
        return self.analysis_state.has_result_set

    @property
    def review_source_expanded(self) -> bool:
        return self.analysis_state.source_expanded

    @review_source_expanded.setter
    def review_source_expanded(self, expanded: bool) -> None:
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
        predicted = (
            "n/a"
            if summary.average_prediction is None
            else f"{summary.average_prediction:.4f}"
        )
        actual = "n/a" if summary.actual_recall is None else f"{summary.actual_recall:.4f}"
        text = f"Reviews: {summary.count}    Predicted: {predicted}    Actual: {actual}"
        if summary.missing_prediction_count:
            text += f"    No prior prediction: {summary.missing_prediction_count}"
        if summary.invalid_count:
            text += f"    Skipped invalid: {summary.invalid_count}"
        return text

    def attach_rerender(self, rerender: Callable[[], Any]) -> None:
        if not callable(rerender):
            raise TypeError("Calibration rerender callback must be callable")
        self._rerender = rerender

    def render_html(self, generation: int) -> str:
        return render_calibration_dialog_html(
            mode=self.mode,
            initial_search=self.initial_search,
            filter_search=self.filter_search,
            include_sameday=self.include_sameday,
            review_interval_operator=self.review_interval_operator,
            review_interval_value=self.review_interval_value,
            summary=self.summary,
            summary_text=self.summary_text,
            error_message=self.error_message,
            running=self.running,
            focus_target=self.focus_target,
            has_result_set=self.has_result_set,
            review_source_expanded=self.review_source_expanded,
            is_dark=self.is_dark,
            generation=generation,
        )

    def handle_command(self, command: WebDialogCommand) -> dict[str, bool]:
        if command.action == CALCULATE_ACTION:
            values = _submission_form_values(command.payload)
            self._set_form_values(values)
            effective_search = self.initial_search.strip() or self.fallback_search
            decision = self.analysis_state.decide(effective_search)
            self.analysis_state.clear_error()
            if decision is AnalysisSubmissionDecision.CALCULATE_THEN_FILTER:
                self.analysis_state.focus_target = "initial_search"
                self._on_calculate_requested(effective_search)
                return {"updated": True}
            self.analysis_state.focus_target = "filter_search"
            updated = self._apply_filters()
            self._request_rerender()
            return {"updated": updated}

        if command.action == SET_SAME_DAY_ACTION:
            values = _submission_form_values(command.payload)
            same_day_changed = self.include_sameday != values[1]
            self._set_form_values(values)
            effective_search = self.initial_search.strip() or self.fallback_search
            self.analysis_state.record_source(effective_search)
            self.analysis_state.clear_error(focus_target="include_sameday")
            if same_day_changed:
                self._refresh_summary()
            self._request_rerender()
            return {"updated": True}

        if command.action == TOGGLE_REVIEW_SOURCE_ACTION:
            expanded = self.analysis_state.set_source_expanded(
                _expanded_value(command.payload)
            )
            return {"expanded": expanded}

        raise BridgePayloadError(f"Unhandled Calibration action: {command.action}")

    def is_action_enabled(self, action: str) -> bool:
        return action in self.actions and not self.running

    def can_close(self, reason: CloseReason) -> bool:
        del reason
        return not self.running

    def begin_calculation(self) -> bool:
        if not self.analysis_state.begin_calculation():
            return False
        self._request_rerender()
        return True

    def finish_calculation(self, *, focus_target: str = "initial_search") -> None:
        if self.analysis_state.finish_calculation(focus_target=focus_target):
            self._apply_filters()
        self._request_rerender()

    def set_prediction_rows(self, rows: Iterable[dict]) -> None:
        self.prediction_rows = list(rows)
        self.analysis_state.accept_result(
            self.initial_search.strip() or self.fallback_search,
            focus_target="filter_search",
        )
        self._apply_filters(replace_on_error=True)
        self._request_rerender()

    def _set_form_values(self, values: tuple[str, bool, str, str, str]) -> None:
        initial_search, include_sameday, filter_search, operator, value_text = values
        self.initial_search = initial_search
        self.include_sameday = include_sameday
        self.filter_search = filter_search.strip()
        self.review_interval_operator = operator
        self.review_interval_value = value_text.strip()

    def _apply_filters(self, *, replace_on_error: bool = False) -> bool:
        if not self.has_result_set:
            self.analysis_state.set_error("Run Calculate first.")
            return False
        try:
            review_interval = review_interval_filter(
                self.review_interval_operator,
                self.review_interval_value,
            )
        except ValueError as exc:
            if replace_on_error:
                self.displayed_rows = list(self.prediction_rows)
                self._refresh_summary()
            self.analysis_state.set_error(
                str(exc),
                focus_target="review_interval_value",
            )
            return False
        rows = list(self.prediction_rows)
        if self.filter_search:
            try:
                allowed = {
                    int(card_id) for card_id in self._search_card_ids(self.filter_search)
                }
            except Exception as exc:  # Anki search parsers use several exception types.
                if replace_on_error:
                    self.displayed_rows = list(self.prediction_rows)
                    self._refresh_summary()
                self.analysis_state.set_error(
                    str(exc).strip() or exc.__class__.__name__,
                    focus_target="filter_search",
                )
                return False
            rows = filter_rows_by_card_ids(rows, allowed)
        self.displayed_rows = filter_rows_by_review_interval(rows, review_interval)
        self._refresh_summary()
        self.analysis_state.clear_error()
        return True

    def _refresh_summary(self) -> None:
        self._summary = summarize_calibration_rows(
            self.displayed_rows,
            bin_count=self.bin_count,
            include_sameday=self.include_sameday,
        )

    def _request_rerender(self) -> None:
        if self._rerender is None:
            raise RuntimeError("Calibration controller is not attached to its dialog")
        self._rerender()


def review_interval_filter(
    operator: str,
    value_text: str,
) -> ReviewIntervalFilter | None:
    if operator not in REVIEW_INTERVAL_OPERATORS:
        raise ValueError("Review interval operator must be one of <, <=, =, >=, >.")
    value = value_text.strip()
    if not value:
        return None
    try:
        days = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Review interval must be a number.") from exc
    return ReviewIntervalFilter(operator=operator, days=days)


def filter_rows_by_card_ids(
    rows: Iterable[dict],
    card_ids: set[int],
) -> list[dict]:
    filtered = []
    for row in rows:
        try:
            card_id = int(row["card_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if card_id in card_ids:
            filtered.append(row)
    return filtered


def _submission_form_values(
    payload: Mapping[str, Any],
) -> tuple[str, bool, str, str, str]:
    _require_payload_keys(
        payload,
        expected={
            "initial_search",
            "include_sameday",
            "filter_search",
            "review_interval_operator",
            "review_interval_value",
        },
        label="Calibration submission",
    )
    initial_search = payload["initial_search"]
    include_sameday = payload["include_sameday"]
    if not isinstance(initial_search, str) or not isinstance(include_sameday, bool):
        raise BridgePayloadError(
            "Calibration search must be text and same-day selection must be true or false."
        )
    values = (
        payload["filter_search"],
        payload["review_interval_operator"],
        payload["review_interval_value"],
    )
    if not all(isinstance(value, str) for value in values):
        raise BridgePayloadError("Calibration filter values must be text.")
    filter_search, operator, value_text = values
    if operator not in REVIEW_INTERVAL_OPERATORS:
        raise BridgePayloadError("Calibration review interval operator is invalid.")
    return initial_search, include_sameday, filter_search, operator, value_text


def _expanded_value(payload: Mapping[str, Any]) -> bool:
    _require_payload_keys(
        payload,
        expected={"expanded"},
        label="Calibration review-source state",
    )
    expanded = payload["expanded"]
    if not isinstance(expanded, bool):
        raise BridgePayloadError(
            "Calibration review-source expanded state must be true or false."
        )
    return expanded


def _require_payload_keys(
    payload: Mapping[str, Any],
    *,
    expected: set[str],
    label: str,
) -> None:
    missing = expected - set(payload)
    extra = set(payload) - expected
    if missing:
        raise BridgePayloadError(
            f"{label} is missing: " + ", ".join(sorted(missing)) + "."
        )
    if extra:
        raise BridgePayloadError(
            f"{label} contains unsupported fields: "
            + ", ".join(sorted(extra))
            + "."
        )
