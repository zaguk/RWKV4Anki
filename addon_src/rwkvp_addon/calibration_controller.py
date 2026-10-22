from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .analysis_workflow import AnalysisSubmissionDecision, AnalysisWorkflowState
from .calibration import (
    CalibrationSummary,
    ReviewIntervalFilter,
    filter_rows_by_review_interval,
    summarize_calibration_comparison,
    summarize_calibration_rows,
)
from .calibration_html import render_calibration_dialog_html
from .metrics import RWKVPredictionMode
from .model_generation import (
    V1_MODEL_GENERATION,
    compatible_evaluation_generations,
    model_generation_label,
    normalize_model_generation,
)
from .rwkv_modes import RetrievabilityMode
from .um_plus import UmPlusSummary, summarize_um_plus_rows
from .web_dialog_bridge import BridgePayloadError, WebDialogCommand
from .web_dialog_controller import BaseWebDialogController, CloseReason

REVIEW_INTERVAL_OPERATORS = ("<", "<=", "=", ">=", ">")

CALCULATE_ACTION = "calculate"
SET_SAME_DAY_ACTION = "set-same-day"
SET_FSRS_ACTION = "set-fsrs"
SET_SMALL_BUCKETS_ACTION = "set-small-buckets"
SET_COMPARISON_ACTION = "set-comparison"
TOGGLE_REVIEW_SOURCE_ACTION = "toggle-review-source"


class CalibrationController(BaseWebDialogController):
    """Shared source/filter workflow for Calibration and its paired UM+ view."""

    actions = frozenset(
        {
            CALCULATE_ACTION,
            SET_SAME_DAY_ACTION,
            SET_FSRS_ACTION,
            SET_COMPARISON_ACTION,
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
        um_plus: bool = False,
        active_generation: str = V1_MODEL_GENERATION,
        include_deleted_reviews: bool = False,
    ) -> None:
        if not callable(search_card_ids):
            raise TypeError("search_card_ids must be callable")
        if not callable(on_calculate_requested):
            raise TypeError("on_calculate_requested must be callable")
        self.mode = mode
        self.prediction_mode = (
            RWKVPredictionMode.PREDICT_AHEAD
            if mode == RetrievabilityMode.FORGETTING_CURVE
            else RWKVPredictionMode.PER_REVIEW
        )
        self.active_generation = normalize_model_generation(active_generation)
        self.compatible_generations = compatible_evaluation_generations(
            self.active_generation,
            prediction_mode=self.prediction_mode.value,
            include_deleted_reviews=include_deleted_reviews,
        )
        self.comparison_generation: str | None = None
        self._loaded_comparison_generation: str | None = None
        self.um_plus = bool(um_plus)
        if self.um_plus:
            self.actions = (self.actions - {SET_FSRS_ACTION}) | {SET_SMALL_BUCKETS_ACTION}
        self.initial_search = str(initial_search)
        self.fallback_search = str(fallback_search)
        self.filter_search = ""
        self.include_sameday = False
        self.include_fsrs = self.um_plus
        self.show_small_buckets = False
        self.fsrs_loaded = False
        self.fsrs_summary: CalibrationSummary | None = None
        self.unpaired_count = 0
        self.review_interval_operator = "<="
        self.review_interval_value = ""
        self.bin_count = int(bin_count)
        self.prediction_rows: list[dict] = []
        self.displayed_rows: list[dict] = []
        self.analysis_state = AnalysisWorkflowState()
        self._summary: CalibrationSummary | UmPlusSummary = (
            summarize_um_plus_rows(())
            if self.um_plus
            else summarize_calibration_rows(
                (),
                bin_count=self.bin_count,
                include_sameday=self.include_sameday,
            )
        )
        self.is_dark = bool(is_dark)
        self._search_card_ids = search_card_ids
        self._on_calculate_requested = on_calculate_requested
        self._rerender: Callable[[], Any] | None = None

    @property
    def summary(self) -> CalibrationSummary | UmPlusSummary:
        return self._summary

    @property
    def comparison_loaded(self) -> bool:
        return self.fsrs_loaded and self._loaded_comparison_generation == self.comparison_generation

    @property
    def active_label(self) -> str:
        return f"RWKV {model_generation_label(self.active_generation)}"

    @property
    def comparison_label(self) -> str:
        return (
            "FSRS"
            if self.comparison_generation is None
            else f"RWKV {model_generation_label(self.comparison_generation)}"
        )

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
        if isinstance(summary, UmPlusSummary):
            text = f"Reviews: {summary.count:,} paired"
            if summary.unpaired_count:
                text += f"    Without paired predictions: {summary.unpaired_count:,}"
            return text
        predicted = (
            "n/a" if summary.average_prediction is None else f"{summary.average_prediction:.4f}"
        )
        actual = "n/a" if summary.actual_recall is None else f"{summary.actual_recall:.4f}"
        text = f"Reviews: {summary.count}    Predicted: {predicted}    Actual: {actual}"
        if self.fsrs_summary is not None:
            fsrs = self.fsrs_summary.average_prediction
            fsrs_text = "n/a" if fsrs is None else f"{fsrs:.4f}"
            text = (
                f"Reviews: {summary.count}    {self.active_label}: {predicted}"
                f"    {self.comparison_label}: {fsrs_text}"
                f"    Actual: {actual}"
            )
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
            include_fsrs=self.include_fsrs,
            fsrs_summary=self.fsrs_summary,
            active_label=self.active_label,
            comparison_generation=self.comparison_generation,
            compatible_generations=self.compatible_generations,
            comparison_loaded=self.comparison_loaded,
            unpaired_count=self.unpaired_count,
            review_interval_operator=self.review_interval_operator,
            review_interval_value=self.review_interval_value,
            summary=self.summary,
            show_small_buckets=self.show_small_buckets,
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
        if command.action not in self.actions:
            raise BridgePayloadError(f"Unhandled graph action: {command.action}")
        if command.action == CALCULATE_ACTION:
            values = self._submission_values(command.payload)
            comparison_changed = self.comparison_generation != values[6]
            self._set_form_values(values)
            if comparison_changed:
                self._refresh_summary()
            effective_search = self.initial_search.strip() or self.fallback_search
            decision = self.analysis_state.decide(effective_search)
            self.analysis_state.clear_error()
            if decision is AnalysisSubmissionDecision.CALCULATE_THEN_FILTER or (
                self.include_fsrs and not self.comparison_loaded
            ):
                self.analysis_state.focus_target = "initial_search"
                self._on_calculate_requested(effective_search)
                return {"updated": True}
            self.analysis_state.focus_target = "filter_search"
            updated = self._apply_filters()
            self._request_rerender()
            return {"updated": updated}

        if command.action in (
            SET_SAME_DAY_ACTION,
            SET_FSRS_ACTION,
            SET_SMALL_BUCKETS_ACTION,
            SET_COMPARISON_ACTION,
        ):
            values = self._submission_values(command.payload)
            summary_changed = (
                self.include_sameday != values[1]
                or (not self.um_plus and self.include_fsrs != values[5])
                or self.comparison_generation != values[6]
            )
            self._set_form_values(values)
            effective_search = self.initial_search.strip() or self.fallback_search
            self.analysis_state.record_source(effective_search)
            self.analysis_state.clear_error(
                focus_target=(
                    "comparison_generation"
                    if command.action == SET_COMPARISON_ACTION
                    else "show_small_buckets"
                    if command.action == SET_SMALL_BUCKETS_ACTION
                    else "include_fsrs"
                    if command.action == SET_FSRS_ACTION
                    else "include_sameday"
                )
            )
            if summary_changed:
                self._refresh_summary()
            self._request_rerender()
            return {"updated": True}

        if command.action == TOGGLE_REVIEW_SOURCE_ACTION:
            expanded = self.analysis_state.set_source_expanded(_expanded_value(command.payload))
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

    def set_prediction_rows(
        self,
        rows: Iterable[dict],
        *,
        fsrs_included: bool = False,
        comparison_generation: str | None = None,
    ) -> None:
        self.prediction_rows = list(rows)
        self.fsrs_loaded = fsrs_included
        self._loaded_comparison_generation = comparison_generation
        self.analysis_state.accept_result(
            self.initial_search.strip() or self.fallback_search,
            focus_target="filter_search",
        )
        self._apply_filters(replace_on_error=True)
        self._request_rerender()

    def _submission_values(self, payload):
        values = _submission_form_values(payload, um_plus=self.um_plus)
        if values[6] is not None and values[6] not in self.compatible_generations:
            raise BridgePayloadError("This comparison generation cannot use the active settings.")
        return values

    def _set_form_values(self, values: tuple[str, bool, str, str, str, bool, str | None]) -> None:
        (
            initial_search,
            include_sameday,
            filter_search,
            operator,
            value_text,
            chart_option,
            comparison_generation,
        ) = values
        self.initial_search = initial_search
        self.include_sameday = include_sameday
        if self.um_plus:
            self.show_small_buckets = chart_option
        else:
            self.include_fsrs = chart_option
            if (
                comparison_generation is not None
                and comparison_generation != self.comparison_generation
            ):
                self.include_fsrs = True
        self.comparison_generation = comparison_generation
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
                allowed = {int(card_id) for card_id in self._search_card_ids(self.filter_search)}
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
        self.fsrs_summary = None
        self.unpaired_count = 0
        if self.um_plus:
            self._summary = summarize_um_plus_rows(
                self.displayed_rows if self.comparison_loaded else (),
                include_sameday=self.include_sameday,
            )
            return
        if self.include_fsrs and self.comparison_loaded:
            comparison = summarize_calibration_comparison(
                self.displayed_rows,
                bin_count=self.bin_count,
                include_sameday=self.include_sameday,
            )
            self._summary, self.fsrs_summary, self.unpaired_count = comparison
            return
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
    *,
    um_plus: bool = False,
) -> tuple[str, bool, str, str, str, bool, str | None]:
    chart_option_key = "show_small_buckets" if um_plus else "include_fsrs"
    _require_payload_keys(
        payload,
        expected={
            "initial_search",
            "include_sameday",
            chart_option_key,
            "filter_search",
            "review_interval_operator",
            "review_interval_value",
        }
        | ({"comparison_generation"} if "comparison_generation" in payload else set()),
        label="Calibration submission",
    )
    initial_search = payload["initial_search"]
    include_sameday = payload["include_sameday"]
    chart_option = payload[chart_option_key]
    if not isinstance(initial_search, str) or not isinstance(include_sameday, bool):
        raise BridgePayloadError(
            "Calibration search must be text and same-day selection must be true or false."
        )
    if not isinstance(chart_option, bool):
        label = "Small-bucket" if um_plus else "FSRS"
        raise BridgePayloadError(f"Graph {label} selection must be true or false.")
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
    comparison = payload.get("comparison_generation", "")
    if not isinstance(comparison, str):
        raise BridgePayloadError("Comparison generation must be text.")
    return (
        initial_search,
        include_sameday,
        filter_search,
        operator,
        value_text,
        chart_option,
        comparison or None,
    )


def _expanded_value(payload: Mapping[str, Any]) -> bool:
    _require_payload_keys(
        payload,
        expected={"expanded"},
        label="Calibration review-source state",
    )
    expanded = payload["expanded"]
    if not isinstance(expanded, bool):
        raise BridgePayloadError("Calibration review-source expanded state must be true or false.")
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
        raise BridgePayloadError(f"{label} is missing: " + ", ".join(sorted(missing)) + ".")
    if extra:
        raise BridgePayloadError(
            f"{label} contains unsupported fields: " + ", ".join(sorted(extra)) + "."
        )
