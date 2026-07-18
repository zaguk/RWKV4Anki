from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .anki_api import FsrsEvaluationMode
from .evaluate_html import render_evaluate_dialog_html
from .evaluation_date_range import EvaluationDateRange
from .evaluation_predictions import EvaluationHistoryRevision
from .evaluation_table import (
    EvaluationResultCache,
    comparison_states,
    format_error,
    format_metric,
    format_relative_ratio,
    is_insufficient_reviews_error,
)
from .metrics import EvaluationScope, MetricResult, RWKVPredictionMode
from .web_dialog_bridge import BridgePayloadError, WebDialogCommand
from .web_dialog_controller import BaseWebDialogController, CloseReason

UPDATE_OPTIONS_ACTION = "update-options"
USE_FULL_RANGE_ACTION = "use-full-range"
RUN_COMPARISON_ACTION = "run-comparison"
RUN_RWKV_ONLY_ACTION = "run-rwkv-only"
TOGGLE_ADVANCED_OPTIONS_ACTION = "toggle-advanced-options"

_FORM_KEYS = frozenset(
    {
        "include_collection",
        "include_presets",
        "include_decks",
        "start_date",
        "end_date",
        "cheating_fsrs",
        "show_insufficient",
    }
)


@dataclass(frozen=True)
class EvaluationScopeSelection:
    include_collection: bool = True
    include_presets: bool = False
    include_decks: bool = False

    @property
    def cache_key(self) -> tuple[bool, bool, bool]:
        return (
            self.include_collection,
            self.include_presets,
            self.include_decks,
        )


@dataclass(frozen=True)
class EvaluationRunRequest:
    include_fsrs: bool
    scopes: tuple[EvaluationScope, ...]
    date_range: EvaluationDateRange
    fsrs_mode: FsrsEvaluationMode
    prediction_mode: RWKVPredictionMode


@dataclass(frozen=True)
class EvaluationDisplayRow:
    scope: str
    reviews_evaluated: str
    fsrs_rmse: str = ""
    rwkv_rmse: str = ""
    rmse_improvement: str = ""
    fsrs_logloss: str = ""
    rwkv_logloss: str = ""
    logloss_improvement: str = ""
    rmse_improvement_state: str = ""
    logloss_improvement_state: str = ""


ScopeBuilder = Callable[[EvaluationScopeSelection], Sequence[EvaluationScope]]
RunCallback = Callable[[EvaluationRunRequest], bool]
WarningCallback = Callable[[str], None]


class EvaluateController(BaseWebDialogController):
    """Qt-independent state, validation, caching, and table shaping for Evaluate."""

    actions = frozenset(
        {
            UPDATE_OPTIONS_ACTION,
            USE_FULL_RANGE_ACTION,
            RUN_COMPARISON_ACTION,
            RUN_RWKV_ONLY_ACTION,
            TOGGLE_ADVANCED_OPTIONS_ACTION,
        }
    )

    def __init__(
        self,
        *,
        title: str,
        rwkv_label: str,
        prediction_mode: RWKVPredictionMode,
        full_collection_start_date: dt.date,
        full_collection_end_date: dt.date,
        build_scopes: ScopeBuilder,
        on_run_requested: RunCallback,
        on_warning: WarningCallback,
        is_dark: bool = False,
    ) -> None:
        if not callable(build_scopes):
            raise TypeError("build_scopes must be callable")
        if not callable(on_run_requested):
            raise TypeError("on_run_requested must be callable")
        if not callable(on_warning):
            raise TypeError("on_warning must be callable")
        if full_collection_end_date < full_collection_start_date:
            raise ValueError("full collection date bounds are reversed")

        self.title = str(title)
        self.rwkv_label = str(rwkv_label)
        self.prediction_mode = prediction_mode
        self.full_collection_start_date = full_collection_start_date
        self.full_collection_end_date = full_collection_end_date
        self.scope_selection = EvaluationScopeSelection()
        self.start_date = full_collection_start_date
        self.end_date = full_collection_end_date
        self.cheating_fsrs = False
        self.show_insufficient = False
        self.show_comparison = True
        self.running = False
        self.focus_target: str | None = None
        self.advanced_options_expanded = False
        self.is_dark = bool(is_dark)

        self._build_scopes = build_scopes
        self._on_run_requested = on_run_requested
        self._on_warning = on_warning
        self._rerender: Callable[[], Any] | None = None
        self._scope_descriptor_cache: dict[
            tuple[bool, bool, bool], tuple[EvaluationScope, ...]
        ] = {}
        self.scope_descriptors = self._current_scope_descriptors()

        self._fsrs_result_cache: EvaluationResultCache[MetricResult] = EvaluationResultCache()
        self._rwkv_result_cache: EvaluationResultCache[MetricResult] = EvaluationResultCache()
        self._fsrs_results: dict[str, MetricResult] = {}
        self._rwkv_results: dict[str, MetricResult] = {}
        self._fsrs_review_counts: dict[str, int] = {}
        self._rwkv_review_counts: dict[str, int] = {}
        self._result_history_revision: EvaluationHistoryRevision | None = None

    @property
    def fsrs_mode(self) -> FsrsEvaluationMode:
        if self.cheating_fsrs:
            return FsrsEvaluationMode.LEGACY_TRAIN_SET
        return FsrsEvaluationMode.TIME_SERIES

    @property
    def date_range_covers_full_collection(self) -> bool:
        return (
            self.start_date <= self.end_date
            and self.start_date <= self.full_collection_start_date
            and self.end_date >= self.full_collection_end_date
        )

    @property
    def display_rows(self) -> tuple[EvaluationDisplayRow, ...]:
        rows: list[EvaluationDisplayRow] = []
        for scope in self.scope_descriptors:
            if not self.show_insufficient and self._scope_has_insufficient_reviews(scope.key):
                continue
            rows.append(self._display_row(scope))
        return tuple(rows)

    @property
    def hidden_insufficient_count(self) -> int:
        if self.show_insufficient:
            return 0
        return sum(
            self._scope_has_insufficient_reviews(scope.key) for scope in self.scope_descriptors
        )

    def attach_rerender(self, rerender: Callable[[], Any]) -> None:
        if not callable(rerender):
            raise TypeError("Evaluate rerender callback must be callable")
        self._rerender = rerender

    def render_html(self, generation: int) -> str:
        return render_evaluate_dialog_html(
            title=self.title,
            rwkv_label=self.rwkv_label,
            scope_selection=self.scope_selection,
            start_date=self.start_date,
            end_date=self.end_date,
            full_collection_start_date=self.full_collection_start_date,
            full_collection_end_date=self.full_collection_end_date,
            cheating_fsrs=self.cheating_fsrs,
            show_insufficient=self.show_insufficient,
            show_comparison=self.show_comparison,
            comparison_available=self.date_range_covers_full_collection,
            rows=self.display_rows,
            selected_scope_count=len(self.scope_descriptors),
            hidden_insufficient_count=self.hidden_insufficient_count,
            running=self.running,
            focus_target=self.focus_target,
            advanced_options_expanded=self.advanced_options_expanded,
            is_dark=self.is_dark,
            generation=generation,
        )

    def handle_command(self, command: WebDialogCommand) -> dict[str, bool]:
        if command.action == TOGGLE_ADVANCED_OPTIONS_ACTION:
            self.advanced_options_expanded = _expanded_value(
                command.payload,
                label="Evaluate advanced options",
            )
            return {"expanded": self.advanced_options_expanded}

        if command.action == UPDATE_OPTIONS_ACTION:
            self._apply_form_payload(command.payload)
            self.focus_target = None
            self._request_rerender()
            return {"updated": True}

        if command.action == USE_FULL_RANGE_ACTION:
            self._apply_form_payload(command.payload)
            self.start_date = self.full_collection_start_date
            self.end_date = self.full_collection_end_date
            self.advanced_options_expanded = True
            self.focus_target = "start_date"
            self._select_cached_results()
            self._request_rerender()
            return {"updated": True}

        if command.action == RUN_COMPARISON_ACTION:
            self._apply_form_payload(command.payload)
            return {"started": self.request_evaluation(include_fsrs=True)}

        if command.action == RUN_RWKV_ONLY_ACTION:
            self._apply_form_payload(command.payload)
            return {"started": self.request_evaluation(include_fsrs=False)}

        raise BridgePayloadError(f"Unhandled Evaluate action: {command.action}")

    def is_action_enabled(self, action: str) -> bool:
        if action not in self.actions or self.running:
            return False
        if action == RUN_COMPARISON_ACTION:
            return self.date_range_covers_full_collection
        return True

    def can_close(self, reason: CloseReason) -> bool:
        del reason
        return not self.running

    def request_evaluation(self, *, include_fsrs: bool) -> bool:
        if self.running:
            return False
        try:
            date_range = EvaluationDateRange.from_local_dates(
                self.start_date,
                self.end_date,
            )
        except (OverflowError, ValueError) as exc:
            self._on_warning(str(exc))
            self.advanced_options_expanded = True
            self.focus_target = "start_date"
            self._request_rerender()
            return False
        if include_fsrs and not self.date_range_covers_full_collection:
            self._on_warning(
                "Anki does not expose date-filtered FSRS-6 metrics. Use Full Range "
                "to compare FSRS-6 and RWKV, or run RWKV Only for the selected dates."
            )
            self.advanced_options_expanded = True
            self.focus_target = "start_date"
            self._request_rerender()
            return False
        if not self.scope_descriptors:
            self._on_warning("Select at least one evaluation scope.")
            self.focus_target = "include_collection"
            self._request_rerender()
            return False
        request = EvaluationRunRequest(
            include_fsrs=bool(include_fsrs),
            scopes=tuple(self.scope_descriptors),
            date_range=date_range,
            fsrs_mode=self.fsrs_mode,
            prediction_mode=self.prediction_mode,
        )
        started = bool(self._on_run_requested(request))
        if not started:
            self._request_rerender()
        return started

    def begin_evaluation(self, *, include_fsrs: bool) -> bool:
        if self.running:
            return False
        self.show_comparison = bool(include_fsrs)
        self.running = True
        self.focus_target = None
        self._select_cached_results()
        self._request_rerender()
        return True

    def finish_evaluation(self) -> None:
        if not self.running:
            return
        self.running = False
        self.focus_target = "run_comparison" if self.show_comparison else "run_rwkv_only"
        self._request_rerender()

    def apply_evaluation_results(
        self,
        request: EvaluationRunRequest,
        *,
        fsrs_results: Mapping[str, MetricResult],
        fsrs_counts: Mapping[str, int],
        rwkv_results: Mapping[str, MetricResult],
        rwkv_counts: Mapping[str, int],
        history_revision: EvaluationHistoryRevision,
    ) -> None:
        self._adopt_history_revision(history_revision)
        if request.include_fsrs:
            self._store_fsrs_results(
                request.fsrs_mode.value,
                fsrs_results,
                fsrs_counts,
            )
        self._store_rwkv_results(
            self.rwkv_cache_key(request.prediction_mode, request.date_range),
            rwkv_results,
            rwkv_counts,
        )
        self._select_cached_results()
        self.running = False
        self.focus_target = "run_comparison" if request.include_fsrs else "run_rwkv_only"
        self._request_rerender()

    def rwkv_cache_key(
        self,
        mode: RWKVPredictionMode | None = None,
        date_range: EvaluationDateRange | None = None,
    ) -> tuple[str, str, str]:
        range_key = (
            (self.start_date.isoformat(), self.end_date.isoformat())
            if date_range is None
            else date_range.cache_key
        )
        return (mode or self.prediction_mode).value, *range_key

    def _apply_form_payload(self, payload: Mapping[str, Any]) -> None:
        values = _evaluation_form_values(payload)
        old_selection = self.scope_selection
        old_dates = (self.start_date, self.end_date)
        old_fsrs_mode = self.fsrs_mode
        old_show_insufficient = self.show_insufficient

        self.scope_selection = values.scope_selection
        self.start_date = values.start_date
        self.end_date = values.end_date
        self.cheating_fsrs = values.cheating_fsrs
        self.show_insufficient = values.show_insufficient

        if self.scope_selection != old_selection:
            self.scope_descriptors = self._current_scope_descriptors()
        if (self.start_date, self.end_date) != old_dates:
            self.advanced_options_expanded = True
            if not self.date_range_covers_full_collection:
                self.show_comparison = False
            self._select_cached_results()
        elif self.fsrs_mode != old_fsrs_mode:
            self._select_cached_fsrs_results()
        if self.fsrs_mode != old_fsrs_mode or self.show_insufficient != old_show_insufficient:
            self.advanced_options_expanded = True

    def _current_scope_descriptors(self) -> tuple[EvaluationScope, ...]:
        key = self.scope_selection.cache_key
        cached = self._scope_descriptor_cache.get(key)
        if cached is None:
            cached = tuple(self._build_scopes(self.scope_selection))
            self._scope_descriptor_cache[key] = cached
        return cached

    def _display_row(self, scope: EvaluationScope) -> EvaluationDisplayRow:
        fsrs = self._fsrs_results.get(scope.key) if self.show_comparison else None
        rwkv = self._rwkv_results.get(scope.key)
        fsrs_rmse, fsrs_logloss = _result_values(fsrs)
        rwkv_rmse, rwkv_logloss = _result_values(rwkv)
        rmse_improvement = ""
        logloss_improvement = ""
        rmse_improvement_state = ""
        logloss_improvement_state = ""
        if _has_metrics(fsrs) and _has_metrics(rwkv):
            rmse_improvement = format_relative_ratio(fsrs.rmse_bins, rwkv.rmse_bins)
            logloss_improvement = format_relative_ratio(fsrs.log_loss, rwkv.log_loss)
            rmse_improvement_state = comparison_states(
                fsrs.rmse_bins,
                rwkv.rmse_bins,
            )[1]
            logloss_improvement_state = comparison_states(
                fsrs.log_loss,
                rwkv.log_loss,
            )[1]
        return EvaluationDisplayRow(
            scope=scope.label,
            reviews_evaluated=self._review_count(scope.key),
            fsrs_rmse=fsrs_rmse,
            rwkv_rmse=rwkv_rmse,
            rmse_improvement=rmse_improvement,
            fsrs_logloss=fsrs_logloss,
            rwkv_logloss=rwkv_logloss,
            logloss_improvement=logloss_improvement,
            rmse_improvement_state=rmse_improvement_state,
            logloss_improvement_state=logloss_improvement_state,
        )

    def _review_count(self, scope_key: str) -> str:
        fsrs_count = self._fsrs_review_counts.get(scope_key)
        rwkv_count = self._rwkv_review_counts.get(scope_key)
        if not self.show_comparison:
            return "" if rwkv_count is None else str(rwkv_count)
        if fsrs_count is not None and rwkv_count is not None and fsrs_count != rwkv_count:
            return f"FSRS {fsrs_count} / RWKV {rwkv_count}"
        count = rwkv_count if rwkv_count is not None else fsrs_count
        return "" if count is None else str(count)

    def _scope_has_insufficient_reviews(self, scope_key: str) -> bool:
        rwkv_insufficient = _is_insufficient(self._rwkv_results.get(scope_key))
        if not self.show_comparison:
            return rwkv_insufficient
        return _is_insufficient(self._fsrs_results.get(scope_key)) or rwkv_insufficient

    def _adopt_history_revision(self, revision: EvaluationHistoryRevision) -> None:
        current = self._result_history_revision
        if current is not None and current != revision:
            self._fsrs_result_cache.clear()
            self._rwkv_result_cache.clear()
            self._fsrs_results = {}
            self._rwkv_results = {}
            self._fsrs_review_counts = {}
            self._rwkv_review_counts = {}
        self._result_history_revision = revision

    def _store_fsrs_results(
        self,
        cache_key: Hashable,
        results: Mapping[str, MetricResult],
        counts: Mapping[str, int],
    ) -> None:
        cached = self._fsrs_result_cache.get(cache_key)
        cached.results.update(results)
        cached.review_counts.update(counts)
        self._fsrs_result_cache.store(
            cache_key,
            cached.results,
            cached.review_counts,
        )

    def _store_rwkv_results(
        self,
        cache_key: Hashable,
        results: Mapping[str, MetricResult],
        counts: Mapping[str, int],
    ) -> None:
        cached = self._rwkv_result_cache.get(cache_key)
        cached.results.update(results)
        cached.review_counts.update(counts)
        self._rwkv_result_cache.store(
            cache_key,
            cached.results,
            cached.review_counts,
        )

    def _select_cached_fsrs_results(self) -> None:
        cached = self._fsrs_result_cache.get(self.fsrs_mode.value)
        self._fsrs_results = cached.results
        self._fsrs_review_counts = cached.review_counts

    def _select_cached_results(self) -> None:
        self._select_cached_fsrs_results()
        cached = self._rwkv_result_cache.get(self.rwkv_cache_key())
        self._rwkv_results = cached.results
        self._rwkv_review_counts = cached.review_counts

    def _request_rerender(self) -> None:
        if self._rerender is None:
            raise RuntimeError("Evaluate controller is not attached to its dialog")
        self._rerender()


@dataclass(frozen=True)
class _EvaluationFormValues:
    scope_selection: EvaluationScopeSelection
    start_date: dt.date
    end_date: dt.date
    cheating_fsrs: bool
    show_insufficient: bool


def _evaluation_form_values(payload: Mapping[str, Any]) -> _EvaluationFormValues:
    missing = _FORM_KEYS - set(payload)
    extra = set(payload) - _FORM_KEYS
    if missing:
        raise BridgePayloadError(
            "Evaluate options are missing: " + ", ".join(sorted(missing)) + "."
        )
    if extra:
        raise BridgePayloadError(
            "Evaluate options contain unsupported fields: " + ", ".join(sorted(extra)) + "."
        )
    boolean_keys = _FORM_KEYS - {"start_date", "end_date"}
    if any(not isinstance(payload[key], bool) for key in boolean_keys):
        raise BridgePayloadError("Evaluate checkbox values must be true or false.")
    if not isinstance(payload["start_date"], str) or not isinstance(payload["end_date"], str):
        raise BridgePayloadError("Evaluate date values must be text.")
    try:
        start_date = dt.date.fromisoformat(payload["start_date"])
        end_date = dt.date.fromisoformat(payload["end_date"])
    except ValueError as exc:
        raise BridgePayloadError("Evaluate dates must use YYYY-MM-DD.") from exc
    return _EvaluationFormValues(
        scope_selection=EvaluationScopeSelection(
            include_collection=payload["include_collection"],
            include_presets=payload["include_presets"],
            include_decks=payload["include_decks"],
        ),
        start_date=start_date,
        end_date=end_date,
        cheating_fsrs=payload["cheating_fsrs"],
        show_insufficient=payload["show_insufficient"],
    )


def _expanded_value(payload: Mapping[str, Any], *, label: str) -> bool:
    if set(payload) != {"expanded"}:
        raise BridgePayloadError(f"{label} payload must contain only an expanded value.")
    expanded = payload["expanded"]
    if not isinstance(expanded, bool):
        raise BridgePayloadError(f"{label} expanded state must be true or false.")
    return expanded


def _result_values(result: MetricResult | None) -> tuple[str, str]:
    if result is None:
        return "", ""
    if result.error:
        return (
            format_error(result.error),
            "" if _is_insufficient(result) else result.error,
        )
    return format_metric(result.rmse_bins), format_metric(result.log_loss)


def _has_metrics(result: MetricResult | None) -> bool:
    return result is not None and not result.error


def _is_insufficient(result: MetricResult | None) -> bool:
    return result is not None and is_insufficient_reviews_error(result.error)
