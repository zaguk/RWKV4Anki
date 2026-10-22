from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
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
from .model_generation import (
    DEFAULT_MODEL_GENERATION,
    compatible_evaluation_generations,
    model_generation_label,
)
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
    comparison_generation: str | None = None

    @property
    def include_comparison(self) -> bool:
        return self.include_fsrs or self.comparison_generation is not None


@dataclass(frozen=True)
class EvaluationSortValues:
    scope: str = ""
    reviews: int | None = None
    fsrs_rmse: float | None = None
    rwkv_rmse: float | None = None
    rmse_improvement: float | None = None
    fsrs_logloss: float | None = None
    rwkv_logloss: float | None = None
    logloss_improvement: float | None = None
    rwkv_auc: float | None = None
    fsrs_auc: float | None = None
    auc_improvement: float | None = None


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
    scope_tooltip: str = ""
    sort_values: EvaluationSortValues = field(default_factory=EvaluationSortValues)
    rwkv_auc: str = ""
    fsrs_auc: str = ""
    auc_improvement: str = ""
    auc_improvement_state: str = ""


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
        active_generation: str = DEFAULT_MODEL_GENERATION,
        include_deleted_reviews: bool = False,
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
        self.active_generation = active_generation
        self.comparison_generations = compatible_evaluation_generations(
            active_generation,
            prediction_mode=prediction_mode.value,
            include_deleted_reviews=include_deleted_reviews,
        )
        self.comparison_generation: str | None = None
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
        self._comparison_model_ids: dict[str, str] = {}

    @property
    def fsrs_mode(self) -> FsrsEvaluationMode:
        if self.cheating_fsrs and self.comparison_generation is None:
            return FsrsEvaluationMode.LEGACY_TRAIN_SET
        return FsrsEvaluationMode.TIME_SERIES

    @property
    def comparison_available(self) -> bool:
        return self.comparison_generation is not None or self.date_range_covers_full_collection

    @property
    def comparison_label(self) -> str:
        return (
            model_generation_label(self.comparison_generation)
            if self.comparison_generation else "FSRS"
        )

    @property
    def comparison_button_label(self) -> str:
        if self.comparison_generation:
            return (
                f"Compare RWKV {self.comparison_label} with active RWKV "
                f"{model_generation_label(self.active_generation)}"
            )
        return f"Compare FSRS-6 and {self.rwkv_label}"

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
            comparison_available=self.comparison_available,
            comparison_generation=self.comparison_generation,
            comparison_generations=self.comparison_generations,
            active_generation=self.active_generation,
            comparison_button_label=self.comparison_button_label,
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
            # Reset also recovers an unfinished/invalid typed date. Preserve
            # the other form choices, but do not parse dates we are replacing.
            self._apply_form_payload({
                **command.payload,
                "start_date": self.full_collection_start_date.isoformat(),
                "end_date": self.full_collection_end_date.isoformat(),
            })
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
            return self.comparison_available
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
        if include_fsrs and not self.comparison_available:
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
            include_fsrs=bool(include_fsrs) and self.comparison_generation is None,
            scopes=tuple(self.scope_descriptors),
            date_range=date_range,
            fsrs_mode=self.fsrs_mode,
            prediction_mode=self.prediction_mode,
            comparison_generation=self.comparison_generation if include_fsrs else None,
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
        comparison_model_id: str | None = None,
    ) -> None:
        if request.comparison_generation and comparison_model_id is not None:
            previous = self._comparison_model_ids.get(request.comparison_generation)
            if previous is not None and previous != comparison_model_id:
                # Another Settings/Setup comparison may have replaced the saved
                # model for this generation while this window remained open.
                self._fsrs_result_cache.clear()
                self._rwkv_result_cache.clear()
            self._comparison_model_ids[request.comparison_generation] = comparison_model_id
        self._adopt_history_revision(history_revision)
        if request.include_comparison:
            self._store_fsrs_results(
                self._comparison_cache_key(
                    request.comparison_generation, request.prediction_mode,
                    request.date_range, request.fsrs_mode,
                ),
                fsrs_results,
                fsrs_counts,
            )
        self._store_rwkv_results(
            self.rwkv_cache_key(
                request.prediction_mode, request.date_range,
                comparison_generation=request.comparison_generation,
            ),
            rwkv_results,
            rwkv_counts,
        )
        self._select_cached_results()
        self.running = False
        self.focus_target = "run_comparison" if request.include_comparison else "run_rwkv_only"
        self._request_rerender()

    def rwkv_cache_key(
        self,
        mode: RWKVPredictionMode | None = None,
        date_range: EvaluationDateRange | None = None,
        *,
        comparison_generation: str | None = None,
    ) -> tuple[str, ...]:
        range_key = (
            (self.start_date.isoformat(), self.end_date.isoformat())
            if date_range is None
            else date_range.cache_key
        )
        key = ((mode or self.prediction_mode).value, *range_key)
        return (*key, comparison_generation) if comparison_generation else key

    def _comparison_cache_key(self, generation, mode=None, date_range=None, fsrs_mode=None):
        if generation:
            return ("rwkv-generation", generation, *self.rwkv_cache_key(mode, date_range))
        return (fsrs_mode or self.fsrs_mode).value

    def _apply_form_payload(self, payload: Mapping[str, Any]) -> None:
        values = _evaluation_form_values(payload)
        if (
            values.comparison_generation is not None
            and values.comparison_generation not in self.comparison_generations
        ):
            raise BridgePayloadError("This generation cannot use the active evaluation settings.")
        old_selection = self.scope_selection
        old_dates = (self.start_date, self.end_date)
        old_fsrs_mode = self.fsrs_mode
        old_show_insufficient = self.show_insufficient
        old_generation = self.comparison_generation

        self.scope_selection = values.scope_selection
        self.start_date = values.start_date
        self.end_date = values.end_date
        self.comparison_generation = values.comparison_generation
        if self.comparison_generation is None:
            self.cheating_fsrs = values.cheating_fsrs
        self.show_insufficient = values.show_insufficient

        if self.scope_selection != old_selection:
            self.scope_descriptors = self._current_scope_descriptors()
        if (self.start_date, self.end_date) != old_dates:
            self.advanced_options_expanded = True
            if not self.comparison_available:
                self.show_comparison = False
            self._select_cached_results()
        elif self.fsrs_mode != old_fsrs_mode:
            self._select_cached_fsrs_results()
        if self.fsrs_mode != old_fsrs_mode or self.show_insufficient != old_show_insufficient:
            self.advanced_options_expanded = True
        if self.comparison_generation != old_generation:
            self.advanced_options_expanded = True
            self.show_comparison = self.comparison_available
            self._select_cached_results()

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
        fsrs_rmse_sort = _sortable_metric(fsrs, "rmse_bins")
        rwkv_rmse_sort = _sortable_metric(rwkv, "rmse_bins")
        fsrs_logloss_sort = _sortable_metric(fsrs, "log_loss")
        rwkv_logloss_sort = _sortable_metric(rwkv, "log_loss")
        rwkv_auc_sort = _sortable_metric(rwkv, "auc")
        fsrs_auc_sort = _sortable_metric(fsrs, "auc")
        auc_improvement = ""
        auc_improvement_state = ""
        auc_improvement_sort = None
        if self.show_comparison and self.comparison_generation is not None:
            auc_improvement = format_relative_ratio(fsrs_auc_sort, rwkv_auc_sort)
            if auc_improvement:
                auc_improvement_state = comparison_states(
                    fsrs_auc_sort, rwkv_auc_sort, higher_is_better=True,
                )[1]
                auc_improvement_sort = _sortable_ratio(fsrs_auc_sort, rwkv_auc_sort)
            else:
                auc_improvement = "—"
        rwkv_auc = ""
        if _has_metrics(rwkv):
            rwkv_auc = "—" if rwkv_auc_sort is None else f"{rwkv_auc_sort:.6f}"
        review_count_sort = self._rwkv_review_counts.get(scope.key)
        if review_count_sort is None:
            review_count_sort = self._fsrs_review_counts.get(scope.key)
        return EvaluationDisplayRow(
            scope=scope.label,
            reviews_evaluated=self._review_count(scope.key),
            scope_tooltip=scope.label_tooltip or "",
            fsrs_rmse=fsrs_rmse,
            rwkv_rmse=rwkv_rmse,
            rmse_improvement=rmse_improvement,
            fsrs_logloss=fsrs_logloss,
            rwkv_logloss=rwkv_logloss,
            logloss_improvement=logloss_improvement,
            rmse_improvement_state=rmse_improvement_state,
            logloss_improvement_state=logloss_improvement_state,
            rwkv_auc=rwkv_auc,
            fsrs_auc=(
                "" if not _has_metrics(fsrs)
                else "—" if fsrs_auc_sort is None else f"{fsrs_auc_sort:.6f}"
            ),
            auc_improvement=auc_improvement,
            auc_improvement_state=auc_improvement_state,
            sort_values=EvaluationSortValues(
                scope=scope.label.casefold(),
                reviews=review_count_sort,
                fsrs_rmse=fsrs_rmse_sort,
                rwkv_rmse=rwkv_rmse_sort,
                rmse_improvement=_sortable_ratio(fsrs_rmse_sort, rwkv_rmse_sort),
                fsrs_logloss=fsrs_logloss_sort,
                rwkv_logloss=rwkv_logloss_sort,
                logloss_improvement=_sortable_ratio(
                    fsrs_logloss_sort,
                    rwkv_logloss_sort,
                ),
                rwkv_auc=rwkv_auc_sort,
                fsrs_auc=fsrs_auc_sort,
                auc_improvement=auc_improvement_sort,
            ),
        )

    def _review_count(self, scope_key: str) -> str:
        fsrs_count = self._fsrs_review_counts.get(scope_key)
        rwkv_count = self._rwkv_review_counts.get(scope_key)
        if not self.show_comparison:
            return "" if rwkv_count is None else str(rwkv_count)
        if fsrs_count is not None and rwkv_count is not None and fsrs_count != rwkv_count:
            active = (
                model_generation_label(self.active_generation)
                if self.comparison_generation else "RWKV"
            )
            return f"{self.comparison_label} {fsrs_count} / {active} {rwkv_count}"
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
        cached = self._fsrs_result_cache.get(self._comparison_cache_key(self.comparison_generation))
        self._fsrs_results = cached.results
        self._fsrs_review_counts = cached.review_counts

    def _select_cached_results(self) -> None:
        self._select_cached_fsrs_results()
        cached = self._rwkv_result_cache.get(self.rwkv_cache_key(
            comparison_generation=self.comparison_generation if self.show_comparison else None,
        ))
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
    comparison_generation: str | None


def _evaluation_form_values(payload: Mapping[str, Any]) -> _EvaluationFormValues:
    missing = _FORM_KEYS - set(payload)
    extra = set(payload) - _FORM_KEYS - {"comparison_generation"}
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
    generation = payload.get("comparison_generation", "")
    if not isinstance(generation, str):
        raise BridgePayloadError("Evaluate comparison generation must be text.")
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
        comparison_generation=generation or None,
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


def _sortable_metric(result: MetricResult | None, attribute: str) -> float | None:
    if not _has_metrics(result):
        return None
    value = getattr(result, attribute)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _sortable_ratio(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or reference <= 0 or candidate < 0:
        return None
    return candidate / reference


def _is_insufficient(result: MetricResult | None) -> bool:
    return result is not None and is_insufficient_reviews_error(result.error)
