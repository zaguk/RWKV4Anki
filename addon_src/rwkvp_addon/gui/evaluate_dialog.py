from __future__ import annotations

from aqt import mw

from ..addon_config import addon_config_for_mw, calculate_forgetting_curves
from ..anki_api import (
    FsrsEvaluationMode,
    build_evaluation_scope_descriptors,
    evaluate_fsrs_scope_plans,
    fsrs_evaluation_review_ids_for_plans,
    fsrs_scope_plans,
    resolve_evaluation_scopes,
)
from ..dataset_export import (
    collection_review_id_bounds,
    ensure_checkpoint_ready_from_load,
    load_review_data_for_checkpoint,
)
from ..evaluate_controller import (
    EvaluateController,
    EvaluationRunRequest,
    EvaluationScopeSelection,
)
from ..evaluation_date_range import (
    filter_target_review_ids_by_date_range,
    review_date_bounds_for_ids,
)
from ..evaluation_predictions import load_current_evaluation_prediction_snapshot
from ..metrics import RWKVPredictionMode, calculate_aligned_scoped_metrics
from ..runtime import manager_for_mw, store_for_mw
from .checkpoint_failure import handle_checkpoint_failure
from .common import ProgressStage, run_with_progress_stages
from .web_dialog import WebDialogHost, widget_uses_dark_palette
from .web_message import show_web_warning


class EvaluateDialog(WebDialogHost):
    def __init__(
        self,
        parent=None,
        *,
        rwkv_mode: RWKVPredictionMode = RWKVPredictionMode.PER_REVIEW,
        title: str = "RWKV Evaluation",
        rwkv_label: str = "RWKV",
    ) -> None:
        self.rwkv_mode = rwkv_mode
        self.rwkv_label = rwkv_label
        first_review_id, last_review_id = collection_review_id_bounds(mw.col)
        full_start, full_end = review_date_bounds_for_ids(
            first_review_id,
            last_review_id,
        )
        self._evaluate_controller = EvaluateController(
            title=title,
            rwkv_label=rwkv_label,
            prediction_mode=rwkv_mode,
            full_collection_start_date=full_start,
            full_collection_end_date=full_end,
            build_scopes=self._build_scope_descriptors,
            on_run_requested=self._start_evaluation,
            on_warning=self._show_warning,
            is_dark=widget_uses_dark_palette(parent),
        )
        super().__init__(
            parent,
            title=title,
            controller=self._evaluate_controller,
            size=(1100, 650),
            requires_collection=True,
        )
        self._evaluate_controller.attach_rerender(self.rerender)

    @property
    def evaluate_controller(self) -> EvaluateController:
        return self._evaluate_controller

    @property
    def scope_descriptors(self):
        return self._evaluate_controller.scope_descriptors

    def run_comparison(self) -> bool:
        return self._evaluate_controller.request_evaluation(include_fsrs=True)

    def run_rwkv_only(self) -> bool:
        return self._evaluate_controller.request_evaluation(include_fsrs=False)

    def _start_evaluation(self, request: EvaluationRunRequest) -> bool:
        if (
            request.prediction_mode == RWKVPredictionMode.PREDICT_AHEAD
            and not calculate_forgetting_curves(addon_config_for_mw(mw))
        ):
            show_web_warning(
                "Calculate Forgetting Curves is disabled in RWKV Settings.",
                title="RWKV Forgetting Curve",
                parent=self,
            )
            return False

        store = store_for_mw(mw)
        manager = manager_for_mw(mw)
        if not self._evaluate_controller.begin_evaluation(
            include_fsrs=request.include_fsrs
        ):
            return False

        scope_descriptors = list(request.scopes)

        def resolve_op(col, progress, _previous):
            return resolve_evaluation_scopes(col, scope_descriptors, progress)

        def plan_op(col, progress, scopes):
            plans = fsrs_scope_plans(col, scopes)
            progress.update(0, max(1, len(plans)), "Prepared evaluation scopes")
            return plans, {}

        def evaluate_fsrs_op(col, progress, previous):
            plans, _fsrs_results = previous
            total_steps = max(1, len(plans))
            progress.update(0, total_steps, "Evaluating FSRS-6")
            fsrs_results = evaluate_fsrs_scope_plans(
                col,
                plans,
                progress,
                mode=request.fsrs_mode,
            )
            progress.update(total_steps, total_steps, "Evaluated FSRS-6")
            return plans, fsrs_results

        def export_op(col, progress, previous):
            plans, fsrs_results = previous
            review_load = load_review_data_for_checkpoint(
                col,
                store,
                manager,
                progress,
                allow_incremental=True,
            )
            return plans, fsrs_results, review_load

        def prepare_op(_col, progress, previous):
            plans, fsrs_results, review_load = previous
            readiness = ensure_checkpoint_ready_from_load(
                manager,
                review_load,
                progress,
                wait_for_save=True,
                capture_prediction_tail=True,
            )
            return plans, fsrs_results, readiness

        def target_op(_col, progress, previous):
            plans, fsrs_results, readiness = previous
            progress.update(0, 1, "Selecting evaluation review items")
            full_rwkv_target_review_ids = fsrs_evaluation_review_ids_for_plans(
                readiness.review_data,
                plans,
                mode=FsrsEvaluationMode.TIME_SERIES,
            )
            rwkv_target_review_ids = filter_target_review_ids_by_date_range(
                full_rwkv_target_review_ids,
                request.date_range,
            )
            if not request.include_fsrs:
                fsrs_target_review_ids = {}
            elif request.fsrs_mode == FsrsEvaluationMode.TIME_SERIES:
                fsrs_target_review_ids = full_rwkv_target_review_ids
            else:
                fsrs_target_review_ids = fsrs_evaluation_review_ids_for_plans(
                    readiness.review_data,
                    plans,
                    mode=request.fsrs_mode,
                )
            progress.update(1, 1, "Selected evaluation review items")
            return (
                plans,
                fsrs_results,
                readiness,
                fsrs_target_review_ids,
                rwkv_target_review_ids,
            )

        def metrics_op(_col, progress, previous):
            (
                plans,
                fsrs_results,
                readiness,
                fsrs_target_review_ids,
                rwkv_target_review_ids,
            ) = previous
            review_data = readiness.review_data
            scopes = [plan.scope for plan in plans]
            prediction_snapshot = load_current_evaluation_prediction_snapshot(
                readiness,
                store,
                manager,
                progress,
                prediction_mode=request.prediction_mode,
            )
            scoped_results = calculate_aligned_scoped_metrics(
                review_data.rows,
                prediction_snapshot.predictions,
                scopes,
                target_review_ids_by_scope=rwkv_target_review_ids,
            )
            return (
                fsrs_results,
                _target_counts(fsrs_target_review_ids),
                scoped_results.metrics,
                scoped_results.counts,
                prediction_snapshot.history_revision,
            )

        def success(payload) -> None:
            if self.cleaned_up:
                return
            fsrs_results, fsrs_counts, rwkv_results, rwkv_counts, revision = payload
            self._evaluate_controller.apply_evaluation_results(
                request,
                fsrs_results=fsrs_results,
                fsrs_counts=fsrs_counts,
                rwkv_results=rwkv_results,
                rwkv_counts=rwkv_counts,
                history_revision=revision,
            )

        def failure(exc: Exception) -> None:
            if self.cleaned_up:
                return
            self._evaluate_controller.finish_evaluation()
            retry = self.run_comparison if request.include_fsrs else self.run_rwkv_only
            handle_checkpoint_failure(exc, retry, parent=self)

        def cancelled() -> None:
            if not self.cleaned_up:
                self._evaluate_controller.finish_evaluation()

        stages = [
            ProgressStage(resolve_op, uses_collection=True),
            ProgressStage(plan_op, uses_collection=True),
        ]
        if request.include_fsrs:
            stages.append(ProgressStage(evaluate_fsrs_op, uses_collection=True))
        stages.extend(
            [
                ProgressStage(export_op, uses_collection=True),
                ProgressStage(prepare_op, uses_collection=False),
                ProgressStage(target_op, uses_collection=False),
                ProgressStage(metrics_op, uses_collection=False),
            ]
        )
        operation_title = (
            f"Compare FSRS-6 and {self.rwkv_label}"
            if request.include_fsrs
            else f"{self.rwkv_label} Evaluation"
        )
        run_with_progress_stages(
            parent=self,
            title=operation_title,
            label=operation_title,
            stages=stages,
            on_success=success,
            on_failure=failure,
            on_cancel=cancelled,
        )
        return True

    def _build_scope_descriptors(
        self,
        selection: EvaluationScopeSelection,
    ):
        return build_evaluation_scope_descriptors(
            mw.col,
            include_collection=selection.include_collection,
            include_presets=selection.include_presets,
            include_decks=selection.include_decks,
        )

    def _show_warning(self, message: str) -> None:
        show_web_warning(
            message,
            title=self.windowTitle(),
            parent=self,
        )


def _target_counts(
    target_review_ids_by_scope: dict[str, set[int]],
) -> dict[str, int]:
    return {
        scope_key: len(review_ids)
        for scope_key, review_ids in target_review_ids_by_scope.items()
    }
