from __future__ import annotations

from aqt import mw

from ..addon_config import (
    addon_config_for_mw,
    calculate_forgetting_curves,
    configured_model_generation,
    evaluation_settings_signature,
    exclude_deleted_card_revlogs,
    retain_multiple_model_generations,
)
from ..anki_api import ACTIVE_CARD_SEARCH, find_cards
from ..calibration_controller import CalibrationController
from ..checkpoint_progress import update_checkpoint_collection_data
from ..dataset_export import (
    ensure_checkpoint_ready_from_load,
    load_review_data_for_checkpoint,
)
from ..evaluation_comparison import (
    ComparisonCheckpointRequired,
    ComparisonEvaluationError,
    GenerationEvaluation,
)
from ..evaluation_predictions import load_current_evaluation_prediction_rows
from ..fsrs_calibration import add_fsrs_calibration_predictions
from ..metrics import RWKVPredictionMode
from ..runtime import manager_for_mw, store_for_mw
from ..rwkv_modes import RetrievabilityMode, mode_spec
from .checkpoint_failure import handle_checkpoint_failure
from .common import ProgressStage, run_with_progress_stages
from .web_dialog import WebDialogHost, widget_uses_dark_palette
from .web_message import ask_web_confirmation, show_web_warning


class CalibrationDialog(WebDialogHost):
    def __init__(
        self,
        parent=None,
        *,
        initial_search: str | None = None,
        mode: RetrievabilityMode = RetrievabilityMode.IMMEDIATE,
        um_plus: bool = False,
    ) -> None:
        self.mode = mode
        self.spec = mode_spec(mode)
        config = addon_config_for_mw(mw)
        self._evaluation_settings = evaluation_settings_signature(config)
        self._calibration_controller = CalibrationController(
            mode=mode,
            initial_search=initial_search or ACTIVE_CARD_SEARCH,
            fallback_search=ACTIVE_CARD_SEARCH,
            search_card_ids=self._matching_card_ids_for_search,
            on_calculate_requested=self.calculate,
            is_dark=widget_uses_dark_palette(parent),
            bin_count=20,
            um_plus=um_plus,
            active_generation=configured_model_generation(config),
            include_deleted_reviews=not exclude_deleted_card_revlogs(config),
        )
        title = f"{self.spec.evaluate_label} {'UM+' if um_plus else 'Calibration Graph'}"
        super().__init__(
            parent,
            title=title,
            controller=self._calibration_controller,
            size=(760, 620),
            requires_collection=True,
        )
        self._calibration_controller.attach_rerender(self.rerender)

    @property
    def calibration_controller(self) -> CalibrationController:
        return self._calibration_controller

    @property
    def prediction_rows(self) -> list[dict]:
        return self._calibration_controller.prediction_rows

    @property
    def displayed_rows(self) -> list[dict]:
        return self._calibration_controller.displayed_rows

    def calculate(
        self,
        search: str | None = None,
        *,
        allow_comparison_build: bool = False,
    ) -> None:
        controller = self._calibration_controller
        if self.cleaned_up or controller.running:
            return
        config = addon_config_for_mw(mw)
        if self._evaluation_settings != evaluation_settings_signature(config):
            show_web_warning(
                "RWKV model or history settings changed. Reopen this graph to use them.",
                title=self.windowTitle(),
                parent=self,
            )
            return
        if self.mode == RetrievabilityMode.FORGETTING_CURVE and not calculate_forgetting_curves(
            config
        ):
            show_web_warning(
                "Calculate Forgetting Curves is disabled in RWKV Settings.",
                title=self.spec.warning_title,
                parent=self,
            )
            return

        effective_search = (
            str(search).strip()
            if search is not None
            else controller.initial_search.strip() or ACTIVE_CARD_SEARCH
        )
        store = store_for_mw(mw)
        manager = manager_for_mw(mw)
        prediction_mode = _prediction_mode_for_calibration(self.mode)
        include_fsrs = controller.include_fsrs
        comparison_generation = controller.comparison_generation
        comparison = None
        comparison_review_load = None
        retain = retain_multiple_model_generations(config)
        if include_fsrs and comparison_generation is not None:
            try:
                comparison = GenerationEvaluation(
                    manager,
                    comparison_generation,
                    prediction_mode=prediction_mode,
                )
                if comparison.needs_build and not allow_comparison_build:
                    self._confirm_comparison_build(effective_search, comparison, retain=retain)
                    return
            except (OSError, ValueError) as exc:
                show_web_warning(str(exc), title=self.windowTitle(), parent=self)
                return
        if not controller.begin_calculation():
            return

        def collect_op(col, progress, _previous):
            nonlocal comparison_review_load
            review_load = load_review_data_for_checkpoint(
                col,
                store,
                manager,
                progress,
                allow_incremental=True,
            )
            if comparison is not None:
                comparison_review_load = comparison.load_review_data(col, progress)
                if (
                    review_load.collection_revision is not None
                    and review_load.collection_revision
                    != comparison_review_load.collection_revision
                ):
                    raise RuntimeError(
                        "Review history changed while loading. Run the comparison again."
                    )
            update_checkpoint_collection_data(progress, "Finding cards")
            card_ids = {int(card_id) for card_id in find_cards(col, effective_search)}
            return review_load, card_ids

        def load_cache_op(_col, progress, previous):
            review_load, card_ids = previous
            if comparison is not None:
                comparison.validate_active_checkpoint()
            readiness = ensure_checkpoint_ready_from_load(
                manager,
                review_load,
                progress,
                wait_for_save=True,
                capture_prediction_tail=True,
            )
            return load_current_evaluation_prediction_rows(
                readiness,
                store,
                manager,
                progress,
                prediction_mode=prediction_mode,
                card_ids=card_ids,
            )

        def fsrs_op(col, progress, prediction_rows):
            return add_fsrs_calibration_predictions(col, prediction_rows, progress)

        def comparison_op(_col, progress, prediction_rows):
            return comparison.calibration_rows(
                comparison_review_load,
                prediction_rows,
                progress,
                allow_build=allow_comparison_build,
                retain=retain,
            )

        def success(prediction_rows) -> None:
            if not self.cleaned_up:
                controller.set_prediction_rows(
                    prediction_rows,
                    fsrs_included=include_fsrs,
                    comparison_generation=comparison_generation,
                )

        def failure(exc: Exception) -> None:
            if self.cleaned_up:
                return
            controller.finish_calculation()
            if isinstance(exc, ComparisonCheckpointRequired) and comparison is not None:
                self._confirm_comparison_build(effective_search, comparison, retain=retain)
                return
            if isinstance(exc, ComparisonEvaluationError):
                show_web_warning(str(exc), title=self.windowTitle(), parent=self)
                return
            handle_checkpoint_failure(exc, self.calculate, parent=self)

        def cancelled() -> None:
            if not self.cleaned_up:
                controller.finish_calculation()

        stages = [
            ProgressStage(collect_op, uses_collection=True),
            ProgressStage(load_cache_op, uses_collection=False),
        ]
        if comparison is not None:
            stages.append(ProgressStage(comparison_op, uses_collection=False))
        elif include_fsrs:
            stages.append(ProgressStage(fsrs_op, uses_collection=True))
        run_with_progress_stages(
            parent=self,
            title=self.windowTitle(),
            label="Calculating UM+" if controller.um_plus else "Calculating calibration",
            stages=stages,
            on_success=success,
            on_failure=failure,
            on_cancel=cancelled,
        )

    def _confirm_comparison_build(self, search, comparison, *, retain: bool) -> None:
        def confirmed(accepted: bool) -> None:
            if (
                accepted
                and not self.cleaned_up
                and self._calibration_controller.comparison_generation == comparison.generation
            ):
                self.calculate(search, allow_comparison_build=True)

        ask_web_confirmation(
            parent=self,
            title="Build comparison model",
            message=comparison.build_warning(retain=retain),
            confirm_label="Okay",
            cancel_label="Cancel",
            on_result=confirmed,
        )

    def _matching_card_ids_for_search(self, query: str) -> tuple[int, ...]:
        if mw.col is None:
            raise RuntimeError("collection unavailable")
        return tuple(int(card_id) for card_id in find_cards(mw.col, query))


def _prediction_mode_for_calibration(mode: RetrievabilityMode) -> RWKVPredictionMode:
    if mode == RetrievabilityMode.FORGETTING_CURVE:
        return RWKVPredictionMode.PREDICT_AHEAD
    return RWKVPredictionMode.PER_REVIEW
