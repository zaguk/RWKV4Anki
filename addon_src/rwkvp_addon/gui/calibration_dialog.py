from __future__ import annotations

from aqt import mw

from ..addon_config import addon_config_for_mw, calculate_forgetting_curves
from ..anki_api import ACTIVE_CARD_SEARCH, find_cards
from ..calibration_controller import CalibrationController
from ..checkpoint_progress import update_checkpoint_collection_data
from ..dataset_export import (
    ensure_checkpoint_ready_from_load,
    load_review_data_for_checkpoint,
)
from ..evaluation_predictions import load_current_evaluation_prediction_rows
from ..metrics import RWKVPredictionMode
from ..runtime import manager_for_mw, store_for_mw
from ..rwkv_modes import RetrievabilityMode, mode_spec
from .checkpoint_failure import handle_checkpoint_failure
from .common import ProgressStage, run_with_progress_stages
from .web_dialog import WebDialogHost, widget_uses_dark_palette
from .web_message import show_web_warning


class CalibrationDialog(WebDialogHost):
    def __init__(
        self,
        parent=None,
        *,
        initial_search: str | None = None,
        mode: RetrievabilityMode = RetrievabilityMode.IMMEDIATE,
    ) -> None:
        self.mode = mode
        self.spec = mode_spec(mode)
        self._calibration_controller = CalibrationController(
            mode=mode,
            initial_search=initial_search or ACTIVE_CARD_SEARCH,
            fallback_search=ACTIVE_CARD_SEARCH,
            search_card_ids=self._matching_card_ids_for_search,
            on_calculate_requested=self.calculate,
            is_dark=widget_uses_dark_palette(parent),
            bin_count=20,
        )
        title = f"{self.spec.evaluate_label} Calibration Graph"
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

    def calculate(self, search: str | None = None) -> None:
        if self.mode == RetrievabilityMode.FORGETTING_CURVE and not calculate_forgetting_curves(
            addon_config_for_mw(mw)
        ):
            show_web_warning(
                "Calculate Forgetting Curves is disabled in RWKV Settings.",
                title=self.spec.warning_title,
                parent=self,
            )
            return

        controller = self._calibration_controller
        effective_search = (
            str(search).strip()
            if search is not None
            else controller.initial_search.strip() or ACTIVE_CARD_SEARCH
        )
        if not controller.begin_calculation():
            return
        store = store_for_mw(mw)
        manager = manager_for_mw(mw)
        prediction_mode = _prediction_mode_for_calibration(self.mode)

        def collect_op(col, progress, _previous):
            review_load = load_review_data_for_checkpoint(
                col,
                store,
                manager,
                progress,
                allow_incremental=True,
            )
            update_checkpoint_collection_data(progress, "Finding cards")
            card_ids = {int(card_id) for card_id in find_cards(col, effective_search)}
            return review_load, card_ids

        def load_cache_op(_col, progress, previous):
            review_load, card_ids = previous
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

        def success(prediction_rows) -> None:
            if not self.cleaned_up:
                controller.set_prediction_rows(prediction_rows)

        def failure(exc: Exception) -> None:
            if self.cleaned_up:
                return
            controller.finish_calculation()
            handle_checkpoint_failure(exc, self.calculate, parent=self)

        def cancelled() -> None:
            if not self.cleaned_up:
                controller.finish_calculation()

        run_with_progress_stages(
            parent=self,
            title=self.windowTitle(),
            label="Calculating calibration",
            stages=[
                ProgressStage(collect_op, uses_collection=True),
                ProgressStage(load_cache_op, uses_collection=False),
            ],
            on_success=success,
            on_failure=failure,
            on_cancel=cancelled,
        )

    def _matching_card_ids_for_search(self, query: str) -> tuple[int, ...]:
        if mw.col is None:
            raise RuntimeError("collection unavailable")
        return tuple(int(card_id) for card_id in find_cards(mw.col, query))


def _prediction_mode_for_calibration(mode: RetrievabilityMode) -> RWKVPredictionMode:
    if mode == RetrievabilityMode.FORGETTING_CURVE:
        return RWKVPredictionMode.PREDICT_AHEAD
    return RWKVPredictionMode.PER_REVIEW
