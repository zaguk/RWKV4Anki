from __future__ import annotations

import json
import math
import time

from aqt import mw

from ..addon_config import addon_config_for_mw, calculate_forgetting_curves
from ..anki_api import ACTIVE_CARD_SEARCH, find_cards
from ..checkpoint_progress import update_checkpoint_collection_data
from ..dataset_export import (
    ensure_checkpoint_ready_from_load,
    latest_collection_review_timestamp_seconds,
    load_review_data_for_checkpoint,
    open_checkpoint_runtime_from_load,
)
from ..retrievability import (
    CardPrediction,
    predict_card_retrievability,
    predict_curve_retrievability,
)
from ..retrievability_controller import RetrievabilityController, format_local_timestamp
from ..review_rows import checkpoint_scope_cards_for_card_ids
from ..runtime import manager_for_mw, store_for_mw
from ..rwkv_modes import RetrievabilityMode
from ..rwkv_processing import predict_many_batched
from ..vendor_bootstrap import require_rwkv_probability
from .checkpoint_failure import handle_checkpoint_failure
from .common import ProgressStage, run_with_progress_stages
from .web_dialog import WebDialogHost, widget_uses_dark_palette
from .web_message import show_web_warning


class RetrievabilityDialog(WebDialogHost):
    def __init__(
        self,
        parent=None,
        *,
        initial_search: str | None = None,
        mode: RetrievabilityMode = RetrievabilityMode.IMMEDIATE,
    ) -> None:
        self.mode = mode
        self._retrievability_bin_size = 0.05
        initial_search_text = initial_search or ACTIVE_CARD_SEARCH
        latest_review_time = (
            latest_collection_review_timestamp_seconds(mw.col) if mw.col else None
        )
        self._retrievability_controller = RetrievabilityController(
            mode=mode,
            initial_search=initial_search_text,
            target_timestamp_seconds=time.time(),
            latest_review_timestamp_seconds=latest_review_time,
            fallback_search=ACTIVE_CARD_SEARCH,
            search_card_ids=self._matching_card_ids_for_search,
            on_calculate_requested=self.calculate,
            on_open_bucket_requested=self._open_bucket_search,
            on_invalid_predictions=self._log_invalid_predictions,
            is_dark=widget_uses_dark_palette(parent),
            bin_size=self._retrievability_bin_size,
        )
        super().__init__(
            parent,
            title=mode.window_title,
            controller=self._retrievability_controller,
            size=(760, 620),
            requires_collection=True,
        )
        self._retrievability_controller.attach_rerender(self.rerender)

    @property
    def retrievability_controller(self) -> RetrievabilityController:
        return self._retrievability_controller

    @property
    def predictions(self) -> list[CardPrediction]:
        return self._retrievability_controller.predictions

    @property
    def displayed_predictions(self) -> list[CardPrediction]:
        return self._retrievability_controller.displayed_predictions

    def calculate(
        self,
        search: str | None = None,
        target_seconds: float | None = None,
    ) -> None:
        if self.mode == RetrievabilityMode.FORGETTING_CURVE and not calculate_forgetting_curves(
            addon_config_for_mw(mw)
        ):
            show_web_warning(
                "Calculate Forgetting Curves is disabled in RWKV Settings.",
                title=self.mode.warning_title,
                parent=self,
            )
            return

        controller = self._retrievability_controller
        effective_search = (
            str(search).strip()
            if search is not None
            else controller.initial_search.strip() or ACTIVE_CARD_SEARCH
        )
        effective_target_seconds = (
            float(target_seconds)
            if target_seconds is not None
            else controller.target_timestamp_seconds
        )
        latest_review_time = (
            latest_collection_review_timestamp_seconds(mw.col) if mw.col else None
        )
        controller.update_latest_review_timestamp(latest_review_time)
        if not controller.prediction_time_is_allowed(effective_target_seconds):
            latest = format_local_timestamp(controller.latest_review_timestamp_seconds)
            controller.show_error(
                f"Prediction time must be at or after the most recent review ({latest}).",
                focus_target="target_time",
            )
            return
        if not controller.begin_calculation():
            return

        store = store_for_mw(mw)
        manager = manager_for_mw(mw)

        def collect_op(col, progress, _previous):
            review_load = load_review_data_for_checkpoint(
                col,
                store,
                manager,
                progress,
                allow_incremental=True,
            )
            update_checkpoint_collection_data(progress, "Finding cards")
            card_ids = find_cards(col, effective_search)
            return review_load, card_ids

        def predict_op(_col, progress, previous):
            review_load, card_ids = previous
            if self.mode == RetrievabilityMode.FORGETTING_CURVE:
                readiness = ensure_checkpoint_ready_from_load(
                    manager,
                    review_load,
                    progress,
                )
                review_data = readiness.review_data
                return predict_curve_retrievability(
                    card_ids=card_ids,
                    review_data=review_data,
                    target_timestamp_seconds=effective_target_seconds,
                    curves_by_card=manager.latest_curves_for_cards(card_ids),
                    curve_predictor=require_rwkv_probability(),
                )

            readiness, runtime = open_checkpoint_runtime_from_load(
                manager,
                review_load,
                progress,
                scope_cards=checkpoint_scope_cards_for_card_ids(
                    card_ids,
                    review_load.review_data,
                ),
            )
            try:
                return self._predict_immediate_retrievability(
                    card_ids,
                    readiness.review_data,
                    effective_target_seconds,
                    progress,
                    runtime,
                )
            finally:
                runtime.close()

        def success(predictions) -> None:
            if self.cleaned_up:
                return
            controller.set_predictions(predictions)

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
            title=self.mode.graph_title,
            label="Calculating retrievability",
            stages=[
                ProgressStage(collect_op, uses_collection=True),
                ProgressStage(predict_op, uses_collection=False),
            ],
            on_success=success,
            on_failure=failure,
            on_cancel=cancelled,
        )

    def _matching_card_ids_for_search(self, query: str) -> tuple[int, ...]:
        if mw.col is None:
            raise RuntimeError("collection unavailable")
        return tuple(int(card_id) for card_id in find_cards(mw.col, query))

    def _open_bucket_search(self, search: str) -> None:
        from aqt import dialogs

        browser = dialogs.open("Browser", mw)
        browser.search_for(search)

    def _predict_immediate_retrievability(
        self,
        card_ids: list[int],
        review_data,
        target_seconds: float,
        progress,
        manager,
    ) -> list[CardPrediction]:
        def predictor(rows):
            return predict_many_batched(
                manager.predict_many,
                rows,
                progress,
                label="Predicting card retrievability",
                chunk_size=getattr(
                    manager,
                    "predict_many_progress_chunk_size",
                    None,
                ),
            )

        return predict_card_retrievability(
            card_ids=card_ids,
            review_data=review_data,
            target_timestamp_seconds=target_seconds,
            predictor=predictor,
        )

    def _log_invalid_predictions(
        self,
        predictions: tuple[CardPrediction, ...],
        prediction_time: float,
        initial_search: str,
        display_filter_search: str,
    ):
        if not predictions:
            return None

        store = store_for_mw(mw)
        path = store.retrievability_nan_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        record_base = {
            "logged_at": int(time.time()),
            "profile_name": store.profile_name,
            "prediction_time": int(prediction_time),
            "initial_search": initial_search,
            "display_filter_search": display_filter_search,
        }
        with path.open("a", encoding="utf-8") as handle:
            for prediction in predictions:
                value = prediction.retrievability
                record = {
                    **record_base,
                    "card_id": prediction.card_id,
                    "deck_id": prediction.deck_id,
                    "preset_id": prediction.preset_id,
                    "retrievability": repr(value),
                    "is_nan": math.isnan(value),
                    "is_infinite": math.isinf(value),
                    "elapsed_days": prediction.elapsed_days,
                    "elapsed_seconds": prediction.elapsed_seconds,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return path
