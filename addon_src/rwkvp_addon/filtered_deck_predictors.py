from __future__ import annotations

from .rwkv_curve_predictions import curve_probabilities_for_prediction_rows
from .rwkv_modes import RetrievabilityMode
from .rwkv_processing import predict_many_batched
from .vendor_bootstrap import require_rwkv_probability


def filtered_deck_predictor_for_mode(mode: RetrievabilityMode, manager, progress):
    if mode == RetrievabilityMode.FORGETTING_CURVE:
        curve_predictor = require_rwkv_probability()

        def predict_from_curves(rows):
            total = max(1, len(rows))
            progress.update(0, total, "Predicting filtered deck candidates")
            curves_by_card = manager.latest_curves_for_cards(
                int(row["card_id"]) for row in rows
            )
            predictions = curve_probabilities_for_prediction_rows(
                rows,
                curves_by_card=curves_by_card,
                curve_predictor=curve_predictor,
            )
            progress.update(len(rows), total, "Predicted filtered deck candidates")
            return predictions

        return predict_from_curves

    def predict_immediate(rows):
        return predict_many_batched(
            manager.predict_many,
            rows,
            progress,
            label="Predicting filtered deck candidates",
            chunk_size=getattr(manager, "predict_many_progress_chunk_size", None),
        )

    return predict_immediate
