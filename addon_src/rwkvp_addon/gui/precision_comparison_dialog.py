from __future__ import annotations

import time
from copy import deepcopy

from aqt import mw

from ..addon_config import (
    EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY,
    PROCESS_MANY_MODE_CONFIG_KEY,
    calculate_forgetting_curves,
    configured_model_generation,
    configured_runtime_model_id,
    normalized_addon_config,
    predict_many_batch_size,
    predict_many_mode,
)
from ..dataset_export import load_review_data_from_collection
from ..precision_comparison import (
    run_state_precision_comparison,
    state_precision_comparison_plan,
)
from ..precision_comparison_html import render_state_precision_comparison_html
from ..review_type_normalization import (
    filtered_review_normalization_policy_for_store,
    filtered_review_policy_from_config_values,
)
from ..runtime import manager_for_mw
from .common import ProgressStage, run_action_with_progress_stages
from .speed_test_dialog import _is_dark_widget, _show_results_after_progress_cleanup
from .state_comparison_dialog import _comparison_can_start, _finish_comparison
from .web_message import ask_web_confirmation, show_web_warning


def show_state_precision_comparison(*, parent, button, config) -> None:
    """Compare a frozen Settings draft without publishing either temporary state."""
    manager = manager_for_mw(mw)
    if not _comparison_can_start(manager, parent=parent):
        return
    draft = normalized_addon_config(deepcopy(config))
    title = "FP16 vs FP32 State Comparison"

    def start(confirmed: bool) -> None:
        if not confirmed or not _comparison_can_start(manager, parent=parent):
            return
        # The confirmation is asynchronous; snapshot the draft before opening it
        # and recheck native ownership before queuing collection/background work.
        def collect(col, progress, _previous):
            progress.check_cancelled()
            progress.update(0, 1, "Reading full review history for FP16 and FP32")
            policy = filtered_review_policy_from_config_values(
                draft,
                fallback=filtered_review_normalization_policy_for_store(manager.store),
            )
            review_data = load_review_data_from_collection(
                col,
                manager.store,
                exclude_deleted_card_revlogs=bool(draft[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY]),
                model_generation=configured_model_generation(draft),
                filtered_review_normalization_policy=policy,
            )
            progress.check_cancelled()
            plan = state_precision_comparison_plan(
                review_data,
                model_id=configured_runtime_model_id(draft),
                model_generation=configured_model_generation(draft),
                process_many_mode=str(draft[PROCESS_MANY_MODE_CONFIG_KEY]),
                # Isolate state quantization, not GPU arithmetic, in this
                # existing storage comparison. Full FP16 still uses GPU here.
                predict_many_mode=predict_many_mode(draft),
                predict_batch_size=predict_many_batch_size(draft),
                calculate_curves=calculate_forgetting_curves(draft),
                target_timestamp_ms=time.time_ns() // 1_000_000,
                temporary_directory=manager.store.cache_dir,
            )
            progress.update(1, 1, "FP16 and FP32 comparison ready")
            return plan

        def compare(_col, progress, plan):
            slot = None
            try:
                progress.check_cancelled()
                progress.update(0, 1, "Reserving RWKV state-building resources")
                slot = manager.reserve_runtime_slot(progress)
                return run_state_precision_comparison(plan, progress)
            finally:
                if slot is not None:
                    slot.close()

        def success(result):
            is_dark = _is_dark_widget(parent)
            _show_results_after_progress_cleanup(
                parent, title=title,
                render_html=lambda generation: render_state_precision_comparison_html(
                    result, is_dark=is_dark, generation=generation,
                ),
            )

        run_action_with_progress_stages(
            button=button, parent=parent, title=title,
            label="Preparing FP16 and FP32 comparison",
            stages=[
                ProgressStage(collect, uses_collection=True),
                ProgressStage(compare, uses_collection=False),
            ],
            on_success=success,
            on_failure=lambda exc: show_web_warning(str(exc), title=title, parent=parent),
            on_finish=lambda: _finish_comparison(button),
        )

    ask_web_confirmation(
        parent=parent, title=title,
        message=(
            "RWKV will build two full-history states using this Settings draft, then "
            "compare state calculation time, saved state size, Immediate LogLoss and "
            "RMSE(bins) for all eligible reviews and the most recent 25% separately, "
            "and warmed predict_many() speed.\n\n"
            "This may take quite some time and temporarily use substantial RAM, GPU "
            "memory and disk space. Progress can be cancelled; partial results are "
            "discarded. Both states are disposable. Your saved checkpoint, settings "
            "and Anki review data will not be changed."
        ),
        confirm_label="Start Comparison", on_result=start,
    )
