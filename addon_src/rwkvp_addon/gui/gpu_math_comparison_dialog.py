from __future__ import annotations

import time
from copy import deepcopy

from aqt import mw

from ..addon_config import (
    EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY,
    PROCESS_MANY_MODE_CONFIG_KEY,
    configured_model_generation,
    configured_runtime_model_id,
    normalized_addon_config,
    predict_many_batch_size,
)
from ..dataset_export import load_review_data_from_collection
from ..gpu_math_comparison import gpu_math_comparison_plan, run_gpu_math_comparison
from ..gpu_math_comparison_html import render_gpu_math_comparison_html
from ..review_type_normalization import (
    filtered_review_normalization_policy_for_store,
    filtered_review_policy_from_config_values,
)
from ..runtime import manager_for_mw
from .common import ProgressStage, run_action_with_progress_stages
from .speed_test_dialog import _is_dark_widget, _show_results_after_progress_cleanup
from .state_comparison_dialog import _comparison_can_start, _finish_comparison
from .web_message import ask_web_confirmation, show_web_warning


def show_gpu_math_comparison(*, parent, button, config, gpu_available: bool = True) -> None:
    draft = normalized_addon_config(deepcopy(config))
    generation = configured_model_generation(draft)
    title = "FP16 GPU Predictions · Compare Predictions"
    if not gpu_available:
        show_web_warning(
            "A compatible prediction GPU is required. No temporary state was built.",
            title=title, parent=parent,
        )
        return
    manager = manager_for_mw(mw)
    if not _comparison_can_start(manager, parent=parent):
        return

    def start(confirmed: bool) -> None:
        if not confirmed or not _comparison_can_start(manager, parent=parent):
            return

        def collect(col, progress, _previous):
            progress.check_cancelled()
            progress.update(0, 1, "Reading review history for GPU math comparison")
            policy = filtered_review_policy_from_config_values(
                draft,
                fallback=filtered_review_normalization_policy_for_store(manager.store),
            )
            data = load_review_data_from_collection(
                col,
                manager.store,
                exclude_deleted_card_revlogs=bool(draft[EXCLUDE_DELETED_CARD_REVLOGS_CONFIG_KEY]),
                model_generation=generation,
                filtered_review_normalization_policy=policy,
            )
            progress.check_cancelled()
            return gpu_math_comparison_plan(
                data,
                model_id=configured_runtime_model_id(draft),
                model_generation=generation,
                process_many_mode=str(draft[PROCESS_MANY_MODE_CONFIG_KEY]),
                predict_batch_size=predict_many_batch_size(draft, "gpu"),
                target_timestamp_ms=time.time_ns() // 1_000_000,
                temporary_directory=manager.store.cache_dir,
            )

        def compare(_col, progress, plan):
            slot = None
            try:
                progress.check_cancelled()
                slot = manager.reserve_runtime_slot(progress)
                return run_gpu_math_comparison(plan, progress)
            finally:
                if slot is not None:
                    slot.close()

        def success(result):
            is_dark = _is_dark_widget(parent)
            _show_results_after_progress_cleanup(
                parent,
                title=title,
                render_html=lambda generation: render_gpu_math_comparison_html(
                    result,
                    is_dark=is_dark,
                    generation=generation,
                ),
            )

        run_action_with_progress_stages(
            button=button,
            parent=parent,
            title=title,
            label="Preparing GPU math comparison",
            stages=[
                ProgressStage(collect, uses_collection=True),
                ProgressStage(compare, uses_collection=False),
            ],
            on_success=success,
            on_failure=lambda exc: show_web_warning(str(exc), title=title, parent=parent),
            on_finish=lambda: _finish_comparison(button),
        )

    ask_web_confirmation(
        parent=parent,
        title=title,
        message=(
            "Build one disposable full-history FP16 state, then compare GPU FP16 vs FP32 "
            "math on identical current-card queries. Reports mean, median, P90, P99 "
            "and maximum absolute percentage-point differences. No performance test.\n\n"
            "Requires a compatible GPU. This may use substantial RAM, GPU memory and "
            "temporary disk space. You can cancel; incomplete results are discarded. "
            "Your settings, saved checkpoint and Anki review data will not change."
        ),
        confirm_label="Start Comparison",
        on_result=start,
    )
