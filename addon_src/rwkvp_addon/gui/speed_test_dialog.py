from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from aqt import mw
from aqt.qt import Qt

from ..anki_api import find_cards
from ..checkpoint_manager import MissingCheckpointError
from ..checkpoint_progress import update_checkpoint_collection_data
from ..dataset_export import (
    latest_collection_review_timestamp_seconds,
    load_review_data_for_checkpoint,
    open_checkpoint_runtime_from_load,
)
from ..live_prediction_benchmark import open_live_prediction_benchmark_session
from ..process_many_benchmark import benchmark_process_many_rows
from ..process_speed_cache import (
    cache_process_many_curve_speed_test,
    cache_process_many_speed_test,
    process_many_speed_cache_path,
)
from ..review_rows import (
    checkpoint_scope_cards_for_card_ids,
    prediction_rows_for_card_ids,
)
from ..runtime import manager_for_mw, store_for_mw
from ..speed_test import (
    PREDICT_MANY_SPEED_TEST_SIZES,
    capped_curve_speed_test_rows,
    capped_process_speed_test_rows,
    run_live_prediction_speed_test,
    run_predict_many_speed_test,
    run_process_many_curve_speed_test,
    run_process_many_speed_test,
    speed_test_checkpoint_is_usable,
    speed_test_modes,
    state_backed_prediction_card_ids,
)
from ..speed_test_html import (
    render_curve_speed_test_html,
    render_live_prediction_speed_test_html,
    render_predict_many_speed_test_html,
    render_process_many_speed_test_html,
)
from ..web_dialog_controller import CloseOnlyReportController
from .checkpoint_failure import handle_checkpoint_failure
from .common import ProgressStage, run_action_with_progress_stages
from .web_dialog import WebDialogHost, widget_uses_dark_palette

_PROGRESS_CLEANUP_DELAY_MS = 50
_is_dark_widget = widget_uses_dark_palette


@dataclass(frozen=True)
class _PredictSpeedTestInput:
    review_load: object
    card_ids: tuple[int, ...]
    collection_card_count: int
    eligible_card_count: int
    target_timestamp_seconds: float


@dataclass(frozen=True)
class _ProcessSpeedTestInput:
    rows: tuple[dict[str, Any], ...]
    available_review_count: int


class SpeedTestResultsDialog(WebDialogHost):
    def __init__(
        self,
        parent,
        *,
        title: str,
        render_html: Callable[[int], str],
    ) -> None:
        self._report_controller = CloseOnlyReportController(render_html)
        super().__init__(
            parent,
            title=title,
            controller=self._report_controller,
            size=(860, 560),
            web_minimum_height=440,
            modality=Qt.WindowModality.WindowModal,
        )


def _show_results_after_progress_cleanup(
    parent,
    *,
    title: str,
    render_html: Callable[[int], str],
) -> None:
    """Open results after the preceding progress presentation has been released.

    The shared progress adapter is asynchronous and uses no nested event loop.
    Retain the short handoff so a standalone progress host can finish its Qt
    deletion/modality cleanup before its result host is opened. In-page
    progress overlays use the same path for deterministic parent checks.
    """

    def show() -> None:
        if not _result_parent_is_visible(parent):
            return
        dialog = SpeedTestResultsDialog(
            parent,
            title=title,
            render_html=render_html,
        )
        dialog.open()

    mw.progress.single_shot(
        _PROGRESS_CLEANUP_DELAY_MS,
        show,
        requires_collection=False,
    )


def _result_parent_is_visible(parent) -> bool:
    is_visible = getattr(parent, "isVisible", None)
    if not callable(is_visible):
        return True
    try:
        return bool(is_visible())
    except RuntimeError:
        return False


def show_predict_many_speed_test(
    *,
    parent,
    button,
    batch_sizes: dict[str, int | None],
    gpu_available: bool,
) -> None:
    manager = manager_for_mw(mw)
    store = store_for_mw(mw)

    def collect_op(col, progress, _previous):
        _require_usable_checkpoint(manager)
        review_load = load_review_data_for_checkpoint(
            col,
            store,
            manager,
            progress,
            allow_incremental=True,
        )
        update_checkpoint_collection_data(
            progress,
            "Finding cards with processed RWKV state for speed test",
        )
        all_card_ids = sorted(int(card_id) for card_id in find_cards(col, ""))
        eligible_card_ids = state_backed_prediction_card_ids(
            all_card_ids,
            review_load.review_data.last_by_card,
        )
        maximum = max(PREDICT_MANY_SPEED_TEST_SIZES)
        latest_review = latest_collection_review_timestamp_seconds(col)
        target_timestamp = max(
            time.time(),
            0.0 if latest_review is None else float(latest_review),
        )
        return _PredictSpeedTestInput(
            review_load=review_load,
            card_ids=eligible_card_ids[:maximum],
            collection_card_count=len(all_card_ids),
            eligible_card_count=len(eligible_card_ids),
            target_timestamp_seconds=target_timestamp,
        )

    def benchmark_op(_col, progress, previous):
        input_data = previous
        lease = None
        try:
            readiness, lease = open_checkpoint_runtime_from_load(
                manager,
                input_data.review_load,
                progress,
                scope_cards=checkpoint_scope_cards_for_card_ids(
                    input_data.card_ids,
                    input_data.review_load.review_data,
                ),
            )
            rows = prediction_rows_for_card_ids(
                input_data.card_ids,
                readiness.review_data,
                target_timestamp_seconds=input_data.target_timestamp_seconds,
            )
            return run_predict_many_speed_test(
                rows,
                collection_card_count=input_data.collection_card_count,
                eligible_card_count=input_data.eligible_card_count,
                model_id=lease.model_id,
                modes=speed_test_modes(gpu_available=gpu_available),
                batch_sizes=batch_sizes,
                open_session=lambda batch, mode, batch_size, refresh_limit: (
                    open_live_prediction_benchmark_session(
                        lease,
                        batch,
                        mode=mode,
                        batch_size=batch_size,
                        refresh_limit=refresh_limit,
                        target_timestamp_seconds=input_data.target_timestamp_seconds,
                    )
                ),
                progress=progress,
            )
        finally:
            # The lease token is the speed test's ownership boundary. Closing
            # it releases only the scoped native runtime opened above. A
            # manager-wide unload could tear down a Browser/Live scope that
            # acquired the manager while this staged operation was finishing.
            if lease is not None:
                lease.close()

    def success(result) -> None:
        is_dark = _is_dark_widget(parent)
        _show_results_after_progress_cleanup(
            parent,
            title="Prediction Throughput Speed Test",
            render_html=lambda generation: render_predict_many_speed_test_html(
                result,
                is_dark=is_dark,
                generation=generation,
            ),
        )

    def failure(exc: Exception) -> None:
        handle_checkpoint_failure(
            exc,
            lambda: show_predict_many_speed_test(
                parent=parent,
                button=button,
                batch_sizes=batch_sizes,
                gpu_available=gpu_available,
            ),
            parent=parent,
        )

    run_action_with_progress_stages(
        button=button,
        parent=parent,
        title="Prediction Throughput Speed Test",
        label="Preparing prediction throughput speed test",
        stages=[
            ProgressStage(collect_op, uses_collection=True),
            ProgressStage(benchmark_op, uses_collection=False),
        ],
        on_success=success,
        on_failure=failure,
        on_finish=lambda: _finish_speed_test(button),
    )


def show_process_many_speed_test(
    *,
    parent,
    button,
    gpu_available: bool,
    calculate_curves: bool,
) -> None:
    manager = manager_for_mw(mw)
    store = store_for_mw(mw)

    def collect_op(col, progress, _previous):
        _require_usable_checkpoint(manager)
        review_load = load_review_data_for_checkpoint(
            col,
            store,
            manager,
            progress,
            allow_incremental=True,
        )
        update_checkpoint_collection_data(progress, "Preparing review rows for speed test")
        review_rows = review_load.review_data.rows
        return _ProcessSpeedTestInput(
            rows=tuple(capped_process_speed_test_rows(review_rows)),
            available_review_count=len(review_rows),
        )

    def benchmark_op(_col, progress, previous):
        input_data = previous
        modes = speed_test_modes(gpu_available=gpu_available, process=True)
        slot = manager.reserve_runtime_slot(progress)
        try:
            result = run_process_many_speed_test(
                review_count=len(input_data.rows),
                available_review_count=input_data.available_review_count,
                model_id=manager.model_id,
                return_curves=bool(calculate_curves),
                modes=modes,
                run_mode=lambda mode, review_count: benchmark_process_many_rows(
                    input_data.rows[:review_count],
                    model_id=manager.model_id,
                    mode=mode,
                    return_curves=bool(calculate_curves),
                ),
                progress=progress,
            )
        finally:
            slot.close()
        # This is a disposable optimization cache. A read-only or unavailable
        # cache directory must not turn a successful benchmark into a failure.
        with suppress(OSError, TypeError, ValueError):
            cache_process_many_speed_test(
                process_many_speed_cache_path(store.cache_dir),
                result,
            )
        return result

    def success(result) -> None:
        is_dark = _is_dark_widget(parent)
        _show_results_after_progress_cleanup(
            parent,
            title="State Building Speed Test",
            render_html=lambda generation: render_process_many_speed_test_html(
                result,
                is_dark=is_dark,
                generation=generation,
            ),
        )

    def failure(exc: Exception) -> None:
        handle_checkpoint_failure(
            exc,
            lambda: show_process_many_speed_test(
                parent=parent,
                button=button,
                gpu_available=gpu_available,
                calculate_curves=calculate_curves,
            ),
            parent=parent,
        )

    run_action_with_progress_stages(
        button=button,
        parent=parent,
        title="State Building Speed Test",
        label="Preparing state-building speed test",
        stages=[
            ProgressStage(collect_op, uses_collection=True),
            ProgressStage(benchmark_op, uses_collection=False),
        ],
        on_success=success,
        on_failure=failure,
        on_finish=lambda: _finish_speed_test(button),
    )


def show_curve_calculation_speed_test(
    *,
    parent,
    button,
    mode: str,
) -> None:
    manager = manager_for_mw(mw)
    store = store_for_mw(mw)

    def collect_op(col, progress, _previous):
        _require_usable_checkpoint(manager)
        review_load = load_review_data_for_checkpoint(
            col,
            store,
            manager,
            progress,
            allow_incremental=True,
        )
        update_checkpoint_collection_data(
            progress,
            "Preparing 10,000 review rows for the curve comparison",
        )
        review_rows = review_load.review_data.rows
        return _ProcessSpeedTestInput(
            rows=tuple(capped_curve_speed_test_rows(review_rows)),
            available_review_count=len(review_rows),
        )

    def benchmark_op(_col, progress, previous):
        input_data = previous
        slot = manager.reserve_runtime_slot(progress)
        try:
            result = run_process_many_curve_speed_test(
                review_count=len(input_data.rows),
                available_review_count=input_data.available_review_count,
                model_id=manager.model_id,
                mode=mode,
                run_once=lambda return_curves: benchmark_process_many_rows(
                    input_data.rows,
                    model_id=manager.model_id,
                    mode=mode,
                    return_curves=return_curves,
                ),
                progress=progress,
            )
        finally:
            slot.close()
        # Retain the exact on/off measurements so a later checkpoint estimate
        # does not substitute a generic curve-overhead assumption for data the
        # user has already measured. Cache failure must not fail the benchmark.
        with suppress(OSError, TypeError, ValueError):
            cache_process_many_curve_speed_test(
                process_many_speed_cache_path(store.cache_dir),
                result,
            )
        return result

    def success(result) -> None:
        is_dark = _is_dark_widget(parent)
        _show_results_after_progress_cleanup(
            parent,
            title="Forgetting Curve Speed Test",
            render_html=lambda generation: render_curve_speed_test_html(
                result,
                is_dark=is_dark,
                generation=generation,
            ),
        )

    def failure(exc: Exception) -> None:
        handle_checkpoint_failure(
            exc,
            lambda: show_curve_calculation_speed_test(
                parent=parent,
                button=button,
                mode=mode,
            ),
            parent=parent,
        )

    run_action_with_progress_stages(
        button=button,
        parent=parent,
        title="Forgetting Curve Speed Test",
        label="Preparing curve calculation speed test",
        stages=[
            ProgressStage(collect_op, uses_collection=True),
            ProgressStage(benchmark_op, uses_collection=False),
        ],
        on_success=success,
        on_failure=failure,
        on_finish=lambda: _finish_speed_test(button),
    )


def show_live_prediction_speed_test(
    *,
    parent,
    button,
    mode: str,
    card_count: int,
    batch_size: int | None,
) -> None:
    manager = manager_for_mw(mw)
    store = store_for_mw(mw)
    requested_card_count = max(1, int(card_count))

    def collect_op(col, progress, _previous):
        _require_usable_checkpoint(manager)
        review_load = load_review_data_for_checkpoint(
            col,
            store,
            manager,
            progress,
            allow_incremental=True,
        )
        update_checkpoint_collection_data(
            progress,
            "Finding state-backed cards for the Live Session speed test",
        )
        all_card_ids = sorted(int(card_id) for card_id in find_cards(col, ""))
        eligible_card_ids = state_backed_prediction_card_ids(
            all_card_ids,
            review_load.review_data.last_by_card,
        )
        latest_review = latest_collection_review_timestamp_seconds(col)
        target_timestamp = max(
            time.time(),
            0.0 if latest_review is None else float(latest_review),
        )
        return _PredictSpeedTestInput(
            review_load=review_load,
            card_ids=eligible_card_ids[:requested_card_count],
            collection_card_count=len(all_card_ids),
            eligible_card_count=len(eligible_card_ids),
            target_timestamp_seconds=target_timestamp,
        )

    def benchmark_op(_col, progress, previous):
        input_data = previous
        lease = None
        try:
            readiness, lease = open_checkpoint_runtime_from_load(
                manager,
                input_data.review_load,
                progress,
                scope_cards=checkpoint_scope_cards_for_card_ids(
                    input_data.card_ids,
                    input_data.review_load.review_data,
                ),
            )
            rows = prediction_rows_for_card_ids(
                input_data.card_ids,
                readiness.review_data,
                target_timestamp_seconds=input_data.target_timestamp_seconds,
            )
            return run_live_prediction_speed_test(
                rows,
                requested_card_count=requested_card_count,
                eligible_card_count=input_data.eligible_card_count,
                model_id=lease.model_id,
                mode=mode,
                batch_size=batch_size,
                open_session=lambda batch, selected_mode, selected_batch_size, refresh_limit: (
                    open_live_prediction_benchmark_session(
                        lease,
                        batch,
                        mode=selected_mode,
                        batch_size=selected_batch_size,
                        refresh_limit=refresh_limit,
                        target_timestamp_seconds=input_data.target_timestamp_seconds,
                    )
                ),
                progress=progress,
            )
        finally:
            if lease is not None:
                lease.close()

    def success(result) -> None:
        is_dark = _is_dark_widget(parent)
        _show_results_after_progress_cleanup(
            parent,
            title="Between-Review Prediction Speed Test",
            render_html=lambda generation: render_live_prediction_speed_test_html(
                result,
                is_dark=is_dark,
                generation=generation,
            ),
        )

    def failure(exc: Exception) -> None:
        handle_checkpoint_failure(
            exc,
            lambda: show_live_prediction_speed_test(
                parent=parent,
                button=button,
                mode=mode,
                card_count=requested_card_count,
                batch_size=batch_size,
            ),
            parent=parent,
        )

    run_action_with_progress_stages(
        button=button,
        parent=parent,
        title="Between-Review Prediction Speed Test",
        label="Preparing Live Session prediction speed test",
        stages=[
            ProgressStage(collect_op, uses_collection=True),
            ProgressStage(benchmark_op, uses_collection=False),
        ],
        on_success=success,
        on_failure=failure,
        on_finish=lambda: _finish_speed_test(button),
    )


def _require_usable_checkpoint(manager) -> None:
    if not speed_test_checkpoint_is_usable(manager):
        raise MissingCheckpointError("No usable RWKV checkpoint is available for a speed test.")


def _finish_speed_test(button) -> None:
    # Manager resources are deliberately not touched here. The prediction
    # benchmark closes its own lease, while process_many releases its disposable
    # runtime before this callback. A staged operation may finish after another
    # owner acquires the shared manager.
    with suppress(RuntimeError):
        button.setEnabled(True)
