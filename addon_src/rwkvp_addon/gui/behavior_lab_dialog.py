from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from aqt import mw
from aqt.qt import Qt
from aqt.utils import tooltip

from ..anki_api import find_cards, is_fsrs_enabled
from ..behavior_lab import (
    BehaviorLabExperiment,
    BehaviorLabResult,
    BehaviorLabScenarioRunner,
    BehaviorLabValidationError,
    ResolvedBehaviorLabExperiment,
    behavior_lab_template,
    build_behavior_lab_result,
    experiment_work_units,
    resolve_experiment,
    runtime_scope_card_ids,
)
from ..behavior_lab_controller import BehaviorLabWebController
from ..behavior_lab_store import BehaviorLabExperimentStore
from ..checkpoint_progress import update_checkpoint_collection_data
from ..dataset_export import (
    ReviewDataLoad,
    load_review_data_for_checkpoint,
    open_checkpoint_runtime_from_load,
)
from ..review_rows import checkpoint_scope_cards_for_card_ids
from ..runtime import manager_for_mw, store_for_mw
from ..vendor_bootstrap import require_rwkv_probability
from ..web_dialog_controller import CloseReason
from .checkpoint_failure import handle_checkpoint_failure, require_checkpoint_for_use
from .common import ProgressStage, run_with_progress_stages, show_fsrs_disabled
from .web_dialog import WebDialogHost, widget_uses_dark_palette
from .web_message import ask_web_confirmation, show_web_warning


@dataclass(frozen=True)
class _PreparedBehaviorLabRun:
    review_load: ReviewDataLoad
    resolved: ResolvedBehaviorLabExperiment


class BehaviorLabDialog(WebDialogHost):
    def __init__(
        self,
        parent=None,
        *,
        initial_card_ids: tuple[int, ...] = (),
    ) -> None:
        self._running = False
        self._last_result: BehaviorLabResult | None = None
        self._experiment_store = BehaviorLabExperimentStore(
            store_for_mw(mw).behavior_lab_experiments_path
        )
        unique_ids = tuple(dict.fromkeys(int(card_id) for card_id in initial_card_ids))
        focal_card_id = unique_ids[0] if unique_ids else _current_card_id()
        experiment = behavior_lab_template(
            "sibling_spillover" if focal_card_id else "custom",
            focal_card_id=focal_card_id,
            selection_card_ids=unique_ids,
        )
        self._controller = BehaviorLabWebController(
            experiment=experiment,
            experiment_store=self._experiment_store,
            on_run_requested=self.run_experiment,
            on_delete_requested=self._request_delete_saved_experiment,
            on_error=self._show_error,
            is_dark=widget_uses_dark_palette(parent),
        )
        super().__init__(
            parent,
            title="RWKV Behavior Lab",
            controller=self._controller,
            size=(1120, 760),
            web_minimum_height=620,
            modality=Qt.WindowModality.NonModal,
            requires_collection=True,
            close_policy=self._close_policy,
        )
        self._controller.attach_rerender(self.rerender)

    @property
    def experiment(self) -> BehaviorLabExperiment:
        return self._controller.experiment

    @experiment.setter
    def experiment(self, experiment: BehaviorLabExperiment) -> None:
        self._controller.experiment = experiment

    def run_experiment(self) -> None:
        if self._running:
            return
        if mw.col is None:
            return
        if not is_fsrs_enabled(mw.col):
            show_fsrs_disabled(self)
            return
        manager = manager_for_mw(mw)
        if not require_checkpoint_for_use(self, manager=manager):
            return
        store = store_for_mw(mw)
        experiment = self.experiment
        self._set_running(True)

        def collect_op(col, progress, _previous):
            review_load = load_review_data_for_checkpoint(
                col,
                store,
                manager,
                progress,
                allow_incremental=True,
            )
            searched_card_ids: tuple[int, ...] = ()
            if experiment.tracked_search.strip():
                update_checkpoint_collection_data(
                    progress,
                    "Finding cards in the Behavior Lab tracking search",
                )
                searched_card_ids = tuple(find_cards(col, experiment.tracked_search.strip()))
            resolved = resolve_experiment(
                experiment,
                review_load.review_data,
                searched_card_ids=searched_card_ids,
                now_timestamp_seconds=time.time(),
            )
            return _PreparedBehaviorLabRun(review_load=review_load, resolved=resolved)

        def simulate_op(_col, progress, previous: _PreparedBehaviorLabRun):
            resolved = previous.resolved
            total = experiment_work_units(resolved)
            completed = 0

            def progress_step(amount: int, label: str) -> None:
                nonlocal completed
                completed = min(total, completed + max(0, int(amount)))
                progress.update(completed, total, label)
                progress.check_cancelled()

            scenario_results = []
            curve_predictor = require_rwkv_probability()
            scope_card_ids = runtime_scope_card_ids(resolved)
            scope_cards = (
                None
                if scope_card_ids is None
                else checkpoint_scope_cards_for_card_ids(
                    scope_card_ids,
                    previous.review_load.review_data,
                )
            )
            for index, scenario in enumerate(resolved.scenarios, start=1):
                progress.check_cancelled()
                progress.update(
                    completed,
                    total,
                    f"Loading branch {index}/{len(resolved.scenarios)}: {scenario.scenario.name}",
                )
                _readiness, runtime = open_checkpoint_runtime_from_load(
                    manager,
                    previous.review_load,
                    progress,
                    scope_cards=scope_cards,
                    wait_for_save=True,
                )
                try:
                    runner = BehaviorLabScenarioRunner(
                        runtime=runtime,
                        review_data=previous.review_load.review_data,
                        resolved_experiment=resolved,
                        curve_predictor=curve_predictor,
                        progress_step=progress_step,
                        check_cancelled=progress.check_cancelled,
                    )
                    scenario_results.append(runner.run(scenario))
                finally:
                    runtime.close()
            progress.update(total, total, "Behavior Lab simulation complete")
            return build_behavior_lab_result(
                resolved,
                scenario_results,
                model_id=manager.model_id,
                checkpoint_fingerprint=_checkpoint_fingerprint(store.manifest()),
            )

        def success(result: BehaviorLabResult) -> None:
            self._finish_run()
            self._last_result = result
            if self.cleaned_up:
                return
            self._controller.show_result(result)

        def failure(exc: Exception) -> None:
            self._finish_run()
            if self.cleaned_up:
                return
            if isinstance(exc, BehaviorLabValidationError):
                show_web_warning(str(exc), title="RWKV Behavior Lab", parent=self)
                return
            handle_checkpoint_failure(exc, self.run_experiment, parent=self)

        def cancelled() -> None:
            self._finish_run()
            if self.cleaned_up:
                return
            tooltip("RWKV Behavior Lab simulation cancelled.", parent=self)

        run_with_progress_stages(
            parent=self,
            title="RWKV Behavior Lab",
            label="Preparing simulation",
            stages=[
                ProgressStage(collect_op, uses_collection=True),
                ProgressStage(simulate_op, uses_collection=False),
            ],
            on_success=success,
            on_failure=failure,
            on_cancel=cancelled,
        )

    def _finish_run(self) -> None:
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self._running = bool(running)
        self._controller.running = bool(running)
        disabled = "true" if running else "false"
        with suppress(AttributeError, RuntimeError):
            self.web.eval(
                "(() => { const button = document.getElementById('rwkv-dialog-close'); "
                f"if (button) {{ button.disabled = {disabled}; "
                f"button.setAttribute('aria-disabled', '{disabled}'); }} }})()"
            )

    def _close_policy(self, reason: CloseReason) -> bool:
        if self._controller.can_close(reason):
            return True
        show_web_warning(
            "Cancel the active Behavior Lab progress operation before closing this window.",
            title="RWKV Behavior Lab",
            parent=self,
        )
        return False

    def _show_error(self, message: str) -> None:
        show_web_warning(str(message), title="RWKV Behavior Lab", parent=self)

    def _request_delete_saved_experiment(self, name: str) -> None:
        ask_web_confirmation(
            parent=self,
            title="Delete saved experiment?",
            message=f"“{name}” will be permanently removed.",
            confirm_label="Delete experiment",
            cancel_label="Keep experiment",
            destructive=True,
            on_result=lambda confirmed: (
                self._controller.delete_saved_experiment(name) if confirmed else None
            ),
        )


def show_behavior_lab(
    *,
    parent=None,
    initial_card_ids: tuple[int, ...] = (),
) -> BehaviorLabDialog | None:
    parent = parent or mw
    if mw.col is None:
        return None
    if not is_fsrs_enabled(mw.col):
        show_fsrs_disabled(parent)
        return None
    manager = manager_for_mw(mw)
    if not require_checkpoint_for_use(parent, manager=manager):
        return None
    dialog = BehaviorLabDialog(parent, initial_card_ids=initial_card_ids)
    dialog.open()
    return dialog


def _current_card_id() -> int:
    reviewer = getattr(mw, "reviewer", None)
    card = getattr(reviewer, "card", None)
    try:
        return int(getattr(card, "id", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _checkpoint_fingerprint(manifest: dict[str, Any]) -> str:
    value = manifest.get("history_fingerprint")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("digest", "sha256", "hash"):
            candidate = value.get(key)
            if candidate:
                return str(candidate)
    binding = manifest.get("evaluation_cache_binding")
    if isinstance(binding, dict):
        candidate = binding.get("checkpoint_history_fingerprint")
        if candidate:
            return str(candidate)
    return ""
