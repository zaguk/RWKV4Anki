from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

from aqt import mw

from ..anki_api import find_cards
from ..dataset_export import load_review_data_from_collection
from ..runtime import manager_for_mw
from ..state_comparison import (
    ImmediateStateComparisonPlan,
    current_review_rows_from_included_history,
    deleted_review_comparison_plan,
    model_comparison_plan,
    run_immediate_state_comparison,
)
from ..state_comparison_html import render_immediate_state_comparison_html
from .common import ProgressStage, run_action_with_progress_stages
from .speed_test_dialog import (
    _is_dark_widget,
    _show_results_after_progress_cleanup,
)
from .web_message import ask_web_confirmation, show_web_warning


@dataclass
class _StateComparisonInput:
    plan: ImmediateStateComparisonPlan

    def release_review_rows(self) -> None:
        seen: set[int] = set()
        for variant in self.plan.variants:
            identity = id(variant.rows)
            if identity in seen:
                continue
            seen.add(identity)
            variant.rows.clear()


def show_deleted_reviews_comparison(
    *,
    parent,
    button,
    model_id: str,
    current_includes_deleted_reviews: bool,
) -> None:
    manager = manager_for_mw(mw)
    if not _comparison_can_start(manager, parent=parent):
        return
    _confirm_comparison(
        parent,
        state_count=2,
        subject="deleted-card history",
        on_result=lambda confirmed: _start_deleted_reviews_comparison(
            parent=parent,
            button=button,
            manager=manager,
            model_id=model_id,
            current_includes_deleted_reviews=current_includes_deleted_reviews,
        )
        if confirmed
        else None,
    )


def _start_deleted_reviews_comparison(
    *,
    parent,
    button,
    manager,
    model_id: str,
    current_includes_deleted_reviews: bool,
) -> None:
    if not _comparison_can_start(manager, parent=parent):
        return
    store = manager.store

    def collect_op(col, progress, _previous):
        progress.update(0, 2, "Finding current collection cards")
        current_card_ids = frozenset(int(card_id) for card_id in find_cards(col, ""))
        progress.update(1, 2, "Loading review history with deleted cards")
        all_data = load_review_data_from_collection(
            col,
            store,
            exclude_deleted_card_revlogs=False,
        )
        current_rows, day_offset_adjustment = current_review_rows_from_included_history(
            all_data.rows,
            current_card_ids,
            source_day_offset_origin=all_data.day_offset_origin,
        )
        reviewed_current_card_ids = _reviewed_card_ids(current_rows)
        progress.update(2, 2, "Prepared deleted-card history comparison")
        return _StateComparisonInput(
            plan=deleted_review_comparison_plan(
                current_rows=current_rows,
                all_rows=all_data.rows,
                current_card_ids=reviewed_current_card_ids,
                model_id=model_id,
                current_includes_deleted_reviews=current_includes_deleted_reviews,
                process_many_mode=manager.process_many_mode,
                without_deleted_day_offset_adjustment=day_offset_adjustment,
            )
        )

    _run_comparison(
        parent=parent,
        button=button,
        manager=manager,
        title="Deleted-card History Comparison",
        initial_label="Preparing deleted-card history comparison",
        collect_op=collect_op,
    )


def show_models_comparison(
    *,
    parent,
    button,
    model_ids: tuple[str, ...],
    current_model_id: str,
    include_deleted_reviews: bool,
) -> None:
    normalized_models = tuple(dict.fromkeys(str(model_id) for model_id in model_ids))
    if len(normalized_models) < 2:
        show_web_warning(
            "Only one RWKV model is available, so there is nothing to compare.",
            title="RWKV Model Comparison",
            parent=parent,
        )
        return
    manager = manager_for_mw(mw)
    if not _comparison_can_start(manager, parent=parent):
        return
    _confirm_comparison(
        parent,
        state_count=len(normalized_models),
        subject="RWKV models",
        on_result=lambda confirmed: _start_models_comparison(
            parent=parent,
            button=button,
            manager=manager,
            normalized_models=normalized_models,
            current_model_id=current_model_id,
            include_deleted_reviews=include_deleted_reviews,
        )
        if confirmed
        else None,
    )


def _start_models_comparison(
    *,
    parent,
    button,
    manager,
    normalized_models: tuple[str, ...],
    current_model_id: str,
    include_deleted_reviews: bool,
) -> None:
    if not _comparison_can_start(manager, parent=parent):
        return
    store = manager.store

    def collect_op(col, progress, _previous):
        progress.update(0, 2, "Finding current collection cards")
        current_card_ids = frozenset(int(card_id) for card_id in find_cards(col, ""))
        progress.update(1, 2, "Loading review history for every model")
        build_data = load_review_data_from_collection(
            col,
            store,
            exclude_deleted_card_revlogs=not include_deleted_reviews,
        )
        reviewed_current_card_ids: set[int] = set()
        current_review_count = 0
        for row in build_data.rows:
            card_id = int(row["card_id"])
            if card_id not in current_card_ids:
                continue
            reviewed_current_card_ids.add(card_id)
            current_review_count += 1
        progress.update(2, 2, "Prepared RWKV model comparison")
        plan = model_comparison_plan(
            rows=build_data.rows,
            current_card_ids=frozenset(reviewed_current_card_ids),
            model_ids=normalized_models,
            current_model_id=current_model_id,
            include_deleted_reviews=include_deleted_reviews,
            current_review_count=current_review_count,
            process_many_mode=manager.process_many_mode,
        )
        return _StateComparisonInput(plan=plan)

    _run_comparison(
        parent=parent,
        button=button,
        manager=manager,
        title="RWKV Model Comparison",
        initial_label="Preparing RWKV model comparison",
        collect_op=collect_op,
    )


def _run_comparison(
    *,
    parent,
    button,
    manager,
    title: str,
    initial_label: str,
    collect_op,
) -> None:
    def compare_op(_col, progress, previous):
        input_data = previous
        slot = None
        try:
            progress.update(0, 1, "Reserving RWKV state-building resources")
            slot = manager.reserve_runtime_slot(progress)
            return run_immediate_state_comparison(input_data.plan, progress)
        finally:
            input_data.release_review_rows()
            if slot is not None:
                slot.close()

    def success(result) -> None:
        is_dark = _is_dark_widget(parent)
        _show_results_after_progress_cleanup(
            parent,
            title=title,
            render_html=lambda generation: render_immediate_state_comparison_html(
                result,
                is_dark=is_dark,
                generation=generation,
            ),
        )

    def failure(exc: Exception) -> None:
        show_web_warning(str(exc), title=title, parent=parent)

    run_action_with_progress_stages(
        button=button,
        parent=parent,
        title=title,
        label=initial_label,
        stages=[
            ProgressStage(collect_op, uses_collection=True),
            ProgressStage(compare_op, uses_collection=False),
        ],
        on_success=success,
        on_failure=failure,
        on_finish=lambda: _finish_comparison(button),
    )


def _confirm_comparison(
    parent,
    *,
    state_count: int,
    subject: str,
    on_result,
) -> None:
    plural = "state" if int(state_count) == 1 else "states"
    ask_web_confirmation(
        parent=parent,
        title="RWKV State Comparison",
        message=(
            f"RWKV will build {int(state_count):,} disposable full-history {plural} to "
            f"compare {subject}. This can take quite some time and may temporarily use "
            "substantial RAM or GPU memory.\n\n"
            "Progress can be cancelled. No checkpoint or Anki review data will be changed."
        ),
        confirm_label="Start Comparison",
        on_result=on_result,
    )


def _comparison_can_start(manager, *, parent) -> bool:
    if not (
        bool(getattr(manager, "runtime_scope_active", False))
        or bool(getattr(manager, "runtime_loaded", False))
        or bool(getattr(manager, "save_in_progress", False))
    ):
        return True
    show_web_warning(
        "Stop the active RWKV operation, Browser load, Live Session, or checkpoint "
        "write before starting this comparison.",
        title="RWKV State Comparison",
        parent=parent,
    )
    return False


def _finish_comparison(button) -> None:
    with suppress(RuntimeError):
        button.setEnabled(True)


def _reviewed_card_ids(rows) -> frozenset[int]:
    return frozenset(int(row["card_id"]) for row in rows)
