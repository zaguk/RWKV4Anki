from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

from aqt import mw

from ..anki_api import find_cards
from ..dataset_export import load_review_data_from_collection
from ..model_generation import (
    DEFAULT_MODEL_GENERATION,
    model_generation_uses_v5o,
    normalize_model_generation,
)
from ..runtime import manager_for_mw
from ..runtime_inputs import RuntimeInputContext, input_context_for_generation
from ..state_comparison import (
    ImmediateStateComparisonPlan,
    current_review_rows_from_included_history,
    deleted_review_comparison_plan,
    generation_comparison_plan,
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
    model_generation: str | None = None,
    current_includes_deleted_reviews: bool,
    state_precision: str = "fp32",
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
            model_generation=_requested_model_generation(model_generation, manager),
            current_includes_deleted_reviews=current_includes_deleted_reviews,
            state_precision=state_precision,
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
    model_generation: str,
    current_includes_deleted_reviews: bool,
    state_precision: str = "fp32",
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
            **({"model_generation": "v5o"} if model_generation_uses_v5o(model_generation) else {}),
        )
        current_rows, day_offset_adjustment = current_review_rows_from_included_history(
            all_data.rows,
            current_card_ids,
            source_day_offset_origin=all_data.day_offset_origin,
        )
        reviewed_current_card_ids = _reviewed_card_ids(current_rows)
        input_context = _input_context_for_generation(all_data, model_generation)
        progress.update(2, 2, "Prepared deleted-card history comparison")
        return _StateComparisonInput(
            plan=deleted_review_comparison_plan(
                current_rows=current_rows,
                all_rows=all_data.rows,
                current_card_ids=reviewed_current_card_ids,
                model_id=str(model_id),
                model_generation=model_generation,
                current_includes_deleted_reviews=current_includes_deleted_reviews,
                process_many_mode=manager.process_many_mode,
                state_precision=state_precision,
                without_deleted_day_offset_adjustment=day_offset_adjustment,
                input_context=input_context,
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
    model_generation: str | None = None,
    include_deleted_reviews: bool,
    state_precision: str = "fp32",
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
            model_generation=_requested_model_generation(model_generation, manager),
            include_deleted_reviews=include_deleted_reviews,
            state_precision=state_precision,
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
    model_generation: str,
    include_deleted_reviews: bool,
    state_precision: str = "fp32",
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
            **({"model_generation": "v5o"} if model_generation_uses_v5o(model_generation) else {}),
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
            model_generation=model_generation,
            include_deleted_reviews=include_deleted_reviews,
            current_review_count=current_review_count,
            process_many_mode=manager.process_many_mode,
            state_precision=state_precision,
            input_context=_input_context_for_generation(build_data, model_generation),
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
    retain_generations: bool = False,
) -> None:
    def compare_op(_col, progress, previous):
        input_data = previous
        slot = None
        try:
            progress.update(0, 1, "Reserving RWKV state-building resources")
            slot = manager.reserve_runtime_slot(progress)
            return run_immediate_state_comparison(
                input_data.plan, progress,
                **({"retention_store": manager.store} if retain_generations else {}),
            )
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
    retain_generations: bool = False,
) -> None:
    plural = "state" if int(state_count) == 1 else "states"
    ask_web_confirmation(
        parent=parent,
        title="RWKV State Comparison",
        message=(
            f"RWKV will build {int(state_count):,} full-history {plural} to "
            f"compare {subject}. This can take quite some time and may temporarily use "
            "substantial RAM or GPU memory.\n\n"
            "Progress can be cancelled. "
            + (
                "On success, one checkpoint and Immediate evaluation cache per generation "
                "will be saved, replacing that generation's previous checkpoint. "
                "V1 curve features require a rebuild with curves enabled."
                if retain_generations else
                "These states are disposable; no checkpoint will be changed."
            )
            + " Anki review data will not be changed."
        ),
        confirm_label="Start Comparison",
        on_result=on_result,
    )


def show_generations_comparison(
    *, parent, button, model_ids_by_generation, current_model_generation: str,
    include_deleted_reviews: bool, retain_generations: bool = False,
    state_precision: str = "fp32",
) -> None:
    models = {g: tuple(ids) for g, ids in model_ids_by_generation.items() if ids}
    if len(models) < 2:
        show_web_warning(
            "Only one RWKV generation is available, so there is nothing to compare.",
            title="RWKV Generation Comparison", parent=parent,
        )
        return
    manager = manager_for_mw(mw)
    if not _comparison_can_start(manager, parent=parent):
        return

    def start(confirmed):
        if not confirmed or not _comparison_can_start(manager, parent=parent):
            return

        def collect(col, progress, _previous):
            progress.update(0, 3, "Finding current collection cards")
            current_ids = frozenset(int(cid) for cid in find_cards(col, ""))
            progress.update(1, 3, "Reading review history for generation comparison")
            data = load_review_data_from_collection(
                col, manager.store, exclude_deleted_card_revlogs=not include_deleted_reviews,
            )
            progress.check_cancelled()
            v5_data = (
                load_review_data_from_collection(
                    col, manager.store, exclude_deleted_card_revlogs=not include_deleted_reviews,
                    model_generation="v5o",
                ) if "v5o" in models else None
            )
            reviewed_ids = _reviewed_card_ids(data.rows) & current_ids
            plan = generation_comparison_plan(
                rows=data.rows, current_card_ids=reviewed_ids,
                model_ids_by_generation=models, current_model_generation=current_model_generation,
                include_deleted_reviews=include_deleted_reviews,
                current_review_count=sum(int(row["card_id"]) in current_ids for row in data.rows),
                process_many_mode=manager.process_many_mode, v5o_review_data=v5_data,
                state_precision=state_precision,
            )
            progress.update(3, 3, "Generation comparison ready")
            return _StateComparisonInput(plan)

        _run_comparison(
            parent=parent, button=button, manager=manager, title="RWKV Generation Comparison",
            initial_label="Preparing RWKV generation comparison", collect_op=collect,
            retain_generations=retain_generations,
        )

    _confirm_comparison(
        parent, state_count=len(models), subject="RWKV generations using their default models",
        on_result=start, retain_generations=retain_generations,
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


def _requested_model_generation(value: str | None, manager) -> str:
    return normalize_model_generation(
        getattr(manager, "model_generation", DEFAULT_MODEL_GENERATION)
        if value is None
        else value
    )


def _input_context_for_generation(
    review_data,
    model_generation: str,
) -> RuntimeInputContext | None:
    return input_context_for_generation(review_data, model_generation)


def _reviewed_card_ids(rows) -> frozenset[int]:
    return frozenset(int(row["card_id"]) for row in rows)
