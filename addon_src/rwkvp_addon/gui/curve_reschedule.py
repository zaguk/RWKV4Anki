from __future__ import annotations

from dataclasses import dataclass

from aqt import mw
from aqt.utils import tooltip

from ..addon_config import (
    addon_config_for_mw,
    curve_rescheduling_enabled,
    experimental_short_term_rescheduling_enabled,
)
from ..anki_api import (
    ACTIVE_CARD_SEARCH,
    active_card_search_for_deck,
    apply_curve_reschedule_plan,
    card_schedule_info_for_ids,
    deck_scheduling_configs_for_decks,
    deck_scheduling_configs_for_subtree,
    find_cards,
    is_filtered_deck,
    is_fsrs_enabled,
)
from ..curve_reschedule import (
    CARD_TYPE_LEARNING,
    CARD_TYPE_RELEARNING,
    CARD_TYPE_REVIEW,
    CardScheduleInfo,
    CurveReschedulePlan,
    DeckSchedulingConfig,
    RescheduleSkipReason,
    build_curve_reschedule_plan,
)
from ..dataset_export import ensure_checkpoint_ready_from_load, load_review_data_for_checkpoint
from ..runtime import manager_for_mw, store_for_mw
from ..vendor_bootstrap import require_rwkv_interval
from .checkpoint_failure import handle_checkpoint_failure, require_checkpoint_for_use
from .common import (
    ProgressStage,
    run_with_progress_stages,
    show_fsrs_disabled,
    show_quiet_info,
)
from .web_message import ask_web_confirmation, show_web_warning


@dataclass(frozen=True)
class _ReschedulePlanningData:
    card_ids: list[int]
    review_load: object
    schedule_infos: dict[int, CardScheduleInfo]
    deck_configs: dict[int, DeckSchedulingConfig]
    today: int
    day_cutoff: int
    allow_short_term_rescheduling: bool


def show_reschedule_all_cards() -> None:
    _start_reschedule(
        parent=mw,
        title="RWKV Forgetting Curve Reschedule",
        search=ACTIVE_CARD_SEARCH,
    )


def show_reschedule_deck(deck_id: int) -> None:
    deck_id = int(deck_id)
    if is_filtered_deck(mw.col, deck_id):
        show_web_warning(
            "RWKV Forgetting Curve rescheduling can only be started from a normal deck.",
            title="RWKV Forgetting Curve",
            parent=mw,
        )
        return
    _start_reschedule(
        parent=mw,
        title="RWKV Forgetting Curve Reschedule",
        search=active_card_search_for_deck(mw.col, deck_id),
        deck_id=deck_id,
    )


def show_reschedule_selected_browser_cards(browser) -> None:
    card_ids = [int(card_id) for card_id in browser.selected_cards()]
    if not card_ids:
        show_web_warning(
            "Select at least one card to reschedule.",
            title="RWKV Forgetting Curve",
            parent=browser,
        )
        return
    _start_reschedule(
        parent=browser,
        title="RWKV Forgetting Curve Reschedule Selected Cards",
        card_ids=card_ids,
    )


def _start_reschedule(
    *,
    parent,
    title: str,
    search: str | None = None,
    card_ids: list[int] | None = None,
    deck_id: int | None = None,
) -> None:
    if not curve_rescheduling_enabled(addon_config_for_mw(mw)):
        show_web_warning(
            "RWKV Forgetting Curve rescheduling is disabled in the add-on config.",
            title="RWKV Forgetting Curve",
            parent=parent,
        )
        return
    if not _require_ready(parent, deck_id):
        return
    store = store_for_mw(mw)
    manager = manager_for_mw(mw)
    allow_short_term_rescheduling = experimental_short_term_rescheduling_enabled(
        addon_config_for_mw(mw)
    )

    def collect_op(col, progress, _previous):
        selected_ids = list(card_ids) if card_ids is not None else find_cards(col, search or "")
        if not selected_ids:
            raise ValueError("No cards matched the selected rescheduling scope.")

        progress.update(0, 1, "Reading card scheduling data")
        schedule_infos = card_schedule_info_for_ids(col, selected_ids)
        deck_configs = (
            deck_scheduling_configs_for_subtree(col, deck_id)
            if deck_id is not None
            else deck_scheduling_configs_for_decks(
                col,
                (info.source_deck_id for info in schedule_infos.values()),
            )
        )
        review_load = load_review_data_for_checkpoint(
            col,
            store,
            manager,
            progress,
            allow_incremental=True,
        )
        return _ReschedulePlanningData(
            card_ids=selected_ids,
            review_load=review_load,
            schedule_infos=schedule_infos,
            deck_configs=deck_configs,
            today=int(getattr(col.sched, "today", 0)),
            day_cutoff=int(getattr(col.sched, "day_cutoff", 0)),
            allow_short_term_rescheduling=allow_short_term_rescheduling,
        )

    def plan_op(_col, progress, previous):
        data = previous
        readiness = ensure_checkpoint_ready_from_load(manager, data.review_load, progress)
        progress.update(0, 1, "Planning card rescheduling")
        plan = build_curve_reschedule_plan(
            card_ids=data.card_ids,
            card_schedule_infos=data.schedule_infos,
            latest_curves_by_card=manager.latest_curves_for_cards(data.card_ids),
            last_reviews_by_card=readiness.review_data.last_by_card,
            deck_configs=data.deck_configs,
            interval_for_curve=require_rwkv_interval(),
            today=data.today,
            day_cutoff=data.day_cutoff,
            allow_short_term_rescheduling=data.allow_short_term_rescheduling,
        )
        progress.update(1, 1, "Planned card rescheduling")
        return plan

    def success(plan: CurveReschedulePlan) -> None:
        _confirm_and_apply_plan(parent, plan)

    def failure(exc: Exception) -> None:
        if isinstance(exc, ValueError):
            show_web_warning(str(exc), title="RWKV Forgetting Curve", parent=parent)
            return
        handle_checkpoint_failure(
            exc,
            lambda: _start_reschedule(
                parent=parent,
                title=title,
                search=search,
                card_ids=card_ids,
                deck_id=deck_id,
            ),
            parent=parent,
        )

    run_with_progress_stages(
        parent=parent,
        title=title,
        label="Planning card rescheduling",
        stages=[
            ProgressStage(collect_op, uses_collection=True),
            ProgressStage(plan_op, uses_collection=False),
        ],
        on_success=success,
        on_failure=failure,
    )


def _confirm_and_apply_plan(parent, plan: CurveReschedulePlan) -> None:
    message = _plan_summary(plan)
    if not plan.updates:
        show_quiet_info(message, title="RWKV Forgetting Curve", parent=parent)
        return
    ask_web_confirmation(
        parent=parent,
        title="RWKV Forgetting Curve",
        message=f"{message}\n\nApply these schedule changes to Anki?",
        confirm_label="Apply Schedule Changes",
        destructive=True,
        on_result=lambda confirmed: _apply_plan(parent, plan) if confirmed else None,
    )


def _apply_plan(parent, plan: CurveReschedulePlan) -> None:

    from aqt.operations import CollectionOp

    CollectionOp(
        parent,
        lambda col: apply_curve_reschedule_plan(col, plan),
    ).success(
        lambda _changes: tooltip(
            f"RWKV Forgetting Curve rescheduled {plan.update_count} cards.",
            parent=parent,
        )
    ).failure(
        lambda exc: show_web_warning(
            str(exc), title="RWKV Forgetting Curve", parent=parent
        )
    ).run_in_background()


def _plan_summary(plan: CurveReschedulePlan) -> str:
    lines = [
        f"Cards to reschedule: {plan.update_count}",
        f"Cards skipped: {len(plan.skipped)}",
    ]
    learning_updates = sum(
        1 for update in plan.updates if update.old_card_type == CARD_TYPE_LEARNING
    )
    relearning_updates = sum(
        1 for update in plan.updates if update.old_card_type == CARD_TYPE_RELEARNING
    )
    review_to_relearning = sum(
        1
        for update in plan.updates
        if update.old_card_type == CARD_TYPE_REVIEW
        and update.new_card_type == CARD_TYPE_RELEARNING
    )
    if learning_updates:
        lines.append(f"- learning cards rescheduled: {learning_updates}")
    if relearning_updates:
        lines.append(f"- relearning cards rescheduled: {relearning_updates}")
    if review_to_relearning:
        lines.append(
            f"- review cards converted to relearning: {review_to_relearning}"
        )
    clamped_subday = sum(1 for update in plan.updates if update.subday_interval_clamped)
    if clamped_subday:
        lines.append(
            f"- sub-day intervals scheduled as one-day review cards: {clamped_subday}"
        )
    for reason, count in sorted(
        plan.skip_counts.items(),
        key=lambda item: item[0].value,
    ):
        lines.append(f"- {_skip_reason_label(reason)}: {count}")
    return "\n".join(lines)


def _skip_reason_label(reason: RescheduleSkipReason) -> str:
    return {
        RescheduleSkipReason.NON_REVIEW: "not review cards",
        RescheduleSkipReason.NO_CURVE: "no RWKV curve",
        RescheduleSkipReason.NO_REVIEW_HISTORY: "no review history",
        RescheduleSkipReason.NO_INTERVAL: "no interval found",
        RescheduleSkipReason.ALREADY_SCHEDULED: "already scheduled",
        RescheduleSkipReason.LEARNING_STEPS_CONFIGURED: "learning steps configured",
        RescheduleSkipReason.RELEARNING_STEPS_CONFIGURED: "relearning steps configured",
        RescheduleSkipReason.ACTIVE_LEARNING_STEPS: "active learning/relearning steps",
        RescheduleSkipReason.FILTERED_DECK_UNSUPPORTED: "cards in filtered decks",
        RescheduleSkipReason.SHORT_TERM_RESCHEDULING_DISABLED: (
            "experimental short-term rescheduling disabled"
        ),
    }[reason]


def _require_ready(parent, deck_id: int | None = None) -> bool:
    if not is_fsrs_enabled(mw.col, deck_id):
        show_fsrs_disabled(parent)
        return False
    return require_checkpoint_for_use(parent, manager=manager_for_mw(mw))
