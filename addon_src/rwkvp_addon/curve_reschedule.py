from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .review_rows import SECONDS_PER_DAY, LastReviewInfo

CARD_TYPE_NEW = 0
CARD_TYPE_LEARNING = 1
CARD_TYPE_REVIEW = 2
CARD_TYPE_RELEARNING = 3

QUEUE_TYPE_NEW = 0
QUEUE_TYPE_LEARNING = 1
QUEUE_TYPE_REVIEW = 2
QUEUE_TYPE_DAY_LEARN_RELEARN = 3


class RescheduleSkipReason(Enum):
    NON_REVIEW = "non_review"
    NO_CURVE = "no_curve"
    NO_REVIEW_HISTORY = "no_review_history"
    NO_INTERVAL = "no_interval"
    ALREADY_SCHEDULED = "already_scheduled"
    LEARNING_STEPS_CONFIGURED = "learning_steps_configured"
    RELEARNING_STEPS_CONFIGURED = "relearning_steps_configured"
    ACTIVE_LEARNING_STEPS = "active_learning_steps"
    FILTERED_DECK_UNSUPPORTED = "filtered_deck_unsupported"
    SHORT_TERM_RESCHEDULING_DISABLED = "short_term_rescheduling_disabled"


@dataclass(frozen=True)
class DeckSchedulingConfig:
    deck_id: int
    desired_retention: float
    max_interval: int | None = None
    name: str = ""
    preset_id: int | None = None
    learning_steps_blank: bool = False
    relearning_steps_blank: bool = False


@dataclass(frozen=True)
class CardScheduleInfo:
    card_id: int
    source_deck_id: int | None
    card_type: int
    queue: int
    due: int
    interval: int
    original_deck_id: int = 0
    original_due: int = 0
    reps: int = 0
    lapses: int = 0
    remaining_steps: int = 0
    note_id: int | None = None
    template_index: int = 0
    modified_secs: int = 0

    @property
    def due_field(self) -> str:
        return "odue" if self.original_deck_id else "due"

    @property
    def active_due(self) -> int:
        return int(self.original_due if self.original_deck_id else self.due)


@dataclass(frozen=True)
class CurveRescheduleUpdate:
    card_id: int
    source_deck_id: int | None
    old_interval: int
    new_interval: int
    old_due: int
    new_due: int
    due_field: str
    desired_retention: float
    curve_review_id: int
    old_card_type: int | None = None
    new_card_type: int | None = None
    old_queue: int | None = None
    new_queue: int | None = None
    old_remaining_steps: int | None = None
    new_remaining_steps: int | None = None
    subday_interval_clamped: bool = False


@dataclass(frozen=True)
class _TargetSchedule:
    card_type: int
    queue: int
    interval: int
    due: int
    due_field: str
    remaining_steps: int
    subday_interval_clamped: bool = False


@dataclass(frozen=True)
class CurveRescheduleSkip:
    card_id: int
    reason: RescheduleSkipReason


@dataclass(frozen=True)
class CurveReschedulePlan:
    updates: list[CurveRescheduleUpdate]
    skipped: list[CurveRescheduleSkip]

    @property
    def update_count(self) -> int:
        return len(self.updates)

    @property
    def skip_counts(self) -> dict[RescheduleSkipReason, int]:
        counts: dict[RescheduleSkipReason, int] = {}
        for skip in self.skipped:
            counts[skip.reason] = counts.get(skip.reason, 0) + 1
        return counts


def build_curve_reschedule_plan(
    *,
    card_ids: Iterable[int],
    card_schedule_infos: Mapping[int, CardScheduleInfo],
    latest_curves_by_card: Mapping[int, Any],
    last_reviews_by_card: Mapping[int, LastReviewInfo],
    deck_configs: Mapping[int, DeckSchedulingConfig],
    interval_for_curve,
    today: int,
    day_cutoff: int,
    allow_short_term_rescheduling: bool = False,
) -> CurveReschedulePlan:
    updates: list[CurveRescheduleUpdate] = []
    skipped: list[CurveRescheduleSkip] = []

    for card_id in [int(value) for value in card_ids]:
        schedule = card_schedule_infos.get(card_id)
        if schedule is None:
            skipped.append(
                CurveRescheduleSkip(card_id, RescheduleSkipReason.NON_REVIEW)
            )
            continue
        if int(schedule.original_deck_id):
            skipped.append(
                CurveRescheduleSkip(
                    card_id,
                    RescheduleSkipReason.FILTERED_DECK_UNSUPPORTED,
                )
            )
            continue
        if int(schedule.remaining_steps) != 0:
            skipped.append(
                CurveRescheduleSkip(card_id, RescheduleSkipReason.ACTIVE_LEARNING_STEPS)
            )
            continue
        if not _is_supported_card_state(schedule):
            skipped.append(
                CurveRescheduleSkip(card_id, RescheduleSkipReason.NON_REVIEW)
            )
            continue

        curve = latest_curves_by_card.get(card_id)
        if curve is None:
            skipped.append(CurveRescheduleSkip(card_id, RescheduleSkipReason.NO_CURVE))
            continue

        last_review = last_reviews_by_card.get(card_id)
        if last_review is None:
            skipped.append(
                CurveRescheduleSkip(card_id, RescheduleSkipReason.NO_REVIEW_HISTORY)
            )
            continue

        deck_config = _deck_config_for_card(schedule, deck_configs)
        desired_retention = deck_config.desired_retention if deck_config else 0.9
        max_interval = deck_config.max_interval if deck_config else None
        interval_seconds = _effective_interval_seconds(
            interval_for_curve(curve, desired_retention),
            max_interval=max_interval,
        )
        if interval_seconds is None:
            skipped.append(CurveRescheduleSkip(card_id, RescheduleSkipReason.NO_INTERVAL))
            continue

        target = _target_schedule_for_card(
            schedule=schedule,
            interval_seconds=interval_seconds,
            last_review=last_review,
            deck_config=deck_config,
            today=today,
            day_cutoff=day_cutoff,
            allow_short_term_rescheduling=allow_short_term_rescheduling,
        )
        if isinstance(target, RescheduleSkipReason):
            skipped.append(CurveRescheduleSkip(card_id, target))
            continue

        if _schedule_already_matches(schedule, target):
            skipped.append(
                CurveRescheduleSkip(
                    card_id,
                    RescheduleSkipReason.ALREADY_SCHEDULED,
                )
            )
            continue

        updates.append(
            CurveRescheduleUpdate(
                card_id=card_id,
                source_deck_id=schedule.source_deck_id,
                old_interval=int(schedule.interval),
                new_interval=target.interval,
                old_due=_old_due_for_target(schedule, target),
                new_due=target.due,
                due_field=target.due_field,
                desired_retention=desired_retention,
                curve_review_id=last_review.review_id,
                old_card_type=int(schedule.card_type),
                new_card_type=target.card_type,
                old_queue=int(schedule.queue),
                new_queue=target.queue,
                old_remaining_steps=int(schedule.remaining_steps),
                new_remaining_steps=target.remaining_steps,
                subday_interval_clamped=target.subday_interval_clamped,
            )
        )

    return CurveReschedulePlan(updates=updates, skipped=skipped)


def interval_days_from_curve_seconds(
    interval_seconds: float | int | None,
    *,
    max_interval: int | None,
) -> int | None:
    if interval_seconds is None:
        return int(max_interval) if max_interval is not None else None
    if not math.isfinite(float(interval_seconds)):
        return None
    interval = max(1, int(round(float(interval_seconds) / SECONDS_PER_DAY)))
    if max_interval is not None:
        interval = min(interval, max(1, int(max_interval)))
    return interval


def _effective_interval_seconds(
    interval_seconds: float | int | None,
    *,
    max_interval: int | None,
) -> float | None:
    if interval_seconds is None:
        if max_interval is None:
            return None
        return float(max(1, int(max_interval)) * SECONDS_PER_DAY)
    interval = float(interval_seconds)
    if not math.isfinite(interval) or interval < 0:
        return None
    if max_interval is not None:
        interval = min(interval, float(max(1, int(max_interval)) * SECONDS_PER_DAY))
    return interval


def anki_day_for_timestamp(
    timestamp_seconds: float,
    *,
    today: int,
    day_cutoff: int,
) -> int:
    current_day_start = int(day_cutoff) - SECONDS_PER_DAY
    return int(
        math.floor((float(timestamp_seconds) - current_day_start) / SECONDS_PER_DAY)
    ) + int(today)


def due_day_from_last_review(last_review_day: int, interval_days: int) -> int:
    due = int(last_review_day) + max(1, int(interval_days))
    return due if due != 0 else 1


def _is_supported_card_state(schedule: CardScheduleInfo) -> bool:
    if _is_normal_review_card(schedule):
        return True
    if int(schedule.original_deck_id):
        return False
    return _is_learning_card(schedule) or _is_relearning_card(schedule)


def _is_normal_review_card(schedule: CardScheduleInfo) -> bool:
    return int(schedule.card_type) == CARD_TYPE_REVIEW and int(schedule.queue) == QUEUE_TYPE_REVIEW


def _is_learning_card(schedule: CardScheduleInfo) -> bool:
    return int(schedule.card_type) == CARD_TYPE_LEARNING and int(schedule.queue) in {
        QUEUE_TYPE_LEARNING,
        QUEUE_TYPE_DAY_LEARN_RELEARN,
    }


def _is_relearning_card(schedule: CardScheduleInfo) -> bool:
    return int(schedule.card_type) == CARD_TYPE_RELEARNING and int(schedule.queue) in {
        QUEUE_TYPE_LEARNING,
        QUEUE_TYPE_DAY_LEARN_RELEARN,
    }


def _target_schedule_for_card(
    *,
    schedule: CardScheduleInfo,
    interval_seconds: float,
    last_review: LastReviewInfo,
    deck_config: DeckSchedulingConfig | None,
    today: int,
    day_cutoff: int,
    allow_short_term_rescheduling: bool,
) -> _TargetSchedule | RescheduleSkipReason:
    if _is_normal_review_card(schedule):
        return _target_schedule_for_review_card(
            schedule=schedule,
            interval_seconds=interval_seconds,
            last_review=last_review,
            deck_config=deck_config,
            today=today,
            day_cutoff=day_cutoff,
            allow_short_term_rescheduling=allow_short_term_rescheduling,
        )

    if _is_learning_card(schedule):
        if not allow_short_term_rescheduling:
            return RescheduleSkipReason.SHORT_TERM_RESCHEDULING_DISABLED
        if deck_config is None or not deck_config.learning_steps_blank:
            return RescheduleSkipReason.LEARNING_STEPS_CONFIGURED
        return _target_schedule_for_learning_queue(
            schedule=schedule,
            card_type=CARD_TYPE_LEARNING,
            interval_seconds=interval_seconds,
            last_review=last_review,
            today=today,
            day_cutoff=day_cutoff,
        )

    if _is_relearning_card(schedule):
        if not allow_short_term_rescheduling:
            return RescheduleSkipReason.SHORT_TERM_RESCHEDULING_DISABLED
        if deck_config is None or not deck_config.relearning_steps_blank:
            return RescheduleSkipReason.RELEARNING_STEPS_CONFIGURED
        return _target_schedule_for_learning_queue(
            schedule=schedule,
            card_type=CARD_TYPE_RELEARNING,
            interval_seconds=interval_seconds,
            last_review=last_review,
            today=today,
            day_cutoff=day_cutoff,
        )

    return RescheduleSkipReason.NON_REVIEW


def _target_schedule_for_review_card(
    *,
    schedule: CardScheduleInfo,
    interval_seconds: float,
    last_review: LastReviewInfo,
    deck_config: DeckSchedulingConfig | None,
    today: int,
    day_cutoff: int,
    allow_short_term_rescheduling: bool,
) -> _TargetSchedule | RescheduleSkipReason:
    subday_interval = interval_seconds < SECONDS_PER_DAY
    relearning_steps_blank = bool(
        deck_config.relearning_steps_blank if deck_config else False
    )
    if subday_interval and allow_short_term_rescheduling and relearning_steps_blank:
        return _target_schedule_for_learning_queue(
            schedule=schedule,
            card_type=CARD_TYPE_RELEARNING,
            interval_seconds=interval_seconds,
            last_review=last_review,
            today=today,
            day_cutoff=day_cutoff,
            interval=1,
            remaining_steps=0,
        )

    new_interval = interval_days_from_curve_seconds(
        interval_seconds,
        max_interval=deck_config.max_interval if deck_config else None,
    )
    if new_interval is None:
        return RescheduleSkipReason.NO_INTERVAL
    last_review_day = anki_day_for_timestamp(
        last_review.timestamp_seconds,
        today=today,
        day_cutoff=day_cutoff,
    )
    return _TargetSchedule(
        card_type=CARD_TYPE_REVIEW,
        queue=QUEUE_TYPE_REVIEW,
        interval=new_interval,
        due=due_day_from_last_review(last_review_day, new_interval),
        due_field=schedule.due_field,
        remaining_steps=0,
        subday_interval_clamped=subday_interval
        and not (allow_short_term_rescheduling and relearning_steps_blank),
    )


def _target_schedule_for_learning_queue(
    *,
    schedule: CardScheduleInfo,
    card_type: int,
    interval_seconds: float,
    last_review: LastReviewInfo,
    today: int,
    day_cutoff: int,
    interval: int | None = None,
    remaining_steps: int = 0,
) -> _TargetSchedule:
    target_due_seconds = float(last_review.timestamp_seconds) + float(interval_seconds)
    target_due_day = anki_day_for_timestamp(
        target_due_seconds,
        today=today,
        day_cutoff=day_cutoff,
    )
    if target_due_day < int(today):
        last_review_day = anki_day_for_timestamp(
            last_review.timestamp_seconds,
            today=today,
            day_cutoff=day_cutoff,
        )
        return _TargetSchedule(
            card_type=CARD_TYPE_REVIEW,
            queue=QUEUE_TYPE_REVIEW,
            interval=max(1, int(schedule.interval if interval is None else interval)),
            due=due_day_from_last_review(last_review_day, 1),
            due_field="due",
            remaining_steps=0,
        )
    if target_due_seconds < int(day_cutoff):
        queue = QUEUE_TYPE_LEARNING
        due = max(0, int(round(target_due_seconds)))
    else:
        queue = QUEUE_TYPE_DAY_LEARN_RELEARN
        due = target_due_day
    return _TargetSchedule(
        card_type=int(card_type),
        queue=queue,
        interval=int(schedule.interval if interval is None else interval),
        due=due,
        due_field="due",
        remaining_steps=int(remaining_steps),
    )


def _schedule_already_matches(
    schedule: CardScheduleInfo,
    target: _TargetSchedule,
) -> bool:
    return (
        int(schedule.card_type) == target.card_type
        and int(schedule.queue) == target.queue
        and int(schedule.interval) == target.interval
        and int(schedule.remaining_steps) == target.remaining_steps
        and _old_due_for_target(schedule, target) == target.due
    )


def _old_due_for_target(schedule: CardScheduleInfo, target: _TargetSchedule) -> int:
    if target.due_field == "odue":
        return schedule.active_due
    return int(schedule.due)


def _deck_config_for_card(
    schedule: CardScheduleInfo,
    deck_configs: Mapping[int, DeckSchedulingConfig],
) -> DeckSchedulingConfig | None:
    if schedule.source_deck_id is not None:
        return deck_configs.get(int(schedule.source_deck_id))
    return None
