from __future__ import annotations

import datetime as dt
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .curve_reschedule import (
    CardScheduleInfo,
    CurveReschedulePlan,
    DeckSchedulingConfig,
)
from .fsrs_targets import (
    FsrsTargetScope,
    fsrs_time_series_target_review_ids_by_scope,
    fsrs_training_item_review_ids_by_scope,
)
from .metrics import EvaluationScope, MetricResult
from .progress import ProgressReporter
from .review_rows import ReviewData


@dataclass(frozen=True)
class PresetInfo:
    config_id: int
    name: str
    desired_retention: float
    param_search: str
    ignore_revlogs_before_ms: int
    fsrs_params_6: list[float]
    learning_steps_blank: bool
    relearning_steps_blank: bool
    relearning_steps_in_day: int


@dataclass(frozen=True)
class DeckRetention:
    deck_id: int
    name: str
    desired_retention: float
    preset_id: int | None
    same_day_desired_retention: float | None = None


@dataclass(frozen=True)
class FsrsEvaluationOptions:
    ignore_revlogs_before_ms: int = 0
    relearning_steps_in_day: int = 0


class FsrsEvaluationMode(Enum):
    TIME_SERIES = "time_series"
    LEGACY_TRAIN_SET = "legacy_train_set"


@dataclass(frozen=True)
class FsrsScopePlan:
    scope: EvaluationScope
    options: FsrsEvaluationOptions
    target_scope: FsrsTargetScope


ACTIVE_CARD_SEARCH = "-is:new -is:suspended"
CARD_SCHEDULE_INFO_QUERY_CHUNK_SIZE = 500


def profile_name(mw) -> str:
    profile_manager = getattr(mw, "pm", None)
    name = getattr(profile_manager, "name", None)
    return str(name or "default")


def current_deck_id(col) -> int:
    try:
        return int(col.decks.get_current_id())
    except Exception:
        try:
            return int(col.decks.current()["id"])
        except Exception:
            return 1


def is_filtered_deck(col, deck_id: int) -> bool:
    try:
        return bool(col.decks.is_filtered(int(deck_id)))
    except Exception:
        return False


def config_lookup_deck_id(col, preferred_deck_id: int | None = None) -> int:
    candidates: list[int] = []
    if preferred_deck_id is not None:
        candidates.append(int(preferred_deck_id))
    candidates.append(current_deck_id(col))
    candidates.extend(did for _name, did in all_normal_deck_name_ids(col))

    seen: set[int] = set()
    for deck_id in candidates:
        if deck_id in seen:
            continue
        seen.add(deck_id)
        if not is_filtered_deck(col, deck_id):
            return deck_id
    return 1


def is_fsrs_enabled(col, deck_id: int | None = None) -> bool:
    try:
        if deck_id is not None and is_filtered_deck(col, deck_id):
            return False
        configs = col.decks.get_deck_configs_for_update(
            config_lookup_deck_id(col, deck_id)
        )
        return bool(configs.fsrs)
    except Exception:
        return False


def all_normal_deck_name_ids(col) -> list[tuple[str, int]]:
    return [
        (entry.name, int(entry.id))
        for entry in col.decks.all_names_and_ids(include_filtered=False)
    ]


def all_deck_name_ids(col) -> list[tuple[str, int]]:
    return [
        (entry.name, int(entry.id))
        for entry in col.decks.all_names_and_ids(include_filtered=True)
    ]


def deck_name(col, deck_id: int) -> str:
    deck = col.decks.get(deck_id)
    if deck and "name" in deck:
        return str(deck["name"])
    for name, did in all_normal_deck_name_ids(col):
        if did == deck_id:
            return name
    return str(deck_id)


def deck_and_child_name_ids(col, deck_id: int) -> list[tuple[str, int]]:
    return [
        (name, int(did))
        for name, did in col.decks.deck_and_child_name_ids(deck_id)
        if not is_filtered_deck(col, int(did))
    ]


def find_cards(col, search: str) -> list[int]:
    return [int(card_id) for card_id in col.find_cards(search, order=False)]


def validate_search(col, search: str) -> None:
    build_search_string = getattr(col, "build_search_string", None)
    if build_search_string is not None:
        build_search_string(search)
        return
    find_cards(col, search)


def active_card_search_for_deck(col, deck_id: int) -> str:
    return f'deck:"{escape_search(deck_name(col, deck_id))}" {ACTIVE_CARD_SEARCH}'


def build_evaluation_scope_descriptors(
    col,
    *,
    include_collection: bool = True,
    include_presets: bool = True,
    include_decks: bool = True,
) -> list[EvaluationScope]:
    scopes: list[EvaluationScope] = []
    if include_collection:
        scopes.append(
            EvaluationScope(
                key="collection",
                label="Collection",
                kind="collection",
                search="-is:suspended",
            )
        )

    if include_presets:
        current_configs = col.decks.get_deck_configs_for_update(config_lookup_deck_id(col))
        for preset in presets_from_configs(current_configs):
            search = preset.param_search or f'preset:"{escape_search(preset.name)}" -is:suspended'
            scopes.append(
                EvaluationScope(
                    key=f"preset:{preset.config_id}",
                    label=f"Preset: {preset.name}",
                    kind="preset",
                    search=search,
                    preset_config_id=preset.config_id,
                )
            )

    if include_decks:
        for name, did in all_normal_deck_name_ids(col):
            scopes.append(
                EvaluationScope(
                    key=f"deck:{did}",
                    label=f"Deck: {name}",
                    kind="deck",
                    search=f'deck:"{escape_search(name)}" -is:suspended',
                    deck_id=did,
                )
            )

    return scopes


def resolve_evaluation_scopes(
    col,
    scopes: list[EvaluationScope],
    progress: ProgressReporter | None = None,
) -> list[EvaluationScope]:
    total = max(1, len(scopes))
    resolved: list[EvaluationScope] = []
    for index, scope in enumerate(scopes, start=1):
        _update_progress(progress, index - 1, total, f"Resolving scope: {scope.label}")
        resolved.append(scope.with_card_ids(find_cards(col, scope.search)))
        _update_progress(progress, index, total, f"Resolved scope: {scope.label}")
    return resolved


def build_evaluation_scopes(
    col,
    *,
    include_collection: bool = True,
    include_presets: bool = True,
    include_decks: bool = True,
    progress: ProgressReporter | None = None,
) -> list[EvaluationScope]:
    return resolve_evaluation_scopes(
        col,
        build_evaluation_scope_descriptors(
            col,
            include_collection=include_collection,
            include_presets=include_presets,
            include_decks=include_decks,
        ),
        progress,
    )


def presets_from_configs(configs) -> list[PresetInfo]:
    presets: list[PresetInfo] = []
    for wrapped in getattr(configs, "all_config", []):
        config = wrapped.config
        presets.append(preset_from_config(config))
    return presets


def preset_from_config(config) -> PresetInfo:
    inner = config.config
    learn_steps = _steps_minutes(getattr(inner, "learn_steps", []))
    relearn_steps = _steps_minutes(getattr(inner, "relearn_steps", []))
    return PresetInfo(
        config_id=int(config.id),
        name=str(config.name),
        desired_retention=float(getattr(inner, "desired_retention", 0.9) or 0.9),
        param_search=str(getattr(inner, "param_search", "") or ""),
        ignore_revlogs_before_ms=_ignore_revlogs_before_ms(
            str(getattr(inner, "ignore_revlogs_before_date", "") or "")
        ),
        fsrs_params_6=[float(value) for value in getattr(inner, "fsrs_params_6", [])],
        learning_steps_blank=not learn_steps,
        relearning_steps_blank=not relearn_steps,
        relearning_steps_in_day=_relearning_steps_in_day(relearn_steps),
    )


def preset_by_id(col, config_id: int | None, deck_id: int | None = None) -> PresetInfo | None:
    configs = col.decks.get_deck_configs_for_update(config_lookup_deck_id(col, deck_id))
    for preset in presets_from_configs(configs):
        if preset.config_id == config_id:
            return preset
    return None


def deck_retention(col, deck_id: int) -> DeckRetention:
    if is_filtered_deck(col, deck_id):
        raise ValueError("RWKV filtered decks can only be generated from normal decks.")
    configs = col.decks.get_deck_configs_for_update(deck_id)
    config_id = int(configs.current_deck.config_id)
    preset = next(
        (preset for preset in presets_from_configs(configs) if preset.config_id == config_id),
        None,
    ) or preset_by_id(col, config_id, deck_id)
    desired = None
    limits = configs.current_deck.limits
    try:
        if limits.HasField("desired_retention"):
            desired = float(limits.desired_retention)
    except Exception:
        desired = float(getattr(limits, "desired_retention", 0) or 0) or None
    if desired is None and preset is not None:
        desired = preset.desired_retention
    return DeckRetention(
        deck_id=int(deck_id),
        name=deck_name(col, deck_id),
        desired_retention=float(desired or 0.9),
        preset_id=preset.config_id if preset else None,
    )


def deck_retentions_for_subtree(col, deck_id: int) -> list[DeckRetention]:
    if is_filtered_deck(col, deck_id):
        raise ValueError("RWKV filtered decks can only be generated from normal decks.")
    return [deck_retention(col, did) for _name, did in deck_and_child_name_ids(col, deck_id)]


def deck_scheduling_config(col, deck_id: int) -> DeckSchedulingConfig:
    retention = deck_retention(col, int(deck_id))
    preset = preset_by_id(col, retention.preset_id, int(deck_id))
    return DeckSchedulingConfig(
        deck_id=retention.deck_id,
        name=retention.name,
        desired_retention=retention.desired_retention,
        preset_id=retention.preset_id,
        max_interval=deck_maximum_review_interval(col, int(deck_id)),
        learning_steps_blank=bool(preset.learning_steps_blank if preset else False),
        relearning_steps_blank=bool(preset.relearning_steps_blank if preset else False),
    )


def deck_scheduling_configs_for_decks(
    col,
    deck_ids: Iterable[int | None],
) -> dict[int, DeckSchedulingConfig]:
    configs: dict[int, DeckSchedulingConfig] = {}
    for deck_id in deck_ids:
        if deck_id is None:
            continue
        deck_id = int(deck_id)
        if deck_id in configs:
            continue
        if is_filtered_deck(col, deck_id):
            continue
        configs[deck_id] = deck_scheduling_config(col, deck_id)
    return configs


def deck_scheduling_configs_for_subtree(
    col,
    deck_id: int,
) -> dict[int, DeckSchedulingConfig]:
    if is_filtered_deck(col, deck_id):
        raise ValueError("RWKV Forgetting Curve rescheduling requires a normal deck.")
    return {
        int(did): deck_scheduling_config(col, int(did))
        for _name, did in deck_and_child_name_ids(col, deck_id)
    }


def deck_maximum_review_interval(col, deck_id: int) -> int | None:
    try:
        config = col.decks.config_dict_for_deck_id(int(deck_id))
        maximum = config.get("rev", {}).get("maxIvl")
        return int(maximum) if maximum is not None else None
    except Exception:
        return None


def card_schedule_info_for_ids(
    col,
    card_ids: Iterable[int],
) -> dict[int, CardScheduleInfo]:
    ids = sorted({int(card_id) for card_id in card_ids})
    if not ids:
        return {}
    infos: dict[int, CardScheduleInfo] = {}
    for id_chunk in _chunks(ids, CARD_SCHEDULE_INFO_QUERY_CHUNK_SIZE):
        rows = col.db.all(
            f"""
            SELECT id, nid, did, odid, type, queue, due, ivl, reps, lapses, odue, "left", ord, mod
            FROM cards
            WHERE id IN {_sql_id_list(id_chunk)}
            """
        )
        for row in rows:
            card_id = int(row[0])
            deck_id = int(row[2]) if row[2] is not None else None
            original_deck_id = int(row[3] or 0)
            source_deck_id = original_deck_id or deck_id
            infos[card_id] = CardScheduleInfo(
                card_id=card_id,
                note_id=int(row[1]) if row[1] is not None else None,
                source_deck_id=source_deck_id,
                original_deck_id=original_deck_id,
                card_type=int(row[4]),
                queue=int(row[5]),
                due=int(row[6]),
                interval=int(row[7]),
                reps=int(row[8]),
                lapses=int(row[9]),
                original_due=int(row[10] or 0),
                remaining_steps=int(row[11] or 0),
                template_index=int(row[12] or 0),
                modified_secs=int(row[13] or 0),
            )
    return infos


def fsrs_difficulties_for_card_ids(
    col,
    card_ids: Iterable[int],
) -> dict[int, float]:
    ids = sorted({int(card_id) for card_id in card_ids})
    if not ids:
        return {}
    difficulties: dict[int, float] = {}
    for id_chunk in _chunks(ids, CARD_SCHEDULE_INFO_QUERY_CHUNK_SIZE):
        try:
            rows = col.db.all(
                f"""
                SELECT id, extract_fsrs_variable(data, 'd')
                FROM cards
                WHERE id IN {_sql_id_list(id_chunk)}
                """
            )
        except Exception:
            continue
        for row in rows:
            try:
                card_id = int(row[0])
                difficulty = float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(difficulty):
                difficulties[card_id] = difficulty
    return difficulties


def fsrs_retrievabilities_for_card_ids(
    col,
    card_ids: Iterable[int],
    *,
    now: int | float | None = None,
) -> dict[int, float]:
    ids = sorted({int(card_id) for card_id in card_ids})
    if not ids:
        return {}
    try:
        today = int(col.sched.today)
        next_day_at = int(col.sched.day_cutoff)
        now_seconds = int(time.time() if now is None else now)
    except Exception:
        return {}

    retrievabilities: dict[int, float] = {}
    for id_chunk in _chunks(ids, CARD_SCHEDULE_INFO_QUERY_CHUNK_SIZE):
        try:
            rows = col.db.all(
                f"""
                SELECT id,
                       extract_fsrs_retrievability(
                           data,
                           case when odue != 0 then odue else due end,
                           ivl,
                           {today},
                           {next_day_at},
                           {now_seconds}
                       )
                FROM cards
                WHERE id IN {_sql_id_list(id_chunk)}
                """
            )
        except Exception:
            continue
        for row in rows:
            try:
                card_id = int(row[0])
                retrievability = float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(retrievability):
                retrievabilities[card_id] = retrievability
    return retrievabilities


def apply_curve_reschedule_plan(col, plan: CurveReschedulePlan):
    cards = []
    for update in plan.updates:
        card = col.get_card(update.card_id)
        if update.new_card_type is not None:
            card.type = update.new_card_type
        if update.new_queue is not None:
            card.queue = update.new_queue
        card.ivl = update.new_interval
        if update.new_remaining_steps is not None:
            card.left = update.new_remaining_steps
        if update.due_field == "odue":
            card.odue = update.new_due
        else:
            card.due = update.new_due
        cards.append(card)
    if not cards:
        raise ValueError("No cards were selected for rescheduling.")
    return col.update_cards(cards)


def evaluate_fsrs_scopes(
    col,
    scopes: list[EvaluationScope],
    progress: ProgressReporter | None = None,
    *,
    mode: FsrsEvaluationMode = FsrsEvaluationMode.TIME_SERIES,
) -> dict[str, MetricResult]:
    return evaluate_fsrs_scope_plans(
        col,
        fsrs_scope_plans(col, scopes),
        progress,
        mode=mode,
    )


def evaluate_fsrs_scope_plans(
    col,
    plans: list[FsrsScopePlan],
    progress: ProgressReporter | None = None,
    *,
    mode: FsrsEvaluationMode = FsrsEvaluationMode.TIME_SERIES,
) -> dict[str, MetricResult]:
    results: dict[str, MetricResult] = {}
    total_steps = max(1, len(plans))
    for index, plan in enumerate(plans, start=1):
        scope = plan.scope
        try:
            _update_progress(progress, index - 1, total_steps, f"Evaluating FSRS-6: {scope.label}")
            response = _evaluate_fsrs_scope(col, plan, mode)
            _update_progress(progress, index, total_steps, f"Evaluated FSRS-6: {scope.label}")
            results[scope.key] = MetricResult(
                rmse_bins=float(response.rmse_bins),
                log_loss=float(response.log_loss),
            )
        except Exception as exc:
            results[scope.key] = MetricResult(None, None, str(exc))
    return results


def fsrs_scope_plans(col, scopes: list[EvaluationScope]) -> list[FsrsScopePlan]:
    needs_presets = any(
        scope.preset_config_id is not None or scope.deck_id is not None for scope in scopes
    )
    presets: dict[int, PresetInfo] = {}
    if needs_presets:
        configs = col.decks.get_deck_configs_for_update(config_lookup_deck_id(col))
        presets = {preset.config_id: preset for preset in presets_from_configs(configs)}
    return [_fsrs_scope_plan(col, scope, presets) for scope in scopes]


def _fsrs_scope_plan(
    col,
    scope: EvaluationScope,
    presets: dict[int, PresetInfo],
) -> FsrsScopePlan:
    options = _fsrs_evaluation_options_for_scope(col, scope, presets)
    return FsrsScopePlan(
        scope=scope,
        options=options,
        target_scope=FsrsTargetScope(
            key=scope.key,
            card_ids=scope.card_ids,
            ignore_revlogs_before_ms=options.ignore_revlogs_before_ms,
        ),
    )


def _evaluate_fsrs_scope(
    col,
    plan: FsrsScopePlan,
    mode: FsrsEvaluationMode,
):
    scope = plan.scope
    options = plan.options
    if mode == FsrsEvaluationMode.LEGACY_TRAIN_SET:
        params = _compute_fsrs_params(col, options, scope.search)
        return _evaluate_fsrs_legacy(
            col,
            params=params,
            search=scope.search,
            ignore_revlogs_before_ms=options.ignore_revlogs_before_ms,
        )
    return _evaluate_fsrs_time_series(
        col,
        search=scope.search,
        ignore_revlogs_before_ms=options.ignore_revlogs_before_ms,
        num_of_relearning_steps=options.relearning_steps_in_day,
    )


def fsrs_target_review_ids_by_scope(
    col,
    review_data: ReviewData,
    scopes: list[EvaluationScope],
    *,
    processed_review_count: int | None = None,
) -> dict[str, set[int]]:
    return _fsrs_review_ids_by_scope(
        col,
        review_data,
        scopes,
        mode=FsrsEvaluationMode.TIME_SERIES,
        processed_review_count=processed_review_count,
    )


def fsrs_training_review_ids_by_scope(
    col,
    review_data: ReviewData,
    scopes: list[EvaluationScope],
    *,
    processed_review_count: int | None = None,
) -> dict[str, set[int]]:
    return _fsrs_review_ids_by_scope(
        col,
        review_data,
        scopes,
        mode=FsrsEvaluationMode.LEGACY_TRAIN_SET,
        processed_review_count=processed_review_count,
    )


def fsrs_evaluation_review_ids_by_scope(
    col,
    review_data: ReviewData,
    scopes: list[EvaluationScope],
    *,
    mode: FsrsEvaluationMode,
    processed_review_count: int | None = None,
) -> dict[str, set[int]]:
    return fsrs_evaluation_review_ids_for_plans(
        review_data,
        fsrs_scope_plans(col, scopes),
        mode=mode,
        processed_review_count=processed_review_count,
    )


def fsrs_evaluation_review_ids_for_plans(
    review_data: ReviewData,
    plans: list[FsrsScopePlan],
    *,
    mode: FsrsEvaluationMode,
    processed_review_count: int | None = None,
) -> dict[str, set[int]]:
    target_scopes = [plan.target_scope for plan in plans]
    if mode == FsrsEvaluationMode.LEGACY_TRAIN_SET:
        return fsrs_training_item_review_ids_by_scope(
            review_data,
            target_scopes,
            processed_review_count=processed_review_count,
        )
    return fsrs_time_series_target_review_ids_by_scope(
        review_data,
        target_scopes,
        processed_review_count=processed_review_count,
    )


def _fsrs_review_ids_by_scope(
    col,
    review_data: ReviewData,
    scopes: list[EvaluationScope],
    *,
    mode: FsrsEvaluationMode,
    processed_review_count: int | None,
) -> dict[str, set[int]]:
    return fsrs_evaluation_review_ids_for_plans(
        review_data,
        fsrs_scope_plans(col, scopes),
        mode=mode,
        processed_review_count=processed_review_count,
    )


def create_filtered_deck_from_card_ids(
    col,
    *,
    name: str,
    card_ids: list[int],
    order_index: int = 1,
) -> int:
    result = create_filtered_deck_from_card_ids_with_changes(
        col,
        name=name,
        card_ids=card_ids,
        order_index=order_index,
    )
    return int(result.id)


def create_filtered_deck_from_card_ids_with_changes(
    col,
    *,
    name: str,
    card_ids: list[int],
    order_index: int = 1,
):
    if not card_ids:
        raise ValueError("No cards were selected for the filtered deck.")
    deck = col.sched.get_or_create_filtered_deck(deck_id=0)
    deck.name = name
    deck.allow_empty = False
    del deck.config.search_terms[:]
    term_type = type(deck.config).SearchTerm
    term = term_type(
        search=cid_search(card_ids),
        limit=len(card_ids),
        order=order_index,
    )
    deck.config.search_terms.extend([term])
    deck.config.reschedule = True
    return col.sched.add_or_update_filtered_deck(deck)


def ensure_no_deck_name_collision(
    col,
    *,
    name: str,
    name_without_expected_prefix: str,
    additional_name_without_expected_prefixes: Iterable[str] = (),
) -> None:
    prefixes = (
        name_without_expected_prefix,
        *(str(prefix) for prefix in additional_name_without_expected_prefixes),
    )
    for existing_name, _deck_id in all_deck_name_ids(col):
        if existing_name == name or any(
            existing_name.startswith(prefix) for prefix in prefixes
        ):
            raise ValueError(
                "A deck already exists for this RWKV filtered deck name. "
                f"Delete or rename it before creating a new one: '{existing_name}'."
            )


def cid_search(card_ids: list[int]) -> str:
    return "cid:" + ",".join(str(int(card_id)) for card_id in card_ids)


def _sql_id_list(ids: Iterable[int]) -> str:
    return "(" + ",".join(str(int(value)) for value in ids) + ")"


def _chunks(values: list[int], chunk_size: int) -> Iterable[list[int]]:
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def escape_search(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _preset_for_scope(
    col,
    scope: EvaluationScope,
    presets: dict[int, PresetInfo],
) -> PresetInfo | None:
    if scope.preset_config_id is not None:
        return presets.get(scope.preset_config_id)
    if scope.deck_id is not None:
        config_id = int(col.decks.get_deck_configs_for_update(scope.deck_id).current_deck.config_id)
        return presets.get(config_id)
    return None


def _fsrs_evaluation_options_for_scope(
    col,
    scope: EvaluationScope,
    presets: dict[int, PresetInfo],
) -> FsrsEvaluationOptions:
    preset = _preset_for_scope(col, scope, presets)
    if preset is None:
        return FsrsEvaluationOptions()
    return FsrsEvaluationOptions(
        ignore_revlogs_before_ms=preset.ignore_revlogs_before_ms,
        relearning_steps_in_day=preset.relearning_steps_in_day,
    )


def _compute_fsrs_params(
    col,
    options: FsrsEvaluationOptions,
    search: str,
) -> list[float]:
    backend = col._backend
    if hasattr(backend, "compute_fsrs_params"):
        response = backend.compute_fsrs_params(
            search=search,
            current_params=[],
            ignore_revlogs_before_ms=options.ignore_revlogs_before_ms,
            num_of_relearning_steps=options.relearning_steps_in_day,
            health_check=False,
        )
        computed = [float(value) for value in response.params]
        if not computed:
            raise ValueError("No evaluable FSRS review items were found for this scope.")
        return computed
    raise RuntimeError("This Anki build does not expose compute_fsrs_params().")


def _evaluate_fsrs_time_series(
    col,
    *,
    search: str,
    ignore_revlogs_before_ms: int,
    num_of_relearning_steps: int,
):
    backend = col._backend
    if hasattr(backend, "evaluate_params"):
        return backend.evaluate_params(
            search=search,
            ignore_revlogs_before_ms=ignore_revlogs_before_ms,
            num_of_relearning_steps=num_of_relearning_steps,
        )
    raise RuntimeError("This Anki build does not expose evaluate_params().")


def _evaluate_fsrs_legacy(
    col,
    *,
    params: list[float],
    search: str,
    ignore_revlogs_before_ms: int,
):
    backend = col._backend
    if hasattr(backend, "evaluate_params_legacy"):
        return backend.evaluate_params_legacy(
            params=params,
            search=search,
            ignore_revlogs_before_ms=ignore_revlogs_before_ms,
        )
    raise RuntimeError("This Anki build does not expose evaluate_params_legacy().")


def _update_progress(
    progress: ProgressReporter | None,
    current: int,
    total: int,
    label: str,
) -> None:
    if progress is not None:
        progress.update(current, total, label)


def _ignore_revlogs_before_ms(value: str) -> int:
    if not value:
        return 0
    try:
        date = dt.date.fromisoformat(value)
    except ValueError:
        return 0
    if date <= dt.date(1970, 1, 1):
        return 0
    try:
        local_midnight = dt.datetime.combine(date, dt.time.min).astimezone()
        return int(local_midnight.timestamp() * 1000)
    except (OSError, OverflowError, ValueError):
        return 0


def _steps_minutes(value) -> list[float]:
    try:
        return [float(step) for step in value or []]
    except TypeError:
        return []


def _relearning_steps_in_day(steps_minutes: list[float]) -> int:
    total = 0.0
    count = 0
    for step in steps_minutes:
        total += step
        if total >= 1440:
            break
        count += 1
    return count
