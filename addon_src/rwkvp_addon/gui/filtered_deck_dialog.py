from __future__ import annotations

import time
from collections.abc import Mapping

from aqt import mw

from ..adaptive_retention import adaptive_retention_card_data_for_card_ids
from ..addon_config import (
    adaptive_desired_retention_enabled,
    addon_config_for_mw,
    calculate_forgetting_curves,
    minimum_review_widening_extra,
)
from ..anki_api import (
    active_card_search_for_deck,
    create_filtered_deck_from_card_ids_with_changes,
    deck_name,
    deck_retentions_for_subtree,
    ensure_no_deck_name_collision,
    find_cards,
    fsrs_difficulties_for_card_ids,
    validate_search,
)
from ..dataset_export import (
    ensure_checkpoint_ready_from_load,
    load_review_data_for_checkpoint,
    open_checkpoint_runtime_from_load,
)
from ..filtered_deck import (
    EmptyFilteredDeckPlanError,
    FilteredDeckSettings,
    FilteredDeckSortInfo,
    build_filtered_deck_plan,
    ensure_filtered_deck_plan_has_selected_cards,
    filtered_deck_candidate_search,
)
from ..filtered_deck_predictors import filtered_deck_predictor_for_mode
from ..filtered_deck_sort import FILTERED_DECK_ORDER_OPTIONS, FilteredDeckOrder
from ..review_rows import checkpoint_scope_cards_for_card_ids
from ..runtime import manager_for_mw, store_for_mw
from ..rwkv_modes import (
    RetrievabilityMode,
    filtered_deck_settings_key,
    generated_deck_name_prefixes_for_mode_collision,
    mode_spec,
)
from ..study_session_controller import (
    StudySessionFormController,
    StudySessionFormSpec,
    StudySessionRequest,
)
from ..vendor_bootstrap import require_rwkv_interval
from .checkpoint_failure import handle_checkpoint_failure
from .common import (
    ProgressStage,
    notify_collection_operation_finished,
    run_with_progress_stages,
    show_quiet_info,
)
from .web_dialog import WebDialogHost, widget_uses_dark_palette
from .web_message import show_web_warning

FILTERED_DECK_CARD_LIMIT_LABEL = "Maximum cards"
FILTERED_DECK_MINIMUM_CARD_LIMIT_LABEL = "Minimum cards"
FILTERED_DECK_MINIMUM_CARD_LIMIT_TOOLTIP = (
    "If fewer cards are below DR, RWKV widens the non-same-day threshold in "
    "small steps until this many cards can be selected, up to Maximum cards. "
    "Same-day cards are not widened."
)
FILTERED_DECK_CARD_LIMIT_TOOLTIP = "Caps how many cards are added to the filtered deck."
FILTERED_DECK_MAX_CARD_LIMIT = 100000
FILTERED_DECK_DEFAULT_CARD_LIMIT = 200
FILTERED_DECK_DEFAULT_MINIMUM_CARD_LIMIT = 0
FILTERED_DECK_DEFAULT_ORDER_INDEX = int(FilteredDeckOrder.RETRIEVABILITY_ASCENDING)
FILTERED_DECK_SAME_DAY_ONLY_TOOLTIP = (
    "Build a filtered deck using only cards that have already been reviewed today."
)
FILTERED_DECK_SEARCH_FILTER_PLACEHOLDER = "Optional Anki search filter, e.g. is:due or prop:ivl>20"
FILTERED_DECK_SEARCH_FILTER_TOOLTIP = (
    "Optional Anki search query that further narrows the source deck cards."
)
FILTERED_DECK_CHECK_FILTER_LABEL = "Check"
FILTERED_DECK_CHECK_FILTER_TOOLTIP = "Count source-deck cards matched by the Anki search filter."


class FilteredDeckDialog(WebDialogHost):
    def __init__(
        self,
        parent,
        deck_id: int,
        *,
        mode: RetrievabilityMode = RetrievabilityMode.IMMEDIATE,
    ) -> None:
        self.deck_id = int(deck_id)
        self.mode = mode
        self.mode_spec = mode_spec(mode)
        self.settings_key = filtered_deck_settings_key(mode, self.deck_id)
        self.source_name = deck_name(mw.col, self.deck_id)
        self.retentions = tuple(deck_retentions_for_subtree(mw.col, self.deck_id))
        self.form_spec = _filtered_deck_form_spec(self.mode_spec.filtered_deck_title)
        self._study_controller = StudySessionFormController(
            spec=self.form_spec,
            source_name=self.source_name,
            retentions=self.retentions,
            order_options=FILTERED_DECK_ORDER_OPTIONS,
            saved_settings=_saved_deck_settings(
                store_for_mw(mw).settings(),
                group_key="filtered_deck_settings",
                settings_key=self.settings_key,
            ),
            adaptive_available=adaptive_desired_retention_enabled(addon_config_for_mw(mw)),
            on_submit_requested=self._start_build,
            on_check_requested=self._matching_candidate_count,
            on_restore_defaults=self._delete_saved_settings,
            on_warning=self._show_warning,
            background_submission=True,
            is_dark=widget_uses_dark_palette(parent),
        )
        super().__init__(
            parent,
            title=self.mode_spec.filtered_deck_title,
            controller=self._study_controller,
            size=(880, 620),
            requires_collection=True,
        )
        self._study_controller.attach_rerender(self.rerender)

    @property
    def study_controller(self) -> StudySessionFormController:
        return self._study_controller

    def _start_build(self, request: StudySessionRequest) -> bool:
        if self.mode == RetrievabilityMode.FORGETTING_CURVE and not calculate_forgetting_curves(
            addon_config_for_mw(mw)
        ):
            self._show_warning("Calculate Forgetting Curves is disabled in RWKV Settings.")
            return False

        settings = self._filtered_deck_settings(request)
        search_filter = request.extra_search
        search = filtered_deck_candidate_search(
            active_card_search_for_deck(mw.col, self.deck_id),
            settings,
            extra_search=search_filter,
        )
        try:
            validate_search(mw.col, search)
        except Exception as exc:
            self._study_controller.clear_filter_count(rerender=False)
            self._show_warning(str(exc))
            return False

        self._run_build(request, settings)
        return True

    def _run_build(
        self,
        request: StudySessionRequest,
        settings: FilteredDeckSettings,
    ) -> None:
        search_filter = request.extra_search
        store = store_for_mw(mw)
        manager = manager_for_mw(mw)

        def collect_op(col, progress, _previous):
            source_name = deck_name(col, self.deck_id)
            deck_name_prefix = self.mode_spec.generated_deck_name_prefix(source_name)
            collision_prefixes = generated_deck_name_prefixes_for_mode_collision(
                self.mode,
                source_name,
            )
            ensure_no_deck_name_collision(
                col,
                name=deck_name_prefix,
                name_without_expected_prefix=deck_name_prefix,
                additional_name_without_expected_prefixes=_additional_collision_prefixes(
                    deck_name_prefix,
                    collision_prefixes,
                ),
            )
            candidate_search = filtered_deck_candidate_search(
                active_card_search_for_deck(col, self.deck_id),
                settings,
                extra_search=search_filter,
            )
            card_ids = find_cards(col, candidate_search)
            if not card_ids:
                raise ValueError(
                    _no_candidate_cards_message(
                        settings,
                        extra_search=search_filter,
                    )
                )
            progress.update(0, 1, "Reading card tie-break metadata")
            card_sort_info = _card_sort_info_for_ids(col, card_ids)
            fsrs_difficulties = (
                fsrs_difficulties_for_card_ids(col, card_ids)
                if _adaptive_retention_enabled(settings)
                else {}
            )
            review_load = load_review_data_for_checkpoint(
                col,
                store,
                manager,
                progress,
                allow_incremental=True,
            )
            return (
                source_name,
                deck_name_prefix,
                collision_prefixes,
                card_ids,
                review_load,
                card_sort_info,
                fsrs_difficulties,
            )

        def plan_op(_col, progress, previous):
            (
                source_name,
                deck_name_prefix,
                collision_prefixes,
                card_ids,
                review_load,
                card_sort_info,
                fsrs_difficulties,
            ) = previous
            runtime = None
            if self.mode == RetrievabilityMode.IMMEDIATE:
                readiness, runtime = open_checkpoint_runtime_from_load(
                    manager,
                    review_load,
                    progress,
                    scope_cards=checkpoint_scope_cards_for_card_ids(
                        card_ids,
                        review_load.review_data,
                    ),
                )
                prediction_source = runtime
            else:
                readiness = ensure_checkpoint_ready_from_load(
                    manager,
                    review_load,
                    progress,
                )
                prediction_source = manager
            try:
                review_data = readiness.review_data
                predictor = filtered_deck_predictor_for_mode(
                    self.mode,
                    prediction_source,
                    progress,
                )
                adaptive_retention_by_card = (
                    adaptive_retention_card_data_for_card_ids(
                        card_ids,
                        latest_curves_by_card=manager.latest_curves_for_cards(card_ids),
                        fsrs_difficulties_by_card=fsrs_difficulties,
                        interval_for_curve=require_rwkv_interval(),
                    )
                    if _adaptive_retention_enabled(settings)
                    else {}
                )

                plan = build_filtered_deck_plan(
                    source_deck_id=self.deck_id,
                    source_deck_name=source_name,
                    card_ids=card_ids,
                    review_data=review_data,
                    retentions=request.retentions,
                    target_timestamp_seconds=time.time() + 600,
                    predictor=predictor,
                    settings=settings,
                    card_sort_info=card_sort_info,
                    adaptive_retention_by_card=adaptive_retention_by_card,
                    deck_name_prefix=deck_name_prefix,
                )
                return plan, deck_name_prefix, collision_prefixes
            finally:
                if runtime is not None:
                    runtime.close()

        def create_op(col, _progress, previous):
            plan, deck_name_prefix, collision_prefixes = previous
            ensure_filtered_deck_plan_has_selected_cards(plan)
            ensure_no_deck_name_collision(
                col,
                name=plan.deck_name,
                name_without_expected_prefix=deck_name_prefix,
                additional_name_without_expected_prefixes=_additional_collision_prefixes(
                    deck_name_prefix,
                    collision_prefixes,
                ),
            )
            selected_ids = [candidate.card_id for candidate in plan.selected]
            op_result = create_filtered_deck_from_card_ids_with_changes(
                col,
                name=plan.deck_name,
                card_ids=selected_ids,
                order_index=1,
            )
            all_settings = store.settings()
            filtered_settings = all_settings.setdefault("filtered_deck_settings", {})
            if not isinstance(filtered_settings, dict):
                filtered_settings = {}
                all_settings["filtered_deck_settings"] = filtered_settings
            filtered_settings[self.settings_key] = request.saved_settings(self.form_spec)
            store.write_settings(all_settings)
            return plan, op_result

        def success(result) -> None:
            if self.cleaned_up:
                return
            self._study_controller.finish_submission(rerender=False)
            plan, op_result = result
            notify_collection_operation_finished(op_result)
            show_quiet_info(
                f"Created filtered deck:\n{plan.deck_name}\n\n"
                f"Selected cards: {len(plan.selected)}\n"
                f"Expected R: {plan.expected_retrievability:.4f}",
                title=self.mode_spec.warning_title,
                parent=self,
            )
            self.accept()

        def failure(exc: Exception) -> None:
            if self.cleaned_up:
                return
            self._study_controller.finish_submission()
            if isinstance(exc, EmptyFilteredDeckPlanError):
                self._show_warning(str(exc))
                return
            handle_checkpoint_failure(
                exc,
                lambda: self._study_controller.retry_submission(request),
                parent=self,
            )

        def cancelled() -> None:
            if not self.cleaned_up:
                self._study_controller.finish_submission()

        run_with_progress_stages(
            parent=self,
            title=self.mode_spec.filtered_deck_title,
            label="Building filtered deck",
            stages=[
                ProgressStage(collect_op, uses_collection=True),
                ProgressStage(plan_op, uses_collection=False),
                ProgressStage(create_op, uses_collection=True),
            ],
            on_success=success,
            on_failure=failure,
            on_cancel=cancelled,
        )

    def _filtered_deck_settings(
        self,
        request: StudySessionRequest,
    ) -> FilteredDeckSettings:
        return FilteredDeckSettings(
            limit=request.maximum,
            order_index=request.order_index,
            same_day_only=request.same_day_only,
            minimum=request.minimum,
            minimum_retention_extra_quantum=minimum_review_widening_extra(addon_config_for_mw(mw)),
            adaptive_retention=request.adaptive_retention_settings,
        )

    def _matching_candidate_count(self, search_filter: str) -> int:
        settings = FilteredDeckSettings(
            limit=FILTERED_DECK_DEFAULT_CARD_LIMIT,
            order_index=FILTERED_DECK_DEFAULT_ORDER_INDEX,
        )
        search = filtered_deck_candidate_search(
            active_card_search_for_deck(mw.col, self.deck_id),
            settings,
            extra_search=search_filter,
        )
        return len(find_cards(mw.col, search))

    def _delete_saved_settings(self) -> None:
        store = store_for_mw(mw)
        settings = store.settings()
        filtered_settings = settings.get("filtered_deck_settings", {})
        if isinstance(filtered_settings, dict):
            filtered_settings.pop(self.settings_key, None)
        store.write_settings(settings)

    def _show_warning(self, message: str) -> None:
        show_web_warning(
            str(message),
            title=self.mode_spec.warning_title,
            parent=self,
        )


def _filtered_deck_form_spec(title: str) -> StudySessionFormSpec:
    return StudySessionFormSpec(
        title=title,
        intro=("Build a temporary deck selected and ordered by RWKV retrievability."),
        size_section_title="Deck Size and Ordering",
        size_section_intro=(
            "Choose how many matching cards may be added and how they are ordered."
        ),
        minimum_label=FILTERED_DECK_MINIMUM_CARD_LIMIT_LABEL,
        minimum_description=FILTERED_DECK_MINIMUM_CARD_LIMIT_TOOLTIP,
        maximum_label=FILTERED_DECK_CARD_LIMIT_LABEL,
        maximum_description=FILTERED_DECK_CARD_LIMIT_TOOLTIP,
        minimum_default=FILTERED_DECK_DEFAULT_MINIMUM_CARD_LIMIT,
        maximum_default=FILTERED_DECK_DEFAULT_CARD_LIMIT,
        maximum_value=FILTERED_DECK_MAX_CARD_LIMIT,
        default_order_index=FILTERED_DECK_DEFAULT_ORDER_INDEX,
        minimum_storage_key="minimum",
        maximum_storage_key="limit",
        primary_action="build-filtered-deck",
        primary_label="Build Filtered Deck",
        same_day_action="build-same-day-deck",
        same_day_label="Build Same-Day-Only Deck",
        same_day_description=FILTERED_DECK_SAME_DAY_ONLY_TOOLTIP,
    )


def _saved_deck_settings(
    settings: Mapping[str, object] | object,
    *,
    group_key: str,
    settings_key: str,
) -> Mapping[str, object]:
    if not isinstance(settings, Mapping):
        return {}
    group = settings.get(group_key)
    if not isinstance(group, Mapping):
        return {}
    saved = group.get(settings_key)
    return saved if isinstance(saved, Mapping) else {}


def _card_sort_info_for_ids(col, card_ids: list[int]) -> dict[int, FilteredDeckSortInfo]:
    sort_info: dict[int, FilteredDeckSortInfo] = {}
    for card_id in card_ids:
        try:
            card = col.get_card(int(card_id))
        except Exception:
            continue
        sort_info[int(card_id)] = FilteredDeckSortInfo(
            card_id=int(card_id),
            modified_secs=int(getattr(card, "mod", 0) or 0),
        )
    return sort_info


def _no_candidate_cards_message(
    settings: FilteredDeckSettings,
    *,
    extra_search: str | None = None,
) -> str:
    suffix = " and the Anki search filter" if str(extra_search or "").strip() else ""
    if settings.same_day_only:
        return f"No candidate cards matched the source deck search{suffix} and rated:1."
    return f"No candidate cards matched the source deck search{suffix}."


def _additional_collision_prefixes(
    primary_prefix: str,
    prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(prefix for prefix in prefixes if prefix != primary_prefix)


def _adaptive_retention_enabled(settings: FilteredDeckSettings) -> bool:
    adaptive = settings.adaptive_retention
    return bool(adaptive is not None and adaptive.enabled)
