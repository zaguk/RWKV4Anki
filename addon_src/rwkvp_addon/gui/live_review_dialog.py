from __future__ import annotations

from collections.abc import Mapping

from aqt import mw

from ..addon_config import adaptive_desired_retention_enabled, addon_config_for_mw
from ..anki_api import (
    deck_name,
    deck_retentions_for_subtree,
    find_cards,
    validate_search,
)
from ..filtered_deck_sort import FILTERED_DECK_ORDER_OPTIONS, FilteredDeckOrder
from ..live_review_candidates import live_review_search_for_deck
from ..runtime import store_for_mw
from ..study_session_controller import (
    StudySessionFormController,
    StudySessionFormSpec,
    StudySessionRequest,
)
from .web_dialog import WebDialogHost, widget_uses_dark_palette
from .web_message import show_web_warning

DEFAULT_LIVE_REVIEW_LIMIT = 9999
DEFAULT_LIVE_REVIEW_MINIMUM_LIMIT = 0
DEFAULT_LIVE_REVIEW_ORDER_INDEX = int(FilteredDeckOrder.RETRIEVABILITY_ASCENDING)
LIVE_REVIEW_MINIMUM_LIMIT_LABEL = "Minimum reviews"
LIVE_REVIEW_LIMIT_LABEL = "Maximum reviews"
LIVE_REVIEW_MINIMUM_LIMIT_TOOLTIP = (
    "If the live queue runs short before this many reviews are done, RWKV "
    "widens the non-same-day threshold in small steps to find more candidates. "
    "Same-day cards are not widened."
)
LIVE_REVIEW_LIMIT_TOOLTIP = (
    "Ends the RWKV Live Session after this many completed reviews. Undoing a "
    "review lowers the count, so re-answering it only counts once."
)
LIVE_REVIEW_SAME_DAY_ONLY_TOOLTIP = (
    "Start a same-day-only Live Session using only cards already reviewed today. "
    "The main button starts the full session with both same-day and other eligible cards."
)
LIVE_REVIEW_SEARCH_FILTER_PLACEHOLDER = "Optional Anki search filter, e.g. is:due or prop:ivl>20"
LIVE_REVIEW_SEARCH_FILTER_TOOLTIP = (
    "Optional Anki search query that further narrows the source deck cards."
)
LIVE_REVIEW_CHECK_FILTER_LABEL = "Check"
LIVE_REVIEW_CHECK_FILTER_TOOLTIP = "Count source-deck cards matched by the Anki search filter."


class LiveReviewDialog(WebDialogHost):
    def __init__(self, parent, deck_id: int) -> None:
        self.deck_id = int(deck_id)
        self.settings_key = f"active_review:{self.deck_id}"
        self.source_name = deck_name(mw.col, self.deck_id)
        self.retentions = tuple(deck_retentions_for_subtree(mw.col, self.deck_id))
        self.form_spec = _live_review_form_spec()
        self._study_controller = StudySessionFormController(
            spec=self.form_spec,
            source_name=self.source_name,
            retentions=self.retentions,
            order_options=FILTERED_DECK_ORDER_OPTIONS,
            saved_settings=_saved_deck_settings(
                store_for_mw(mw).settings(),
                group_key="active_review_settings",
                settings_key=self.settings_key,
            ),
            adaptive_available=adaptive_desired_retention_enabled(addon_config_for_mw(mw)),
            on_submit_requested=self._start_live_review,
            on_check_requested=self._matching_candidate_count,
            on_restore_defaults=self._delete_saved_settings,
            on_warning=self._show_warning,
            background_submission=False,
            is_dark=widget_uses_dark_palette(parent),
        )
        super().__init__(
            parent,
            title="RWKV Live Session",
            controller=self._study_controller,
            size=(880, 620),
            requires_collection=True,
        )
        self._study_controller.attach_rerender(self.rerender)

    @property
    def study_controller(self) -> StudySessionFormController:
        return self._study_controller

    def _start_live_review(self, request: StudySessionRequest) -> bool:
        search = live_review_search_for_deck(
            mw.col,
            self.deck_id,
            same_day_only=request.same_day_only,
            extra_search=request.extra_search,
        )
        try:
            validate_search(mw.col, search)
        except Exception as exc:
            self._study_controller.clear_filter_count(rerender=False)
            self._show_warning(str(exc))
            return False

        store = store_for_mw(mw)
        all_settings = store.settings()
        active_settings = all_settings.setdefault("active_review_settings", {})
        if not isinstance(active_settings, dict):
            active_settings = {}
            all_settings["active_review_settings"] = active_settings
        active_settings[self.settings_key] = request.saved_settings(self.form_spec)
        store.write_settings(all_settings)

        from .live_review_bridge import show_active_review_prototype

        self.accept()
        show_active_review_prototype(
            self.deck_id,
            retentions=request.retentions,
            review_limit=request.maximum,
            minimum_review_limit=request.minimum,
            order_index=request.order_index,
            same_day_only=request.same_day_only,
            extra_search=request.extra_search,
            adaptive_retention_settings=request.adaptive_retention_settings,
        )
        return True

    def _matching_candidate_count(self, search_filter: str) -> int:
        search = live_review_search_for_deck(
            mw.col,
            self.deck_id,
            same_day_only=False,
            extra_search=search_filter,
        )
        return len(find_cards(mw.col, search))

    def _delete_saved_settings(self) -> None:
        store = store_for_mw(mw)
        settings = store.settings()
        active_settings = settings.get("active_review_settings", {})
        if isinstance(active_settings, dict):
            active_settings.pop(self.settings_key, None)
        store.write_settings(settings)

    def _show_warning(self, message: str) -> None:
        show_web_warning(str(message), title="RWKV Live Session", parent=self)


def _live_review_form_spec() -> StudySessionFormSpec:
    return StudySessionFormSpec(
        title="RWKV Live Session",
        intro=("Study with a queue that refreshes RWKV retrievability after every review."),
        size_section_title="Session Size and Ordering",
        size_section_intro=(
            "Choose when minimum-review widening applies, when the session ends, "
            "and how selected cards are ordered."
        ),
        minimum_label=LIVE_REVIEW_MINIMUM_LIMIT_LABEL,
        minimum_description=LIVE_REVIEW_MINIMUM_LIMIT_TOOLTIP,
        maximum_label=LIVE_REVIEW_LIMIT_LABEL,
        maximum_description=LIVE_REVIEW_LIMIT_TOOLTIP,
        minimum_default=DEFAULT_LIVE_REVIEW_MINIMUM_LIMIT,
        maximum_default=DEFAULT_LIVE_REVIEW_LIMIT,
        maximum_value=DEFAULT_LIVE_REVIEW_LIMIT,
        default_order_index=DEFAULT_LIVE_REVIEW_ORDER_INDEX,
        minimum_storage_key="minimum_review_limit",
        maximum_storage_key="review_limit",
        primary_action="start-live-session",
        primary_label="Start Live Session",
        same_day_action="start-same-day-live-session",
        same_day_label="Start Same-Day-Only Live Session",
        same_day_description=LIVE_REVIEW_SAME_DAY_ONLY_TOOLTIP,
        split_same_day_action=True,
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
