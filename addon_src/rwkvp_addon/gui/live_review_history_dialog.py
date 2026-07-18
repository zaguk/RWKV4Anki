from __future__ import annotations

from ..anki_api import find_cards
from ..live_review_history import load_live_review_history
from ..live_review_history_controller import (
    LiveReviewHistoryController,
    LiveReviewHistorySearchUnavailable,
)
from .web_dialog import WebDialogHost, widget_uses_dark_palette


class LiveReviewHistoryDialog(WebDialogHost):
    def __init__(self, parent, store) -> None:
        self.col = getattr(parent, "col", None)
        self.store = store
        self.overview = load_live_review_history(
            store.live_review_history_path,
            session_limit=None,
        )
        self._history_controller = LiveReviewHistoryController(
            overview=self.overview,
            search_card_ids=self._matching_card_ids_for_search,
            include_fsrs=True,
            is_dark=widget_uses_dark_palette(parent),
            next_day_at_seconds=_collection_day_cutoff(self.col),
        )
        super().__init__(
            parent,
            title="RWKV Live Review History",
            controller=self._history_controller,
            size=(1040, 700),
            requires_collection=True,
        )
        self._history_controller.attach_rerender(self.rerender)

    @property
    def history_controller(self) -> LiveReviewHistoryController:
        return self._history_controller

    def _matching_card_ids_for_search(self, query: str) -> tuple[int, ...]:
        if self.col is None:
            raise LiveReviewHistorySearchUnavailable("collection unavailable")
        return tuple(int(card_id) for card_id in find_cards(self.col, query))


def _collection_day_cutoff(col) -> int | None:
    if col is None:
        return None
    try:
        value = int(col.sched.day_cutoff)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value > 0 else None
