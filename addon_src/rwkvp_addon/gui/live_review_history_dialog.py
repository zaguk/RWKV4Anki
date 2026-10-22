from __future__ import annotations

from ..anki_api import find_cards
from ..live_review_history import load_live_review_history
from ..live_review_history_controller import (
    LiveReviewHistoryController,
    LiveReviewHistorySearchUnavailable,
)
from ..live_review_rmse import load_live_rmse_bin_context
from ..live_review_stats import summarize_live_retention_records
from ..progress import CancelledError
from .web_dialog import WebDialogHost, widget_uses_dark_palette


def open_live_review_history(parent, store) -> None:
    """Prepare the saved snapshot before constructing the modeless report."""
    from .common import run_with_progress
    from .web_message import show_web_warning

    collection = getattr(parent, "col", None)
    is_dark = widget_uses_dark_palette(parent)
    cutoff = _collection_day_cutoff(collection)

    def search_card_ids(query: str) -> tuple[int, ...]:
        if collection is None or getattr(parent, "col", None) is not collection:
            raise LiveReviewHistorySearchUnavailable("collection unavailable")
        return tuple(int(card_id) for card_id in find_cards(collection, query))

    def operation(_col, progress):
        overview = load_live_review_history(
            store.live_review_history_path, session_limit=None, progress=progress,
        )
        controller = LiveReviewHistoryController(
            overview=overview,
            search_card_ids=search_card_ids,
            include_fsrs=True,
            is_dark=is_dark,
            next_day_at_seconds=cutoff,
        )
        progress.update(0, 1, "Preparing Live Review History display…")
        controller.prepare_display(progress)
        progress.update(1, 1, "Live Review History ready")
        return controller

    def success(controller):
        if getattr(parent, "col", None) is not collection:
            return
        dialog = LiveReviewHistoryDialog(parent, store, controller=controller)
        dialog.show()

    def failure(exc):
        if getattr(parent, "col", None) is collection:
            show_web_warning(
                f"Unable to open RWKV Live Review History: {exc}",
                title="RWKV", parent=parent,
            )

    run_with_progress(
        parent=parent,
        title="Live Review History",
        label="Loading saved Live Review History…",
        op=operation,
        uses_collection=False,
        on_success=success,
        on_failure=failure,
    )


class LiveReviewHistoryDialog(WebDialogHost):
    def __init__(self, parent, store, *, controller: LiveReviewHistoryController) -> None:
        self.col = getattr(parent, "col", None)
        self._collection_owner = parent
        self.store = store
        self._rmse_loading = False
        self._rmse_prepared_snapshot = None
        self._rmse_context = {}
        self._rmse_message = ""
        self.overview = controller.overview
        self._history_controller = controller
        super().__init__(
            parent,
            title="RWKV Live Review History",
            controller=self._history_controller,
            size=(1040, 700),
            requires_collection=True,
        )
        self._history_controller.attach_rerender(self.rerender)
        self._history_controller.attach_metrics_loader(self.prepare_rmse_bins)

    @property
    def history_controller(self) -> LiveReviewHistoryController:
        return self._history_controller

    def prepare_rmse_bins(self) -> None:
        """On demand, load bins once and calculate the selected metrics off-thread."""
        from .common import run_with_progress

        if self.cleaned_up or self._rmse_loading:
            return
        controller = self._history_controller
        request = controller.metrics_request()
        if request is None:
            return
        snapshot = controller.overview
        needs_context = self._rmse_prepared_snapshot is not snapshot
        context = self._rmse_context
        message = self._rmse_message
        self._rmse_loading = True
        collection = self.col

        def operation(col, progress):
            prepared_context, notice = context, message
            if needs_context:
                prepared_context, notice = {}, ""
                if snapshot.summary.records:
                    try:
                        if col is None or col is not collection:
                            raise RuntimeError("The collection is no longer available.")
                        prepared_context = load_live_rmse_bin_context(
                            col, self.store, snapshot.summary.records, progress,
                        )
                    except CancelledError:
                        raise
                    except Exception as exc:
                        # LogLoss/AUC require no Anki history and remain usable.
                        notice = (
                            f"RMSE(bins) is unavailable: {str(exc)[:500]}. Reopen History to retry."
                        )
            summaries = []
            for selected, label in (
                (request.summary, "Calculating All Reviews evaluation metrics…"),
                (request.session_summary, "Calculating session evaluation metrics…"),
            ):
                progress.check_cancelled()
                if selected is None:
                    summaries.append(None)
                    continue
                progress.update(0, 1, label)
                summaries.append(summarize_live_retention_records(
                    selected.records, skipped_count=selected.skipped_count,
                    rmse_bin_context=prepared_context, include_auc=True, progress=progress,
                ))
            progress.check_cancelled()
            progress.update(1, 1, "Live evaluation metrics ready")
            return prepared_context, notice, *summaries

        def still_current():
            return (
                not self.cleaned_up and controller.overview is snapshot
                and getattr(self._collection_owner, "col", None) is collection
            )

        def finish(result):
            self._rmse_loading = False
            if not still_current():
                return
            self._rmse_context, self._rmse_message, summary, session_summary = result
            self._rmse_prepared_snapshot = snapshot
            if not controller.finish_metrics(
                request, summary, session_summary=session_summary, message=self._rmse_message,
            ):
                # A newer session/filter selection may have arrived while busy.
                self.prepare_rmse_bins()

        def unsuccessful(message):
            self._rmse_loading = False
            if still_current() and not controller.cancel_metrics(request, message):
                self.prepare_rmse_bins()

        run_with_progress(
            parent=self,
            title="Live Review History",
            label="Preparing evaluation metrics…",
            op=operation,
            uses_collection=needs_context and bool(snapshot.summary.records),
            on_success=finish,
            on_failure=lambda exc: unsuccessful(
                f"Evaluation metrics are unavailable: {str(exc)[:500]}. "
                "Switch to Retention and back to Evaluation Metrics to retry.",
            ),
            on_cancel=lambda: unsuccessful(
                "Evaluation metrics cancelled. Switch to Retention and back "
                "to Evaluation Metrics to retry.",
            ),
        )


def _collection_day_cutoff(col) -> int | None:
    if col is None:
        return None
    try:
        value = int(col.sched.day_cutoff)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value > 0 else None
