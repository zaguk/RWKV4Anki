from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from aqt import mw
from aqt.utils import tooltip

from ..progress import TimedProgressReporter
from .background_stages import ProgressStage, run_background_stages
from .progress_action import ButtonLike, ProgressActionLifecycle
from .web_message import show_web_info, show_web_warning
from .web_progress import WebProgressSession, start_web_progress

T = TypeVar("T")


def run_with_progress(
    *,
    parent,
    title: str,
    label: str,
    op: Callable[[object, TimedProgressReporter], T],
    on_success: Callable[[T], None],
    on_failure: Callable[[Exception], None] | None = None,
    on_cancel: Callable[[], None] | None = None,
    uses_collection: bool = True,
) -> None:
    def stage_op(col, progress, _previous):
        return op(col, progress)

    run_with_progress_stages(
        parent=parent,
        title=title,
        label=label,
        stages=[ProgressStage(stage_op, uses_collection=uses_collection)],
        on_success=on_success,
        on_failure=on_failure,
        on_cancel=on_cancel,
    )


def run_action_with_progress(
    *,
    button: ButtonLike,
    parent,
    title: str,
    label: str,
    op: Callable[[object, TimedProgressReporter], T],
    on_success: Callable[[T], None],
    on_failure: Callable[[Exception], None] | None = None,
    on_cancel: Callable[[], None] | None = None,
    uses_collection: bool = True,
    before_start: Callable[[], None] | None = None,
    on_finish: Callable[[], None] | None = None,
) -> bool:
    def stage_op(col, progress, _previous):
        return op(col, progress)

    return run_action_with_progress_stages(
        button=button,
        parent=parent,
        title=title,
        label=label,
        stages=[ProgressStage(stage_op, uses_collection=uses_collection)],
        on_success=on_success,
        on_failure=on_failure,
        on_cancel=on_cancel,
        before_start=before_start,
        on_finish=on_finish,
    )


def run_action_with_progress_stages(
    *,
    button: ButtonLike,
    parent,
    title: str,
    label: str,
    stages: list[ProgressStage],
    on_success: Callable[[object], None],
    on_failure: Callable[[Exception], None] | None = None,
    on_cancel: Callable[[], None] | None = None,
    before_start: Callable[[], None] | None = None,
    on_finish: Callable[[], None] | None = None,
) -> bool:
    lifecycle = ProgressActionLifecycle(
        button,
        before_start=before_start,
        on_finish=on_finish,
    )
    if not lifecycle.begin():
        return False

    def failure(exception: Exception) -> None:
        if on_failure:
            on_failure(exception)
            return
        show_web_warning(str(exception), title=title, parent=parent)

    run_with_progress_stages(
        parent=parent,
        title=title,
        label=label,
        stages=stages,
        on_success=lifecycle.wrap_success(on_success),
        on_failure=lifecycle.wrap_failure(failure),
        on_cancel=lifecycle.wrap_cancel(on_cancel),
    )
    return True


def run_with_progress_stages(
    *,
    parent,
    title: str,
    label: str,
    stages: list[ProgressStage],
    on_success: Callable[[object], None],
    on_failure: Callable[[Exception], None] | None = None,
    on_cancel: Callable[[], None] | None = None,
    cancelled_tooltip: str | None = None,
    cancelled_tooltip_parent=None,
) -> None:
    if not stages:
        on_success(None)
        return

    closed = False
    presentation: WebProgressSession | None = None

    def progress_callback(current: int, total: int, text: str, eta: float | None) -> None:
        if closed:
            return
        active_presentation = presentation
        if active_presentation is not None:
            active_presentation.post_update(current, total, text or label, eta)

    reporter = TimedProgressReporter(progress_callback)
    try:
        active_presentation = start_web_progress(
            parent=parent,
            title=title,
            label=label,
            schedule_on_main=mw.taskman.run_on_main,
            on_cancel=reporter.cancel,
        )
    except Exception as exc:
        if on_failure:
            on_failure(exc)
        else:
            show_web_warning(str(exc), title=title, parent=parent)
        return
    presentation = active_presentation

    def close_as_cancelled() -> None:
        nonlocal closed
        closed = True
        was_visible = active_presentation.finish()
        if on_cancel:
            on_cancel()
        if was_visible:
            tooltip(
                cancelled_tooltip or f"{title}: cancelled.",
                parent=(parent if cancelled_tooltip_parent is None else cancelled_tooltip_parent),
            )

    def close_as_failed(exception: Exception) -> None:
        nonlocal closed
        closed = True
        active_presentation.finish()
        if on_failure:
            on_failure(exception)
            return
        show_web_warning(str(exception), title=title, parent=parent)

    def close_as_success(result: object) -> None:
        nonlocal closed
        closed = True
        active_presentation.finish()
        on_success(result)

    run_background_stages(
        stages=stages,
        reporter=reporter,
        on_success=close_as_success,
        on_failure=close_as_failed,
        on_cancel=close_as_cancelled,
    )


def notify_collection_operation_finished(result: object, initiator: object | None = None) -> None:
    from aqt.operations import on_op_finished

    on_op_finished(mw, result, initiator)


def show_quiet_info(message: str, *, title: str, parent=None) -> None:
    show_web_info(message, title=title, parent=parent)


def show_fsrs_disabled(parent) -> None:
    show_web_warning(
        "RWKV features require FSRS to be enabled for this profile/deck.",
        title="RWKV",
        parent=parent,
    )
