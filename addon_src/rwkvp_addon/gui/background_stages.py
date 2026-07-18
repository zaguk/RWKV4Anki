from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from importlib import import_module
from threading import Lock

from ..progress import CancelledError, TimedProgressReporter


@dataclass(frozen=True)
class ProgressStage:
    op: Callable[[object | None, TimedProgressReporter, object | None], object]
    uses_collection: bool = True


def run_background_stages(
    *,
    stages: list[ProgressStage],
    reporter: TimedProgressReporter,
    on_success: Callable[[object], None],
    on_failure: Callable[[Exception], None],
    on_cancel: Callable[[], None],
) -> None:
    """Run progress stages without creating or managing presentation UI."""

    # Settings GUI tests deliberately replace the lightweight ``aqt`` module
    # between cases. Resolve the main window when an operation begins instead
    # of retaining an import-time object; in Anki this is the same singleton.
    main_window = import_module("aqt").mw
    completion_lock = Lock()
    completed = False

    def begin_completion() -> bool:
        nonlocal completed
        with completion_lock:
            if completed:
                return False
            completed = True
            return True

    def finish_success(result: object) -> None:
        if not begin_completion():
            return
        try:
            on_success(result)
        except Exception as exc:
            _safe_failure_callback(on_failure, exc)

    def finish_failure(exception: Exception) -> None:
        if not begin_completion():
            return
        if isinstance(exception, CancelledError):
            try:
                on_cancel()
            except Exception as exc:
                _safe_failure_callback(on_failure, exc)
            return
        _safe_failure_callback(on_failure, exception)

    if not stages:
        finish_success(None)
        return

    def run_stage(index: int, previous: object | None = None) -> None:
        stage = stages[index]
        done_lock = Lock()
        done_called = False

        def task():
            reporter.check_cancelled()
            col = main_window.col if stage.uses_collection else None
            result = stage.op(col, reporter, previous)
            reporter.check_cancelled()
            return result

        def done(future: Future) -> None:
            nonlocal done_called
            with done_lock:
                if done_called or completed:
                    return
                done_called = True

            try:
                exception = future.exception()
            except Exception as exc:
                finish_failure(exc)
                return
            if exception:
                finish_failure(exception)
                return

            try:
                result = future.result()
            except Exception as exc:
                finish_failure(exc)
                return
            next_index = index + 1
            if next_index >= len(stages):
                finish_success(result)
                return
            run_stage(next_index, result)

        try:
            main_window.taskman.run_in_background(
                task,
                done,
                uses_collection=stage.uses_collection,
            )
        except Exception as exc:
            finish_failure(exc)

    run_stage(0)


def _safe_failure_callback(
    on_failure: Callable[[Exception], None],
    exception: Exception,
) -> None:
    # Terminal callbacks execute on Anki's GUI thread. A presentation error
    # must not escape into Anki's global exception/modal handling and recreate
    # the input-lock failures this runner is designed to avoid.
    try:
        on_failure(exception)
    except Exception:
        return


__all__ = ["ProgressStage", "run_background_stages"]
