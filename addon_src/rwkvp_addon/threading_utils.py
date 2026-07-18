from __future__ import annotations

from contextlib import suppress


def detach_exception_from_thread(exc: BaseException) -> BaseException:
    """Drop traceback links before moving an exception across thread boundaries."""

    with suppress(Exception):
        exc.__traceback__ = None
    with suppress(Exception):
        exc.__cause__ = None
    with suppress(Exception):
        exc.__context__ = None
    return exc
