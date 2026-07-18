from __future__ import annotations

from contextlib import suppress
from typing import Any


def runtime_is_rust(runtime) -> bool:
    module = type(runtime).__module__
    return (
        module == "rwkv_srs.backends.rust"
        or module.endswith(".backends.rust")
        or _object_has_native_runtime(runtime)
    )


def release_runtime_resources(runtime) -> None:
    """Drop native Rust handles while still on their owner thread."""

    if not runtime_is_rust(runtime):
        return
    # The native runtime would also drop this derived cache during close, but
    # invoking the public release explicitly makes the GPU-memory boundary
    # deterministic and directly testable for scoped operations and Live
    # Sessions. A device-loss error must never prevent the CPU runtime closing.
    release_gpu = getattr(runtime, "release_gpu", None)
    if release_gpu is not None:
        with suppress(Exception):
            release_gpu()
    close = getattr(runtime, "close", None)
    if close is not None:
        with suppress(Exception):
            close()
    _clear_native_runtime_handles(runtime, seen=set())
    with suppress(Exception):
        object.__setattr__(runtime, "_rnn", None)


def _clear_native_runtime_handles(
    obj: Any,
    *,
    seen: set[int],
    depth: int = 0,
) -> None:
    if obj is None or depth > 8:
        return
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    attrs = _safe_vars(obj)
    if not attrs:
        return

    for name, value in list(attrs.items()):
        if name == "_native_runtime" or _is_native_runtime_value(value):
            with suppress(Exception):
                object.__setattr__(obj, name, None)
            continue
        if _should_descend_into(value):
            _clear_native_runtime_handles(value, seen=seen, depth=depth + 1)
        elif isinstance(value, dict):
            _clear_native_runtime_handles_in_mapping(value, seen=seen, depth=depth + 1)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                if _should_descend_into(item):
                    _clear_native_runtime_handles(item, seen=seen, depth=depth + 1)


def _clear_native_runtime_handles_in_mapping(
    mapping: dict[Any, Any],
    *,
    seen: set[int],
    depth: int,
) -> None:
    for key, value in list(mapping.items()):
        if _is_native_runtime_value(value):
            with suppress(Exception):
                mapping[key] = None
            continue
        if _should_descend_into(value):
            _clear_native_runtime_handles(value, seen=seen, depth=depth + 1)


def _object_has_native_runtime(obj: Any) -> bool:
    attrs = _safe_vars(obj)
    if not attrs:
        return False
    if "_native_runtime" in attrs:
        return True
    rnn = attrs.get("_rnn")
    return "_native_runtime" in (_safe_vars(rnn) or {})


def _safe_vars(obj: Any) -> dict[str, Any] | None:
    try:
        attrs = object.__getattribute__(obj, "__dict__")
    except Exception:
        return None
    return attrs if isinstance(attrs, dict) else None


def _should_descend_into(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return False
    return _safe_vars(value) is not None


def _is_native_runtime_value(value: Any) -> bool:
    if value is None:
        return False
    cls = type(value)
    module = getattr(cls, "__module__", "")
    name = getattr(cls, "__name__", "")
    return module.endswith("._native") or "NativeRuntime" in name
