from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

RWKV_SRS_BACKEND_ENV_VAR = "RWKV_SRS_BACKEND"
DEFAULT_RWKV_SRS_BACKEND = "rust"

_BACKEND_ALIASES = {
    "py": "python",
    "python": "python",
    "torch": "python",
    "rs": "rust",
    "rust": "rust",
}


def normalize_rwkv_backend(value: str | None) -> str:
    raw = (value or DEFAULT_RWKV_SRS_BACKEND).strip().lower()
    if not raw:
        raw = DEFAULT_RWKV_SRS_BACKEND
    try:
        return _BACKEND_ALIASES[raw]
    except KeyError as exc:
        supported = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(
            f"Unsupported {RWKV_SRS_BACKEND_ENV_VAR}={value!r}; expected one of: {supported}."
        ) from exc


def configured_rwkv_backend() -> str:
    return normalize_rwkv_backend(os.environ.get(RWKV_SRS_BACKEND_ENV_VAR))


def ensure_default_rwkv_backend() -> str:
    os.environ.setdefault(RWKV_SRS_BACKEND_ENV_VAR, DEFAULT_RWKV_SRS_BACKEND)
    return configured_rwkv_backend()


def rwkv_checkpoint_suffix(backend: str | None = None) -> str:
    resolved = configured_rwkv_backend() if backend is None else normalize_rwkv_backend(backend)
    return ".bin" if resolved == "rust" else ".pt"


def is_checkpoint_path_for_backend(path, backend: str | None = None) -> bool:
    return str(path).endswith(rwkv_checkpoint_suffix(backend))


@contextmanager
def rwkv_backend_scope(backend: str | None) -> Iterator[str]:
    if backend is None:
        yield ensure_default_rwkv_backend()
        return

    previous = os.environ.get(RWKV_SRS_BACKEND_ENV_VAR)
    normalized = normalize_rwkv_backend(backend)
    os.environ[RWKV_SRS_BACKEND_ENV_VAR] = normalized
    try:
        yield normalized
    finally:
        if previous is None:
            os.environ.pop(RWKV_SRS_BACKEND_ENV_VAR, None)
        else:
            os.environ[RWKV_SRS_BACKEND_ENV_VAR] = previous
