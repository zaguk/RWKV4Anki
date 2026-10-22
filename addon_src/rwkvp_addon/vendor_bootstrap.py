from __future__ import annotations

import os
import sys

from .constants import VENDOR_ROOT, VENDOR_RUNTIME_ROOT
from .rwkv_backend import (
    RWKV_SRS_BACKEND_ENV_VAR,
    configured_rwkv_backend,
    ensure_default_rwkv_backend,
    rwkv_backend_scope,
)
from .windows_native_cache import (
    WindowsNativeLoadError,
    prepare_windows_native_extension,
)


def ensure_vendor_paths() -> None:
    ensure_default_rwkv_backend()
    for root in (VENDOR_RUNTIME_ROOT, VENDOR_ROOT):
        path = str(root)
        if root.exists() and path not in sys.path:
            sys.path.insert(0, path)


class DependencyError(RuntimeError):
    pass


_gpu_available_cache: dict[str, bool] = {}


def require_rwkv_srs(*, backend: str | None = None):
    with rwkv_backend_scope(backend) as selected_backend:
        ensure_vendor_paths()
        _prepare_native_backend(selected_backend)
        try:
            from rwkv_srs import RWKV_SRS
            if selected_backend == "rust":
                from rwkv_srs._native import native_api_version

                if native_api_version() < 80:
                    raise DependencyError(
                        "RWKV4Anki requires a matching RWKV-SRS native API 80+ bundle "
                        "for checkpoint loading and saving. Reinstall the current add-on bundle "
                        "and restart Anki."
                    )
        except DependencyError:
            raise
        except Exception as exc:  # pragma: no cover - depends on Anki runtime binaries
            raise DependencyError(
                "RWKV-SRS could not be imported. The add-on vendors rwkv_srs, but "
                "the selected backend also needs its matching bundled runtime files."
            ) from exc
        return RWKV_SRS


def require_rwkv_live_candidate_seed(*, backend: str | None = None):
    """Return RWKV-SRS's public live-candidate seed type lazily.

    Keeping this import behind the vendor bootstrap lets the add-on's pure
    Python engine and unit tests load without importing the platform-specific
    native extension.
    """

    with rwkv_backend_scope(backend) as selected_backend:
        ensure_vendor_paths()
        _prepare_native_backend(selected_backend)
        try:
            from rwkv_srs import LiveCandidateSeed
        except Exception as exc:  # pragma: no cover - depends on bundled runtime
            raise DependencyError(
                "RWKV-SRS live prediction-session support could not be imported."
            ) from exc
        return LiveCandidateSeed


def require_rwkv_review_batch(*, backend: str | None = None):
    """Return RWKV-SRS's native immutable review-batch type lazily."""

    with rwkv_backend_scope(backend) as selected_backend:
        ensure_vendor_paths()
        _prepare_native_backend(selected_backend)
        try:
            from rwkv_srs import ReviewBatch
        except Exception as exc:  # pragma: no cover - depends on bundled native assets
            raise DependencyError(
                "RWKV-SRS native review-batch support could not be imported."
            ) from exc
        return ReviewBatch


def require_rwkv_v5o_review_batch(*, backend: str | None = None):
    """Load the distinct V5 transport from a matching API 76+ native payload."""

    with rwkv_backend_scope(backend) as selected_backend:
        ensure_vendor_paths()
        _prepare_native_backend(selected_backend)
        try:
            from rwkv_srs import V5oReviewBatch
            from rwkv_srs._native import native_api_version

            if native_api_version() < 76:
                raise RuntimeError("V5 requires RWKV-SRS native API 76 or newer")
        except Exception as exc:  # pragma: no cover - depends on bundled native assets
            raise DependencyError("V5 requires a matching RWKV-SRS API 76+ bundle.") from exc
        return V5oReviewBatch




def require_rwkv_checkpoint_history_consistency(*, backend: str | None = None):
    """Return RWKV-SRS's runtime-free native checkpoint history helper."""

    with rwkv_backend_scope(backend) as selected_backend:
        ensure_vendor_paths()
        _prepare_native_backend(selected_backend)
        try:
            from rwkv_srs import check_checkpoint_history_consistency
        except Exception as exc:  # pragma: no cover - depends on bundled native assets
            raise DependencyError(
                "RWKV-SRS native checkpoint-history support could not be imported."
            ) from exc
        return check_checkpoint_history_consistency


def require_rwkv_probability(*, backend: str | None = None):
    with rwkv_backend_scope(backend) as selected_backend:
        ensure_vendor_paths()
        _prepare_native_backend(selected_backend)
        try:
            from rwkv_srs import get_probability
        except Exception as exc:  # pragma: no cover - depends on Anki runtime binaries
            raise DependencyError("RWKV-SRS probability helper could not be imported.") from exc
        return get_probability


def require_rwkv_interval(*, backend: str | None = None):
    with rwkv_backend_scope(backend) as selected_backend:
        ensure_vendor_paths()
        _prepare_native_backend(selected_backend)
        try:
            from rwkv_srs import get_interval
        except Exception as exc:  # pragma: no cover - depends on Anki runtime binaries
            raise DependencyError("RWKV-SRS interval helper could not be imported.") from exc
        return get_interval


def seed_rwkv_srs_torch(seed: int, *, required: bool = True) -> bool:
    ensure_vendor_paths()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on Anki runtime binaries
        if required:
            raise DependencyError("Torch could not be imported for RWKV-SRS seeding.") from exc
        return False
    torch.manual_seed(seed)
    return True


def rwkv_backend_name() -> str:
    ensure_vendor_paths()
    try:
        from rwkv_srs import backend_name
    except Exception:
        return os.environ.get(RWKV_SRS_BACKEND_ENV_VAR, "")
    return str(backend_name())


def rwkv_gpu_available(operation: str = "predict") -> bool:
    """Return one cached, operation-specific rwkv-srs GPU probe result."""

    if operation not in {"predict", "process"}:
        raise ValueError("GPU operation must be 'predict' or 'process'.")
    if operation in _gpu_available_cache:
        return _gpu_available_cache[operation]
    if configured_rwkv_backend() != "rust":
        _gpu_available_cache[operation] = False
        return False

    try:
        ensure_vendor_paths()
        _prepare_native_backend("rust")
        from rwkv_srs import gpu_available

        available = bool(gpu_available(operation))
    except Exception:
        available = False
    _gpu_available_cache[operation] = available
    return available


def reset_gpu_availability_cache() -> None:
    """Forget the machine-local probe result after replacing native assets."""

    _gpu_available_cache.clear()


def _prepare_native_backend(selected_backend: str) -> None:
    if selected_backend != "rust":
        return
    try:
        prepare_windows_native_extension()
    except Exception as exc:
        summary = (
            "RWKV-SRS verified its Windows native-extension cache, but could not "
            "load the native module."
            if isinstance(exc, WindowsNativeLoadError)
            else "RWKV-SRS could not prepare its verified Windows native-extension cache."
        )
        # Background workers detach exception chains to release their runtime
        # frames. Preserve the diagnostic as text before that happens: setup
        # and checkpoint failure dialogs intentionally display only str(exc).
        raise DependencyError(f"{summary}\n\n{_native_failure_details(exc)}") from exc


def _native_failure_details(exc: BaseException) -> str:
    details = []
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        description = str(current).strip()
        name = type(current).__name__
        details.append(f"{name}: {description}" if description else name)
        if current.__cause__ is not None:
            current = current.__cause__
        else:
            current = None if current.__suppress_context__ else current.__context__
    return "\nCaused by: ".join(details)
