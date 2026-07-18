from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .constants import ADDON_PACKAGE, VENDOR_ROOT

NATIVE_MODULE_NAME = "rwkv_srs._native"
NATIVE_CACHE_LAYOUT = "native-v1"
NATIVE_CACHE_MARKER = ".rwkv4anki-native-cache.json"

_CACHE_MARKER_CONTENT = {
    "layout": 1,
    "owner": ADDON_PACKAGE,
    "purpose": "Windows native extension cache",
}
_CONTENT_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_scheduled_cleanup_roots: set[str] = set()
_native_cache_lock = threading.RLock()


class WindowsNativeCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class WindowsNativeCacheEntry:
    source_path: Path
    cached_path: Path
    content_digest: str
    cache_root: Path
    content_root: Path


def prepare_windows_native_extension(
    *,
    vendor_root: Path = VENDOR_ROOT,
    application_cache_dir: Path | None = None,
    platform_name: str | None = None,
    runtime_tag: str | None = None,
) -> Path | None:
    """Load ``rwkv_srs._native`` from a disposable cache on Windows.

    Windows locks an imported ``.pyd`` until the Python process exits. Loading a
    byte-for-byte cached copy keeps Anki's installed add-on directory replaceable.
    Other platforms retain the ordinary bundled import path.
    """

    if (platform_name or sys.platform) != "win32":
        return None

    with _native_cache_lock:
        existing = sys.modules.get(NATIVE_MODULE_NAME)
        if existing is not None:
            loaded_path = getattr(existing, "__file__", None)
            return None if not loaded_path else Path(loaded_path)

        source_path = find_bundled_windows_native_extension(vendor_root)
        cache_dir = (
            windows_application_cache_dir()
            if application_cache_dir is None
            else Path(application_cache_dir)
        )
        entry = stage_windows_native_extension(
            source_path,
            cache_dir,
            runtime_tag=runtime_tag,
        )
        prune_obsolete_windows_native_entries(entry)
        if _sha256_file(entry.cached_path) != entry.content_digest:
            raise WindowsNativeCacheError(
                f"Cached RWKV-SRS native extension failed verification: {entry.cached_path}"
            )
        load_cached_windows_native_module(entry.cached_path)
        return entry.cached_path


def prune_windows_native_cache_at_startup(
    *,
    vendor_root: Path = VENDOR_ROOT,
    application_cache_dir: Path | None = None,
    platform_name: str | None = None,
    runtime_tag: str | None = None,
) -> bool:
    """Prune old unlocked hashes without importing or copying the current DLL."""

    if (platform_name or sys.platform) != "win32":
        return False
    cache_dir = (
        windows_application_cache_dir()
        if application_cache_dir is None
        else Path(application_cache_dir)
    )
    cache_root = windows_native_cache_root(cache_dir)
    if not cache_root.exists() or not _cache_root_is_owned(cache_root):
        return False
    source_path = find_bundled_windows_native_extension(vendor_root)
    digest = _sha256_file(source_path)
    tag = _safe_path_component(runtime_tag or windows_native_runtime_tag())
    content_root = cache_root / tag / digest
    entry = WindowsNativeCacheEntry(
        source_path=source_path,
        cached_path=content_root / "rwkv_srs" / source_path.name,
        content_digest=digest,
        cache_root=cache_root,
        content_root=content_root,
    )
    prune_obsolete_windows_native_entries(entry)
    return True


def find_bundled_windows_native_extension(vendor_root: Path = VENDOR_ROOT) -> Path:
    package_root = Path(vendor_root) / "rwkv_srs"
    candidates = sorted(path for path in package_root.glob("_native*.pyd") if path.is_file())
    if len(candidates) != 1:
        raise WindowsNativeCacheError(
            "Expected exactly one bundled rwkv_srs/_native*.pyd, "
            f"found {len(candidates)} under {package_root}."
        )
    return candidates[0]


def windows_application_cache_dir() -> Path:
    """Return Anki's application-specific, machine-local Qt cache directory."""

    try:
        from aqt.qt import QStandardPaths

        try:
            cache_location = QStandardPaths.StandardLocation.CacheLocation
        except AttributeError:  # pragma: no cover - compatibility with older Qt
            cache_location = QStandardPaths.CacheLocation
        value = QStandardPaths.writableLocation(cache_location)
    except Exception as exc:  # pragma: no cover - depends on the packaged Anki Qt API
        raise WindowsNativeCacheError(
            "Anki's Qt cache location is unavailable for the Windows native runtime."
        ) from exc
    if not value:
        raise WindowsNativeCacheError(
            "Anki returned an empty Qt cache location for the Windows native runtime."
        )
    return Path(value)


def windows_native_cache_root(application_cache_dir: Path) -> Path:
    return Path(application_cache_dir) / ADDON_PACKAGE / NATIVE_CACHE_LAYOUT


def stage_windows_native_extension(
    source_path: Path,
    application_cache_dir: Path,
    *,
    runtime_tag: str | None = None,
) -> WindowsNativeCacheEntry:
    source_path = Path(source_path)
    if not source_path.is_file() or source_path.suffix.lower() != ".pyd":
        raise WindowsNativeCacheError(
            f"Bundled RWKV-SRS native extension is missing or invalid: {source_path}"
        )

    digest = _sha256_file(source_path)
    cache_root = windows_native_cache_root(application_cache_dir)
    _ensure_owned_cache_root(cache_root)
    tag = _safe_path_component(runtime_tag or windows_native_runtime_tag())
    runtime_root = cache_root / tag
    _ensure_cache_directory(runtime_root)
    content_root = runtime_root / digest
    _ensure_cache_directory(content_root)
    package_root = content_root / "rwkv_srs"
    _ensure_cache_directory(package_root)
    cached_path = package_root / source_path.name

    if not _verified_regular_file(cached_path, digest):
        _copy_file_atomically(source_path, cached_path, expected_digest=digest)
    if not _verified_regular_file(cached_path, digest):
        raise WindowsNativeCacheError(
            f"Unable to create a verified RWKV-SRS native cache entry: {cached_path}"
        )

    return WindowsNativeCacheEntry(
        source_path=source_path,
        cached_path=cached_path,
        content_digest=digest,
        cache_root=cache_root,
        content_root=content_root,
    )


def windows_native_runtime_tag() -> str:
    machine = platform.machine().strip().lower() or "unknown"
    python_tag = str(getattr(sys.implementation, "cache_tag", "") or "python")
    return f"windows-{machine}-{python_tag}"


def load_cached_windows_native_module(cached_path: Path) -> ModuleType:
    existing = sys.modules.get(NATIVE_MODULE_NAME)
    if existing is not None:
        return existing

    package_name, attribute_name = NATIVE_MODULE_NAME.rsplit(".", 1)
    package = importlib.import_module(package_name)
    spec = importlib.util.spec_from_file_location(NATIVE_MODULE_NAME, cached_path)
    if spec is None or spec.loader is None:
        raise WindowsNativeCacheError(
            f"Unable to create an import loader for cached extension: {cached_path}"
        )
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous_attribute = getattr(package, attribute_name, missing)
    sys.modules[NATIVE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        setattr(package, attribute_name, module)
    except BaseException:
        if sys.modules.get(NATIVE_MODULE_NAME) is module:
            sys.modules.pop(NATIVE_MODULE_NAME, None)
        if previous_attribute is missing:
            with suppress(AttributeError):
                delattr(package, attribute_name)
        else:
            setattr(package, attribute_name, previous_attribute)
        raise
    return module


def prune_obsolete_windows_native_entries(entry: WindowsNativeCacheEntry) -> None:
    """Best-effort removal of old content hashes before the current DLL is loaded."""

    if not _cache_root_is_owned(entry.cache_root):
        raise WindowsNativeCacheError(
            f"Refusing to prune an unrecognized native cache: {entry.cache_root}"
        )
    keep = _absolute_path_key(entry.content_root)
    for runtime_root in entry.cache_root.iterdir():
        if runtime_root.name == NATIVE_CACHE_MARKER or not runtime_root.is_dir():
            continue
        if _path_is_link_or_reparse(runtime_root):
            continue
        for content_root in runtime_root.iterdir():
            if (
                not _CONTENT_DIGEST_RE.fullmatch(content_root.name)
                or not content_root.is_dir()
                or _path_is_link_or_reparse(content_root)
            ):
                continue
            if _absolute_path_key(content_root) == keep:
                continue
            try:
                # Re-check immediately before recursive deletion so a verified
                # directory cannot be swapped for a junction after enumeration.
                if _path_is_link_or_reparse(content_root):
                    continue
                shutil.rmtree(content_root)
            except OSError:
                # Another Anki process may still have this content hash loaded.
                continue
        with suppress(OSError):
            runtime_root.rmdir()


def schedule_windows_native_cache_removal_after_exit(
    *,
    application_cache_dir: Path | None = None,
    parent_pid: int | None = None,
    platform_name: str | None = None,
    powershell_path: Path | None = None,
    launcher: Callable[..., Any] = subprocess.Popen,
) -> bool:
    """Start a detached Windows janitor that removes the cache after Anki exits."""

    if (platform_name or sys.platform) != "win32":
        return False
    cache_dir = (
        windows_application_cache_dir()
        if application_cache_dir is None
        else Path(application_cache_dir)
    )
    cache_root = windows_native_cache_root(cache_dir)
    if not cache_root.exists() or not _cache_root_is_owned(cache_root):
        return False
    cache_key = _absolute_path_key(cache_root)
    if cache_key in _scheduled_cleanup_roots:
        return True

    executable = powershell_path or _system_powershell_path()
    if executable is None:
        return False
    script = _powershell_cleanup_script(cache_root, parent_pid or os.getpid())
    encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    creation_flags = (
        int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        | int(getattr(subprocess, "DETACHED_PROCESS", 0))
        | int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    )
    try:
        launcher(
            [
                str(executable),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-EncodedCommand",
                encoded_script,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creation_flags,
        )
    except OSError:
        return False
    _scheduled_cleanup_roots.add(cache_key)
    return True


def _ensure_owned_cache_root(cache_root: Path) -> None:
    cache_namespace = cache_root.parent
    cache_namespace.parent.mkdir(parents=True, exist_ok=True)
    _ensure_cache_directory(cache_namespace)
    _ensure_cache_directory(cache_root)
    marker_path = cache_root / NATIVE_CACHE_MARKER
    if marker_path.exists():
        if not _cache_root_is_owned(cache_root):
            raise WindowsNativeCacheError(f"Native cache marker is invalid: {marker_path}")
        return
    marker_bytes = (json.dumps(_CACHE_MARKER_CONTENT, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes_atomically(marker_path, marker_bytes)
    if not _cache_root_is_owned(cache_root):
        raise WindowsNativeCacheError(f"Unable to verify native cache marker: {marker_path}")


def _ensure_cache_directory(path: Path) -> None:
    try:
        path.mkdir(exist_ok=True)
    except OSError as exc:
        raise WindowsNativeCacheError(f"Unable to create native cache directory: {path}") from exc
    if not path.is_dir() or _path_is_link_or_reparse(path):
        raise WindowsNativeCacheError(
            f"Native cache directories must not be links or reparse points: {path}"
        )


def _path_is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    except OSError:
        return False
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(reparse_flag and attributes & reparse_flag)


def _cache_root_is_owned(cache_root: Path) -> bool:
    cache_root = Path(cache_root)
    cache_namespace = cache_root.parent
    if (
        not cache_root.is_dir()
        or _path_is_link_or_reparse(cache_root)
        or not cache_namespace.is_dir()
        or _path_is_link_or_reparse(cache_namespace)
    ):
        return False
    marker_path = cache_root / NATIVE_CACHE_MARKER
    if _path_is_link_or_reparse(marker_path) or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return marker == _CACHE_MARKER_CONTENT


def _absolute_path_key(path: Path) -> str:
    """Compare cache entries without resolving attacker-controlled links."""

    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _copy_file_atomically(source: Path, destination: Path, *, expected_digest: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target, source.open("rb") as source_file:
            shutil.copyfileobj(source_file, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if _sha256_file(temporary_path) != expected_digest:
            raise WindowsNativeCacheError(
                f"Copied RWKV-SRS native extension failed verification: {temporary_path}"
            )
        os.replace(temporary_path, destination)
    finally:
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def _write_bytes_atomically(destination: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, destination)
    finally:
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def _verified_regular_file(path: Path, digest: str) -> bool:
    return path.is_file() and not path.is_symlink() and _sha256_file(path) == digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    return safe or "unknown"


def _system_powershell_path() -> Path | None:
    system_root = os.environ.get("SYSTEMROOT")
    if not system_root:
        return None
    candidate = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return candidate if candidate.is_file() else None


def _powershell_cleanup_script(cache_root: Path, parent_pid: int) -> str:
    # Preserve the lexical cache path. Resolving here could follow a junction
    # swapped in after the caller's safety check and hand the janitor the
    # external target path instead of the reparse point it must reject.
    root_literal = _powershell_single_quoted(_absolute_path_key(cache_root))
    parent_literal = _powershell_single_quoted(_absolute_path_key(cache_root.parent))
    marker_literal = _powershell_single_quoted(NATIVE_CACHE_MARKER)
    owner_literal = _powershell_single_quoted(str(_CACHE_MARKER_CONTENT["owner"]))
    purpose_literal = _powershell_single_quoted(str(_CACHE_MARKER_CONTENT["purpose"]))
    return "\n".join(
        [
            "$ErrorActionPreference = 'SilentlyContinue'",
            f"$cacheRoot = {root_literal}",
            f"$cacheParent = {parent_literal}",
            f"$markerName = {marker_literal}",
            "function Test-RwkvOwnedCacheRoot {",
            "  try {",
            "    $rootItem = Get-Item -LiteralPath $cacheRoot -Force -ErrorAction Stop",
            "    $parentItem = Get-Item -LiteralPath $cacheParent -Force -ErrorAction Stop",
            "    $reparse = [IO.FileAttributes]::ReparsePoint",
            "    if (($rootItem.Attributes -band $reparse) -ne 0) { return $false }",
            "    if (($parentItem.Attributes -band $reparse) -ne 0) { return $false }",
            "    $markerPath = Join-Path $cacheRoot $markerName",
            "    $markerItem = Get-Item -LiteralPath $markerPath -Force -ErrorAction Stop",
            "    if (($markerItem.Attributes -band $reparse) -ne 0) { return $false }",
            (
                "    $marker = Get-Content -LiteralPath $markerPath -Raw "
                "-ErrorAction Stop | ConvertFrom-Json"
            ),
            "    return (",
            "      [int]$marker.layout -eq 1 -and",
            f"      [string]$marker.owner -eq {owner_literal} -and",
            f"      [string]$marker.purpose -eq {purpose_literal}",
            "    )",
            "  } catch { return $false }",
            "}",
            (
                f"try {{ Wait-Process -Id {int(parent_pid)} "
                "-ErrorAction SilentlyContinue }} catch {}"
            ),
            "if (-not (Test-RwkvOwnedCacheRoot)) { exit 0 }",
            "for ($attempt = 0; $attempt -lt 120; $attempt++) {",
            "  try {",
            "    if (-not (Test-RwkvOwnedCacheRoot)) { break }",
            "    Remove-Item -LiteralPath $cacheRoot -Recurse -Force -ErrorAction Stop",
            "    break",
            "  } catch {",
            "    Start-Sleep -Milliseconds 500",
            "  }",
            "}",
            "try {",
            "  $parentItem = Get-Item -LiteralPath $cacheParent -Force -ErrorAction Stop",
            "  if (($parentItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {",
            "    Remove-Item -LiteralPath $cacheParent -Force -ErrorAction Stop",
            "  }",
            "} catch {}",
        ]
    )


def _powershell_single_quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
