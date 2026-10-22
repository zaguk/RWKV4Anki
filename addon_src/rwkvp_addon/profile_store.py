from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes
from uuid import uuid4

from .constants import USER_FILES_ROOT
from .model_generation import (
    DEFAULT_MODEL_GENERATION,
    MODEL_GENERATIONS,
    normalize_model_generation,
)
from .model_storage import legacy_model_generation
from .rwkv_backend import is_checkpoint_path_for_backend, rwkv_checkpoint_suffix

SCHEMA_VERSION = 1
_SNAPSHOT_ID = re.compile(r"[0-9a-f]{32}\Z")

_PROFILE_FOLDER_SAFE_BYTES = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def slugify_profile_name(profile_name: str) -> str:
    return profile_folder_name(profile_name)


def _legacy_slugify_profile_name(profile_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile_name.strip())
    return slug.strip("._") or "default"


def profile_folder_name(profile_name: str) -> str:
    raw = profile_name.encode("utf-8")
    if not raw:
        return "%00"

    encoded = "".join(
        chr(byte) if byte in _PROFILE_FOLDER_SAFE_BYTES else f"%{byte:02X}" for byte in raw
    )
    if encoded.upper() in _WINDOWS_RESERVED_NAMES:
        return "".join(f"%{byte:02X}" for byte in raw)
    return encoded


def profile_name_from_folder_name(folder_name: str) -> str:
    if folder_name == "%00":
        return ""
    return unquote_to_bytes(folder_name).decode("utf-8")


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    # Independent writers must never share a temporary file: one writer could
    # otherwise publish another's partial output or remove its pending write.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            tmp = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        # The replacement is already committed. A platform/filesystem that
        # cannot sync directory entries must not turn it into a reported failure.
        with suppress(OSError):
            _sync_directory(path.parent)
    finally:
        if tmp is not None:
            with suppress(OSError):
                tmp.unlink(missing_ok=True)


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        # Windows does not support opening a directory through os.open().
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_linked_path(path: Path) -> bool:
    """Recognize Windows junctions as well as ordinary symbolic links."""
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (is_junction is not None and is_junction())


@dataclass(frozen=True)
class ProfileStore:
    root: Path
    profile_name: str
    model_generation: str = DEFAULT_MODEL_GENERATION
    checkpoint_directory: Path | None = None

    def for_generation(self, generation: str) -> ProfileStore:
        return replace(
            self, model_generation=normalize_model_generation(generation), checkpoint_directory=None
        )

    def with_checkpoint_directory(self, directory: Path) -> ProfileStore:
        """Use an unpublished checkpoint directory, keeping profile settings shared."""
        return replace(self, checkpoint_directory=directory)

    @classmethod
    def for_profile(cls, profile_name: str) -> ProfileStore:
        profiles_root = USER_FILES_ROOT / "profiles"
        root = profiles_root / profile_folder_name(profile_name)
        _migrate_legacy_profile_root(profiles_root, profile_name, root)
        store = cls(root=root, profile_name=profile_name)
        store.ensure()
        return store

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_directory is None:
            self._recover_checkpoint_publication()
            _migrate_legacy_checkpoint(self)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.behavior_lab_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoints_dir(self) -> Path:
        generation = normalize_model_generation(self.model_generation)
        folder = {"v1": "v1", "v5o": "v5"}[generation]
        return self.checkpoint_directory or self.root / "checkpoints" / folder

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def evaluation_cache_dir(self) -> Path:
        return self._checkpoint_data_directory() / "evaluation_cache"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def behavior_lab_dir(self) -> Path:
        return self.root / "behavior_lab"

    @property
    def behavior_lab_experiments_path(self) -> Path:
        return self.behavior_lab_dir / "experiments.json"

    @property
    def retrievability_nan_log_path(self) -> Path:
        return self.logs_dir / "retrievability_nonfinite_predictions.jsonl"

    @property
    def evaluation_cache_path(self) -> Path:
        return self.evaluation_cache_dir / "rwkv_evaluation_cache.sqlite"

    @property
    def partial_evaluation_cache_path(self) -> Path:
        return self.evaluation_cache_dir / "rwkv_evaluation_cache.partial.sqlite"

    @property
    def live_review_history_path(self) -> Path:
        # Session reports span generations and remain profile-wide.
        return self.root / "evaluation_cache" / "live_review_history.sqlite"

    @property
    def prediction_cache_path(self) -> Path:
        return self.evaluation_cache_dir / "rwkv_review_predictions.jsonl.gz"

    @property
    def review_tail_context_path(self) -> Path:
        """Disposable context for replaying a cold checkpoint review tail."""

        return self.checkpoints_dir / "review_tail_context.bin"

    @property
    def partial_prediction_cache_path(self) -> Path:
        return self.evaluation_cache_dir / "rwkv_review_predictions.partial.jsonl.gz"

    @property
    def predict_ahead_prediction_cache_path(self) -> Path:
        return self.evaluation_cache_dir / "rwkv_predict_ahead_predictions.jsonl.gz"

    @property
    def partial_predict_ahead_prediction_cache_path(self) -> Path:
        return self.evaluation_cache_dir / "rwkv_predict_ahead_predictions.partial.jsonl.gz"

    @property
    def predict_ahead_curve_cache_path(self) -> Path:
        return self.evaluation_cache_dir / "rwkv_predict_ahead_curves.pkl"

    @property
    def partial_predict_ahead_curve_cache_path(self) -> Path:
        return self.evaluation_cache_dir / "rwkv_predict_ahead_curves.partial.pkl"

    @property
    def latest_checkpoint_path(self) -> Path:
        return self._checkpoint_data_directory() / f"latest{rwkv_checkpoint_suffix()}"

    @property
    def partial_checkpoint_path(self) -> Path:
        return self._checkpoint_data_directory() / f"partial{rwkv_checkpoint_suffix()}"

    @property
    def manifest_path(self) -> Path:
        return self.checkpoints_dir / "checkpoint_manifest.json"

    def _checkpoint_data_directory(self, manifest: dict[str, Any] | None = None) -> Path:
        manifest = self.manifest() if manifest is None else manifest
        if "snapshot_id" not in manifest:
            return self.checkpoints_dir
        snapshot_id = manifest["snapshot_id"]
        if not isinstance(snapshot_id, str) or _SNAPSHOT_ID.fullmatch(snapshot_id) is None:
            raise ValueError("The checkpoint manifest has an invalid snapshot identity")
        snapshots = self.checkpoints_dir / "snapshots"
        directory = snapshots / snapshot_id
        if _is_linked_path(snapshots) or _is_linked_path(directory):
            raise ValueError("Checkpoint snapshot directories must not be symbolic links")
        return directory

    def create_checkpoint_snapshot(self) -> ProfileStore:
        """Allocate an unpublished snapshot whose native backing paths stay stable.

        The caller holds the exclusive runtime slot. Never rename this directory
        or remove it while a scoped runtime may use its checkpoint as backing.
        """
        self.ensure()
        snapshots = self.checkpoints_dir / "snapshots"
        if _is_linked_path(snapshots):
            raise ValueError("Checkpoint snapshot directories must not be symbolic links")
        snapshots.mkdir(exist_ok=True)
        directory = snapshots / uuid4().hex
        directory.mkdir()
        staged = self.with_checkpoint_directory(directory)
        staged.ensure()
        return staged

    def publish_checkpoint_snapshot(self, staged: ProfileStore) -> bool:
        """Commit a complete snapshot by replacing only the root manifest.

        Publication never moves native backing files. Pre-publication failure
        leaves the previous manifest authoritative; orphan deletion is a separate
        operation because a live runtime may already be backed by the new file.
        Return whether syncing the committed pointer succeeded, allowing cleanup.
        """
        if self.root != staged.root or self.model_generation != staged.model_generation:
            raise ValueError("Checkpoint publication must keep the profile and generation")
        snapshots = self.checkpoints_dir / "snapshots"
        directory = staged.checkpoints_dir
        if (
            staged.checkpoint_directory is None
            or directory.parent != snapshots
            or _SNAPSHOT_ID.fullmatch(directory.name) is None
            or _is_linked_path(snapshots)
            or _is_linked_path(directory)
            or _is_linked_path(staged.manifest_path)
        ):
            raise ValueError("Checkpoint publication requires a managed snapshot directory")
        manifest = staged.manifest()
        status = manifest.get("status")
        if status not in {"valid", "partial", "invalid"}:
            raise ValueError("Only complete checkpoint snapshots can be published")
        if manifest.get("model_generation") != self.model_generation:
            raise ValueError("The staged checkpoint belongs to another generation")
        if "snapshot_id" in manifest:
            raise ValueError("An unpublished snapshot must not select another snapshot")
        if _is_linked_path(staged.evaluation_cache_dir):
            raise ValueError("Checkpoint snapshot directories must not be symbolic links")
        partial = status == "partial"
        expected = {
            "checkpoint_path": (
                staged.partial_checkpoint_path if partial else staged.latest_checkpoint_path
            ),
            "evaluation_cache_path": (
                staged.partial_evaluation_cache_path if partial else staged.evaluation_cache_path
            ),
        }
        for key, path in expected.items():
            if (
                manifest.get(key) != str(path)
                or not path.is_file()
                or _is_linked_path(path)
                or not path.resolve().is_relative_to(directory.resolve())
            ):
                raise ValueError("A snapshot requires its own complete checkpoint and cache")
            # Windows FlushFileBuffers (used by fsync) requires a writable handle.
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
        with staged.manifest_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        for path in (staged.evaluation_cache_dir, directory, snapshots, self.checkpoints_dir):
            _sync_directory(path)
        manifest["snapshot_id"] = directory.name
        self.write_manifest(manifest)
        try:
            _sync_directory(self.checkpoints_dir)
        except OSError:
            # Publication happened. Retain the previous snapshot until a later
            # successful sync rather than reporting an already committed save
            # as failed, or removing the power-loss fallback prematurely.
            return False
        return True

    def cleanup_checkpoint_snapshots(self, *, keep_paths: Iterable[Path] = ()) -> None:
        """Best-effort cleanup at a caller-proven exclusive runtime boundary.

        Preserve selected state and any explicitly protected native backing
        paths. Unknown names and links are never followed or deleted.
        """
        try:
            manifest = self.manifest()
            if "snapshot_id" in manifest:
                if self._snapshot_checkpoint_path(manifest) is None:
                    return
                if not Path(manifest["evaluation_cache_path"]).is_file():
                    return
            selected = self._checkpoint_data_directory(manifest).resolve()
            protected = tuple(Path(path).resolve() for path in keep_paths)
            snapshots = self.checkpoints_dir / "snapshots"
            if _is_linked_path(snapshots):
                return
            # A previous post-commit sync may have failed. Never delete its
            # fallback until the current pointer is durably discoverable.
            _sync_directory(self.checkpoints_dir)
            directories = tuple(snapshots.iterdir())
        except (OSError, ValueError):
            return
        for directory in directories:
            try:
                if (
                    _SNAPSHOT_ID.fullmatch(directory.name) is None
                    or _is_linked_path(directory)
                    or not directory.is_dir()
                    or directory.resolve() == selected
                    or any(path.is_relative_to(directory.resolve()) for path in protected)
                ):
                    continue
                shutil.rmtree(directory)
            except OSError:
                # A leftover snapshot is harmless; a later safe save can retry.
                continue

    @property
    def _previous_checkpoint_directory(self) -> Path:
        return self.checkpoints_dir.with_name(f".{self.checkpoints_dir.name}.previous")

    def _recover_checkpoint_publication(self) -> None:
        previous = self._previous_checkpoint_directory
        if previous.exists() and not self.checkpoints_dir.exists():
            previous.rename(self.checkpoints_dir)

    def publish_checkpoint(self, staged: ProfileStore) -> None:
        """Publish a complete checkpoint/cache pair; recover the old pair on failure.

        The caller must hold the profile's exclusive runtime slot. A directory
        rename keeps caches and manifests together, including after interruption.
        """
        if self.checkpoint_directory is not None or staged.checkpoint_directory is None:
            raise ValueError("Checkpoint publication requires a staged source and canonical target")
        if self.root != staged.root or self.model_generation != staged.model_generation:
            raise ValueError("Checkpoint publication must keep the profile and generation")
        manifest = staged.manifest()
        if manifest.get("status") != "valid":
            raise ValueError("Only complete comparison checkpoints can be published")
        if _checkpoint_manifest_generation(manifest) != self.model_generation:
            raise ValueError("The staged checkpoint belongs to another generation")
        for key in ("checkpoint_path", "evaluation_cache_path"):
            if not manifest.get(key) or not Path(manifest[key]).is_file():
                raise ValueError("A complete checkpoint and evaluation cache are required")
        for key, value in tuple(manifest.items()):
            if key.endswith("_path") and value:
                relative = Path(value).relative_to(staged.checkpoints_dir)
                manifest[key] = str(self.checkpoints_dir / relative)
        staged.write_manifest(manifest)
        self.ensure()
        previous = self._previous_checkpoint_directory
        if previous.exists():
            shutil.rmtree(previous)
        self.checkpoints_dir.rename(previous)
        try:
            staged.checkpoints_dir.rename(self.checkpoints_dir)
        except BaseException:
            previous.rename(self.checkpoints_dir)
            raise
        # A failed cleanup is harmless and can be retried on the next publish.
        shutil.rmtree(previous, ignore_errors=True)

    @property
    def settings_path(self) -> Path:
        return self.root / "settings.json"

    def manifest(self) -> dict[str, Any]:
        return read_json(self.manifest_path)

    def write_manifest(self, data: dict[str, Any]) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "profile_name": self.profile_name}
        payload.update(data)
        atomic_write_json(self.manifest_path, payload)

    def settings(self) -> dict[str, Any]:
        return read_json(self.settings_path)

    def write_settings(self, data: dict[str, Any]) -> None:
        atomic_write_json(self.settings_path, data)

    def checkpoint_status(self) -> str:
        manifest = self.manifest()
        if "snapshot_id" in manifest:
            return str(manifest.get("status") or "missing") if self._snapshot_checkpoint_path(
                manifest
            ) is not None else "missing"
        status = str(manifest.get("status") or "missing")
        checkpoint = manifest.get("checkpoint_path")
        if _usable_checkpoint_path(checkpoint):
            return status
        if self.latest_checkpoint_path.exists():
            return "valid"
        if self.partial_checkpoint_path.exists():
            return "partial"
        return "missing"

    def active_checkpoint_path(self) -> Path | None:
        manifest = self.manifest()
        if "snapshot_id" in manifest:
            return self._snapshot_checkpoint_path(manifest)
        checkpoint = manifest.get("checkpoint_path")
        if _usable_checkpoint_path(checkpoint):
            return Path(checkpoint)
        if self.latest_checkpoint_path.exists():
            return self.latest_checkpoint_path
        if self.partial_checkpoint_path.exists():
            return self.partial_checkpoint_path
        return None

    def _snapshot_checkpoint_path(self, manifest: dict[str, Any]) -> Path | None:
        directory = self._checkpoint_data_directory(manifest)
        partial = manifest.get("status") == "partial" or (
            manifest.get("status") == "invalid"
            and Path(str(manifest.get("checkpoint_path", ""))).name
            == f"partial{rwkv_checkpoint_suffix()}"
        )
        name = "partial" if partial else "latest"
        expected = directory / f"{name}{rwkv_checkpoint_suffix()}"
        if manifest.get("checkpoint_path") != str(expected):
            raise ValueError("The checkpoint manifest points outside its selected snapshot")
        cache_name = (
            "rwkv_evaluation_cache.partial.sqlite" if partial else "rwkv_evaluation_cache.sqlite"
        )
        expected_cache = directory / "evaluation_cache" / cache_name
        if manifest.get("evaluation_cache_path") != str(expected_cache):
            raise ValueError("The cache manifest points outside its selected snapshot")
        if any(_is_linked_path(path) for path in (expected, expected_cache.parent, expected_cache)):
            raise ValueError("Checkpoint snapshot files must not be symbolic links")
        return expected if expected.is_file() else None


def _usable_checkpoint_path(value: Any) -> bool:
    if not value:
        return False
    path = Path(value)
    return path.exists() and is_checkpoint_path_for_backend(path)


def _checkpoint_manifest_generation(manifest: dict[str, Any]) -> str | None:
    """Resolve legacy state without applying forgiving user-config defaults."""
    generation = manifest.get("model_generation")
    if generation is not None:
        return generation if generation in MODEL_GENERATIONS else None
    model_id = manifest.get("model_id")
    if model_id == "rwkv-p-v2o":
        return None
    return legacy_model_generation(model_id)


def _migrate_legacy_checkpoint(store: ProfileStore) -> None:
    """Move the old single checkpoint into its owning generation, manifest last.

    Renames are cheap even for very large states. Until the final manifest is
    written, the original manifest doubles as a restartable migration journal.
    Never replace a slot that already has its own manifest or conflicting file.
    """
    legacy_manifest = store.root / "checkpoint_manifest.json"
    if not legacy_manifest.exists():
        return
    manifest = read_json(legacy_manifest)
    generation = _checkpoint_manifest_generation(manifest)
    if generation is None:
        # Preserve retired/unknown state in place. It must not become a valid
        # V1 checkpoint merely because stale configuration falls back to V1.
        return
    target = store.for_generation(generation)
    target._recover_checkpoint_publication()
    if target.manifest_path.exists():
        # A previous migration may have committed before removing its journal.
        # An independently created slot takes precedence; leave legacy data alone.
        return
    moves: list[tuple[Path, Path]] = []
    for name in ("latest.bin", "partial.bin", "latest.pt", "partial.pt"):
        moves.append((store.root / "checkpoints" / name, target.checkpoints_dir / name))
    for attribute in (
        "evaluation_cache_path", "partial_evaluation_cache_path", "prediction_cache_path",
        "partial_prediction_cache_path", "predict_ahead_prediction_cache_path",
        "partial_predict_ahead_prediction_cache_path", "predict_ahead_curve_cache_path",
        "partial_predict_ahead_curve_cache_path",
    ):
        destination = getattr(target, attribute)
        moves.append((store.root / "evaluation_cache" / destination.name, destination))
    moves.append((
        store.root / "cache" / "review_tail_context.bin", target.review_tail_context_path
    ))
    for source, destination in moves:
        if source.exists() and destination.exists():
            raise FileExistsError(f"Checkpoint migration would overwrite {destination}")
    for source, destination in moves:
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
    destinations = {source.name: destination for source, destination in moves}
    for key, value in tuple(manifest.items()):
        if key.endswith("_path") and value and Path(value).name in destinations:
            manifest[key] = str(destinations[Path(value).name])
    manifest["model_generation"] = generation
    target.write_manifest(manifest)
    legacy_manifest.unlink()


def _migrate_legacy_profile_root(
    profiles_root: Path,
    profile_name: str,
    new_root: Path,
) -> None:
    legacy_root = profiles_root / _legacy_slugify_profile_name(profile_name)
    if legacy_root == new_root or new_root.exists() or not legacy_root.exists():
        return

    manifest = read_json(legacy_root / "checkpoint_manifest.json")
    if manifest.get("profile_name") != profile_name:
        return

    profiles_root.mkdir(parents=True, exist_ok=True)
    legacy_root.rename(new_root)
