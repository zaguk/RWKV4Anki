from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes

from .constants import USER_FILES_ROOT
from .rwkv_backend import is_checkpoint_path_for_backend, rwkv_checkpoint_suffix

SCHEMA_VERSION = 1

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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ProfileStore:
    root: Path
    profile_name: str

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
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.behavior_lab_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def evaluation_cache_dir(self) -> Path:
        return self.root / "evaluation_cache"

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
        return self.evaluation_cache_dir / "live_review_history.sqlite"

    @property
    def prediction_cache_path(self) -> Path:
        return self.evaluation_cache_dir / "rwkv_review_predictions.jsonl.gz"

    @property
    def review_tail_context_path(self) -> Path:
        """Disposable context for replaying a cold checkpoint review tail."""

        return self.cache_dir / "review_tail_context.bin"

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
        return self.checkpoints_dir / f"latest{rwkv_checkpoint_suffix()}"

    @property
    def partial_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / f"partial{rwkv_checkpoint_suffix()}"

    @property
    def manifest_path(self) -> Path:
        return self.root / "checkpoint_manifest.json"

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
        checkpoint = manifest.get("checkpoint_path")
        if _usable_checkpoint_path(checkpoint):
            return Path(checkpoint)
        if self.latest_checkpoint_path.exists():
            return self.latest_checkpoint_path
        if self.partial_checkpoint_path.exists():
            return self.partial_checkpoint_path
        return None


def _usable_checkpoint_path(value: Any) -> bool:
    if not value:
        return False
    path = Path(value)
    return path.exists() and is_checkpoint_path_for_backend(path)


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
