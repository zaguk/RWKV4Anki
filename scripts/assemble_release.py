#!/usr/bin/env python3
"""Validate and assemble the complete RWKV4Anki GitHub Release asset set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from scripts.package_addon import (
        COMMIT_PATTERN,
        DEFAULT_MODEL_LOCK,
        DEFAULT_RELEASE_LOCK,
        SUPPORTED_TARGETS,
        VERSION_PATTERN,
        ModelLock,
        ReleaseLock,
        _load_model_lock,
        _load_release_lock,
    )
    from scripts.validate_addon_bundle import validate_stage
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from package_addon import (  # type: ignore[import-not-found,no-redef]
        COMMIT_PATTERN,
        DEFAULT_MODEL_LOCK,
        DEFAULT_RELEASE_LOCK,
        SUPPORTED_TARGETS,
        VERSION_PATTERN,
        ModelLock,
        ReleaseLock,
        _load_model_lock,
        _load_release_lock,
    )
    from validate_addon_bundle import validate_stage  # type: ignore[import-not-found,no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_PROVENANCE = PROJECT_ROOT / "PUBLIC_EXPORT_PROVENANCE.json"
CHECKSUMS_NAME = "SHA256SUMS"
RELEASE_PROVENANCE_NAME = "RELEASE_PROVENANCE.json"
ADDON_PACKAGE = "RWKV4Anki"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ReleaseAssemblyError(ValueError):
    """Raised when a candidate GitHub Release is incomplete or inconsistent."""


@dataclass(frozen=True)
class BundleRecord:
    target: str
    filename: str
    sha256: str
    size: int
    backend_commit: str
    backend_version: str
    wheel_name: str
    wheel_sha256: str


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all six target-native RWKV4Anki bundles and create "
            "aggregate checksums and release provenance."
        )
    )
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--public-commit", required=True)
    parser.add_argument("--release-lock", type=Path, default=DEFAULT_RELEASE_LOCK)
    parser.add_argument("--model-lock", type=Path, default=DEFAULT_MODEL_LOCK)
    parser.add_argument(
        "--export-provenance",
        type=Path,
        default=DEFAULT_EXPORT_PROVENANCE,
    )
    return parser


def main() -> int:
    args = _argument_parser().parse_args()
    try:
        result = assemble_release(
            assets_dir=args.assets_dir,
            output_dir=args.output_dir,
            version=args.version,
            public_commit=args.public_commit,
            release_lock_path=args.release_lock,
            model_lock_path=args.model_lock,
            export_provenance_path=args.export_provenance,
        )
    except (OSError, ReleaseAssemblyError, zipfile.BadZipFile, ValueError) as error:
        raise SystemExit(f"release assembly failed: {error}") from error
    print(result["checksums"])
    print(result["provenance"])
    return 0


def assemble_release(
    *,
    assets_dir: Path,
    output_dir: Path,
    version: str,
    public_commit: str,
    release_lock_path: Path = DEFAULT_RELEASE_LOCK,
    model_lock_path: Path = DEFAULT_MODEL_LOCK,
    export_provenance_path: Path = DEFAULT_EXPORT_PROVENANCE,
) -> dict[str, Path]:
    if VERSION_PATTERN.fullmatch(version) is None:
        raise ReleaseAssemblyError(f"unsafe add-on version: {version!r}")
    if COMMIT_PATTERN.fullmatch(public_commit) is None:
        raise ReleaseAssemblyError("public source commit must be a lowercase 40-character SHA")

    assets_dir = assets_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not assets_dir.is_dir():
        raise ReleaseAssemblyError(f"release asset directory does not exist: {assets_dir}")
    if (
        assets_dir == output_dir
        or assets_dir.is_relative_to(output_dir)
        or output_dir.is_relative_to(assets_dir)
    ):
        raise ReleaseAssemblyError(
            "release assets and aggregate output must use separate, non-overlapping directories"
        )

    release_lock = _load_release_lock(release_lock_path.expanduser().resolve(), require_ready=True)
    model_lock = _load_model_lock(model_lock_path.expanduser().resolve())
    export = _load_export_provenance(export_provenance_path.expanduser().resolve())
    _validate_exported_files(export, export_provenance_path.expanduser().resolve().parent)

    expected = {target: f"rwkv4anki-{version}-{target}.ankiaddon" for target in SUPPORTED_TARGETS}
    actual_entries = sorted(assets_dir.iterdir(), key=lambda path: path.name)
    actual_names = {
        path.name for path in actual_entries if path.is_file() and not path.is_symlink()
    }
    expected_names = set(expected.values())
    if actual_names != expected_names or len(actual_entries) != len(expected_names):
        missing = sorted(expected_names - actual_names)
        extra = sorted(path.name for path in actual_entries if path.name not in expected_names)
        raise ReleaseAssemblyError(
            f"release assets must contain exactly the six expected bundles; "
            f"missing={missing}, extra={extra}"
        )

    records = [
        _inspect_bundle(
            assets_dir / expected[target],
            target=target,
            version=version,
            release_lock=release_lock,
            model_lock=model_lock,
        )
        for target in SUPPORTED_TARGETS
    ]
    backend_commits = {record.backend_commit for record in records}
    backend_versions = {record.backend_version for record in records}
    if len(backend_commits) != 1 or len(backend_versions) != 1:
        raise ReleaseAssemblyError(
            "release bundles disagree on the RWKV-SRS source commit or package version"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checksums_path = output_dir / CHECKSUMS_NAME
    provenance_path = output_dir / RELEASE_PROVENANCE_NAME
    checksums = "".join(
        f"{record.sha256}  {record.filename}\n"
        for record in sorted(records, key=lambda record: record.filename)
    )
    _atomic_write(checksums_path, checksums.encode("utf-8"))

    assert release_lock.release_manifest is not None
    provenance = {
        "addon": {
            "package": ADDON_PACKAGE,
            "version": version,
        },
        "artifacts": [
            {
                "filename": record.filename,
                "sha256": record.sha256,
                "size": record.size,
                "target": record.target,
                "wheel": {
                    "filename": record.wheel_name,
                    "sha256": record.wheel_sha256,
                },
            }
            for record in records
        ],
        "models": {
            "assets": [
                {
                    "filename": model.name,
                    "sha256": model.sha256,
                    "url": model.url,
                }
                for model in model_lock.models
            ],
            "lock_sha256": model_lock.sha256,
        },
        "rwkv_srs": {
            "package_version": records[0].backend_version,
            "pgo": False,
            "release_manifest": {
                "filename": release_lock.release_manifest.name,
                "sha256": release_lock.release_manifest.sha256,
            },
            "release_tag": release_lock.release_tag,
            "repository": release_lock.repository,
            "source_commit": records[0].backend_commit,
        },
        "schema_version": 1,
        "source": {
            "private_commit": export["source_commit"],
            "public_commit": public_commit,
            "public_export_provenance_sha256": _sha256(
                export_provenance_path.expanduser().resolve()
            ),
            "source_repository": export["source_repository"],
        },
    }
    _atomic_write(
        provenance_path,
        (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {"checksums": checksums_path, "provenance": provenance_path}


def _load_export_provenance(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReleaseAssemblyError(f"public export provenance does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ReleaseAssemblyError(f"invalid public export provenance: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ReleaseAssemblyError("unsupported public export provenance schema")
    if not isinstance(raw.get("source_repository"), str) or not raw["source_repository"]:
        raise ReleaseAssemblyError("public export provenance has no source repository")
    if (
        not isinstance(raw.get("source_commit"), str)
        or COMMIT_PATTERN.fullmatch(raw["source_commit"]) is None
    ):
        raise ReleaseAssemblyError("public export provenance has an invalid source commit")
    if not isinstance(raw.get("files"), dict) or not raw["files"]:
        raise ReleaseAssemblyError("public export provenance has no file records")
    return raw


def _validate_exported_files(provenance: dict[str, Any], root: Path) -> None:
    casefolded: set[str] = set()
    for relative, record in provenance["files"].items():
        if not isinstance(relative, str):
            raise ReleaseAssemblyError("public export provenance contains a non-string path")
        path = _safe_relative_path(relative, label="public export path")
        key = path.as_posix().casefold()
        if key in casefolded:
            raise ReleaseAssemblyError("public export provenance contains colliding paths")
        casefolded.add(key)
        if not isinstance(record, dict):
            raise ReleaseAssemblyError(f"invalid public export record: {relative}")
        expected_size = record.get("size")
        expected_hash = record.get("sha256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ReleaseAssemblyError(f"invalid public export size: {relative}")
        if not isinstance(expected_hash, str) or SHA256_PATTERN.fullmatch(expected_hash) is None:
            raise ReleaseAssemblyError(f"invalid public export checksum: {relative}")
        current = root.joinpath(*path.parts)
        if current.is_symlink() or not current.is_file():
            raise ReleaseAssemblyError(f"public export file is missing or linked: {relative}")
        if current.stat().st_size != expected_size or _sha256(current) != expected_hash:
            raise ReleaseAssemblyError(f"public export file changed after export: {relative}")


def _inspect_bundle(
    archive_path: Path,
    *,
    target: str,
    version: str,
    release_lock: ReleaseLock,
    model_lock: ModelLock,
) -> BundleRecord:
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ReleaseAssemblyError(f"release bundle is missing or linked: {archive_path.name}")
    with tempfile.TemporaryDirectory(prefix=f"rwkv4anki-{target}-") as directory:
        stage = Path(directory)
        _extract_bundle(archive_path, stage)
        errors = validate_stage(stage, require_vendor=True)
        if errors:
            raise ReleaseAssemblyError(f"invalid {target} add-on bundle:\n- " + "\n- ".join(errors))
        addon_manifest = _read_json(stage / "manifest.json", label="add-on manifest")
        if addon_manifest.get("package") != ADDON_PACKAGE:
            raise ReleaseAssemblyError(
                f"{target} add-on manifest does not identify package {ADDON_PACKAGE}"
            )
        if addon_manifest.get("human_version") != version:
            raise ReleaseAssemblyError(
                f"{target} add-on manifest version does not match the release version"
            )
        vendor = _read_json(
            stage / "vendor" / "RWKV_SRS_VENDOR.json",
            label=f"{target} vendor manifest",
        )
        _validate_vendor_release(
            vendor,
            target=target,
            release_lock=release_lock,
            model_lock=model_lock,
        )

    wheel = vendor["wheel"]
    return BundleRecord(
        target=target,
        filename=archive_path.name,
        sha256=_sha256(archive_path),
        size=archive_path.stat().st_size,
        backend_commit=vendor["backend_commit"],
        backend_version=vendor["rwkv_srs_version"],
        wheel_name=wheel["name"],
        wheel_sha256=wheel["sha256"],
    )


def _extract_bundle(archive_path: Path, stage: Path) -> None:
    exact: set[str] = set()
    casefolded: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            path = _safe_relative_path(info.filename, label="add-on archive member")
            if info.filename in exact or path.as_posix().casefold() in casefolded:
                raise ReleaseAssemblyError(
                    f"add-on archive contains a duplicate or colliding member: {info.filename}"
                )
            exact.add(info.filename)
            casefolded.add(path.as_posix().casefold())
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ReleaseAssemblyError(
                    f"add-on archive contains a symbolic link: {info.filename}"
                )
            if info.flag_bits & 0x1:
                raise ReleaseAssemblyError(
                    f"add-on archive contains an encrypted member: {info.filename}"
                )
            if info.is_dir():
                continue
            destination = stage.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _validate_vendor_release(
    vendor: dict[str, Any],
    *,
    target: str,
    release_lock: ReleaseLock,
    model_lock: ModelLock,
) -> None:
    expected_wheel = release_lock.targets[target]
    assert release_lock.release_manifest is not None
    expected = {
        "backend_ref": release_lock.release_tag,
        "backend_repo": release_lock.repository,
        "model_lock_sha256": model_lock.sha256,
        "model_source": "fixed-download",
        "pgo": False,
        "platform": target,
        "release_manifest_sha256": release_lock.release_manifest.sha256,
        "source_kind": "official-release",
    }
    for field, value in expected.items():
        if vendor.get(field) != value:
            raise ReleaseAssemblyError(
                f"{target} vendor manifest {field} does not match the release lock"
            )
    backend_commit = vendor.get("backend_commit")
    backend_version = vendor.get("rwkv_srs_version")
    if not isinstance(backend_commit, str) or COMMIT_PATTERN.fullmatch(backend_commit) is None:
        raise ReleaseAssemblyError(f"{target} vendor manifest has an invalid backend commit")
    if not isinstance(backend_version, str) or VERSION_PATTERN.fullmatch(backend_version) is None:
        raise ReleaseAssemblyError(f"{target} vendor manifest has an invalid package version")
    wheel = vendor.get("wheel")
    if wheel != {"name": expected_wheel.name, "sha256": expected_wheel.sha256}:
        raise ReleaseAssemblyError(f"{target} vendor wheel does not match the release lock")

    models = vendor.get("models")
    if not isinstance(models, list):
        raise ReleaseAssemblyError(f"{target} vendor manifest has no model records")
    actual_models: dict[str, tuple[str, str]] = {}
    for record in models:
        if not isinstance(record, dict):
            raise ReleaseAssemblyError(f"{target} vendor manifest has an invalid model record")
        path = record.get("path")
        digest = record.get("sha256")
        url = record.get("url")
        if not isinstance(path, str) or not isinstance(digest, str) or not isinstance(url, str):
            raise ReleaseAssemblyError(f"{target} vendor manifest has an invalid model record")
        name = PurePosixPath(path).name
        if name in actual_models:
            raise ReleaseAssemblyError(f"{target} vendor manifest has duplicate models")
        actual_models[name] = (digest, url)
    expected_models = {model.name: (model.sha256, model.url) for model in model_lock.models}
    if actual_models != expected_models:
        raise ReleaseAssemblyError(f"{target} vendor models do not match the model lock")


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ReleaseAssemblyError(f"unsafe {label}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ReleaseAssemblyError(f"unsafe {label}: {value!r}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReleaseAssemblyError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseAssemblyError(f"{label} must be a JSON object")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
