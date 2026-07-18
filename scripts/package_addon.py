#!/usr/bin/env python3
"""Package RWKV4Anki from an official release, local wheel, or local native module."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

try:
    from scripts.validate_addon_bundle import (
        MIN_ANKI_POINT_VERSION,
        TESTED_ANKI_POINT_VERSION,
        validate_stage,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from validate_addon_bundle import (  # type: ignore[import-not-found,no-redef]
        MIN_ANKI_POINT_VERSION,
        TESTED_ANKI_POINT_VERSION,
        validate_stage,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADDON_SRC = PROJECT_ROOT / "addon_src"
BUILD_ROOT = PROJECT_ROOT / "build"
DIST_ROOT = PROJECT_ROOT / "dist"
DEFAULT_RELEASE_LOCK = PROJECT_ROOT / "rwkv-srs.lock.json"
DEFAULT_MODEL_LOCK = PROJECT_ROOT / "rwkv-models.lock.json"
DEFAULT_CACHE_ROOT = BUILD_ROOT / "downloads" / "rwkv-srs"
SUPPORTED_TARGETS = (
    "linux-x86_64",
    "linux-aarch64",
    "macos-x86_64",
    "macos-aarch64",
    "windows-x86_64",
    "windows-aarch64",
)
NATIVE_SUFFIXES = (".abi3.so", ".so", ".pyd", ".dll", ".dylib")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
ASSET_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
NOTICE_NAMES = {"THIRD_PARTY_LICENSES.txt", "THIRD_PARTY_NOTICES.md"}


class PackagingError(ValueError):
    """Raised when an add-on artifact cannot be assembled safely."""


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    sha256: str


@dataclass(frozen=True)
class ReleaseLock:
    repository: str
    release_tag: str
    release_manifest: ReleaseAsset | None
    targets: dict[str, ReleaseAsset]


@dataclass(frozen=True)
class ModelAsset:
    name: str
    url: str
    sha256: str


@dataclass(frozen=True)
class ModelLock:
    models: tuple[ModelAsset, ...]
    sha256: str


@dataclass(frozen=True)
class BackendArtifact:
    source_kind: str
    wheel: Path
    wheel_sha256: str
    repository: str | None = None
    release_tag: str | None = None
    release_manifest_sha256: str | None = None
    source_commit: str | None = None
    package_version: str | None = None
    native_override: Path | None = None


@dataclass(frozen=True)
class PackageRequest:
    version: str = "dev"
    target: str = ""
    release_lock: Path = DEFAULT_RELEASE_LOCK
    model_lock: Path = DEFAULT_MODEL_LOCK
    cache_dir: Path = DEFAULT_CACHE_ROOT
    wheel: Path | None = None
    native_extension: Path | None = None
    runtime_wheel: Path | None = None
    models_dir: Path | None = None
    output_dir: Path = DIST_ROOT
    stage_dir: Path | None = None
    offline: bool = False
    import_smoke: bool = True


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained RWKV4Anki .ankiaddon from one verified "
            "RWKV-SRS artifact source."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--release",
        action="store_true",
        help="Use the pinned official RWKV-SRS GitHub Release (default).",
    )
    source.add_argument(
        "--wheel",
        type=Path,
        help="Use a complete local rwkv_srs wheel.",
    )
    source.add_argument(
        "--native-extension",
        type=Path,
        help=(
            "Use a local _native extension over --runtime-wheel, or over the "
            "pinned official wheel when --runtime-wheel is omitted."
        ),
    )
    parser.add_argument(
        "--runtime-wheel",
        type=Path,
        help="Python runtime wheel used only with --native-extension.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        help=(
            "Use local .safetensors models instead of the fixed, hash-verified "
            "downloads declared by the model lock."
        ),
    )
    parser.add_argument("--version", default="dev", help="Version used in the output name.")
    parser.add_argument(
        "--platform",
        choices=SUPPORTED_TARGETS,
        default=None,
        help="Target platform. Defaults to the current machine.",
    )
    parser.add_argument(
        "--release-lock",
        type=Path,
        default=DEFAULT_RELEASE_LOCK,
        help=f"Pinned RWKV-SRS release lock. Default: {DEFAULT_RELEASE_LOCK}",
    )
    parser.add_argument(
        "--model-lock",
        type=Path,
        default=DEFAULT_MODEL_LOCK,
        help=f"Pinned stable-model download lock. Default: {DEFAULT_MODEL_LOCK}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Verified download cache. Default: {DEFAULT_CACHE_ROOT}",
    )
    parser.add_argument("--output-dir", type=Path, default=DIST_ROOT)
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Override the clean staging directory retained for inspection.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid downloads and require all official assets in the verified cache.",
    )
    parser.add_argument(
        "--skip-import-smoke",
        action="store_true",
        help="Development-only: do not import the staged native backend before zipping.",
    )
    return parser


def _source_kind(args: argparse.Namespace) -> str:
    if args.wheel is not None:
        return "local-wheel"
    if args.native_extension is not None:
        return "local-native"
    return "official-release"


def _request_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> PackageRequest:
    if args.runtime_wheel is not None and args.native_extension is None:
        parser.error("--runtime-wheel requires --native-extension")
    if not VERSION_PATTERN.fullmatch(args.version):
        parser.error("--version must contain only filename-safe version characters")
    target = args.platform or _default_platform_tag()
    if target not in SUPPORTED_TARGETS:
        parser.error(
            f"unsupported current platform {target!r}; pass one of: " + ", ".join(SUPPORTED_TARGETS)
        )
    return PackageRequest(
        version=args.version,
        target=target,
        release_lock=args.release_lock,
        model_lock=args.model_lock,
        cache_dir=args.cache_dir,
        wheel=args.wheel,
        native_extension=args.native_extension,
        runtime_wheel=args.runtime_wheel,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        stage_dir=args.stage_dir,
        offline=args.offline,
        import_smoke=not args.skip_import_smoke,
    )


def main() -> int:
    parser = _argument_parser()
    request = _request_from_args(parser, parser.parse_args())
    try:
        archive = package_addon(request)
    except (PackagingError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        raise SystemExit(f"packaging failed: {error}") from error
    print(archive)
    return 0


def package_addon(request: PackageRequest) -> Path:
    target = request.target or _default_platform_tag()
    if target not in SUPPORTED_TARGETS:
        raise PackagingError(f"unsupported target: {target!r}")
    if not VERSION_PATTERN.fullmatch(request.version):
        raise PackagingError(f"unsafe package version: {request.version!r}")

    stage = (
        request.stage_dir.expanduser().resolve()
        if request.stage_dir is not None
        else (BUILD_ROOT / "stage" / target / "addon").resolve()
    )
    output_dir = request.output_dir.expanduser().resolve()
    cache_dir = request.cache_dir.expanduser().resolve()
    lock_path = request.release_lock.expanduser().resolve()
    model_lock_path = request.model_lock.expanduser().resolve()

    lock: ReleaseLock | None = None
    source_kind = _request_source_kind(request)
    if source_kind == "official-release" or (
        source_kind == "local-native" and request.runtime_wheel is None
    ):
        lock = _load_release_lock(lock_path, require_ready=True)

    backend = _resolve_backend(
        request,
        source_kind=source_kind,
        target=target,
        lock=lock,
        cache_dir=cache_dir,
    )
    _verify_wheel_target(backend.wheel, target)
    _validate_output_paths(
        stage,
        output_dir=output_dir,
        cache_dir=cache_dir,
        inputs=(
            backend.wheel,
            backend.native_override,
            request.models_dir.expanduser().resolve() if request.models_dir else None,
            lock_path,
            model_lock_path,
        ),
    )
    if backend.native_override is not None:
        _verify_native_extension_target(backend.native_override, target)
    _prepare_stage(stage)
    _write_addon_release_metadata(stage, version=request.version)

    vendor_root = stage / "vendor"
    package_root = vendor_root / "rwkv_srs"
    notice_paths = _extract_runtime_package(backend.wheel, vendor_root)
    if backend.release_manifest_sha256 is not None and not notice_paths:
        raise PackagingError(
            "official RWKV-SRS wheel does not contain required third-party notices"
        )
    if backend.native_override is not None:
        _overlay_native_extension(package_root, backend.native_override)

    native_paths = _native_modules(package_root)
    if len(native_paths) != 1:
        found = ", ".join(path.name for path in native_paths) or "none"
        raise PackagingError(
            f"selected runtime must contain exactly one rwkv_srs native extension; found {found}"
        )
    _verify_native_extension_target(native_paths[0], target)

    model_records, model_source, model_lock_sha256 = _stage_models(
        package_root,
        models_dir=request.models_dir,
        model_lock_path=model_lock_path,
        cache_dir=cache_dir,
        offline=request.offline,
    )
    native_relative = native_paths[0].relative_to(vendor_root).as_posix()
    manifest = {
        "artifact_sha256": _sha256(native_paths[0]),
        "backend_name": "RWKV-SRS",
        "backend_ref": backend.release_tag,
        "backend_repo": backend.repository,
        "backend_commit": backend.source_commit,
        "included_packages": ["rwkv_srs"],
        "models": model_records,
        "model_source": model_source,
        "model_lock_sha256": model_lock_sha256,
        "native_module": native_relative,
        "notices": [path.relative_to(vendor_root).as_posix() for path in notice_paths],
        "pgo": False if backend.source_kind == "official-release" else None,
        "platform": target,
        "python_package": "rwkv_srs",
        "safetensors_models": [record["path"] for record in model_records],
        "schema_version": 2,
        "source_kind": backend.source_kind,
        "release_manifest_sha256": backend.release_manifest_sha256,
        "rwkv_srs_version": backend.package_version,
        "wheel": {
            "name": backend.wheel.name,
            "sha256": backend.wheel_sha256,
        },
    }
    (vendor_root / "RWKV_SRS_VENDOR.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    errors = validate_stage(stage, require_vendor=True)
    if errors:
        raise PackagingError("invalid staged add-on:\n- " + "\n- ".join(errors))
    if request.import_smoke:
        if target != _default_platform_tag():
            raise PackagingError(
                "native import smoke tests require a target-native machine; "
                "run on the target or pass --skip-import-smoke for development only"
            )
        _smoke_import(vendor_root)
        errors = validate_stage(stage, require_vendor=True)
        if errors:
            raise PackagingError(
                "native import smoke polluted or invalidated the staged add-on:\n- "
                + "\n- ".join(errors)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"rwkv4anki-{request.version}-{target}.ankiaddon"
    _zip_stage_contents(stage, archive)
    return archive


def _request_source_kind(request: PackageRequest) -> str:
    if request.wheel is not None and request.native_extension is not None:
        raise PackagingError("select only one of a local wheel or local native extension")
    if request.runtime_wheel is not None and request.native_extension is None:
        raise PackagingError("runtime_wheel requires native_extension")
    if request.wheel is not None:
        return "local-wheel"
    if request.native_extension is not None:
        return "local-native"
    return "official-release"


def _load_release_lock(path: Path, *, require_ready: bool) -> ReleaseLock:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PackagingError(f"RWKV-SRS release lock does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise PackagingError(f"invalid RWKV-SRS release lock {path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise PackagingError(f"unsupported RWKV-SRS release lock: {path}")
    status = raw.get("status")
    if require_ready and status != "ready":
        raise PackagingError(
            "the official RWKV-SRS release is not configured yet; use --wheel or "
            "--native-extension with local artifacts"
        )
    repository = raw.get("repository")
    release_tag = raw.get("release_tag")
    if status == "ready":
        if not isinstance(repository, str) or REPOSITORY_PATTERN.fullmatch(repository) is None:
            raise PackagingError("release lock has an invalid repository")
        if not isinstance(release_tag, str) or VERSION_PATTERN.fullmatch(release_tag) is None:
            raise PackagingError("release lock has an invalid release_tag")
    else:
        repository = repository if isinstance(repository, str) else ""
        release_tag = release_tag if isinstance(release_tag, str) else ""

    targets_raw = raw.get("targets", {})
    if not isinstance(targets_raw, dict):
        raise PackagingError("release lock targets must be an object")
    targets = {
        str(target): _parse_release_asset(value, label=f"target {target}")
        for target, value in targets_raw.items()
    }
    if status == "ready" and set(targets) != set(SUPPORTED_TARGETS):
        missing = sorted(set(SUPPORTED_TARGETS) - set(targets))
        extra = sorted(set(targets) - set(SUPPORTED_TARGETS))
        raise PackagingError(
            "ready release lock must define exactly the supported targets; "
            f"missing={missing}, extra={extra}"
        )
    release_manifest_raw = raw.get("release_manifest")
    release_manifest = (
        _parse_release_asset(release_manifest_raw, label="release_manifest")
        if release_manifest_raw is not None
        else None
    )
    if status == "ready" and release_manifest is None:
        raise PackagingError("ready release lock is missing release_manifest")
    declared_names = [asset.name for asset in targets.values()]
    if release_manifest is not None:
        declared_names.append(release_manifest.name)
    if len(declared_names) != len(set(declared_names)):
        raise PackagingError("release lock contains duplicate asset names")
    return ReleaseLock(
        repository=repository,
        release_tag=release_tag,
        release_manifest=release_manifest,
        targets=targets,
    )


def _parse_release_asset(value: object, *, label: str) -> ReleaseAsset:
    if not isinstance(value, dict):
        raise PackagingError(f"release lock {label} must be an object")
    name = value.get("asset")
    digest = value.get("sha256")
    if (
        not isinstance(name, str)
        or ASSET_NAME_PATTERN.fullmatch(name) is None
        or PurePosixPath(name).name != name
    ):
        raise PackagingError(f"release lock {label} has an unsafe asset name")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise PackagingError(f"release lock {label} has an invalid sha256")
    return ReleaseAsset(name=name, sha256=digest)


def _load_model_lock(path: Path) -> ModelLock:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise PackagingError(f"RWKV model lock does not exist: {path}") from error
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackagingError(f"invalid RWKV model lock {path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise PackagingError(f"unsupported RWKV model lock: {path}")
    models_raw = raw.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise PackagingError("RWKV model lock must declare at least one model")
    models = tuple(
        _parse_model_asset(value, label=f"model {index}") for index, value in enumerate(models_raw)
    )
    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise PackagingError("RWKV model lock contains duplicate model names")
    return ModelLock(models=models, sha256=hashlib.sha256(payload).hexdigest())


def _parse_model_asset(value: object, *, label: str) -> ModelAsset:
    if not isinstance(value, dict):
        raise PackagingError(f"RWKV model lock {label} must be an object")
    name = value.get("name")
    url = value.get("url")
    digest = value.get("sha256")
    if (
        not isinstance(name, str)
        or ASSET_NAME_PATTERN.fullmatch(name) is None
        or PurePosixPath(name).name != name
        or not name.endswith(".safetensors")
    ):
        raise PackagingError(f"RWKV model lock {label} has an unsafe model name")
    if not isinstance(url, str):
        raise PackagingError(f"RWKV model lock {label} has an invalid URL")
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or PurePosixPath(parsed.path).name != name
    ):
        raise PackagingError(f"RWKV model lock {label} has an invalid URL")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise PackagingError(f"RWKV model lock {label} has an invalid sha256")
    return ModelAsset(name=name, url=url, sha256=digest)


def _resolve_backend(
    request: PackageRequest,
    *,
    source_kind: str,
    target: str,
    lock: ReleaseLock | None,
    cache_dir: Path,
) -> BackendArtifact:
    if source_kind == "local-wheel":
        assert request.wheel is not None
        wheel = _require_local_file(request.wheel, label="wheel")
        return BackendArtifact(
            source_kind=source_kind,
            wheel=wheel,
            wheel_sha256=_sha256(wheel),
        )

    if source_kind == "local-native":
        assert request.native_extension is not None
        native = _require_native_extension(request.native_extension)
        release_record: dict[str, str] | None = None
        if request.runtime_wheel is not None:
            wheel = _require_local_file(request.runtime_wheel, label="runtime wheel")
            repository = None
            release_tag = None
        else:
            assert lock is not None
            wheel = _acquire_target_wheel(
                lock,
                target=target,
                cache_dir=cache_dir,
                offline=request.offline,
            )
            repository = lock.repository
            release_tag = lock.release_tag
            release_record = _verify_official_release(
                lock,
                target=target,
                wheel=wheel,
                cache_dir=cache_dir,
                offline=request.offline,
            )
        return BackendArtifact(
            source_kind=source_kind,
            wheel=wheel,
            wheel_sha256=_sha256(wheel),
            repository=repository,
            release_tag=release_tag,
            release_manifest_sha256=(
                lock.release_manifest.sha256
                if request.runtime_wheel is None and lock and lock.release_manifest
                else None
            ),
            source_commit=(
                release_record["source_commit"] if request.runtime_wheel is None else None
            ),
            package_version=(
                release_record["package_version"] if request.runtime_wheel is None else None
            ),
            native_override=native,
        )

    assert lock is not None
    wheel = _acquire_target_wheel(
        lock,
        target=target,
        cache_dir=cache_dir,
        offline=request.offline,
    )
    release_record = _verify_official_release(
        lock,
        target=target,
        wheel=wheel,
        cache_dir=cache_dir,
        offline=request.offline,
    )
    return BackendArtifact(
        source_kind=source_kind,
        wheel=wheel,
        wheel_sha256=_sha256(wheel),
        repository=lock.repository,
        release_tag=lock.release_tag,
        release_manifest_sha256=(lock.release_manifest.sha256 if lock.release_manifest else None),
        source_commit=release_record["source_commit"],
        package_version=release_record["package_version"],
    )


def _require_local_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PackagingError(f"{label} does not exist: {resolved}")
    return resolved


def _require_native_extension(path: Path) -> Path:
    resolved = _require_local_file(path, label="native extension")
    if not _is_native_name(resolved.name):
        raise PackagingError(
            "native extension must be named _native and use a supported extension suffix: "
            f"{resolved.name}"
        )
    return resolved


def _acquire_target_wheel(
    lock: ReleaseLock,
    *,
    target: str,
    cache_dir: Path,
    offline: bool,
) -> Path:
    try:
        asset = lock.targets[target]
    except KeyError as error:
        raise PackagingError(
            f"RWKV-SRS release {lock.release_tag} does not define target {target}"
        ) from error
    if not asset.name.endswith(".whl"):
        raise PackagingError(f"release target {target} is not a wheel: {asset.name}")
    return _acquire_release_asset(lock, asset, cache_dir=cache_dir, offline=offline)


def _verify_official_release(
    lock: ReleaseLock,
    *,
    target: str,
    wheel: Path,
    cache_dir: Path,
    offline: bool,
) -> dict[str, str]:
    if lock.release_manifest is None:
        raise PackagingError("ready release lock is missing release_manifest")
    manifest_path = _acquire_release_asset(
        lock,
        lock.release_manifest,
        cache_dir=cache_dir,
        offline=offline,
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PackagingError(f"invalid RWKV-SRS release provenance: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise PackagingError("unsupported RWKV-SRS release provenance schema")
    if manifest.get("release_tag") != lock.release_tag:
        raise PackagingError("RWKV-SRS release provenance tag does not match the lock")
    if manifest.get("published_artifact_policy") != "wheels-and-manifests-only":
        raise PackagingError("RWKV-SRS release provenance has an unexpected artifact policy")

    wheel_asset = lock.targets[target]
    records = manifest.get("wheels")
    if not isinstance(records, list):
        raise PackagingError("RWKV-SRS release provenance wheels must be a list")
    matches = []
    for record in records:
        if not isinstance(record, dict):
            continue
        artifact = record.get("artifact")
        if isinstance(artifact, dict) and artifact.get("filename") == wheel_asset.name:
            matches.append(record)
    if len(matches) != 1:
        raise PackagingError(
            f"RWKV-SRS release provenance does not uniquely identify {wheel_asset.name}"
        )
    record = matches[0]
    artifact = _required_manifest_object(record, "artifact")
    target_record = _required_manifest_object(record, "target")
    package = _required_manifest_object(record, "package")
    build = _required_manifest_object(record, "build")
    source = _required_manifest_object(record, "source")
    wheel_hash = _sha256(wheel)
    if (
        artifact.get("type") != "wheel"
        or artifact.get("sha256") != wheel_hash
        or artifact.get("sha256") != wheel_asset.sha256
        or artifact.get("size") != wheel.stat().st_size
    ):
        raise PackagingError("RWKV-SRS release provenance does not match the selected wheel")
    if target_record.get("id") != target:
        raise PackagingError("RWKV-SRS release provenance target does not match the package target")
    package_version = package.get("version")
    if (
        package.get("distribution") != "rwkv-srs"
        or not isinstance(package_version, str)
        or VERSION_PATTERN.fullmatch(package_version) is None
    ):
        raise PackagingError("RWKV-SRS release provenance has invalid package metadata")
    if record.get("schema_version") != 1:
        raise PackagingError("unsupported RWKV-SRS wheel provenance schema")
    if (
        build.get("cargo_profile") != "release-ci"
        or build.get("cpu_tuning") != "portable"
        or build.get("pgo") is not False
    ):
        raise PackagingError("RWKV-SRS official wheel is not a portable non-PGO build")
    source_commit = source.get("commit")
    if not isinstance(source_commit, str) or COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise PackagingError("RWKV-SRS release provenance has an invalid source commit")
    if manifest.get("source_commit") != source_commit:
        raise PackagingError("RWKV-SRS aggregate and wheel source commits disagree")
    if manifest.get("package_version") != package_version:
        raise PackagingError("RWKV-SRS aggregate and wheel package versions disagree")
    return {"package_version": package_version, "source_commit": source_commit}


def _required_manifest_object(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise PackagingError(f"RWKV-SRS release provenance field {field!r} is invalid")
    return value


def _release_asset_url(lock: ReleaseLock, asset: ReleaseAsset) -> str:
    repository = "/".join(urllib.parse.quote(part, safe="") for part in lock.repository.split("/"))
    tag = urllib.parse.quote(lock.release_tag, safe="")
    name = urllib.parse.quote(asset.name, safe="")
    return f"https://github.com/{repository}/releases/download/{tag}/{name}"


def _acquire_release_asset(
    lock: ReleaseLock,
    asset: ReleaseAsset,
    *,
    cache_dir: Path,
    offline: bool,
) -> Path:
    destination = cache_dir / lock.release_tag / asset.name
    return _acquire_verified_url(
        url=_release_asset_url(lock, asset),
        destination=destination,
        expected_sha256=asset.sha256,
        offline=offline,
        label=f"release asset {asset.name}",
    )


def _acquire_model_asset(
    asset: ModelAsset,
    *,
    cache_dir: Path,
    offline: bool,
) -> Path:
    destination = cache_dir / "models" / asset.sha256 / asset.name
    return _acquire_verified_url(
        url=asset.url,
        destination=destination,
        expected_sha256=asset.sha256,
        offline=offline,
        label=f"model {asset.name}",
    )


def _acquire_verified_url(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    offline: bool,
    label: str,
) -> Path:
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return destination.resolve()
    if offline:
        raise PackagingError(f"verified {label} is not available offline: {destination}")
    if destination.exists():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "RWKV4Anki-packager/1"})
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output, length=1024 * 1024)
        except (urllib.error.URLError, TimeoutError) as error:
            raise PackagingError(f"could not download {url}: {error}") from error
        actual = _sha256(temporary)
        if actual != expected_sha256:
            raise PackagingError(
                f"checksum mismatch for {label}: expected {expected_sha256}, got {actual}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def _prepare_stage(stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(ADDON_SRC, stage, ignore=_ignore_addon_source)


def _write_addon_release_metadata(stage: Path, *, version: str) -> None:
    manifest_path = stage / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackagingError(f"invalid add-on manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise PackagingError("add-on manifest must be a JSON object")
    manifest["human_version"] = version
    manifest["min_point_version"] = MIN_ANKI_POINT_VERSION
    manifest["max_point_version"] = TESTED_ANKI_POINT_VERSION
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _paths_overlap(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _validate_output_paths(
    stage: Path,
    *,
    output_dir: Path,
    cache_dir: Path,
    inputs: tuple[Path | None, ...],
) -> None:
    project_root = PROJECT_ROOT.resolve()
    build_root = BUILD_ROOT.resolve()
    stage = stage.resolve()
    output_dir = output_dir.resolve()
    cache_dir = cache_dir.resolve()
    if stage == project_root or project_root.is_relative_to(stage):
        raise PackagingError("staging directory must not be the project or its ancestor")
    if _is_within(stage, project_root) and not _is_within(stage, build_root):
        raise PackagingError(f"staging directories inside the project must be under {build_root}")
    if _paths_overlap(stage, output_dir):
        raise PackagingError("staging and output directories must not overlap")
    if _paths_overlap(stage, cache_dir):
        raise PackagingError("staging and download-cache directories must not overlap")
    for selected in inputs:
        if selected is None:
            continue
        selected = selected.resolve()
        destructive_overlap = (
            _paths_overlap(stage, selected) if selected.is_dir() else _is_within(selected, stage)
        )
        if destructive_overlap:
            raise PackagingError(f"staging cleanup would remove an input artifact: {selected}")


def _ignore_addon_source(_directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in {
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "profiles",
            "vendor",
            "vendor_runtime",
        }:
            ignored.add(name)
        if name.endswith((".pyc", ".pyo")):
            ignored.add(name)
    return ignored


def _safe_wheel_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise PackagingError(f"unsafe wheel member: {name!r}")
    return path


def _extract_runtime_package(wheel: Path, vendor_root: Path) -> list[Path]:
    if vendor_root.exists():
        shutil.rmtree(vendor_root)
    vendor_root.mkdir(parents=True)
    extracted: set[str] = set()
    destinations: set[str] = set()
    notices: list[Path] = []
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            path = _safe_wheel_member(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise PackagingError(f"wheel contains a symbolic link: {info.filename}")
            if info.flag_bits & 0x1:
                raise PackagingError(f"wheel contains an encrypted member: {info.filename}")
            if info.is_dir():
                continue
            if info.filename in extracted:
                raise PackagingError(f"wheel contains a duplicate member: {info.filename}")
            extracted.add(info.filename)

            destination: Path | None = None
            if path.parts and path.parts[0] == "rwkv_srs":
                destination = vendor_root.joinpath(*path.parts)
            elif _is_notice_member(path):
                dist_info = path.parts[0]
                relative = PurePosixPath(*path.parts[1:])
                destination = vendor_root / "licenses" / dist_info / relative.as_posix()
            if destination is None:
                continue
            destination_key = destination.relative_to(vendor_root).as_posix().casefold()
            if destination_key in destinations:
                raise PackagingError(
                    f"wheel members collide at one extracted path: {info.filename}"
                )
            destinations.add(destination_key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if path.parts[0] != "rwkv_srs":
                notices.append(destination)

    package_root = vendor_root / "rwkv_srs"
    if not (package_root / "__init__.py").is_file():
        raise PackagingError(f"wheel does not contain rwkv_srs/__init__.py: {wheel}")
    return sorted(notices)


def _is_notice_member(path: PurePosixPath) -> bool:
    if len(path.parts) < 2 or not path.parts[0].endswith(".dist-info"):
        return False
    return "licenses" in path.parts[1:] or path.name in NOTICE_NAMES


def _overlay_native_extension(package_root: Path, native_extension: Path) -> None:
    for existing in _native_modules(package_root):
        existing.unlink()
    shutil.copy2(native_extension, package_root / native_extension.name)


def _native_modules(package_root: Path) -> list[Path]:
    if not package_root.is_dir():
        return []
    return sorted(
        path for path in package_root.iterdir() if path.is_file() and _is_native_name(path.name)
    )


def _is_native_name(name: str) -> bool:
    return name.startswith("_native") and name.endswith(NATIVE_SUFFIXES)


def _stage_models(
    package_root: Path,
    *,
    models_dir: Path | None,
    model_lock_path: Path,
    cache_dir: Path,
    offline: bool,
) -> tuple[list[dict[str, object]], str, str | None]:
    destination = package_root / "pretrained"
    if models_dir is not None:
        source = models_dir.expanduser().resolve()
        if not source.is_dir():
            raise PackagingError(f"models directory does not exist: {source}")
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()
        copied = _copy_local_models(source, destination)
        return _model_records(copied, package_root.parent), "local-directory", None

    embedded = sorted(destination.glob("*.safetensors")) if destination.is_dir() else []
    if embedded:
        return _model_records(embedded, package_root.parent), "wheel", None

    model_lock = _load_model_lock(model_lock_path)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    source_urls: dict[str, str] = {}
    for asset in model_lock.models:
        downloaded = _acquire_model_asset(
            asset,
            cache_dir=cache_dir,
            offline=offline,
        )
        target = destination / asset.name
        shutil.copy2(downloaded, target)
        copied.append(target)
        source_urls[asset.name] = asset.url
    records = _model_records(copied, package_root.parent)
    for record in records:
        record["url"] = source_urls[PurePosixPath(str(record["path"])).name]
    return records, "fixed-download", model_lock.sha256


def _copy_local_models(source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for model in sorted(source.glob("*.safetensors")):
        if model.is_symlink() or not model.is_file():
            raise PackagingError(f"model must be an ordinary file: {model}")
        target = destination / model.name
        shutil.copy2(model, target)
        copied.append(target)
    if not copied:
        raise PackagingError(f"no .safetensors model files found in {source}")
    return copied


def _model_records(models: list[Path], vendor_root: Path) -> list[dict[str, object]]:
    records = []
    for model in models:
        if model.stat().st_size == 0:
            raise PackagingError(f"model file is empty: {model}")
        records.append(
            {
                "path": model.relative_to(vendor_root).as_posix(),
                "sha256": _sha256(model),
                "size": model.stat().st_size,
            }
        )
    return records


def _verify_wheel_target(wheel: Path, target: str) -> None:
    name = wheel.name.lower()
    if not name.endswith(".whl") or "-abi3-" not in name:
        raise PackagingError(f"RWKV-SRS wheel must use the stable abi3 tag: {wheel.name}")
    markers = {
        "linux-x86_64": ("manylinux", "x86_64"),
        "linux-aarch64": ("manylinux", "aarch64"),
        "macos-x86_64": ("macosx", "x86_64"),
        "macos-aarch64": ("macosx", "arm64"),
        "windows-x86_64": ("win_amd64",),
        "windows-aarch64": ("win_arm64",),
    }
    if not all(marker in name for marker in markers[target]):
        raise PackagingError(f"wheel {wheel.name} does not match target {target}")


def _verify_native_extension_target(native: Path, target: str) -> None:
    name = native.name.lower()
    if target.startswith("windows-") and not name.endswith(".pyd"):
        raise PackagingError(f"Windows native extensions must use .pyd: {native.name}")
    if not target.startswith("windows-") and not name.endswith(".so"):
        raise PackagingError(f"Linux/macOS native extensions must use .so: {native.name}")


def _smoke_import(vendor_root: Path) -> None:
    code = (
        "import os, sys; "
        "sys.dont_write_bytecode = True; "
        f"sys.path.insert(0, {str(vendor_root)!r}); "
        "os.environ['RWKV_SRS_BACKEND'] = 'rust'; "
        "import rwkv_srs; "
        "from rwkv_srs.backends import rust; "
        "assert rwkv_srs.backend_name() == 'rust'; "
        "assert rust.__name__ == 'rwkv_srs.backends.rust'"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="rwkv4anki-import-smoke-") as directory:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise PackagingError(f"staged RWKV-SRS import smoke failed: {detail}")


def _zip_stage_contents(stage: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}.", suffix=".tmp", dir=archive.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as output:
            for path in sorted(stage.rglob("*")):
                if not path.is_file():
                    continue
                if path.is_symlink():
                    raise PackagingError(f"refusing symbolic link in add-on stage: {path}")
                relative = path.relative_to(stage).as_posix()
                info = zipfile.ZipInfo(relative, date_time=ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 & 0xFFFF) << 16
                output.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    system_name = {"darwin": "macos", "linux": "linux", "windows": "windows"}.get(system, system)
    machine_name = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }.get(machine, machine)
    return f"{system_name}-{machine_name}"


if __name__ == "__main__":
    raise SystemExit(main())
