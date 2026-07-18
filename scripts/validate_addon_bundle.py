#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from pathlib import Path, PurePosixPath

FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "tests",
    "vendor_runtime",
    "profiles",
    "native-v1",
}
FORBIDDEN_FILE_NAMES = {
    "meta.json",
    "coverage.xml",
    "layout.txt",
    "RWKV_P_VENDOR.json",
    ".rwkv4anki-native-cache.json",
}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pth",
    ".bin",
    ".sqlite",
    ".db",
    ".jsonl",
}
REQUIRED_ADDON_PATHS = (
    "__init__.py",
    "manifest.json",
    "config.json",
    "rwkvp_addon",
    "rwkvp_addon/modal_components.js",
    "rwkvp_addon/modal_styles.css",
    "user_files/README.txt",
)
REQUIRED_VENDOR_PATHS = (
    "vendor/rwkv_srs",
    "vendor/RWKV_SRS_VENDOR.json",
)
REQUIRED_UI_ASSET_SENTINELS = {
    "rwkvp_addon/modal_components.js": "window.RWKVModal",
    "rwkvp_addon/modal_styles.css": ".rwkv-modal-shell",
}
NATIVE_SUFFIXES = (".abi3.so", ".so", ".pyd", ".dll", ".dylib")
SHA256_LENGTH = 64
COMMIT_LENGTH = 40
SUPPORTED_TARGETS = {
    "linux-x86_64",
    "linux-aarch64",
    "macos-x86_64",
    "macos-aarch64",
    "windows-x86_64",
    "windows-aarch64",
}
ADDON_PACKAGE = "RWKV4Anki"
MIN_ANKI_POINT_VERSION = 260500
TESTED_ANKI_POINT_VERSION = 260500
HUMAN_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a staged RWKV4Anki addon directory before zipping."
    )
    parser.add_argument("stage", type=Path, help="Directory containing staged addon files.")
    parser.add_argument(
        "--require-vendor",
        action="store_true",
        help="Require staged RWKV-SRS vendor packages and native extension.",
    )
    parser.add_argument(
        "--require-pgo",
        action="store_true",
        help="Require RWKV_SRS_VENDOR.json to identify a PGO-built native extension.",
    )
    args = parser.parse_args()

    errors = validate_stage(
        args.stage,
        require_vendor=args.require_vendor,
        require_pgo=args.require_pgo,
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"validated addon stage: {args.stage}")
    return 0


def validate_stage(
    stage: Path,
    *,
    require_vendor: bool = False,
    require_pgo: bool = False,
) -> list[str]:
    errors: list[str] = []
    stage = stage.resolve()
    if not stage.exists() or not stage.is_dir():
        return [f"stage directory does not exist: {stage}"]

    for relative in REQUIRED_ADDON_PATHS:
        if not (stage / relative).exists():
            errors.append(f"missing required addon path: {relative}")

    manifest_path = stage / "manifest.json"
    if manifest_path.is_file():
        errors.extend(_validate_addon_manifest(manifest_path))

    for relative, sentinel in REQUIRED_UI_ASSET_SENTINELS.items():
        path = stage / relative
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8") if path.is_file() else ""
        except OSError:
            content = ""
        if sentinel not in content:
            errors.append(f"missing or invalid shared UI asset: {relative}")

    for path in stage.rglob("*"):
        relative = path.relative_to(stage).as_posix()
        if path.is_dir():
            if path.name in FORBIDDEN_DIR_NAMES:
                errors.append(f"forbidden directory in bundle: {relative}")
            if relative == "vendor/rwkv_p":
                errors.append("stale RWKV-P vendor package found: vendor/rwkv_p")
            if relative == "vendor/srs_metrics":
                errors.append("obsolete external metrics package found: vendor/srs_metrics")
            continue

        if path.name in FORBIDDEN_FILE_NAMES:
            errors.append(f"forbidden file in bundle: {relative}")
        if path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden generated/cache file in bundle: {relative}")
        if "rwkv_p_cpu_rs" in path.name:
            errors.append(f"stale native distribution name in bundle: {relative}")

    if require_vendor:
        for relative in REQUIRED_VENDOR_PATHS:
            if not (stage / relative).exists():
                errors.append(f"missing required vendor path: {relative}")
        native_dir = stage / "vendor" / "rwkv_srs"
        native_files = [
            path
            for path in native_dir.glob("_native*")
            if path.is_file() and path.name.endswith(NATIVE_SUFFIXES)
        ]
        if not native_files:
            errors.append("missing RWKV-SRS native extension under vendor/rwkv_srs/")
        elif len(native_files) > 1:
            errors.append("multiple RWKV-SRS native extensions found under vendor/rwkv_srs/")
        model_dir = stage / "vendor" / "rwkv_srs" / "pretrained"
        safetensors_files = list(model_dir.glob("*.safetensors")) if model_dir.exists() else []
        if not safetensors_files:
            errors.append("missing RWKV-SRS safetensors models under vendor/rwkv_srs/pretrained/")
        manifest_path = stage / "vendor" / "RWKV_SRS_VENDOR.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("missing or invalid vendor manifest: vendor/RWKV_SRS_VENDOR.json")
        else:
            if not isinstance(manifest, dict):
                errors.append("vendor manifest must be a JSON object")
            elif manifest.get("schema_version") == 2:
                errors.extend(_validate_vendor_manifest_v2(stage / "vendor", manifest))

    if require_pgo:
        manifest_path = stage / "vendor" / "RWKV_SRS_VENDOR.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("missing or invalid PGO vendor manifest: vendor/RWKV_SRS_VENDOR.json")
        else:
            if not isinstance(manifest, dict) or manifest.get("pgo") is not True:
                errors.append("vendor native extension was not built with PGO")

    return errors


def _validate_addon_manifest(path: Path) -> list[str]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ["missing or invalid add-on manifest: manifest.json"]
    if not isinstance(manifest, dict):
        return ["add-on manifest must be a JSON object"]

    errors: list[str] = []
    if manifest.get("package") != ADDON_PACKAGE or manifest.get("name") != ADDON_PACKAGE:
        errors.append(f"add-on manifest must identify package and name as {ADDON_PACKAGE}")
    if manifest.get("min_point_version") != MIN_ANKI_POINT_VERSION:
        errors.append(f"add-on manifest min_point_version must be {MIN_ANKI_POINT_VERSION}")
    if manifest.get("max_point_version") != TESTED_ANKI_POINT_VERSION:
        errors.append(f"add-on manifest max_point_version must be {TESTED_ANKI_POINT_VERSION}")
    human_version = manifest.get("human_version")
    if not isinstance(human_version, str) or HUMAN_VERSION_PATTERN.fullmatch(human_version) is None:
        errors.append("add-on manifest human_version is missing or invalid")
    return errors


def _validate_vendor_manifest_v2(vendor_root: Path, manifest: dict[str, object]) -> list[str]:
    errors: list[str] = []
    source_kind = manifest.get("source_kind")
    if source_kind not in {
        "official-release",
        "local-wheel",
        "local-native",
    }:
        errors.append("vendor manifest has an invalid source_kind")
    if manifest.get("platform") not in SUPPORTED_TARGETS:
        errors.append("vendor manifest has an invalid platform")
    if manifest.get("model_source") not in {
        "fixed-download",
        "local-directory",
        "wheel",
    }:
        errors.append("vendor manifest has an invalid model_source")
    model_source = manifest.get("model_source")
    if model_source == "fixed-download" and not _is_sha256(manifest.get("model_lock_sha256")):
        errors.append("fixed-download model lock checksum is invalid")
    if source_kind == "official-release":
        if manifest.get("pgo") is not False:
            errors.append("official-release vendor manifest must record pgo=false")
        if not _nonempty_string(manifest.get("backend_repo")) or not _nonempty_string(
            manifest.get("backend_ref")
        ):
            errors.append("official-release vendor provenance is incomplete")
        if not _is_commit(manifest.get("backend_commit")):
            errors.append("official-release vendor source commit is invalid")
        if not _is_sha256(manifest.get("release_manifest_sha256")):
            errors.append("official-release provenance manifest checksum is invalid")
        if not _nonempty_string(manifest.get("rwkv_srs_version")):
            errors.append("official-release RWKV-SRS version is invalid")

    native_relative = _manifest_relative_path(manifest.get("native_module"))
    if native_relative is None:
        errors.append("vendor manifest has an invalid native_module path")
    else:
        native = vendor_root / native_relative
        actual_native = [
            path
            for path in (vendor_root / "rwkv_srs").glob("_native*")
            if path.is_file() and path.name.endswith(NATIVE_SUFFIXES)
        ]
        if (
            not native.is_file()
            or not native.name.startswith("_native")
            or not native.name.endswith(NATIVE_SUFFIXES)
            or actual_native != [native]
        ):
            errors.append("vendor manifest native_module does not exist")
        elif manifest.get("artifact_sha256") != _sha256(native):
            errors.append("vendor manifest native extension checksum does not match")

    wheel = manifest.get("wheel")
    if not isinstance(wheel, dict):
        errors.append("vendor manifest wheel provenance is missing")
    else:
        wheel_name = wheel.get("name")
        wheel_hash = wheel.get("sha256")
        if (
            not isinstance(wheel_name, str)
            or PurePosixPath(wheel_name).name != wheel_name
            or not wheel_name.endswith(".whl")
        ):
            errors.append("vendor manifest wheel name is invalid")
        if not _is_sha256(wheel_hash):
            errors.append("vendor manifest wheel checksum is invalid")

    models = manifest.get("models")
    model_paths: list[str] = []
    if not isinstance(models, list) or not models:
        errors.append("vendor manifest models are missing")
    else:
        for index, record in enumerate(models):
            if not isinstance(record, dict):
                errors.append(f"vendor manifest model {index} is invalid")
                continue
            relative = _manifest_relative_path(record.get("path"))
            if (
                relative is None
                or not relative.startswith("rwkv_srs/pretrained/")
                or not relative.endswith(".safetensors")
            ):
                errors.append(f"vendor manifest model {index} has an invalid path")
                continue
            model_paths.append(relative)
            model = vendor_root / relative
            if not model.is_file():
                errors.append(f"vendor manifest model does not exist: {relative}")
                continue
            if record.get("sha256") != _sha256(model):
                errors.append(f"vendor manifest model checksum does not match: {relative}")
            if record.get("size") != model.stat().st_size:
                errors.append(f"vendor manifest model size does not match: {relative}")
            if model_source == "fixed-download" and not _is_https_model_url(
                record.get("url"), PurePosixPath(relative).name
            ):
                errors.append(f"vendor manifest model URL is invalid: {relative}")
    if manifest.get("safetensors_models") != model_paths:
        errors.append("vendor manifest safetensors_models does not match model records")
    actual_models = sorted(
        path.relative_to(vendor_root).as_posix()
        for path in (vendor_root / "rwkv_srs" / "pretrained").glob("*.safetensors")
        if path.is_file()
    )
    if sorted(model_paths) != actual_models:
        errors.append("vendor manifest model records do not match staged models")

    notices = manifest.get("notices")
    if not isinstance(notices, list):
        errors.append("vendor manifest notices must be a list")
    else:
        for value in notices:
            relative = _manifest_relative_path(value)
            if relative is None or not (vendor_root / relative).is_file():
                errors.append(f"vendor manifest notice does not exist: {value!r}")
        if source_kind == "official-release" and not notices:
            errors.append("official-release vendor manifest has no third-party notices")
    return errors


def _manifest_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == COMMIT_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_https_model_url(value: object, name: str) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urllib.parse.urlsplit(value)
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and PurePosixPath(parsed.path).name == name
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
