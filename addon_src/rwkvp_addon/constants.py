from __future__ import annotations

import os
from pathlib import Path

ADDON_ROOT = Path(__file__).resolve().parents[1]
DEV_VENDOR_ROOT_ENV_VAR = "RWKV4ANKI_DEV_VENDOR_ROOT"


def _configured_vendor_root() -> Path:
    override = os.environ.get(DEV_VENDOR_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return ADDON_ROOT / "vendor"


VENDOR_ROOT = _configured_vendor_root()
VENDOR_RUNTIME_ROOT = ADDON_ROOT / "vendor_runtime"
USER_FILES_ROOT = ADDON_ROOT / "user_files"
ADDON_PACKAGE = "RWKV4Anki"
ADDON_NAME = "RWKV4Anki"
DEFAULT_MODEL_ID = "RWKV_trained_on_101_4999"
RWKV_SRS_TORCH_SEED = 12345
CHECKPOINT_SAVE_INTERVAL = 1000
PREDICT_BATCH_SIZE = 192
# Anki currently retains at most 30 undoable operations. Keep the Rust-side
# live-review undo queue comfortably above that fixed application limit.
LIVE_REVIEW_RUST_UNDO_LIMIT = 64
