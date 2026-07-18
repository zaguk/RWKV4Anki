from __future__ import annotations

PREDICT_MANY_ORACLE_MODE = "oracle"
PREDICT_MANY_FAST_MODE = "fast"
PREDICT_MANY_GPU_MODE = "gpu"
PREDICT_MANY_MODES = (
    PREDICT_MANY_GPU_MODE,
    PREDICT_MANY_FAST_MODE,
    PREDICT_MANY_ORACLE_MODE,
)
DEFAULT_PREDICT_MANY_MODE = PREDICT_MANY_FAST_MODE
_LEGACY_PREDICT_MANY_LIGHTNING_MODE = "lightning"

# These mirror rwkv-srs's public automatic policies. They are used only to
# choose responsive outer progress chunks; passing batch_size=None still lets
# rwkv-srs own the native batching policy itself.
ORACLE_PREDICTION_PROGRESS_CHUNK_SIZE = 192
FAST_PREDICTION_PROGRESS_CHUNK_SIZE = 1_536
GPU_PREDICTION_PROGRESS_CHUNK_SIZE = 8_192

PROCESS_MANY_FAST_MODE = "fast"
PROCESS_MANY_GPU_MODE = "gpu"
PROCESS_MANY_MODES = (
    PROCESS_MANY_GPU_MODE,
    PROCESS_MANY_FAST_MODE,
)
DEFAULT_PROCESS_MANY_MODE = PROCESS_MANY_FAST_MODE
_LEGACY_PROCESS_MANY_ORACLE_MODE = "oracle"


def normalize_predict_many_mode(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == _LEGACY_PREDICT_MANY_LIGHTNING_MODE:
        return PREDICT_MANY_FAST_MODE
    return _normalize_mode(normalized, PREDICT_MANY_MODES, DEFAULT_PREDICT_MANY_MODE)


def normalize_process_many_mode(value: object) -> str:
    # Oracle was previously exposed as a State Building mode. Existing profile
    # settings migrate to Fast so every add-on-owned processing runtime shares
    # one CPU profile and can be benchmarked safely inside Anki's process.
    if str(value or "").strip().lower() == _LEGACY_PROCESS_MANY_ORACLE_MODE:
        return PROCESS_MANY_FAST_MODE
    return _normalize_mode(value, PROCESS_MANY_MODES, DEFAULT_PROCESS_MANY_MODE)


def predict_many_uses_fast(mode: object) -> bool:
    return normalize_predict_many_mode(mode) == PREDICT_MANY_FAST_MODE


def predict_many_uses_gpu(mode: object) -> bool:
    return normalize_predict_many_mode(mode) == PREDICT_MANY_GPU_MODE


def available_predict_many_modes(*, gpu_available: bool) -> tuple[str, ...]:
    if gpu_available:
        return PREDICT_MANY_MODES
    return tuple(mode for mode in PREDICT_MANY_MODES if mode != PREDICT_MANY_GPU_MODE)


def available_process_many_modes(*, gpu_available: bool) -> tuple[str, ...]:
    if gpu_available:
        return PROCESS_MANY_MODES
    return tuple(mode for mode in PROCESS_MANY_MODES if mode != PROCESS_MANY_GPU_MODE)


def prediction_progress_chunk_size(
    mode: object,
    batch_size_override: int | None = None,
) -> int:
    normalized = normalize_predict_many_mode(mode)
    override = _positive_optional_int(batch_size_override)
    if normalized == PREDICT_MANY_GPU_MODE:
        return override or GPU_PREDICTION_PROGRESS_CHUNK_SIZE
    if normalized == PREDICT_MANY_FAST_MODE:
        if override is None:
            return FAST_PREDICTION_PROGRESS_CHUNK_SIZE
        # Give the native fast route multiple batches to distribute while
        # retaining bounded progress/cancellation latency.
        return max(
            ORACLE_PREDICTION_PROGRESS_CHUNK_SIZE,
            min(GPU_PREDICTION_PROGRESS_CHUNK_SIZE, override * 16),
        )
    return override or ORACLE_PREDICTION_PROGRESS_CHUNK_SIZE


def cpu_mode_for_process_many_mode(mode: object) -> str:
    normalize_process_many_mode(mode)
    return PROCESS_MANY_FAST_MODE


def process_many_uses_gpu(mode: object) -> bool:
    return normalize_process_many_mode(mode) == PROCESS_MANY_GPU_MODE


def _normalize_mode(value: object, choices: tuple[str, ...], default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in choices else default


def _positive_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
