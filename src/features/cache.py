"""Cache log-mel dạng .npy (float16) theo track_id - chạy lại an toàn."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def cache_path(cache_dir: str | Path, track_id: int) -> Path:
    return Path(cache_dir) / f"{int(track_id):06d}.npy"


def save_logmel(cache_dir: str | Path, track_id: int, mel: np.ndarray, dtype: str = "float16") -> Path:
    p = cache_path(cache_dir, track_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npy")
    np.save(tmp, mel.astype(dtype))
    tmp.replace(p)  # ghi nguyên tử: tiến trình bị ngắt không để lại file dở
    return p


def load_logmel(cache_dir: str | Path, track_id: int) -> np.ndarray:
    """Đọc cache -> float32 (n_mels, n_frames)."""
    return np.load(cache_path(cache_dir, track_id)).astype(np.float32)


def has_logmel(cache_dir: str | Path, track_id: int) -> bool:
    return cache_path(cache_dir, track_id).exists()
