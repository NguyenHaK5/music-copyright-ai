"""Kiểm tra chất lượng từng file audio (giải mã thử, độ dài, im lặng)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.logmel import load_audio


def fma_audio_path(audio_dir: str | Path, track_id: int) -> Path:
    """Quy ước FMA: <audio_dir>/<3 chữ số đầu>/<track_id 6 chữ số>.mp3."""
    tid = f"{int(track_id):06d}"
    return Path(audio_dir) / tid[:3] / f"{tid}.mp3"


def check_audio(path: str | Path, sr: int, min_seconds: float, max_seconds: float,
                silence_peak: float) -> tuple[str | None, float, np.ndarray | None]:
    """Trả (drop_reason, duration_s, y). drop_reason=None nghĩa là file đạt.

    Lý do: missing_file | decode_error | too_short | silent | nonfinite.
    y chỉ có khi đạt; đã cắt tối đa max_seconds, CHƯA chuẩn hóa biên độ.
    """
    p = Path(path)
    if not p.exists():
        return "missing_file", 0.0, None
    try:
        y = load_audio(p, sr=sr, mono=True, max_seconds=max_seconds)
    except Exception:  # librosa/audioread ném nhiều loại lỗi khác nhau
        return "decode_error", 0.0, None
    dur = len(y) / sr
    if not np.all(np.isfinite(y)):
        return "nonfinite", dur, None
    if dur < min_seconds:
        return "too_short", dur, None
    if float(np.max(np.abs(y))) < silence_peak:
        return "silent", dur, None
    return None, dur, y
