"""Đọc audio và tính log-mel spectrogram."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(path: str | Path, sr: int = 22050, mono: bool = True,
               max_seconds: float | None = None) -> np.ndarray:
    """Đọc file audio -> mảng float32 1-D (mono) ở tần số sr. Ném lỗi nếu giải mã thất bại."""
    y, _ = librosa.load(str(path), sr=sr, mono=mono, duration=max_seconds)
    return y.astype(np.float32, copy=False)


def peak_normalize(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Chia cho biên độ đỉnh -> đỉnh = 1. Tín hiệu im lặng được giữ nguyên."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return y if peak < eps else (y / peak).astype(np.float32)


def logmel(y: np.ndarray, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512,
           n_mels: int = 128) -> np.ndarray:
    """Log-mel (dB, ref=1.0, amin=1e-10) -> float32, shape (n_mels, n_frames)."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_mels, power=2.0)
    return librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=None).astype(np.float32)
