"""Đặc trưng thủ công cho M1, tính CHỈ từ log-mel đã cache (không giải mã lại audio).

Khác thiết kế Phase 0 (có chroma, spectral contrast, ZCR): các đặc trưng đó cần audio/STFT.
Để M1 nhanh và dùng cùng đầu vào với M3, M1 dùng MFCC + delta + thống kê dải mel.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct


def feature_dim(n_mels: int = 128, n_mfcc: int = 20) -> int:
    return n_mfcc * 4 + n_mels * 2


def mel_stats_features(mel: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """mel: log-mel dB (n_mels, n_frames), n_frames >= 2 -> vector float32.
    Gồm mean/std của MFCC, delta-MFCC, và mean/std từng dải mel."""
    if mel.ndim != 2 or mel.shape[1] < 2:
        raise ValueError("mel phải có shape (n_mels, n_frames) với n_frames >= 2")
    if n_mfcc > mel.shape[0]:
        raise ValueError("n_mfcc không được lớn hơn n_mels")
    mfcc = dct(mel.astype(np.float64), type=2, axis=0, norm="ortho")[:n_mfcc]
    delta = np.gradient(mfcc, axis=1)
    parts = [mfcc.mean(1), mfcc.std(1), delta.mean(1), delta.std(1), mel.mean(1), mel.std(1)]
    return np.concatenate(parts).astype(np.float32)


def window_starts(n_frames: int, win: int, hop: int) -> list[int]:
    """Vị trí bắt đầu (khung) của các cửa sổ trượt; luôn có ít nhất 1 cửa sổ nếu n_frames >= win."""
    if win < 2 or hop < 1:
        raise ValueError("win >= 2 và hop >= 1")
    if n_frames < win:
        raise ValueError(f"n_frames={n_frames} < win={win}")
    return list(range(0, n_frames - win + 1, hop))
