"""Biến đổi tín hiệu cho bộ truy vấn robustness (miền audio, float32 mono)."""
from __future__ import annotations

import shutil
import warnings
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


def _power(y: np.ndarray) -> float:
    return float(np.mean(np.square(y.astype(np.float64))))


def add_noise_snr(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Cộng nhiễu trắng Gauss sao cho SNR = snr_db (công suất tín hiệu / công suất nhiễu)."""
    p = _power(y)
    if p <= 0:
        raise ValueError("Tín hiệu im lặng, không xác định được SNR")
    noise = rng.standard_normal(y.shape[0])
    noise *= np.sqrt(p / (10 ** (snr_db / 10)) / _power(noise))
    return (y + noise).astype(np.float32)


def lowpass(y: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    return sosfilt(butter(order, cutoff_hz, "low", fs=sr, output="sos"), y).astype(np.float32)


def highpass(y: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    return sosfilt(butter(order, cutoff_hz, "high", fs=sr, output="sos"), y).astype(np.float32)


def overlay_snr(y: np.ndarray, other: np.ndarray, snr_db: float) -> np.ndarray:
    """Trộn `other` (lặp/cắt cho đủ dài) vào y sao cho công suất y / công suất other = snr_db."""
    if other.size == 0 or _power(other) <= 0 or _power(y) <= 0:
        raise ValueError("Tín hiệu trộn hoặc tín hiệu gốc im lặng")
    reps = int(np.ceil(y.size / other.size))
    o = np.tile(other, reps)[: y.size].astype(np.float64)
    o *= np.sqrt(_power(y) / (10 ** (snr_db / 10)) / _power(o))
    return (y + o).astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """rate > 1: nhanh hơn (ngắn hơn); rate < 1: chậm hơn."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # cảnh báo tham số cũ trong librosa 1.0
        return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def mp3_roundtrip(y: np.ndarray, sr: int, bitrate_kbps: int) -> np.ndarray:
    """Nén MP3 ở bitrate_kbps rồi giải nén về sr (cần ffmpeg trong PATH)."""
    if not ffmpeg_available():
        raise RuntimeError("Không tìm thấy ffmpeg trong PATH")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        sf.write(d / "in.wav", y, sr)
        base = ["ffmpeg", "-y", "-loglevel", "error"]
        subprocess.run(base + ["-i", str(d / "in.wav"), "-b:a", f"{bitrate_kbps}k", str(d / "x.mp3")],
                       check=True)
        subprocess.run(base + ["-i", str(d / "x.mp3"), "-ar", str(sr), "-ac", "1", str(d / "out.wav")],
                       check=True)
        out, _ = sf.read(d / "out.wav", dtype="float32")
    return out if out.ndim == 1 else out.mean(1)
