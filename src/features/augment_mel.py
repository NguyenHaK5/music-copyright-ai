"""Augmentation trên miền log-mel (dB) cho M3 (Phase 3A).

Mọi hàm nhận mel dạng (n_mels, n_frames) float, giá trị dB công suất (10*log10), trả float32.
Đây là XẤP XỈ các biến đổi audio thật; đánh giá vẫn dùng biến đổi thật (bộ truy vấn 2D).
Không import gì từ repo để test độc lập.

Họ giữ ra khỏi huấn luyện (ISSUE-05): mix/overlay (chồng âm) và codec (nén MP3). MelAugmenter từ chối chúng.
"""
from __future__ import annotations

import math

import numpy as np

ALLOWED_OPS = ("time_warp", "freq_warp", "eq", "noise", "gain_ramp", "spec_mask")
HELD_OUT_OPS = ("mix", "overlay", "codec")
_EPS_POWER = 1e-20

DEFAULTS = {
    "time_warp": {"p": 0.0, "rate_min": 0.85, "rate_max": 1.15},
    "freq_warp": {"p": 0.0, "semitone_min": -2.5, "semitone_max": 2.5},
    "eq": {"p": 0.0, "lowpass_hz": [2000.0, 6000.0], "highpass_hz": [200.0, 1000.0], "max_atten_db": 60.0},
    "noise": {"p": 0.0, "snr_db_min": 3.0, "snr_db_max": 25.0, "mode": "additive", "coherent_k": 1},
    "gain_ramp": {"p": 0.0, "total_db_abs": 6.0},
    "spec_mask": {"p": 0.0, "n_freq": 2, "max_freq_width": 12, "n_time": 2, "max_time_width": 20},
}


def hz_to_mel(f):
    """Hz -> mel (công thức HTK). Xấp xỉ thang Slaney của librosa; đủ cho augmentation."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_bin_freqs(n_mels: int, sr: int) -> np.ndarray:
    """Tần số tâm (Hz) của n_mels dải, cách đều theo mel từ 0 đến sr/2."""
    top = float(hz_to_mel(sr / 2.0))
    return mel_to_hz(np.linspace(0.0, top, n_mels + 2)[1:-1])


def time_warp(mel: np.ndarray, rate: float, n_out: int) -> np.ndarray:
    """Khung ra t đọc khung nguồn t*rate (nội suy tuyến tính). rate>1 = nhanh hơn. Trả (n_mels, n_out)."""
    t_src = mel.shape[1]
    pos = np.clip(np.arange(n_out, dtype=np.float64) * rate, 0.0, t_src - 1.0)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, t_src - 1)
    w = (pos - i0).astype(np.float32)
    return (mel[:, i0] * (1.0 - w) + mel[:, i1] * w).astype(np.float32)


def freq_warp(mel: np.ndarray, semitones: float, sr: int) -> np.ndarray:
    """Dịch nội dung theo tỷ số tần số 2^(semitones/12) (xấp xỉ pitch shift giữ nguyên độ dài).

    Dải ra có tần số f đọc từ f/ratio. Vùng không có nguồn được điền bằng giá trị nhỏ nhất của crop.
    """
    n_mels = mel.shape[0]
    freqs = mel_bin_freqs(n_mels, sr)
    ratio = 2.0 ** (semitones / 12.0)
    m0 = float(hz_to_mel(freqs[0]))
    dm = float(hz_to_mel(freqs[1])) - m0
    idx = (hz_to_mel(freqs / ratio) - m0) / dm
    invalid = (idx < -1e-6) | (idx > n_mels - 1 + 1e-6)
    idxc = np.clip(idx, 0.0, n_mels - 1.0)
    i0 = np.floor(idxc).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_mels - 1)
    w = (idxc - i0).astype(np.float32)[:, None]
    out = mel[i0] * (1.0 - w) + mel[i1] * w
    out[invalid] = float(mel.min())
    return out.astype(np.float32)


def eq_gain_db(freqs: np.ndarray, kind: str, fc: float, order: int = 4, max_atten_db: float = 60.0) -> np.ndarray:
    """Độ lợi công suất (dB, <= 0) của bộ lọc Butterworth theo từng tần số, chặn dưới -max_atten_db."""
    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-6)
    if kind == "lowpass":
        g = -10.0 * np.log10(1.0 + (f / fc) ** (2 * order))
    elif kind == "highpass":
        g = -10.0 * np.log10(1.0 + (fc / f) ** (2 * order))
    else:
        raise ValueError(f"kind phải là lowpass/highpass, nhận {kind!r}")
    return np.maximum(g, -max_atten_db)


def add_noise(mel: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Cộng nhiễu trắng ở miền công suất: công suất nhiễu trung bình = công suất tín hiệu trung bình / 10^(snr/10)."""
    p = np.power(10.0, mel.astype(np.float64) / 10.0)
    noise_p = p.mean() * 10.0 ** (-snr_db / 10.0)
    p2 = p + noise_p * rng.exponential(1.0, size=p.shape)
    return (10.0 * np.log10(np.maximum(p2, _EPS_POWER))).astype(np.float32)


def add_noise_coherent(mel: np.ndarray, snr_db: float, rng: np.random.Generator, k: int = 1) -> np.ndarray:
    """Nhiễu 'kết hợp': mỗi ô mel là trung bình của k số phức, P' = mean_i |sqrt(P) + sqrt(Pn)*z_i|^2, z_i ~ CN(0,1).

    Cùng công suất kỳ vọng với add_noise (E[P'] = P + Pn) nhưng thêm dao động giao thoa pha ở các ô tín hiệu mạnh
    (đúng với nhiễu cộng ở miền audio), thứ mà mô hình cộng công suất thuần không có. k là số bin FFT độc lập
    được xấp xỉ trong một ô mel: k=1 dao động mạnh nhất (thường QUÁ khó), k lớn tiến về add_noise (thường QUÁ dễ).
    Thực tế nằm giữa; chọn k bằng chẩn đoán trên Val (m3_noise_check), không đặt tùy ý.
    """
    if int(k) < 1:
        raise ValueError("k phải >= 1")
    p = np.power(10.0, mel.astype(np.float64) / 10.0)
    noise_p = p.mean() * 10.0 ** (-snr_db / 10.0)
    shape = (int(k),) + p.shape
    z = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)
    p2 = (np.abs(np.sqrt(p)[None] + np.sqrt(noise_p) * z) ** 2).mean(axis=0)
    return (10.0 * np.log10(np.maximum(p2, _EPS_POWER))).astype(np.float32)


NOISE_MODES = {"additive": add_noise, "coherent": add_noise_coherent}


def gain_ramp(mel: np.ndarray, total_db: float) -> np.ndarray:
    """Cộng đường dốc dB tuyến tính theo thời gian, trung bình bằng 0 (từ -total/2 đến +total/2)."""
    ramp = np.linspace(-total_db / 2.0, total_db / 2.0, mel.shape[1], dtype=np.float32)[None, :]
    return (mel + ramp).astype(np.float32)


def spec_mask(mel: np.ndarray, rng: np.random.Generator, n_freq: int, max_freq_width: int,
              n_time: int, max_time_width: int) -> np.ndarray:
    """SpecAugment: che n_freq dải tần và n_time đoạn thời gian bằng giá trị trung bình của crop."""
    out = mel.astype(np.float32, copy=True)
    fill = float(out.mean())
    n_mels, n_frames = out.shape
    for _ in range(n_freq):
        w = int(rng.integers(1, max_freq_width + 1))
        s = int(rng.integers(0, max(1, n_mels - w + 1)))
        out[s:s + w, :] = fill
    for _ in range(n_time):
        w = int(rng.integers(1, max_time_width + 1))
        s = int(rng.integers(0, max(1, n_frames - w + 1)))
        out[:, s:s + w] = fill
    return out


class MelAugmenter:
    """Chuỗi augmentation theo thứ tự cố định: time_warp -> cắt -> freq_warp -> eq -> noise -> gain_ramp -> spec_mask.

    cfg: dict {tên_op: {tham số}}. Op có p<=0 hoặc vắng mặt là tắt. Op ngoài ALLOWED_OPS báo ValueError
    (riêng mix/overlay/codec có thông báo riêng vì là họ giữ ra).
    """

    def __init__(self, cfg: dict | None, sr: int = 22050, n_mels: int = 128):
        cfg = cfg or {}
        for k in cfg:
            if k in HELD_OUT_OPS:
                raise ValueError(f"'{k}' là họ giữ ra khỏi huấn luyện (ISSUE-05); không đưa vào augment.")
            if k not in ALLOWED_OPS:
                raise ValueError(f"Op augmentation không hợp lệ: {k!r}. Hợp lệ: {ALLOWED_OPS}")
        self.sr = int(sr)
        self.n_mels = int(n_mels)
        self.params: dict[str, dict] = {}
        for k, v in cfg.items():
            merged = {**DEFAULTS[k], **(v or {})}
            if k == "noise" and merged["mode"] not in NOISE_MODES:
                raise ValueError(f"noise.mode phải thuộc {sorted(NOISE_MODES)}, nhận {merged['mode']!r}")
            if k == "noise" and int(merged["coherent_k"]) < 1:
                raise ValueError("noise.coherent_k phải >= 1")
            if float(merged["p"]) > 0.0:
                self.params[k] = merged
        self._freqs = mel_bin_freqs(self.n_mels, self.sr)

    @property
    def active_ops(self) -> list[str]:
        return [k for k in ALLOWED_OPS if k in self.params]

    def source_frames(self, out_frames: int) -> int:
        """Số khung nguồn cần cắt để sau time_warp vẫn đủ out_frames khung."""
        if "time_warp" in self.params:
            return int(math.ceil(out_frames * float(self.params["time_warp"]["rate_max"]))) + 2
        return int(out_frames)

    def _hit(self, op: str, rng: np.random.Generator) -> bool:
        return op in self.params and rng.random() < float(self.params[op]["p"])

    def __call__(self, mel_src: np.ndarray, out_frames: int, rng: np.random.Generator):
        """mel_src: (n_mels, >= out_frames) -> (mel (n_mels, out_frames) float32, danh sách op đã áp dụng)."""
        if mel_src.shape[1] < out_frames:
            raise ValueError(f"mel_src có {mel_src.shape[1]} khung < out_frames={out_frames}")
        x = np.asarray(mel_src, dtype=np.float32)
        applied: list[str] = []
        if self._hit("time_warp", rng):
            prm = self.params["time_warp"]
            rate = float(rng.uniform(prm["rate_min"], prm["rate_max"]))
            rate = min(rate, float(prm["rate_max"]))
            if (out_frames - 1) * rate <= x.shape[1] - 1:
                x = time_warp(x, rate, out_frames)
                applied.append("time_warp")
        x = x[:, :out_frames]
        if self._hit("freq_warp", rng):
            prm = self.params["freq_warp"]
            x = freq_warp(x, float(rng.uniform(prm["semitone_min"], prm["semitone_max"])), self.sr)
            applied.append("freq_warp")
        if self._hit("eq", rng):
            prm = self.params["eq"]
            kind = "lowpass" if rng.random() < 0.5 else "highpass"
            lo, hi = prm["lowpass_hz"] if kind == "lowpass" else prm["highpass_hz"]
            fc = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            g = eq_gain_db(self._freqs, kind, fc, max_atten_db=float(prm["max_atten_db"]))
            x = (x + g[:, None].astype(np.float32)).astype(np.float32)
            applied.append("eq")
        if self._hit("noise", rng):
            prm = self.params["noise"]
            snr = float(rng.uniform(prm["snr_db_min"], prm["snr_db_max"]))
            x = (add_noise_coherent(x, snr, rng, k=int(prm["coherent_k"])) if prm["mode"] == "coherent"
                 else NOISE_MODES[prm["mode"]](x, snr, rng))
            applied.append("noise")
        if self._hit("gain_ramp", rng):
            prm = self.params["gain_ramp"]
            x = gain_ramp(x, float(rng.uniform(-prm["total_db_abs"], prm["total_db_abs"])))
            applied.append("gain_ramp")
        if self._hit("spec_mask", rng):
            prm = self.params["spec_mask"]
            x = spec_mask(x, rng, int(prm["n_freq"]), int(prm["max_freq_width"]),
                          int(prm["n_time"]), int(prm["max_time_width"]))
            applied.append("spec_mask")
        return np.ascontiguousarray(x, dtype=np.float32), applied
