"""Đỉnh phổ (constellation) và băm cặp điểm neo-đích, tính trên log-mel (dB).

Đỉnh = cực đại cục bộ cao hơn nền cục bộ (trung bình lân cận) ít nhất `prominence_db`.
Cả hai đại lượng đều dịch theo cùng độ lợi, nên tập đỉnh bất biến khi đổi âm lượng.
Băm 20 bit: (f_neo << 13) | (f_dich << 6) | dt, với f < 128 dải, dt < 64 khung.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter


def extract_peaks(mel: np.ndarray, peak_freq_size: int = 11, peak_time_size: int = 11,
                  bg_freq_size: int = 31, bg_time_size: int = 43, prominence_db: float = 3.0,
                  max_peaks_per_sec: float = 15.0, frames_per_sec: float = 43.07
                  ) -> tuple[np.ndarray, np.ndarray]:
    """mel (n_mels, n_frames) dB -> (t, f) int arrays, sắp theo (t, f)."""
    if mel.ndim != 2:
        raise ValueError("mel phải 2 chiều")
    m = mel.astype(np.float32)
    is_max = m == maximum_filter(m, size=(peak_freq_size, peak_time_size), mode="nearest")
    bg = uniform_filter(m, size=(bg_freq_size, bg_time_size), mode="nearest")
    f_idx, t_idx = np.nonzero(is_max & (m > bg + prominence_db))
    if f_idx.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    val = m[f_idx, t_idx]
    block = max(1, int(round(frames_per_sec)))
    cap = max(1, int(round(max_peaks_per_sec * block / frames_per_sec)))
    keep = []
    blk = t_idx // block
    for b in np.unique(blk):
        ix = np.nonzero(blk == b)[0]
        if ix.size > cap:
            ix = ix[np.argsort(-val[ix], kind="stable")[:cap]]
        keep.append(ix)
    ix = np.concatenate(keep)
    t, f = t_idx[ix].astype(np.int32), f_idx[ix].astype(np.int32)
    order = np.lexsort((f, t))
    return t[order], f[order]


def hashes_from_peaks(t: np.ndarray, f: np.ndarray, dt_min: int = 1, dt_max: int = 63,
                      df_max: int = 40, fan_out: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Cặp (neo, đích): đích nằm sau neo dt_min..dt_max khung, |df| <= df_max, tối đa fan_out
    đích đầu tiên theo thời gian. Trả (hash int32, t_neo int32). t, f phải sắp theo (t, f)."""
    if not (1 <= dt_min <= dt_max <= 63):
        raise ValueError("Cần 1 <= dt_min <= dt_max <= 63")
    if f.size and (f.min() < 0 or f.max() > 127):
        raise ValueError("Chỉ số dải mel phải trong [0, 127]")
    hs, ts = [], []
    n = t.size
    for i in range(n):
        ti, fi, got = int(t[i]), int(f[i]), 0
        j = i + 1
        while j < n and got < fan_out:
            dt = int(t[j]) - ti
            if dt > dt_max:
                break
            if dt >= dt_min and abs(int(f[j]) - fi) <= df_max:
                hs.append((fi << 13) | (int(f[j]) << 6) | dt)
                ts.append(ti)
                got += 1
            j += 1
    return np.array(hs, dtype=np.int32), np.array(ts, dtype=np.int32)
