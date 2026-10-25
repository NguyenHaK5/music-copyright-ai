"""M2: fingerprint tự cài (constellation hashing + bỏ phiếu theo độ lệch thời gian).

Chỉ mục: mảng băm đã sắp (H), thời điểm neo (T), chỉ số bài (TR). Truy vấn: tra các băm khớp,
bỏ phiếu theo (bài, t_chỉ_mục - t_truy_vấn); điểm bài = phiếu cực đại / số băm của truy vấn.
Tính trên log-mel đã cache (khác thiết kế Phase 0: không dựng lại STFT từ audio).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.features.peaks import extract_peaks, hashes_from_peaks
from src.models.base import MelLoader

_SHIFT = 100_000
_WIDTH = 2 * _SHIFT + 1


class M2Fingerprint:
    name = "m2"

    def __init__(self, peak_freq_size=11, peak_time_size=11, bg_freq_size=31, bg_time_size=43,
                 prominence_db=3.0, max_peaks_per_sec=15.0, fan_out=5, dt_min=1, dt_max=63,
                 df_max=40, delta_bin=1, frames_per_sec=43.07):
        self.pk = dict(peak_freq_size=peak_freq_size, peak_time_size=peak_time_size,
                       bg_freq_size=bg_freq_size, bg_time_size=bg_time_size,
                       prominence_db=prominence_db, max_peaks_per_sec=max_peaks_per_sec,
                       frames_per_sec=frames_per_sec)
        self.hs = dict(dt_min=dt_min, dt_max=dt_max, df_max=df_max, fan_out=fan_out)
        if delta_bin < 1:
            raise ValueError("delta_bin >= 1")
        self.bin = delta_bin
        self.track_ids: list[int] = []
        self.H = self.T = self.TR = None

    def _hashes(self, mel: np.ndarray):
        t, f = extract_peaks(mel, **self.pk)
        return hashes_from_peaks(t, f, **self.hs)

    @property
    def index_size(self) -> int:
        return 0 if self.H is None else int(self.H.size)

    def build_index(self, track_ids: Sequence[int], mel_loader: MelLoader) -> None:
        self.track_ids = [int(t) for t in track_ids]
        hh, tt, rr = [], [], []
        for k, tid in enumerate(self.track_ids):
            h, t = self._hashes(mel_loader(tid))
            hh.append(h)
            tt.append(t)
            rr.append(np.full(h.size, k, dtype=np.int32))
        H, T, TR = np.concatenate(hh), np.concatenate(tt), np.concatenate(rr)
        order = np.argsort(H, kind="stable")
        self.H, self.T, self.TR = H[order], T[order], TR[order]

    def _score_one(self, mel: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.track_ids), dtype=np.float32)
        h, tq = self._hashes(mel)
        if h.size == 0:
            return out
        lo = np.searchsorted(self.H, h, "left")
        hi = np.searchsorted(self.H, h, "right")
        cnt = hi - lo
        total = int(cnt.sum())
        if total == 0:
            return out
        starts = np.repeat(lo, cnt)
        within = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ix = starts + within
        delta = self.T[ix] - np.repeat(tq, cnt)
        key = self.TR[ix].astype(np.int64) * _WIDTH + (delta + _SHIFT) // self.bin
        uk, votes = np.unique(key, return_counts=True)
        np.maximum.at(out, (uk // _WIDTH).astype(int), votes.astype(np.float32))
        return out / h.size

    def score_matrix(self, queries: Sequence[np.ndarray]) -> np.ndarray:
        if self.H is None:
            raise RuntimeError("Chưa build_index")
        return np.stack([self._score_one(q) for q in queries])
