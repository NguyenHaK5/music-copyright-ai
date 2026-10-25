"""M1: đặc trưng thủ công (MFCC/delta/thống kê dải mel) -> chuẩn hóa -> cosine.

Chỉ mục: mỗi bài = các cửa sổ trượt (win_sec, hop_sec) trên toàn clip. Điểm bài = max cosine
giữa truy vấn và các cửa sổ của bài. Scaler (mean/std) chỉ fit trên bài Train.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.features.cache import load_logmel
from src.features.handcrafted import mel_stats_features, window_starts
from src.models.base import MelLoader


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def sample_train_features(train_track_ids: Sequence[int], cache_dir: str | Path, n_tracks: int,
                          windows_per_track: int, win: int, n_mfcc: int, seed: int) -> np.ndarray:
    """Đặc trưng của cửa sổ ngẫu nhiên từ bài Train, để fit scaler."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(int(t) for t in train_track_ids))
    if ids.size == 0:
        raise ValueError("Không có bài Train")
    pick = rng.choice(ids, size=min(n_tracks, ids.size), replace=False)
    feats = []
    for tid in sorted(pick):
        mel = load_logmel(cache_dir, int(tid))
        if mel.shape[1] < win:
            continue
        for _ in range(windows_per_track):
            s = int(rng.integers(0, mel.shape[1] - win + 1))
            feats.append(mel_stats_features(mel[:, s:s + win], n_mfcc))
    if not feats:
        raise ValueError("Không lấy được cửa sổ Train nào")
    return np.stack(feats)


class M1Classical:
    name = "m1"

    def __init__(self, win_frames: int, hop_frames: int, n_mfcc: int = 20):
        self.win, self.hop, self.n_mfcc = win_frames, hop_frames, n_mfcc
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.track_ids: list[int] = []
        self._W: np.ndarray | None = None
        self._offsets: np.ndarray | None = None

    def fit(self, train_features: np.ndarray) -> "M1Classical":
        if train_features.ndim != 2 or train_features.shape[0] < 2:
            raise ValueError("Cần ít nhất 2 vector đặc trưng Train")
        self.mu = train_features.mean(0)
        self.sd = np.maximum(train_features.std(0), 1e-6)
        return self

    def _norm(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None:
            raise RuntimeError("Chưa fit scaler")
        return _l2((X - self.mu) / self.sd)

    def build_index(self, track_ids: Sequence[int], mel_loader: MelLoader) -> None:
        feats, offsets = [], []
        self.track_ids = [int(t) for t in track_ids]
        n = 0
        for tid in self.track_ids:
            mel = mel_loader(tid)
            starts = window_starts(mel.shape[1], self.win, self.hop)
            offsets.append(n)
            for s in starts:
                feats.append(mel_stats_features(mel[:, s:s + self.win], self.n_mfcc))
            n += len(starts)
        self._W = self._norm(np.stack(feats)).astype(np.float32)
        self._offsets = np.array(offsets)

    def score_matrix(self, queries: Sequence[np.ndarray], chunk: int = 512) -> np.ndarray:
        if self._W is None:
            raise RuntimeError("Chưa build_index")
        Q = self._norm(np.stack([mel_stats_features(q, self.n_mfcc) for q in queries])).astype(np.float32)
        out = np.empty((Q.shape[0], len(self.track_ids)), dtype=np.float32)
        for i in range(0, Q.shape[0], chunk):
            sim = Q[i:i + chunk] @ self._W.T
            out[i:i + chunk] = np.maximum.reduceat(sim, self._offsets, axis=1)
        return out
