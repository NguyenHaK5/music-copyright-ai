"""Bộ truy vấn SẠCH (crop không biến đổi) dùng chung cho M0/M1/M2/M3.

Giao thức (Phase 0, mục 3.4): chỉ mục chứa toàn bộ clip của bài known; truy vấn known là crop
ngẫu nhiên của chính clip đó; truy vấn unknown là crop của bài unknown (không có trong chỉ mục).
Đặc tả truy vấn lưu theo KHUNG và GIÂY để cả pipeline log-mel lẫn audio (M2) dùng chung.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.features.cache import cache_path, load_logmel


def crop_frames(crop_sec: float, sr: int, hop: int) -> int:
    return int(round(crop_sec * sr / hop))


def sample_query_specs(tracks: pd.DataFrame, split: str, cache_dir: str | Path, crop_sec: float,
                       sr: int, hop: int, n_known: int, n_unknown: int, seed: int) -> pd.DataFrame:
    """Trả DataFrame: query_id, source_track_id, is_known, start_frame, n_frames, start_sec, dur_sec.
    Xác định theo seed (duyệt track_id tăng dần)."""
    need = {"track_id", "split", "role"}
    if not need <= set(tracks.columns):
        raise ValueError(f"tracks thiếu cột: {need - set(tracks.columns)}")
    sub = tracks[(tracks["split"] == split) & (tracks["role"].isin(["known", "unknown"]))]
    sub = sub.sort_values("track_id")
    if sub.empty:
        raise ValueError(f"Không có bài known/unknown trong split={split}")
    nf = crop_frames(crop_sec, sr, hop)
    rng = np.random.default_rng(seed)
    rows = []
    for tid, role in zip(sub["track_id"].astype(int), sub["role"]):
        p = cache_path(cache_dir, tid)
        if not p.exists():
            raise FileNotFoundError(f"Thiếu cache log-mel: {p}")
        T = int(np.load(p, mmap_mode="r").shape[1])
        if T < nf:
            raise ValueError(f"track {tid}: {T} khung < truy vấn {nf} khung")
        is_known = role == "known"
        for _ in range(n_known if is_known else n_unknown):
            s = int(rng.integers(0, T - nf + 1))
            rows.append((tid, is_known, s))
    df = pd.DataFrame(rows, columns=["source_track_id", "is_known", "start_frame"])
    df.insert(0, "query_id", np.arange(len(df)))
    df["n_frames"] = nf
    df["start_sec"] = df["start_frame"] * hop / sr
    df["dur_sec"] = nf * hop / sr
    return df


class QueryMels:
    """Chuỗi truy vấn log-mel nạp lười từ cache: len() và duyệt theo thứ tự query_id."""

    def __init__(self, cache_dir: str | Path, specs: pd.DataFrame):
        self.cache_dir, self.specs = cache_dir, specs.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator[np.ndarray]:
        last_tid, mel = None, None
        for tid, s, n in zip(self.specs["source_track_id"], self.specs["start_frame"],
                             self.specs["n_frames"]):
            if tid != last_tid:
                mel, last_tid = load_logmel(self.cache_dir, int(tid)), tid
            yield mel[:, int(s):int(s) + int(n)]
