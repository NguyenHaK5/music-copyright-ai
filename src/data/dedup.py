"""Khử trùng: trùng chính xác (SHA1 file) + nghi trùng gần (chữ ký phổ trung bình).

- Trùng chính xác: giữ track_id nhỏ nhất, loại các bản còn lại (drop_reason='duplicate_exact').
- Nghi trùng gần: chỉ GẮN CỜ và gom nhóm (dup_group). Bước chia tập (1D) bắt buộc giữ cả nhóm
  trong cùng một tập. Không tự xóa vì chữ ký phổ thô có thể báo nhầm.
Chạy:  python -m src.data.dedup
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.log import get_logger
from src.common.paths import resolve
from src.data.clean import fma_audio_path
from src.data.download import sha1_file
from src.data.smoke_check import load_fma_tracks
from src.features.cache import load_logmel

log = get_logger(__name__)


def spectral_signatures(cache_dir: Path, ids: list[int]) -> np.ndarray:
    """Mỗi track -> mean log-mel theo thời gian (n_mels,), trừ trung bình toàn tập, chuẩn hóa L2.

    Trả ma trận float32 (n_tracks, n_mels). Trừ trung bình để cosine không bị 'dính' gần 1
    do mọi bản nhạc đều có hình dạng phổ tương tự nhau.
    """
    X = np.stack([load_logmel(cache_dir, t).mean(axis=1) for t in ids]).astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, 1e-9)


def find_near_pairs(X: np.ndarray, ids: list[int], thr: float, block: int = 1024):
    """Trả (pairs, nn_cos): pairs=[(id_a,id_b,cos)] với a<b và cos>=thr; nn_cos[i]=cosine láng giềng gần nhất."""
    n = len(ids)
    pairs: list[tuple[int, int, float]] = []
    nn = np.full(n, -1.0, dtype=np.float32)
    for s in range(0, n, block):
        sim = X[s:s + block] @ X.T
        for r in range(sim.shape[0]):
            sim[r, s + r] = -1.0  # bỏ chính nó
        nn[s:s + block] = sim.max(axis=1) if n > 1 else -1.0
        rows, cols = np.where(sim >= thr)
        for r, c in zip(rows, cols):
            a, b = s + int(r), int(c)
            if a < b:
                pairs.append((ids[a], ids[b], float(sim[r, c])))
    return pairs, nn


def group_components(ids: list[int], edges: list[tuple[int, int]]) -> dict[int, int]:
    """Union-find; nhãn nhóm = track_id nhỏ nhất của thành phần."""
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in ids}


def run_dedup(ok_ids: list[int], path_fn: Callable[[int], Path], cache_dir: Path,
              thr: float, block: int = 1024):
    """Trả (DataFrame theo track_id, near_pairs DataFrame).

    Cột: track_id, file_sha1, exact_dup_of (0 nếu không), dup_group, nn_cosine.
    """
    ids = sorted(int(t) for t in ok_ids)
    sha = {t: sha1_file(path_fn(t)) for t in ids}
    first: dict[str, int] = {}
    exact_of = {}
    for t in ids:
        exact_of[t] = 0 if sha[t] not in first else first[sha[t]]
        first.setdefault(sha[t], t)

    X = spectral_signatures(cache_dir, ids)
    pairs, nn = find_near_pairs(X, ids, thr, block)
    edges = [(a, b) for a, b, _ in pairs] + [(t, o) for t, o in exact_of.items() if o]
    groups = group_components(ids, edges)
    df = pd.DataFrame({"track_id": ids, "file_sha1": [sha[t] for t in ids],
                       "exact_dup_of": [exact_of[t] for t in ids],
                       "dup_group": [groups[t] for t in ids], "nn_cosine": nn})
    return df, pd.DataFrame(pairs, columns=["track_a", "track_b", "cosine"])


def main() -> int:
    dcfg, fcfg = load_config("data"), load_config("features")
    tracks_csv = resolve(dcfg["paths"]["metadata_csv"])
    report_p = resolve(fcfg["paths"]["clean_report"])
    if not tracks_csv.exists() or not report_p.exists():
        log.error("CHƯA CHẠY: cần tracks.csv (1A) và clean_report.csv (1B).")
        return 2
    meta = load_fma_tracks(tracks_csv, dcfg["dataset"]["subset"]).reset_index()
    rep = pd.read_csv(report_p)
    rep["file_ok"] = rep["file_ok"].astype(bool)
    audio_dir = resolve(dcfg["paths"]["audio_dir"])
    cache_dir = resolve(fcfg["paths"]["logmel_dir"])
    ok_ids = rep.loc[rep["file_ok"], "track_id"].astype(int).tolist()

    dd, pairs = run_dedup(ok_ids, lambda t: fma_audio_path(audio_dir, t), cache_dir,
                          dcfg["dedup"]["near_dup_cosine"], dcfg["dedup"]["block_size"])
    df = meta.merge(rep, on="track_id", how="left").merge(dd, on="track_id", how="left")
    df["drop_reason"] = df["drop_reason"].fillna("").astype(str)
    df["file_ok"] = df["file_ok"].fillna(False).astype(bool)
    is_dup = df["exact_dup_of"].fillna(0).astype(int) > 0
    df.loc[is_dup, "drop_reason"] = "duplicate_exact"
    df.loc[is_dup, "file_ok"] = False
    df.loc[~df["file_ok"] & (df["drop_reason"] == ""), "drop_reason"] = "not_processed"
    df["dup_group"] = df["dup_group"].fillna(df["track_id"]).astype(int)

    art = meta.set_index("track_id")["artist_id"]
    pairs["same_artist"] = pairs["track_a"].map(art).values == pairs["track_b"].map(art).values
    out = resolve(dcfg["interim"]["tracks_clean"]); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    pairs.to_csv(resolve(dcfg["interim"]["near_pairs"]), index=False)
    log.info("exact dup: %d; cặp nghi trùng gần: %d; file_ok cuối: %d/%d",
             int(is_dup.sum()), len(pairs), int(df["file_ok"].sum()), len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
