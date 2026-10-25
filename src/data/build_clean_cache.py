"""Kiểm tra file -> chuẩn hóa -> log-mel -> cache; ghi clean_report.csv. Chạy lại được (checkpoint).

Chạy:  python -m src.data.build_clean_cache [--limit N]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

from src.common.config import load_config
from src.common.log import get_logger
from src.common.paths import resolve
from src.data.clean import check_audio, fma_audio_path
from src.data.smoke_check import load_fma_tracks
from src.features.cache import has_logmel, save_logmel
from src.features.logmel import logmel, peak_normalize

log = get_logger(__name__)
_COLS = ["track_id", "file_ok", "drop_reason", "duration_s"]


def build(track_ids, path_fn: Callable[[int], Path], cache_dir: Path, report_path: Path,
          fcfg: dict, limit: int | None = None) -> pd.DataFrame:
    """Xử lý các track_id chưa có trong report. Trả report đầy đủ."""
    sr, c = fcfg["sample_rate"], fcfg["clean"]
    report = pd.read_csv(report_path) if report_path.exists() else pd.DataFrame(columns=_COLS)
    done = set(report["track_id"].astype(int))
    todo = [int(t) for t in track_ids if int(t) not in done]
    if limit:
        todo = todo[:limit]
    rows: list[dict] = []

    def flush() -> pd.DataFrame:
        nonlocal report, rows
        if rows:
            report = pd.concat([report, pd.DataFrame(rows)], ignore_index=True)
            rows = []
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
        return report

    for i, tid in enumerate(tqdm(todo, desc="clean+logmel"), start=1):
        reason, dur, y = check_audio(path_fn(tid), sr, c["min_seconds"], c["max_seconds"],
                                     c["silence_peak_threshold"])
        if reason is None:
            if not has_logmel(cache_dir, tid):
                mel = logmel(peak_normalize(y), sr, fcfg["n_fft"], fcfg["hop_length"], fcfg["n_mels"])
                save_logmel(cache_dir, tid, mel, fcfg["cache_dtype"])
        rows.append({"track_id": tid, "file_ok": reason is None, "drop_reason": reason or "",
                     "duration_s": round(dur, 2)})
        if i % c["checkpoint_every"] == 0:
            flush()
    return flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="chỉ xử lý N file (chạy thử)")
    args = ap.parse_args()
    dcfg, fcfg = load_config("data"), load_config("features")
    tracks_csv = resolve(dcfg["paths"]["metadata_csv"])
    if not tracks_csv.exists():
        log.error("CHƯA CHẠY: thiếu %s (xem Phase 1A).", tracks_csv)
        return 2
    audio_dir = resolve(dcfg["paths"]["audio_dir"])
    if not audio_dir.exists():
        log.error("CHƯA CHẠY: thiếu thư mục audio %s (chạy src.data.fetch_audio trước).", audio_dir)
        return 2
    df = load_fma_tracks(tracks_csv, dcfg["dataset"]["subset"])
    rep = build(df.index, lambda t: fma_audio_path(audio_dir, t), resolve(fcfg["paths"]["logmel_dir"]),
                resolve(fcfg["paths"]["clean_report"]), fcfg, args.limit)
    log.info("Tổng %d file; đạt %d", len(rep), int(rep["file_ok"].sum()))
    log.info("Lý do loại:\n%s", rep.loc[~rep["file_ok"].astype(bool), "drop_reason"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
