"""Smoke check metadata FMA TRƯỚC khi tải audio (ISSUE-07 của Phase 0).

Chạy:  python -m src.data.smoke_check [--tracks đường_dẫn/tracks.csv]
Kết quả: experiments/results/e0_smoke_check.json
Ngưỡng cổng là ĐỀ XUẤT (configs/data.yaml), không phải chuẩn.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.log import get_logger
from src.common.paths import resolve
from src.data.license_norm import license_family

log = get_logger(__name__)

# (mức 0, mức 1) trong tracks.csv của FMA -> tên cột phẳng
_COLUMN_MAP = {
    ("artist", "id"): "artist_id",
    ("artist", "name"): "artist_name",
    ("album", "id"): "album_id",
    ("track", "title"): "title",
    ("track", "genre_top"): "genre",
    ("track", "license"): "license_raw",
    ("track", "duration"): "duration",
    ("set", "subset"): "subset",
}


def load_fma_tracks(path: str | Path, subset: str | None = "small") -> pd.DataFrame:
    """Đọc tracks.csv (header 2 tầng) -> DataFrame phẳng, index = track_id.

    subset: lọc theo cột ('set','subset'); None = giữ tất cả.
    """
    raw = pd.read_csv(path, index_col=0, header=[0, 1], low_memory=False)
    missing = [k for k in _COLUMN_MAP if k not in raw.columns]
    if missing:
        raise KeyError(f"tracks.csv thiếu cột {missing}; cấu trúc file khác kỳ vọng - cần kiểm tra tay")
    df = raw[list(_COLUMN_MAP)].copy()
    df.columns = [_COLUMN_MAP[c] for c in df.columns]
    df.index.name = "track_id"
    if subset is not None:
        df = df[df["subset"] == subset]
    df["license_family"] = df["license_raw"].map(license_family)
    return df


def _gate(name: str, value, threshold, passed: bool, action: str) -> dict:
    return {"gate": name, "value": value, "threshold": threshold, "passed": bool(passed),
            "action_if_failed": action}


def evaluate_gates(df: pd.DataFrame, gates_cfg: dict, free_disk_gb: float) -> list[dict]:
    """Hàm thuần: tính các cổng từ DataFrame phẳng (cột: artist_id, genre, license_family)."""
    out: list[dict] = []
    n = len(df)

    miss = float(df["artist_id"].isna().mean()) if n else 1.0
    out.append(_gate("artist_id_missing_rate", round(miss, 4), gates_cfg["max_missing_artist_rate"],
                     miss <= gates_cfg["max_missing_artist_rate"],
                     "Tự suy nhóm theo artist_name, ghi vào Data Card"))

    n_art = int(df["artist_id"].nunique())
    out.append(_gate("n_artists", n_art, gates_cfg["min_artists"], n_art >= gates_cfg["min_artists"],
                     "Thu nhỏ known/unknown hoặc thêm nguồn"))

    per_genre = df.dropna(subset=["genre"]).groupby("genre")["artist_id"].nunique()
    min_pg = int(per_genre.min()) if len(per_genre) else 0
    out.append(_gate("min_artists_per_genre", min_pg, gates_cfg["min_artists_per_genre"],
                     min_pg >= gates_cfg["min_artists_per_genre"],
                     "Unknown cùng thể loại khó đảm bảo: gộp/bỏ thể loại ít nghệ sĩ"))

    fam = df["license_family"].value_counts()
    big = fam[(fam >= gates_cfg["min_tracks_per_license_family"]) & (fam.index != "unknown")]
    out.append(_gate("license_families_ge_min_tracks", int(len(big)), gates_cfg["min_license_families"],
                     len(big) >= gates_cfg["min_license_families"], "Bỏ thí nghiệm T3 (không dừng dự án)"))

    out.append(_gate("free_disk_gb", round(free_disk_gb, 1), gates_cfg["min_free_disk_gb"],
                     free_disk_gb >= gates_cfg["min_free_disk_gb"],
                     "Chỉ cache crop thay vì cả bài, hoặc dọn ổ đĩa"))
    return out


def summarize(df: pd.DataFrame) -> dict:
    """Thống kê mô tả để đưa vào Data Card."""
    per_artist = df.groupby("artist_id").size()
    return {
        "n_tracks": int(len(df)),
        "n_artists": int(df["artist_id"].nunique()),
        "n_albums": int(df["album_id"].nunique()),
        "tracks_per_artist": {"min": int(per_artist.min()), "median": float(per_artist.median()),
                              "max": int(per_artist.max())} if len(per_artist) else {},
        "genre_counts": df["genre"].value_counts(dropna=False).astype(int).to_dict(),
        "license_family_counts": df["license_family"].value_counts().astype(int).to_dict(),
        "duration_sec": {"min": float(df["duration"].min()), "median": float(df["duration"].median()),
                         "max": float(df["duration"].max())} if len(df) else {},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", default=None, help="đường dẫn tracks.csv (mặc định theo configs/data.yaml)")
    args = ap.parse_args()

    cfg = load_config("data")
    tracks_path = resolve(args.tracks or cfg["paths"]["metadata_csv"])
    if not tracks_path.exists():
        log.error("CHƯA CHẠY: không thấy %s. Tải fma_metadata.zip (xem configs/data.yaml), giải nén vào "
                  "data/raw/fma_metadata/ rồi chạy lại.", tracks_path)
        return 2

    df = load_fma_tracks(tracks_path, cfg["dataset"]["subset"])
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(raw_dir).free / 1e9

    gates = evaluate_gates(df, cfg["smoke_gates"], free_gb)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "tracks_csv": str(tracks_path),
              "subset": cfg["dataset"]["subset"], "gates": gates, "summary": summarize(df)}
    out = resolve(cfg["paths"]["results_dir"]) / "e0_smoke_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for g in gates:
        log.info("%s %s = %s (ngưỡng %s)", "PASS" if g["passed"] else "FAIL", g["gate"], g["value"], g["threshold"])
    log.info("Đã ghi %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
