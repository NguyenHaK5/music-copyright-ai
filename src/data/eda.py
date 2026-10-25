"""EDA: bảng + hình + e0_eda.json từ tracks_clean.parquet. Logic ở đây, notebook chỉ gọi hàm.

Chạy:  python -m src.data.eda
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.log import get_logger
from src.common.paths import resolve
from src.features.cache import has_logmel, load_logmel

log = get_logger(__name__)


def eda_summary(df: pd.DataFrame, near_pairs: pd.DataFrame | None = None) -> dict:
    """Thống kê mô tả (dict JSON-được). df: tracks_clean (cột file_ok, drop_reason, ...)."""
    ok = df[df["file_ok"]]
    per_artist = ok.groupby("artist_id").size()
    s = {
        "n_tracks_metadata": int(len(df)),
        "n_tracks_ok": int(len(ok)),
        "drop_reason_counts": df.loc[~df["file_ok"], "drop_reason"].value_counts().astype(int).to_dict(),
        "n_artists_ok": int(ok["artist_id"].nunique()),
        "n_albums_ok": int(ok["album_id"].nunique()),
        "tracks_per_artist": {"min": int(per_artist.min()), "median": float(per_artist.median()),
                              "max": int(per_artist.max())} if len(per_artist) else {},
        "genre_counts": ok["genre"].value_counts(dropna=False).astype(int).to_dict(),
        "license_family_counts": ok["license_family"].value_counts().astype(int).to_dict(),
        "duration_sec": {"min": float(ok["duration_s"].min()), "median": float(ok["duration_s"].median()),
                         "max": float(ok["duration_s"].max())} if len(ok) else {},
        "n_dup_groups_gt1": int((ok.groupby("dup_group").size() > 1).sum()),
    }
    if near_pairs is not None:
        s["near_pairs_flagged"] = int(len(near_pairs))
        if "same_artist" in near_pairs and len(near_pairs):
            s["near_pairs_same_artist"] = int(near_pairs["same_artist"].sum())
    if "nn_cosine" in ok and len(ok):
        q = ok["nn_cosine"].quantile([0.5, 0.9, 0.99, 1.0])
        s["nn_cosine_quantiles"] = {str(k): float(v) for k, v in q.items()}
    return s


def _bar(series: pd.Series, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot.bar(ax=ax)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _hist(values, title: str, path: Path, bins: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def make_figures(df: pd.DataFrame, cache_dir: Path, fig_dir: Path, tab_dir: Path,
                 n_examples: int = 3, seed: int = 42) -> list[Path]:
    """Sinh hình + bảng CSV; trả danh sách file đã tạo."""
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)
    ok = df[df["file_ok"]]
    made: list[Path] = []

    for name, col in (("genre", "genre"), ("license_family", "license_family")):
        vc = ok[col].value_counts(dropna=False)
        vc.rename("n_tracks").to_csv(tab_dir / f"eda_{name}.csv", header=True)
        p = fig_dir / f"eda_{name}.png"; _bar(vc, f"Số bài theo {name}", p); made.append(p)

    p = fig_dir / "eda_tracks_per_artist.png"
    _hist(ok.groupby("artist_id").size(), "Số bài mỗi nghệ sĩ", p, bins=30); made.append(p)
    p = fig_dir / "eda_duration.png"; _hist(ok["duration_s"], "Thời lượng (s)", p); made.append(p)
    if "nn_cosine" in ok and ok["nn_cosine"].notna().any():
        p = fig_dir / "eda_nn_cosine.png"
        _hist(ok["nn_cosine"].dropna(), "Cosine láng giềng gần nhất (chữ ký phổ)", p); made.append(p)

    df.loc[~df["file_ok"], "drop_reason"].value_counts().rename("n").to_csv(tab_dir / "eda_drop_reason.csv", header=True)

    rng = np.random.default_rng(seed)
    cand = [int(t) for t in ok["track_id"] if has_logmel(cache_dir, int(t))]
    if cand:
        pick = rng.choice(cand, size=min(n_examples, len(cand)), replace=False)
        fig, axes = plt.subplots(1, len(pick), figsize=(5 * len(pick), 3.5), squeeze=False)
        for ax, t in zip(axes[0], pick):
            ax.imshow(load_logmel(cache_dir, int(t)), origin="lower", aspect="auto")
            ax.set_title(f"track {int(t)}"); ax.set_xlabel("khung"); ax.set_ylabel("mel bin")
        fig.tight_layout(); p = fig_dir / "eda_logmel_examples.png"; fig.savefig(p, dpi=120); plt.close(fig)
        made.append(p)
    return made


def main() -> int:
    dcfg, fcfg = load_config("data"), load_config("features")
    tc = resolve(dcfg["interim"]["tracks_clean"])
    if not tc.exists():
        log.error("CHƯA CHẠY: thiếu %s (chạy python -m src.data.dedup).", tc)
        return 2
    df = pd.read_parquet(tc)
    pp = resolve(dcfg["interim"]["near_pairs"])
    pairs = pd.read_csv(pp) if pp.exists() else None
    summ = eda_summary(df, pairs)
    make_figures(df, resolve(fcfg["paths"]["logmel_dir"]), resolve(dcfg["interim"]["figures_dir"]),
                 resolve(dcfg["interim"]["tables_dir"]), seed=dcfg["seed"])
    out = resolve(dcfg["paths"]["results_dir"]) / "e0_eda.json"
    out.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Đã ghi %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
