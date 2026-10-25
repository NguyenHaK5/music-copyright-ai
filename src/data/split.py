"""Chia Train/Val/Test và known/unknown theo ĐƠN VỊ rò rỉ (nghệ sĩ + nhóm trùng), phân tầng thể loại.

Thứ tự bắt buộc: khử trùng (1C) -> chia (file này) -> cắt crop -> augment (các phase sau).
Chạy:  python -m src.data.split
Ghi data/processed/tracks.parquet CHỈ KHI leakage check pass.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.log import get_logger
from src.common.paths import resolve
from src.data.dedup import group_components
from src.data.leakage_check import check_split

log = get_logger(__name__)


def add_artist_key(df: pd.DataFrame) -> pd.Series:
    """artist_id nếu có; không thì theo tên nghệ sĩ; không thì riêng từng track."""
    key = df["artist_id"].map(lambda x: f"id:{int(x)}" if pd.notna(x) else None)
    if "artist_name" in df:
        name = df["artist_name"].map(lambda x: f"name:{x}" if pd.notna(x) else None)
        key = key.fillna(name)
    return key.fillna(df["track_id"].map(lambda t: f"track:{int(t)}"))


def build_units(df: pd.DataFrame) -> pd.Series:
    """Đơn vị chia = thành phần liên thông theo (cùng artist_key) hoặc (cùng dup_group). Trả Series unit theo index df."""
    ids = df["track_id"].astype(int).tolist()
    edges: list[tuple[int, int]] = []
    for col in ("artist_key", "dup_group", "album_id"):
        for _, g in df.groupby(col):
            t = g["track_id"].astype(int).tolist()
            edges += [(t[0], x) for x in t[1:]]
    root = group_components(ids, edges)
    return df["track_id"].astype(int).map(root)


def allocate(units: pd.DataFrame, fractions: dict[str, float], seed: int) -> dict[int, str]:
    """units: cột unit, size, stratum. Gán mỗi unit một nhãn sao cho tổng SỐ BÀI mỗi nhãn ~ tỷ lệ, theo từng stratum.

    Tham lam: unit lớn trước, gán cho nhãn đang thiếu nhiều nhất (deficit). Xác định theo seed.
    """
    rng = np.random.default_rng(seed)
    labels = list(fractions)
    out: dict[int, str] = {}
    for _, g in units.groupby("stratum", sort=True):
        order = rng.permutation(len(g))
        g = g.iloc[order].sort_values("size", ascending=False, kind="stable")
        total = int(g["size"].sum())
        target = {l: fractions[l] * total for l in labels}
        count = {l: 0 for l in labels}
        for u, s in zip(g["unit"], g["size"]):
            pick = max(labels, key=lambda l: target[l] - count[l])
            out[int(u)] = pick
            count[pick] += int(s)
    return out


def _unit_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("unit")
    mode_genre = g["genre"].agg(lambda s: s.fillna("unknown_genre").mode().iloc[0])
    return pd.DataFrame({"unit": mode_genre.index, "size": g.size().values, "stratum": mode_genre.values})


def make_split(clean: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """clean: tracks_clean (có file_ok, dup_group, genre...). Trả bảng tracks theo schema Phase 0 (+ cột phụ)."""
    ok = clean[clean["file_ok"]].copy()
    ok["artist_key"] = add_artist_key(ok)
    ok["unit"] = build_units(ok).values

    ut = _unit_table(ok)
    s = cfg["split"]
    top = allocate(ut, {"train": s["train"], "val": s["val"], "test": s["test"]}, seed)
    ok["split"] = ok["unit"].map(top)

    ok["role"] = "train"
    kf = s["known_fraction"]
    for sp in ("val", "test"):
        sub = ok[ok["split"] == sp]
        role = allocate(_unit_table(sub), {"known": kf, "unknown": 1 - kf}, seed + (1 if sp == "val" else 2))
        ok.loc[sub.index, "role"] = sub["unit"].map(role)

    ex = clean[~clean["file_ok"]].copy()
    ex["artist_key"] = add_artist_key(ex)
    ex["split"], ex["role"] = "excluded", "excluded"
    full = pd.concat([ok.drop(columns="unit"), ex], ignore_index=True)

    full["fingerprint_hash"] = full["file_sha1"].fillna("").astype(str)
    full["duration"] = full["duration_s"]
    full["source"] = "FMA"
    cols = ["track_id", "artist_id", "artist_key", "album_id", "genre", "license_raw", "license_family",
            "duration", "fingerprint_hash", "file_ok", "drop_reason", "dup_group", "split", "role", "source"]
    return full[cols].sort_values("track_id").reset_index(drop=True)


def main() -> int:
    cfg = load_config("data")
    src = resolve(cfg["interim"]["tracks_clean"])
    if not src.exists():
        log.error("CHƯA CHẠY: thiếu %s (chạy python -m src.data.dedup).", src)
        return 2
    out = make_split(pd.read_parquet(src), cfg, cfg["seed"])
    res = check_split(out)
    summ = {"counts": res["counts"], "passed": res["passed"], "violations": res["violations"],
            "genre_by_split": pd.crosstab(out.loc[out["split"] != "excluded", "genre"].fillna("unknown_genre"),
                                          out.loc[out["split"] != "excluded", "split"]).to_dict()}
    (resolve(cfg["paths"]["results_dir"]) / "e0_split.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
    if not res["passed"]:
        log.error("Rò rỉ - KHÔNG ghi tracks.parquet: %s", res["violations"])
        return 1
    p = resolve(cfg["split"]["tracks_parquet"]); p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    log.info("Đã ghi %s; %s", p, res["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
