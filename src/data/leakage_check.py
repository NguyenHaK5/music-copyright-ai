"""Kiểm tra rò rỉ giữa Train/Val/Test và giữa known/unknown. Hàm thuần, dùng cho test và CLI.

Chạy:  python -m src.data.leakage_check   (đọc data/processed/tracks.parquet)
"""
from __future__ import annotations

import json
from itertools import combinations

import pandas as pd

from src.common.config import load_config
from src.common.log import get_logger
from src.common.paths import resolve

log = get_logger(__name__)
SPLITS = ("train", "val", "test")


def _overlaps(df: pd.DataFrame, col: str, group_col: str, groups: tuple[str, ...], skip_empty=False) -> list[dict]:
    out = []
    sets = {}
    for g in groups:
        s = df.loc[df[group_col] == g, col].dropna()
        if skip_empty:
            s = s[s.astype(str) != ""]
        sets[g] = set(s)
    for a, b in combinations(groups, 2):
        inter = sets[a] & sets[b]
        if inter:
            out.append({"check": f"{col}_overlap", "between": [a, b], "n": len(inter),
                        "examples": sorted(map(str, inter))[:5]})
    return out


def check_split(df: pd.DataFrame) -> dict:
    """Trả {'passed': bool, 'violations': [...], 'counts': {...}}.

    Cần cột: track_id, artist_key, album_id, fingerprint_hash, dup_group, file_ok, split, role.
    """
    v: list[dict] = []
    if df["track_id"].duplicated().any():
        v.append({"check": "track_id_unique", "n": int(df["track_id"].duplicated().sum())})
    used = df[df["split"].isin(SPLITS)]

    if not used["file_ok"].all():
        v.append({"check": "only_file_ok_in_splits", "n": int((~used["file_ok"]).sum())})
    excl = df[~df["split"].isin(SPLITS)]
    if len(excl) and excl["split"].ne("excluded").any():
        v.append({"check": "unknown_split_label", "n": int(excl["split"].ne("excluded").sum())})

    for col, skip in (("artist_key", False), ("album_id", False), ("fingerprint_hash", True), ("dup_group", False)):
        v += _overlaps(used, col, "split", SPLITS, skip_empty=skip)

    # Vai trò: known/unknown tách nghệ sĩ với nhau và với Train
    roles = ("train", "known", "unknown")
    r = used[used["role"].isin(roles)]
    for col in ("artist_key", "dup_group"):
        for x in _overlaps(r, col, "role", roles):
            x["check"] = f"{col}_role_overlap"
            v.append(x)
    # nhất quán split-role
    bad = used[((used["split"] == "train") & (used["role"] != "train")) |
               ((used["split"] != "train") & (~used["role"].isin(["known", "unknown"])))]
    if len(bad):
        v.append({"check": "split_role_consistent", "n": int(len(bad))})

    counts = {s: {"tracks": int((used["split"] == s).sum()),
                  **{role: int(((used["split"] == s) & (used["role"] == role)).sum())
                     for role in ("known", "unknown") if s != "train"}} for s in SPLITS}
    return {"passed": not v, "violations": v, "counts": counts}


def assert_no_leakage(df: pd.DataFrame) -> dict:
    res = check_split(df)
    if not res["passed"]:
        raise AssertionError(f"Phát hiện rò rỉ: {json.dumps(res['violations'], ensure_ascii=False)[:800]}")
    return res


def main() -> int:
    cfg = load_config("data")
    p = resolve(cfg["split"]["tracks_parquet"])
    if not p.exists():
        log.error("CHƯA CHẠY: thiếu %s (chạy python -m src.data.split).", p)
        return 2
    res = check_split(pd.read_parquet(p))
    out = resolve(cfg["paths"]["results_dir"]) / "e0_leakage.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Leakage check: %s; %s", "PASS" if res["passed"] else "FAIL", res["counts"])
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
