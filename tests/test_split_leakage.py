"""Test chống rò rỉ. Phần 'dữ liệu thật' bị SKIP nếu chưa có tracks.parquet (nghĩa là CHƯA CHẠY trên dữ liệu thật)."""
import numpy as np
import pandas as pd
import pytest

from src.common.config import load_config
from src.common.paths import resolve
from src.data.leakage_check import assert_no_leakage, check_split
from src.data.split import make_split


def _clean(n_artists=300, seed=0):
    rng = np.random.default_rng(seed)
    rows, tid = [], 1
    for a in range(n_artists):
        for _ in range(int(rng.integers(1, 7))):
            rows.append({"track_id": tid, "artist_id": a, "artist_name": f"A{a}", "album_id": a * 10,
                         "genre": ["Rock", "Pop", "Jazz", "Folk"][a % 4], "license_raw": "x",
                         "license_family": "by", "file_ok": True, "drop_reason": "", "duration_s": 30.0,
                         "file_sha1": f"h{tid}", "dup_group": tid})
            tid += 1
    df = pd.DataFrame(rows)
    # nhóm trùng chéo nghệ sĩ: track của artist 0 và artist 1 cùng dup_group
    t0 = df.index[df.artist_id == 0][0]; t1 = df.index[df.artist_id == 1][0]
    df.loc[t1, "dup_group"] = df.loc[t0, "dup_group"]
    # vài track bị loại
    df.loc[df.index[-5:], ["file_ok", "drop_reason"]] = [False, "too_short"]
    return df


CFG = {"split": {"train": 0.6, "val": 0.2, "test": 0.2, "known_fraction": 0.5}}


def test_split_no_leakage_and_proportions():
    out = make_split(_clean(), CFG, seed=42)
    res = assert_no_leakage(out)
    c = res["counts"]
    n = sum(c[s]["tracks"] for s in c)
    assert abs(c["train"]["tracks"] / n - 0.6) < 0.05
    assert abs(c["val"]["tracks"] / n - 0.2) < 0.05
    for s in ("val", "test"):
        k, u = c[s]["known"], c[s]["unknown"]
        assert abs(k / (k + u) - 0.5) < 0.1
    assert (out.loc[out.split == "excluded", "role"] == "excluded").all()
    assert (out.loc[out.split == "excluded", "file_ok"] == False).all()  # noqa: E712


def test_cross_artist_dup_group_stays_together():
    df = _clean()
    out = make_split(df, CFG, seed=1)
    g = out[out.dup_group == df.loc[df.artist_id == 0, "dup_group"].iloc[0]]
    assert g["split"].nunique() == 1 and g["role"].nunique() == 1


def test_split_deterministic():
    a = make_split(_clean(), CFG, seed=5)
    b = make_split(_clean(), CFG, seed=5)
    assert a.equals(b)
    c = make_split(_clean(), CFG, seed=6)
    assert not a["split"].equals(c["split"])


def test_unknown_artists_never_in_train_or_known():
    out = make_split(_clean(), CFG, seed=3)
    u = set(out.loc[out.role == "unknown", "artist_key"])
    assert not u & set(out.loc[out.role.isin(["train", "known"]), "artist_key"])


def test_detects_injected_violations():
    out = make_split(_clean(), CFG, seed=42)
    # 1) cùng nghệ sĩ ở hai split
    bad = out.copy()
    a = bad.loc[bad.split == "train", "artist_key"].iloc[0]
    i = bad.index[(bad.split == "test")][0]
    bad.loc[i, "artist_key"] = a
    r = check_split(bad)
    assert not r["passed"] and any(v["check"] == "artist_key_overlap" for v in r["violations"])
    # 2) cùng fingerprint_hash ở hai split
    bad = out.copy()
    h = bad.loc[bad.split == "train", "fingerprint_hash"].iloc[0]
    bad.loc[bad.index[bad.split == "val"][0], "fingerprint_hash"] = h
    assert any(v["check"] == "fingerprint_hash_overlap" for v in check_split(bad)["violations"])
    # 3) dup_group chéo split
    bad = out.copy()
    g = bad.loc[bad.split == "train", "dup_group"].iloc[0]
    bad.loc[bad.index[bad.split == "test"][0], "dup_group"] = g
    assert any(v["check"] == "dup_group_overlap" for v in check_split(bad)["violations"])
    # 4) unknown trùng nghệ sĩ với known
    bad = out.copy()
    k = bad.loc[(bad.split == "val") & (bad.role == "known"), "artist_key"].iloc[0]
    bad.loc[bad.index[(bad.split == "val") & (bad.role == "unknown")][0], "artist_key"] = k
    assert any("role_overlap" in v["check"] for v in check_split(bad)["violations"])
    # 5) track bị loại lọt vào split
    bad = out.copy()
    bad.loc[bad.index[bad.split == "train"][0], "file_ok"] = False
    assert any(v["check"] == "only_file_ok_in_splits" for v in check_split(bad)["violations"])


def test_real_data_tracks_parquet():
    p = resolve(load_config("data")["split"]["tracks_parquet"])
    if not p.exists():
        pytest.skip("CHƯA CHẠY trên dữ liệu thật: chưa có data/processed/tracks.parquet")
    assert_no_leakage(pd.read_parquet(p))
