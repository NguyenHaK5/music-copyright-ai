import numpy as np
import pandas as pd

from src.common.config import load_config
from src.data.smoke_check import evaluate_gates, summarize, load_fma_tracks


def _synthetic(n_artists=400, per_artist=2, genres=("Rock", "Pop")):
    rows = []
    for a in range(n_artists):
        for t in range(per_artist):
            rows.append({"artist_id": a, "album_id": a, "genre": genres[a % len(genres)],
                         "duration": 30.0, "license_family": ["by", "by-nc", "by-sa"][a % 3]})
    return pd.DataFrame(rows)


def test_gates_pass():
    cfg = load_config("data")["smoke_gates"]
    gates = evaluate_gates(_synthetic(), cfg, free_disk_gb=100)
    assert all(g["passed"] for g in gates), gates


def test_gates_fail_few_artists_and_disk():
    cfg = load_config("data")["smoke_gates"]
    gates = {g["gate"]: g for g in evaluate_gates(_synthetic(n_artists=50), cfg, free_disk_gb=5)}
    assert not gates["n_artists"]["passed"]
    assert not gates["free_disk_gb"]["passed"]


def test_gates_missing_artist():
    cfg = load_config("data")["smoke_gates"]
    df = _synthetic()
    df.loc[df.index[:50], "artist_id"] = np.nan
    gates = {g["gate"]: g for g in evaluate_gates(df, cfg, 100)}
    assert not gates["artist_id_missing_rate"]["passed"]


def test_summarize_keys():
    s = summarize(_synthetic())
    assert s["n_tracks"] == 800 and s["n_artists"] == 400


def test_load_fma_tracks_two_level_header(tmp_path):
    # Mô phỏng cấu trúc tracks.csv của FMA: header 2 tầng + hàng nhãn index.
    cols = [("artist", "id"), ("artist", "name"), ("album", "id"), ("track", "title"),
            ("track", "genre_top"), ("track", "license"), ("track", "duration"), ("set", "subset")]
    df = pd.DataFrame([[1, "A", 10, "t1", "Rock", "http://creativecommons.org/licenses/by/4.0/", 30, "small"],
                       [2, "B", 11, "t2", "Pop", "http://creativecommons.org/licenses/by-nc/4.0/", 30, "medium"]],
                      index=pd.Index([100, 200], name="track_id"),
                      columns=pd.MultiIndex.from_tuples(cols))
    p = tmp_path / "tracks.csv"
    df.to_csv(p)
    out = load_fma_tracks(p, subset="small")
    assert list(out.index) == [100]
    assert out.loc[100, "license_family"] == "by"
