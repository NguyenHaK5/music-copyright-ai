import numpy as np
import pandas as pd
import soundfile as sf

from src.data.dedup import find_near_pairs, group_components, run_dedup, spectral_signatures
from src.data.data_card import render_data_card
from src.data.eda import eda_summary, make_figures
from src.features.cache import save_logmel

SR = 22050


def _mk(tmp_path):
    """Track 1 và 4: cùng file. Track 2: phổ gần giống 1 (nhưng file khác). Track 3: khác hẳn."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(128, 50)).astype(np.float32) * 5 + np.linspace(-40, 0, 128)[:, None]
    mels = {1: base, 2: base + rng.normal(scale=0.01, size=base.shape).astype(np.float32),
            3: rng.normal(size=(128, 50)).astype(np.float32) * 5 + np.linspace(0, -60, 128)[:, None],
            4: base}
    cache = tmp_path / "cache"
    for t, m in mels.items():
        save_logmel(cache, t, m, "float32")
    wav = {1: rng.normal(size=SR).astype(np.float32) * 0.1, 2: rng.normal(size=SR).astype(np.float32) * 0.1,
           3: rng.normal(size=SR).astype(np.float32) * 0.1}
    wav[4] = wav[1]
    for t, y in wav.items():
        sf.write(tmp_path / f"{t}.wav", y, SR)
    return cache, (lambda t: tmp_path / f"{t}.wav")


def test_group_components():
    g = group_components([1, 2, 3, 4, 5], [(2, 1), (3, 2)])
    assert g == {1: 1, 2: 1, 3: 1, 4: 4, 5: 5}


def test_near_pairs_and_signatures(tmp_path):
    cache, _ = _mk(tmp_path)
    ids = [1, 2, 3, 4]
    X = spectral_signatures(cache, ids)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
    pairs, nn = find_near_pairs(X, ids, thr=0.99, block=2)
    got = {(a, b) for a, b, _ in pairs}
    assert (1, 2) in got and (1, 4) in got and (2, 4) in got
    assert not any(3 in p for p in got)
    assert nn[2] < 0.99  # track 3 không có láng giềng gần


def test_run_dedup(tmp_path):
    cache, pf = _mk(tmp_path)
    df, pairs = run_dedup([1, 2, 3, 4], pf, cache, thr=0.99, block=2)
    r = df.set_index("track_id")
    assert r.loc[4, "exact_dup_of"] == 1 and r.loc[1, "exact_dup_of"] == 0
    assert r.loc[2, "exact_dup_of"] == 0 and r.loc[2, "dup_group"] == 1
    assert r.loc[3, "dup_group"] == 3
    assert len(pairs) >= 3


def _tc():
    return pd.DataFrame({"track_id": [1, 2, 3, 4], "artist_id": [10, 10, 11, 12], "album_id": [1, 1, 2, 3],
                         "genre": ["Rock", "Rock", "Pop", "Pop"], "license_family": ["by", "by", "by-nc", "by"],
                         "file_ok": [True, True, True, False], "drop_reason": ["", "", "", "too_short"],
                         "duration_s": [30.0, 29.5, 30.0, 5.0], "dup_group": [1, 1, 3, 4],
                         "nn_cosine": [0.99, 0.99, 0.2, 0.1]})


def test_eda_and_figures(tmp_path):
    df = _tc()
    pairs = pd.DataFrame({"track_a": [1], "track_b": [2], "cosine": [0.99], "same_artist": [True]})
    s = eda_summary(df, pairs)
    assert s["n_tracks_ok"] == 3 and s["drop_reason_counts"] == {"too_short": 1}
    assert s["n_dup_groups_gt1"] == 1 and s["near_pairs_same_artist"] == 1
    cache, _ = _mk(tmp_path)
    made = make_figures(df, cache, tmp_path / "f", tmp_path / "t", seed=1)
    assert all(p.exists() for p in made) and (tmp_path / "t" / "eda_genre.csv").exists()


def test_data_card_missing_and_filled():
    assert "CHƯA CHẠY" in render_data_card(None, None)
    card = render_data_card({"gates": [{"gate": "n_artists", "passed": True}]}, eda_summary(_tc()))
    assert "n_artists=PASS" in card and "too_short: 1" in card
