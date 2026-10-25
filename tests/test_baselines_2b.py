import numpy as np
import pandas as pd
import pytest

from src.data.queries import QueryMels, crop_frames, sample_query_specs
from src.evaluation.runner import conservative_ranks, evaluate_scores
from src.features.cache import load_logmel, save_logmel
from src.features.handcrafted import feature_dim, mel_stats_features, window_starts
from src.models.m0_trivial import M0Random
from src.models.m1_classical import M1Classical, sample_train_features

SR, HOP, NMEL, T = 22050, 512, 128, 400
WIN = 60


def _mel(rng, profile):
    return (profile[:, None] + rng.normal(0, 1.0, (NMEL, T))).astype(np.float32)


@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for tid in range(60):
        split, role = ("train", "train") if tid < 30 else \
                      ("val", "known") if tid < 45 else ("val", "unknown")
        profile = rng.normal(-40, 8, NMEL)
        save_logmel(tmp_path, tid, _mel(rng, profile))
        rows.append((tid, split, role))
    return tmp_path, pd.DataFrame(rows, columns=["track_id", "split", "role"])


def test_feature_dim_and_shape():
    rng = np.random.default_rng(1)
    f = mel_stats_features(rng.normal(size=(NMEL, 50)).astype(np.float32), 20)
    assert f.shape == (feature_dim(NMEL, 20),) and f.dtype == np.float32
    with pytest.raises(ValueError):
        mel_stats_features(np.zeros((NMEL, 1)))


def test_window_starts():
    assert window_starts(10, 4, 3) == [0, 3, 6]
    assert window_starts(4, 4, 3) == [0]
    with pytest.raises(ValueError):
        window_starts(3, 4, 1)


def test_specs_deterministic_and_valid(world):
    d, df = world
    kw = dict(split="val", cache_dir=d, crop_sec=WIN * HOP / SR, sr=SR, hop=HOP,
              n_known=3, n_unknown=2, seed=7)
    a, b = sample_query_specs(df, **kw), sample_query_specs(df, **kw)
    pd.testing.assert_frame_equal(a, b)
    assert a["is_known"].sum() == 15 * 3 and (~a["is_known"]).sum() == 15 * 2
    assert (a["start_frame"] + a["n_frames"] <= T).all() and (a["start_frame"] >= 0).all()
    assert set(a["source_track_id"]) == set(range(30, 60))   # không có bài Train
    assert crop_frames(5.0, SR, HOP) == 215


def test_query_mels_matches_cache(world):
    d, df = world
    specs = sample_query_specs(df, "val", d, WIN * HOP / SR, SR, HOP, 2, 2, 3)
    qm = QueryMels(d, specs)
    assert len(qm) == len(specs)
    first = next(iter(qm))
    r = specs.iloc[0]
    ref = load_logmel(d, int(r.source_track_id))[:, int(r.start_frame):int(r.start_frame) + WIN]
    np.testing.assert_array_equal(first, ref)


def test_conservative_ranks_ties():
    S = np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.5]])
    assert conservative_ranks(S, np.array([1, 2])).tolist() == [3, 2]


def test_evaluate_scores_guards():
    S = np.zeros((2, 2))
    with pytest.raises(ValueError):  # unknown nằm trong chỉ mục
        evaluate_scores(S, [1, 2], [1, 2], [True, False], 0.01, 0.05, n_boot=10)
    with pytest.raises(ValueError):  # chỉ có known
        evaluate_scores(S, [1, 2], [1, 2], [True, True], 0.01, 0.05, n_boot=10)


def _pipeline(world, model):
    d, df = world
    specs = sample_query_specs(df, "val", d, WIN * HOP / SR, SR, HOP, 3, 3, 5)
    index_ids = sorted(df[(df.split == "val") & (df.role == "known")].track_id)
    model.build_index(index_ids, lambda t: load_logmel(d, t))
    S = model.score_matrix(QueryMels(d, specs))
    return evaluate_scores(S, index_ids, specs.source_track_id, specs.is_known,
                           0.05, 0.05, n_boot=100, seed=1)


def test_m0_is_near_chance(world):
    res = _pipeline(world, M0Random(seed=3))
    assert res["recall_at_1"]["value"] < 0.3
    assert 0.2 < res["roc_auc_known_vs_unknown"] < 0.8


def test_m1_works_on_synthetic(world):
    d, df = world
    X = sample_train_features(df[df.split == "train"].track_id, d, 30, 3, WIN, 20, 0)
    res = _pipeline(world, M1Classical(WIN, WIN // 2, 20).fit(X))
    assert res["recall_at_1"]["value"] > 0.9
    assert res["roc_auc_known_vs_unknown"] > 0.9
    op = res["operating_point"]
    assert op["fmr_at_tau_high"] <= 0.05 + 1e-9
    assert op["tau_low"] <= op["tau_high"]
    lo, hi = res["recall_at_1"]["ci95"]
    assert lo <= res["recall_at_1"]["value"] <= hi


def test_m1_requires_fit_and_index():
    m = M1Classical(WIN, 30)
    with pytest.raises(RuntimeError):
        m.build_index([0], lambda t: np.zeros((NMEL, T), np.float32))
    with pytest.raises(RuntimeError):
        m.score_matrix([np.zeros((NMEL, WIN))])
