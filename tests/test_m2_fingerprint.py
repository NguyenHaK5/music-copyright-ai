import numpy as np
import pandas as pd
import pytest

from src.data.queries import QueryMels, sample_query_specs
from src.evaluation.runner import evaluate_scores
from src.features.cache import load_logmel, save_logmel
from src.features.peaks import extract_peaks, hashes_from_peaks
from src.models.m2_fingerprint import M2Fingerprint

NMEL, T, SR, HOP = 128, 600, 22050, 512
PK = dict(prominence_db=1.0)


def test_hash_packing_and_constraints():
    t = np.array([0, 5, 6, 200], dtype=np.int32)
    f = np.array([10, 20, 100, 30], dtype=np.int32)
    h, ta = hashes_from_peaks(t, f, dt_min=1, dt_max=63, df_max=40, fan_out=5)
    assert h.tolist() == [(10 << 13) | (20 << 6) | 5]   # (0,10)->(5,20); (6,100) vượt df; (5,20)->(6,100) df=80 loại
    assert ta.tolist() == [0]
    # fan_out giới hạn
    t2 = np.arange(10, dtype=np.int32)
    h2, _ = hashes_from_peaks(t2, np.full(10, 5, np.int32), fan_out=2)
    assert len(h2) <= 2 * 10
    assert (np.bincount(np.repeat(np.arange(10), 1)) >= 0).all()
    with pytest.raises(ValueError):
        hashes_from_peaks(t, f, dt_max=64)
    with pytest.raises(ValueError):
        hashes_from_peaks(t, np.array([0, 1, 2, 200], np.int32))


def test_peaks_flat_none_and_gain_invariant():
    flat = np.full((NMEL, 200), -50.0, np.float32)
    t, f = extract_peaks(flat)
    assert t.size == 0
    rng = np.random.default_rng(0)
    m = rng.normal(-40, 6, (NMEL, 300)).astype(np.float32)
    a = extract_peaks(m, prominence_db=1.0)
    b = extract_peaks(m + 17.0, prominence_db=1.0)
    assert a[0].size > 0
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for tid in range(40):
        role = "known" if tid < 20 else "unknown"
        save_logmel(tmp_path, tid, rng.normal(-40, 6, (NMEL, T)).astype(np.float32))
        rows.append((tid, "val", role))
    return tmp_path, pd.DataFrame(rows, columns=["track_id", "split", "role"])


def _run(world, shift_db=0.0):
    d, df = world
    specs = sample_query_specs(df, "val", d, 200 * HOP / SR, SR, HOP, 3, 3, 4)
    ids = sorted(df[df.role == "known"].track_id)
    m = M2Fingerprint(**PK)
    m.build_index(ids, lambda t: load_logmel(d, t))
    qs = [q + shift_db for q in QueryMels(d, specs)]
    S = m.score_matrix(qs)
    return m, S, evaluate_scores(S, ids, specs.source_track_id, specs.is_known, 0.05, 0.05,
                                 n_boot=50, seed=1)


def test_m2_recognises_and_rejects(world):
    m, S, res = _run(world)
    assert m.index_size > 0
    assert res["recall_at_1"]["value"] > 0.95
    assert res["roc_auc_known_vs_unknown"] > 0.95
    assert S.min() >= 0 and S.max() <= 1


def test_m2_gain_invariant(world):
    _, S0, _ = _run(world)
    _, S1, _ = _run(world, shift_db=12.0)
    np.testing.assert_allclose(S0, S1)


def test_m2_requires_index():
    with pytest.raises(RuntimeError):
        M2Fingerprint().score_matrix([np.zeros((NMEL, 50), np.float32)])


def test_m2_empty_query_scores_zero(world):
    d, df = world
    m = M2Fingerprint(**PK)
    m.build_index([0, 1], lambda t: load_logmel(d, t))
    S = m.score_matrix([np.full((NMEL, 100), -50.0, np.float32)])
    assert (S == 0).all()
