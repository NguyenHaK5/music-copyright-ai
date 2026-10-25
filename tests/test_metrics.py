import numpy as np
import pytest

from src.evaluation.metrics import (hit_indicators, mrr, rank_of_true, ranks,
                                    recall_at_k)
from src.evaluation.bootstrap import bootstrap_ci, bootstrap_diff_ci


def test_rank_of_true():
    assert rank_of_true([5, 7, 9], 7) == 2
    assert rank_of_true([5, 7, 9], 1) is None


def test_recall_mrr_hand_computed():
    # đáp án tính tay: hạng = [1, 3, None, 2]
    rl = ranks([[1, 2, 3], [9, 8, 4], [7, 7, 7], [0, 5, 6]], [1, 4, 99, 5])
    assert rl == [1, 3, None, 2]
    assert recall_at_k(rl, 1) == pytest.approx(0.25)
    assert recall_at_k(rl, 2) == pytest.approx(0.5)
    assert recall_at_k(rl, 5) == pytest.approx(0.75)
    assert mrr(rl) == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
    assert hit_indicators(rl, 3).tolist() == [1, 1, 0, 1]


def test_errors():
    with pytest.raises(ValueError):
        recall_at_k([], 1)
    with pytest.raises(ValueError):
        ranks([[1]], [1, 2])


def test_bootstrap_ci_basic_and_deterministic():
    vals = np.array([1, 1, 0, 0, 1, 1, 1, 0])
    cl = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    m, lo, hi = bootstrap_ci(vals, cl, n_boot=500, seed=1)
    assert m == pytest.approx(5 / 8)
    assert lo <= m <= hi
    assert bootstrap_ci(vals, cl, n_boot=500, seed=1) == (m, lo, hi)


def test_bootstrap_diff_detects_clear_gain():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(100), 5)
    a = np.ones(500)
    b = (rng.random(500) < 0.3).astype(float)
    d, lo, hi = bootstrap_diff_ci(a, b, cl, n_boot=500, seed=2)
    assert lo > 0 and d == pytest.approx(1 - b.mean())


def test_bootstrap_diff_no_gain_contains_zero():
    cl = np.repeat(np.arange(50), 4)
    a = np.tile([1.0, 0.0], 100)
    d, lo, hi = bootstrap_diff_ci(a, a.copy(), cl, n_boot=300, seed=3)
    assert d == 0 and lo == 0 and hi == 0
