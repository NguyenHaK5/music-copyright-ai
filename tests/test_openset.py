import numpy as np
import pytest

from src.evaluation.openset import (LABEL_IN, LABEL_OUT, LABEL_UNK, choose_tau_high,
                                    choose_tau_low, decide_labels, eer, fmr,
                                    known_outcomes, roc_auc)


def test_fmr_and_outcomes_hand_computed():
    unk = [0.1, 0.2, 0.6, 0.9]
    assert fmr(unk, 0.6) == pytest.approx(0.5)        # >= tau được chấp nhận
    ks = [0.95, 0.8, 0.4, 0.7]
    kc = [True, False, True, True]
    o = known_outcomes(ks, kc, 0.7)
    assert o == {"correct_accept": 0.5, "wrong_id": 0.25, "rejected": 0.25}
    assert sum(o.values()) == pytest.approx(1.0)


def test_roc_auc_perfect_random_ties():
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert roc_auc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)
    # known=[3,1], unk=[2]: cặp (3>2)=1, (1>2)=0 -> 0.5
    assert roc_auc([3, 1], [2]) == pytest.approx(0.5)


def test_choose_tau_high_respects_max_fmr():
    rng = np.random.default_rng(0)
    u = rng.random(1000)
    for target in (0.0, 0.01, 0.05, 0.2):
        t = choose_tau_high(u, target)
        assert fmr(u, t) <= target + 1e-12
        # là ngưỡng nhỏ nhất: hạ xuống một chút sẽ vượt mục tiêu (trừ target=1)
        if np.isfinite(t):
            # tối thiểu: hạ ngưỡng xuống ngay dưới t làm FMR tăng (t là điểm nhảy)
            assert fmr(u, np.nextafter(t, -np.inf)) > fmr(u, t)
    assert choose_tau_high(u, 1.0) == float("-inf")


def test_choose_tau_high_with_ties():
    u = [0.5] * 10
    t = choose_tau_high(u, 0.0)
    assert fmr(u, t) == 0.0


def test_choose_tau_low():
    s = np.arange(1, 101) / 100.0
    t = choose_tau_low(s, 0.05)
    assert (s < t).mean() <= 0.05
    with pytest.raises(ValueError):
        choose_tau_low(s, 1.5)


def test_decide_labels():
    out = decide_labels([0.9, 0.5, 0.1], tau_low=0.3, tau_high=0.8)
    assert out == [LABEL_IN, LABEL_UNK, LABEL_OUT]
    assert decide_labels([0.8], 0.3, 0.8) == [LABEL_IN]   # biên: >= tau_high
    assert decide_labels([0.3], 0.3, 0.8) == [LABEL_UNK]  # biên: không < tau_low
    with pytest.raises(ValueError):
        decide_labels([0.5], 0.9, 0.1)


def test_eer_separable_is_zero_and_symmetric_case():
    r = eer([0.9, 0.8, 0.95], [True, True, True], [0.1, 0.2, 0.3])
    assert r["eer"] == 0.0 and r["fmr"] == 0.0 and r["fnmr"] == 0.0
    # trộn hoàn toàn: known=unk -> eer ~ 0.5
    x = np.linspace(0, 1, 200)
    r2 = eer(x, np.ones(200, bool), x)
    assert r2["eer"] == pytest.approx(0.5, abs=0.02)


def test_eer_penalises_wrong_id():
    # tách hoàn toàn theo điểm nhưng top-1 sai ID một nửa -> FNMR >= 0.5
    r = eer([0.9, 0.9, 0.8, 0.8], [True, False, True, False], [0.1, 0.2, 0.1, 0.2])
    assert r["eer"] == pytest.approx(0.5)


def test_invalid_input():
    with pytest.raises(ValueError):
        fmr([], 0.5)
    with pytest.raises(ValueError):
        fmr([np.nan], 0.5)
