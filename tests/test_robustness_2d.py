import numpy as np
import pandas as pd
import pytest

from src.data import transforms as T
from src.data.robust_queries import CONFIGS, GROUPS, build_robust_set, select_sources
from src.evaluation.make_targets import build_targets
from src.evaluation.robustness import choose_tau_pooled, evaluate_robust

SR = 22050


def _sine(sec=12, f=440.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * f * t) + 0.1 * rng.standard_normal(t.size)).astype(np.float32)


def _snr(clean, noisy):
    n = noisy - clean
    return 10 * np.log10(np.mean(clean.astype(float) ** 2) / np.mean(n.astype(float) ** 2))


def test_configs_cover_seven_groups():
    assert {c.group for c in CONFIGS} == set(GROUPS)
    assert len({c.name for c in CONFIGS}) == len(CONFIGS) == 17


@pytest.mark.parametrize("snr", [5, 10, 20])
def test_noise_snr(snr):
    y = _sine()
    z = T.add_noise_snr(y, snr, np.random.default_rng(1))
    assert _snr(y, z) == pytest.approx(snr, abs=0.05)


def test_overlay_snr_and_silence_guard():
    y, o = _sine(5), _sine(5, 1000.0, 3)
    z = T.overlay_snr(y, o, 5.0)
    assert _snr(y, z) == pytest.approx(5.0, abs=0.05)
    with pytest.raises(ValueError):
        T.overlay_snr(y, np.zeros(100, np.float32), 5.0)


def test_filters_remove_band():
    t = np.arange(SR * 2) / SR
    lo = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    hi = np.sin(2 * np.pi * 8000 * t).astype(np.float32)
    assert np.std(T.lowpass(hi, SR, 3000)) < 0.01 * np.std(hi)
    assert np.std(T.highpass(lo, SR, 500)) < 0.2 * np.std(lo)
    assert np.std(T.lowpass(lo, SR, 3000)) > 0.9 * np.std(lo)


def test_time_stretch_length():
    y = _sine(5)
    assert len(T.time_stretch(y, 1.10)) == pytest.approx(len(y) / 1.10, rel=0.02)
    assert len(T.time_stretch(y, 0.90)) == pytest.approx(len(y) / 0.90, rel=0.02)


@pytest.mark.skipif(not T.ffmpeg_available(), reason="không có ffmpeg")
def test_mp3_roundtrip_shape():
    y = _sine(5)
    z = T.mp3_roundtrip(y, SR, 64)
    assert abs(len(z) - len(y)) < SR * 0.2 and np.isfinite(z).all()


def test_select_sources_deterministic():
    df = pd.DataFrame({"track_id": range(60), "split": "val",
                       "role": ["known"] * 30 + ["unknown"] * 30})
    a, b = select_sources(df, "val", 10, 5), select_sources(df, "val", 10, 5)
    pd.testing.assert_frame_equal(a, b)
    assert a["is_known"].sum() == 10 and len(a) == 20


def test_build_robust_set_small(tmp_path):
    src = pd.DataFrame({"track_id": [1, 2, 3, 4], "is_known": [True, True, False, False]})
    audio = {i: _sine(12, 300 + 50 * i, i) for i in range(1, 6)}
    msgs = []
    man = build_robust_set(src, [5], lambda t: audio[t], SR, 2048, 512, 128, tmp_path, "val", 7,
                           configs=[c for c in CONFIGS if c.kind != "mp3"][:8], log=msgs.append)
    assert man["skipped"] == {} and len(man["configs"]) == 8
    z = np.load(tmp_path / "robust_val_crop_5s.npz")
    assert z["mels"].shape[0] == 4 and z["mels"].dtype == np.float16
    assert z["is_known"].tolist() == [True, True, False, False]
    # tái lập
    build_robust_set(src, [5], lambda t: audio[t], SR, 2048, 512, 128, tmp_path / "b", "val", 7,
                     configs=[c for c in CONFIGS if c.name == "snr_10db"], log=msgs.append)
    build_robust_set(src, [5], lambda t: audio[t], SR, 2048, 512, 128, tmp_path / "c", "val", 7,
                     configs=[c for c in CONFIGS if c.name == "snr_10db"], log=msgs.append)
    np.testing.assert_array_equal(np.load(tmp_path / "b" / "robust_val_snr_10db.npz")["mels"],
                                  np.load(tmp_path / "c" / "robust_val_snr_10db.npz")["mels"])


def _fake_scores(rng, n_known=20, n_unk=20, n_idx=15, good=0.9):
    idx = list(range(n_idx))
    src = np.array(list(range(n_known)) + list(range(100, 100 + n_unk)))
    # truy vấn known thuộc chỉ mục: dùng id 0..n_idx-1 tuần hoàn
    src = np.array([i % n_idx for i in range(n_known)] + list(range(100, 100 + n_unk)))
    known = np.array([True] * n_known + [False] * n_unk)
    S = rng.random((n_known + n_unk, n_idx)) * 0.3
    for q in range(n_known):
        S[q, src[q]] = good + rng.random() * 0.05
    return {"S": S, "index_ids": idx, "source_ids": src, "is_known": known}


def test_pooled_tau_and_macro():
    rng = np.random.default_rng(0)
    by = {c.name: _fake_scores(rng) for c in CONFIGS}
    tau = choose_tau_pooled(by, 0.01)
    r = evaluate_robust(by, tau=None, max_fmr=0.01)
    assert r["tau"] == tau and r["macro"]["n_groups"] == 7 and not r["missing_groups"]
    pooled_unk = np.concatenate([d["S"][~d["is_known"]].max(1) for d in by.values()])
    assert (pooled_unk >= tau).mean() <= 0.01 + 1e-9
    assert r["macro"]["recall_at_1"] == pytest.approx(1.0)
    # tau cho trước (Test) không bị chọn lại
    assert evaluate_robust(by, tau=0.5)["tau"] == 0.5


def test_evaluate_rejects_unknown_in_index():
    rng = np.random.default_rng(1)
    d = _fake_scores(rng)
    d["source_ids"] = d["source_ids"].copy()
    d["source_ids"][-1] = 3  # unknown trùng chỉ mục
    with pytest.raises(ValueError):
        evaluate_robust({"crop_5s": d}, tau=0.5)


def test_missing_group_blocks_targets():
    rng = np.random.default_rng(2)
    full = {c.name: _fake_scores(rng) for c in CONFIGS}
    part = {c.name: _fake_scores(rng) for c in CONFIGS if c.group != "codec"}
    ok = {n: _jsonable(evaluate_robust(full)) for n in ("m0", "m1", "m2")}
    t = build_targets(ok, 0.01)
    assert t["frozen"] is True and t["targets"]["m3_vs_m1_min_gain"]["macro_recall_at_1"] >= 0.02
    bad = dict(ok)
    bad["m1"] = _jsonable(evaluate_robust(part))
    with pytest.raises(ValueError):
        build_targets(bad, 0.01)


def _jsonable(r):
    return {"tau": r["tau"], "macro": r["macro"], "missing_groups": r["missing_groups"]}
