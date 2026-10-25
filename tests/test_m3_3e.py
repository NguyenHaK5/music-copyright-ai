"""Test 3E: ghi đè cấu hình/ablation, tổng hợp seed/ablation, chẩn đoán nhiễu."""
import json

import numpy as np
import pytest
import yaml

from src.evaluation.summarize_m3 import GROUPS, collect, summarize, to_markdown

torch = pytest.importorskip("torch")
from src.train_m3 import apply_overrides


def test_apply_overrides_and_validation():
    cfg = {"augment": {"noise": {"p": 0.6, "snr_db_min": 3.0}}, "model": {"pool": "freq_flatten", "channels": [16, 32]}}
    out = apply_overrides(cfg, ["augment.noise.p=0", "model.pool=gap", "model.channels=[8,16]", "augment.noise.snr_db_min=-5.0"])
    assert out["augment"]["noise"] == {"p": 0, "snr_db_min": -5.0} and out["model"] == {"pool": "gap", "channels": [8, 16]}
    assert cfg["augment"]["noise"]["p"] == 0.6                      # không sửa bản gốc
    for bad in ("augment.nois.p=0", "augment.noise.q=1", "khongco=1", "noequals"):
        with pytest.raises((KeyError, ValueError)):
            apply_overrides(cfg, [bad])


def test_all_ablations_valid_against_m3_yaml():
    cfg = yaml.safe_load(open("configs/m3.yaml", encoding="utf-8"))
    ab = yaml.safe_load(open("configs/m3_ablations.yaml", encoding="utf-8"))
    assert set(ab["priority"]) <= set(ab["ablations"])
    for name, v in ab["ablations"].items():
        assert v["kind"] in ("ablation", "tuning") and v["desc"]
        apply_overrides(cfg, v["set"])                             # không ném lỗi


def _res(r1, ca, groups=None):
    groups = groups or {g: r1 for g in GROUPS}
    return {"tau": 0.8, "macro": {"recall_at_1": r1, "correct_accept_at_tau": ca},
            "groups": {g: {"recall_at_1": v, "correct_accept_at_tau": v * 0.9} for g, v in groups.items()}}


def test_summarize_seeds_ablations_and_markdown(tmp_path):
    for label, payload in {"m3_seed42": _res(0.90, 0.80), "m3_seed43": _res(0.92, 0.82), "m3_seed44": _res(0.88, 0.78),
                           "m3_abl_no_noise_seed42": _res(0.85, 0.70), "m1": _res(0.62, 0.10), "m2": _res(0.79, 0.71)}.items():
        (tmp_path / f"e5_robust_val_{label}.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "e5_robust_val_m3_seed42_perquery.npz").write_bytes(b"x")     # phải bị bỏ qua
    res = collect(tmp_path)
    assert "m3_seed42_perquery" not in res and len(res) == 6
    s = summarize(res)
    ms = s["m3_mean_std"]
    assert ms["n_seeds"] == 3 and ms["macro_recall_at_1"]["mean"] == pytest.approx(0.90)
    assert ms["macro_recall_at_1"]["std"] == pytest.approx(0.02)
    d = s["ablations"]["no_noise"]["delta_vs_seed_baseline"]
    assert d["macro_recall_at_1"] == pytest.approx(-0.05) and d["group_recall_at_1"]["noise"] == pytest.approx(-0.05)
    md = to_markdown(s)
    assert "M3 (n=3 seed" in md and "no_noise" in md and "| m1 |" in md


def test_summarize_single_seed_has_no_std(tmp_path):
    (tmp_path / "e5_robust_val_m3_seed42.json").write_text(json.dumps(_res(0.9, 0.8)), encoding="utf-8")
    s = summarize(collect(tmp_path))
    assert s["m3_mean_std"]["macro_recall_at_1"]["std"] is None and s["ablations"] == {}


def test_noise_check_core_on_synthetic(tmp_path):
    from src.analysis.m3_noise_check import run_check, tilt_db
    from src.models.m3_cnn import build_model
    from src.models.m3_encoder import M3Encoder
    from src.models.m3_retrieval import M3Retriever
    F = 128
    cfg = {"data": {"n_mels": F, "sr": 22050, "hop_length": 512},
           "model": {"channels": [4, 8], "blocks_per_stage": 1, "embed_dim": 16, "pool": "freq_flatten", "dropout": 0.0}}
    torch.manual_seed(0)
    m = build_model(cfg).train()
    for _ in range(3):
        m(torch.randn(8, 1, F, 60))
    p = tmp_path / "c.pt"
    torch.save({"model": m.state_dict(), "cfg": cfg, "norm_std": 12.0}, p)
    r = M3Retriever(M3Encoder.from_checkpoint(p, device="cpu"), 215, 107)
    tr = {10 + i: (np.linspace(-10 - 5 * i, -60, F)[:, None] + np.random.default_rng(i).normal(0, 4, (F, 500))).astype(np.float32)
          for i in range(5)}
    r.build_index(list(tr), lambda t: tr[t])
    clean = np.stack([tr[t][:, :215] for t in tr])
    true = np.arange(5)
    real = {"snr_10db": clean + np.random.default_rng(1).normal(0, 1, clean.shape).astype(np.float32)}
    out = run_check(r, clean, true, real, snr_grid=(0, 10, 20), tilt_sample=5, recall_grid=(0, 10))
    c = out["conditions"]["snr_10db"]
    assert out["clean"]["recall_at_1"] == 1.0 and c["nominal_snr_db"] == 10.0
    assert set(c) >= {"real", "synthetic_same_nominal", "synthetic_coherent_same_nominal", "equivalent_synthetic_snr_db_by_tilt",
                      "equivalent_snr_by_recall_additive", "equivalent_snr_by_recall_coherent"}
    assert set(out["recall_by_snr"]) == {"additive", "coherent"} and set(out["recall_by_snr"]["coherent"]) == {"0", "10"}
    assert c["equivalent_synthetic_snr_db_by_tilt"] in (0, 10, 20)
    assert tilt_db(np.vstack([np.zeros((96, 10)), np.ones((32, 10))])) == pytest.approx(1.0)


def test_coherent_noise_properties():
    from src.features.augment_mel import MelAugmenter, add_noise, add_noise_coherent
    rng = np.random.default_rng(0)
    base = np.full((128, 400), -30.0, dtype=np.float32)
    base[10:20, :] += 10.0                                  # vùng tín hiệu mạnh hơn nền 10 dB
    p = 10 ** (base.astype(np.float64) / 10)
    a = add_noise(base, 5.0, np.random.default_rng(1))
    c = add_noise_coherent(base, 5.0, np.random.default_rng(1))
    pa, pc = 10 ** (a.astype(np.float64) / 10), 10 ** (c.astype(np.float64) / 10)
    assert (pa - p).mean() == pytest.approx((pc - p).mean(), rel=0.1)       # cùng công suất nhiễu kỳ vọng
    strong = slice(10, 20)
    assert (c - base)[strong].std() > 3 * (a - base)[strong].std()          # giao thoa pha làm ô mạnh dao động hơn nhiều
    assert (a - base)[strong].min() >= -1e-4                                 # cộng công suất thuần chỉ làm tăng
    assert (c - base)[strong].min() < -0.5                                   # nhiễu kết hợp có thể làm giảm
    with pytest.raises(ValueError, match="noise.mode"):
        MelAugmenter({"noise": {"p": 1.0, "mode": "bogus"}})
    aug = MelAugmenter({"noise": {"p": 1.0, "mode": "coherent"}}, 22050, 128)
    out, ops = aug(base.copy(), 215, np.random.default_rng(3))
    assert ops == ["noise"] and np.isfinite(out).all()


def test_coherent_k_interpolates_toward_additive():
    from src.features.augment_mel import MelAugmenter, add_noise_coherent
    base = np.full((128, 400), -30.0, dtype=np.float32)
    base[10:20, :] += 10.0
    sd = {k: (add_noise_coherent(base, 5.0, np.random.default_rng(1), k=k) - base)[10:20].std() for k in (1, 4, 16)}
    assert sd[1] > sd[4] > sd[16]                                            # k lớn -> dao động giảm
    p = 10 ** (base.astype(np.float64) / 10)
    p4 = 10 ** (add_noise_coherent(base, 5.0, np.random.default_rng(2), k=4).astype(np.float64) / 10)
    assert (p4 - p).mean() == pytest.approx(p.mean() * 10 ** -0.5, rel=0.1)  # giữ công suất nhiễu kỳ vọng
    with pytest.raises(ValueError):
        add_noise_coherent(base, 5.0, np.random.default_rng(0), k=0)
    with pytest.raises(ValueError, match="coherent_k"):
        MelAugmenter({"noise": {"p": 1.0, "mode": "coherent", "coherent_k": 0}})
    aug = MelAugmenter({"noise": {"p": 1.0, "mode": "coherent", "coherent_k": 4}}, 22050, 128)
    assert aug(base.copy(), 215, np.random.default_rng(0))[1] == ["noise"]


def test_noise_check_k_candidates(tmp_path):
    from src.analysis.m3_noise_check import run_check
    from src.models.m3_cnn import build_model
    from src.models.m3_encoder import M3Encoder
    from src.models.m3_retrieval import M3Retriever
    F = 128
    cfg = {"data": {"n_mels": F, "sr": 22050, "hop_length": 512},
           "model": {"channels": [4, 8], "blocks_per_stage": 1, "embed_dim": 16, "pool": "freq_flatten", "dropout": 0.0}}
    torch.manual_seed(0)
    m = build_model(cfg).train()
    for _ in range(3):
        m(torch.randn(8, 1, F, 60))
    torch.save({"model": m.state_dict(), "cfg": cfg, "norm_std": 12.0}, tmp_path / "c.pt")
    r = M3Retriever(M3Encoder.from_checkpoint(tmp_path / "c.pt", device="cpu"), 215, 107)
    tr = {10 + i: (np.linspace(-10 - 5 * i, -60, F)[:, None] + np.random.default_rng(i).normal(0, 4, (F, 500))).astype(np.float32)
          for i in range(5)}
    r.build_index(list(tr), lambda t: tr[t])
    clean = np.stack([tr[t][:, :215] for t in tr])
    real = {"snr_10db": clean + np.random.default_rng(1).normal(0, 1, clean.shape).astype(np.float32)}
    out = run_check(r, clean, np.arange(5), real, snr_grid=(0, 10), tilt_sample=5, recall_grid=(0,), k_candidates=(2, 4))
    assert set(out["coherent_k"]) == {"1", "2", "4"} and str(out["best_k_by_l1"]) in out["coherent_k"]
    assert all(v["l1_to_real"] >= 0 for v in out["coherent_k"].values())


def test_summarize_multiseed_ablation_mean_std_and_delta(tmp_path):
    base = {"m3_seed42": _res(0.90, 0.80), "m3_seed43": _res(0.92, 0.82)}
    var = {"m3_abl_noise_coherent_k2_seed42": _res(0.95, 0.88), "m3_abl_noise_coherent_k2_seed43": _res(0.97, 0.90),
           "m3_abl_noise_coherent_k2_seed44": _res(0.96, 0.89)}
    for label, payload in {**base, **var}.items():
        (tmp_path / f"e5_robust_val_{label}.json").write_text(json.dumps(payload), encoding="utf-8")
    s = summarize(collect(tmp_path))
    e = s["ablations"]["noise_coherent_k2"]
    assert set(e["per_seed"]) == {"42", "43", "44"}                       # không còn ghi đè nhau
    assert e["mean_std"]["n_seeds"] == 3 and e["mean_std"]["macro_recall_at_1"]["mean"] == pytest.approx(0.96)
    assert e["delta_mean_vs_baseline_mean"]["macro_recall_at_1"] == pytest.approx(0.96 - 0.91)
    assert e["delta_vs_seed_baseline"]["macro_recall_at_1"] == pytest.approx(0.05)
    md = to_markdown(s)
    assert "noise_coherent_k2 (n=3 seed" in md and "noise_coherent_k2 (seed 42)" in md
