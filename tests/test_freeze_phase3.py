"""Test 3F: kiểm tra/đóng băng Phase 3."""
import json
import shutil

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.freeze_phase3 import FINAL, freeze, sha256_file, verify_freeze

GROUPS = ["cut", "codec", "noise", "eq", "pitch", "tempo", "overlay"]


def _e5(label, r1, ca):
    return {"label": label, "tau": 0.79, "n_queries": 6800, "missing_groups": [], "skipped_configs": {},
            "macro": {"recall_at_1": r1, "correct_accept_at_tau": ca, "n_groups": 7},
            "groups": {g: {"recall_at_1": r1, "correct_accept_at_tau": ca} for g in GROUPS}}


def _setup(tmp_path, seeds=(42, 43, 44), noise=None, test_touched=False):
    res, runs, cfgd = tmp_path / "results", tmp_path / "runs", tmp_path / "configs"
    for d in (res, runs, cfgd):
        d.mkdir()
    for name in ("targets.yaml", "m3.yaml", "m3_ablations.yaml", "baseline.yaml", "features.yaml"):
        (cfgd / name).write_text(f"# {name}\n", encoding="utf-8")
    for i, s in enumerate(seeds):
        (res / f"e3_train_abl_{FINAL}_seed{s}.json").write_text(json.dumps({
            "split_used": ["train", "val"], "test_touched": test_touched, "epochs_run": 60, "epochs_planned": 60,
            "best_epoch": 60, "stopped_early": False, "ablation": {"name": FINAL}}), encoding="utf-8")
        (res / f"e5_robust_val_m3_abl_{FINAL}_seed{s}.json").write_text(json.dumps(_e5(f"m3_abl_{FINAL}_seed{s}", 0.96 + i * 0.001, 0.88)), encoding="utf-8")
        d = runs / f"m3_abl_{FINAL}_seed{s}"
        d.mkdir()
        torch.save({"cfg": {"augment": {"noise": noise or {"p": 0.6, "mode": "coherent", "coherent_k": 2}}}, "seed": s,
                    "norm_std": 15.189896072421144, "model": {}}, d / "best.pt")
    for s in (42, 43, 44):                       # baseline để bộ tổng hợp có tham chiếu
        (res / f"e5_robust_val_m3_seed{s}.json").write_text(json.dumps(_e5(f"m3_seed{s}", 0.91, 0.82)), encoding="utf-8")
    return res, runs, cfgd


def test_freeze_ok_and_verify(tmp_path):
    res, runs, cfgd = _setup(tmp_path)
    m = freeze(res, runs, cfgd)
    assert m["ok"], [c for c in m["checks"] if not c["ok"]]
    assert m["val_mean_std"]["n_seeds"] == 3 and len(m["sha256"]) == 3 + 5
    mp = tmp_path / "freeze.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    assert verify_freeze(mp) == []
    (cfgd / "targets.yaml").write_text("# đã sửa\n", encoding="utf-8")        # sửa sau khi đóng băng
    assert any("targets.yaml" in x for x in verify_freeze(mp))
    shutil.rmtree(runs / f"m3_abl_{FINAL}_seed44")
    assert any("thiếu" in x for x in verify_freeze(mp))


def test_freeze_fails_on_missing_seed(tmp_path):
    res, runs, cfgd = _setup(tmp_path, seeds=(43, 44))                          # thiếu seed 42 như trường hợp thật
    m = freeze(res, runs, cfgd)
    assert not m["ok"]
    failed = [c["name"] for c in m["checks"] if not c["ok"]]
    assert any("seed42" in n for n in failed) and any("đủ 3 seed" in n for n in failed)
    assert verify_freeze_path(tmp_path, m) != []


def verify_freeze_path(tmp_path, m):
    p = tmp_path / "f.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    return verify_freeze(p)


def test_freeze_fails_on_wrong_noise_test_touch_or_test_results(tmp_path):
    res, runs, cfgd = _setup(tmp_path, noise={"p": 0.6, "mode": "additive", "coherent_k": 1})
    assert not freeze(res, runs, cfgd)["ok"]
    res2, runs2, cfg2 = _setup(tmp_path / "b", seeds=(42, 43, 44)) if (tmp_path / "b").mkdir() is None else (None,) * 3
    (res2 / "e5_robust_test_m3_seed42.json").write_text("{}", encoding="utf-8")
    m = freeze(res2, runs2, cfg2)
    assert not m["ok"] and any("Test" in c["name"] and not c["ok"] for c in m["checks"])
    res3, runs3, cfg3 = _setup(tmp_path / "c", test_touched=True) if (tmp_path / "c").mkdir() is None else (None,) * 3
    assert not freeze(res3, runs3, cfg3)["ok"]


def test_sha256_file(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
