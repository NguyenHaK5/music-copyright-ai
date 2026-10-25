"""Test 4A/4B: cổng Test một lần, bootstrap cặp đôi, kết luận theo quy tắc, FMR, đọc gain."""
import json

import numpy as np
import pytest

from src.evaluation import freeze_phase3 as fz
from src.evaluation.test_gate import FINAL_TAG_PREFIX, check_test_allowed, label_of, val_tau
from src.evaluation.test_verdict import (build_verdict, compare, fmr_pooled, load_gains, paired_bootstrap,
                                         per_source_macro, to_markdown)

GROUP_OF = {"cut_a": "cut", "cut_b": "cut", "noise_a": "noise"}


def _manifest(tmp_path, ok=True):
    runs, cfgd = tmp_path / "runs", tmp_path / "configs"
    runs.mkdir(exist_ok=True); cfgd.mkdir(exist_ok=True)
    (cfgd / "targets.yaml").write_text("x: 1\n", encoding="utf-8")
    m = {"ok": ok, "final_config": fz.FINAL, "sha256": {"config_targets.yaml": fz.sha256_file(cfgd / "targets.yaml")},
         "paths": {"runs_dir": str(runs), "config_dir": str(cfgd)}}
    p = tmp_path / "freeze.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    return p, cfgd


def _results(tmp_path, labels, with_test=()):
    rd = tmp_path / "results"
    rd.mkdir(exist_ok=True)
    for l in labels:
        (rd / f"e5_robust_val_{l}.json").write_text(json.dumps({"tau": 0.8}), encoding="utf-8")
    for l in with_test:
        (rd / f"e5_robust_test_{l}.json").write_text("{}", encoding="utf-8")
    return rd


def test_gate_val_always_allowed(tmp_path):
    assert check_test_allowed("val", "eval", False, ["m3"], None, tmp_path, tmp_path / "nope.json") == []


def test_gate_test_conditions(tmp_path):
    mp, cfgd = _manifest(tmp_path)
    tag = f"{FINAL_TAG_PREFIX}42"
    rd = _results(tmp_path, ["m1", f"m3_{tag}"])
    ok = check_test_allowed("test", "eval", True, ["m3"], tag, rd, mp)
    assert ok == []
    assert any("--final-test" in e for e in check_test_allowed("test", "eval", False, ["m1"], None, rd, mp))
    assert any("cấu hình cuối" in e for e in check_test_allowed("test", "eval", True, ["m3"], "seed42", rd, mp))
    assert any("danh sách" in e for e in check_test_allowed("test", "eval", True, ["m9"], None, rd, mp))
    assert any("thiếu" in e for e in check_test_allowed("test", "eval", True, ["m2"], None, rd, mp))      # m2 chưa có tau Val
    (cfgd / "targets.yaml").write_text("x: 2\n", encoding="utf-8")                                       # sửa sau đóng băng
    assert any("Đóng băng" in e for e in check_test_allowed("test", "eval", True, ["m1"], None, rd, mp))


def test_gate_refuses_overwrite_and_unfrozen(tmp_path):
    mp, _ = _manifest(tmp_path)
    rd = _results(tmp_path, ["m1"], with_test=["m1"])
    assert any("một lần" in e for e in check_test_allowed("test", "eval", True, ["m1"], None, rd, mp))
    bad, _ = _manifest(tmp_path / "x") if (tmp_path / "x").mkdir() is None else (None, None)
    m = json.loads(bad.read_text()); m["ok"] = False; bad.write_text(json.dumps(m))
    assert any("Đóng băng" in e for e in check_test_allowed("test", "eval", True, ["m1"], None, _results(tmp_path / "x", ["m1"]), bad))
    assert any("freeze_phase3" in e for e in check_test_allowed("test", "build", True, [], None, rd, tmp_path / "none.json"))


def test_gate_build_refuses_existing_test_set(tmp_path):
    mp, _ = _manifest(tmp_path)
    rb = tmp_path / "robust"; rb.mkdir()
    assert check_test_allowed("test", "build", True, [], None, tmp_path, mp, robust_dir=rb) == []
    (rb / "robust_test_manifest.json").write_text("{}")
    assert any("đã được sinh" in e for e in check_test_allowed("test", "build", True, [], None, tmp_path, mp, robust_dir=rb))


def test_label_and_val_tau(tmp_path):
    assert label_of("m3", "abl_x") == "m3_abl_x" and label_of("m1", "abl_x") == "m1" and label_of("m3", None) == "m3"
    rd = _results(tmp_path, ["m1"])
    assert val_tau(rd, "m1") == 0.8


def _pq(hit_by_cfg: dict, srcs, with_unknown=None, ca=None):
    out = {}
    for c, h in hit_by_cfg.items():
        out[f"{c}__hit1"] = np.asarray(h, float)
        out[f"{c}__ca"] = np.asarray(ca if ca is not None else h, float)
        out[f"{c}__src"] = np.asarray(srcs)
        if with_unknown is not None:
            out[f"{c}__unk_top1"] = np.asarray(with_unknown, float)
            out[f"{c}__unk_src"] = np.arange(len(with_unknown)) + 1000
    return out


def test_per_source_macro_group_then_mean():
    pq = _pq({"cut_a": [1, 0], "cut_b": [0, 0], "noise_a": [1, 1]}, [10, 11])
    ps = per_source_macro(pq, "hit1", GROUP_OF)
    assert ps[10] == pytest.approx(np.mean([np.mean([1, 0]), 1.0])) and ps[11] == pytest.approx(np.mean([0.0, 1.0]))


def test_paired_bootstrap_and_compare_rules():
    rng = np.random.default_rng(0)
    m1 = {i: float(x) for i, x in enumerate(rng.uniform(0.4, 0.6, 300))}
    big = [{i: min(1.0, v + 0.3) for i, v in m1.items()} for _ in range(3)]
    c = compare(m1, big, gain=0.1)
    assert c["passed"] and c["ci95"][0] > 0 and c["diff"] == pytest.approx(0.3, abs=0.02) and all(p["meets_gain"] for p in c["per_seed"])
    small = [{i: v + 0.02 for i, v in m1.items()} for _ in range(3)]
    c2 = compare(m1, small, gain=0.1)
    assert not c2["passed"] and c2["ci95"][0] > 0                      # CI > 0 nhưng chưa đạt gain -> KHÔNG ĐẠT
    zero = [{i: v + rng.normal(0, 0.2) * 0 for i, v in m1.items()} for _ in range(2)]
    assert not compare(m1, zero, gain=0.0)["passed"]                   # hiệu 0: cận dưới CI không > 0
    b1, b2 = paired_bootstrap(np.arange(50.0)), paired_bootstrap(np.arange(50.0))
    assert b1 == b2                                                    # xác định theo seed


def test_fmr_pooled():
    pq = _pq({"cut_a": [1, 1], "noise_a": [1, 1]}, [1, 2], with_unknown=[0.9, 0.5, 0.95, 0.1])
    f = fmr_pooled(pq, tau=0.8)
    assert f["fmr"] == pytest.approx(0.5) and f["per_config"]["cut_a"] == pytest.approx(0.5)
    assert f["ci95"][0] <= 0.5 <= f["ci95"][1] and f["n_unknown_sources"] == 4


def test_load_gains(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("targets:\n  m3_vs_m1_min_gain_macro_recall_at_1: 0.0945\n  m3_vs_m1_min_gain_macro_correct_accept: 0.2233\n  fmr: 0.01\n")
    g = load_gains(p)
    assert g["gain_recall_at_1"] == 0.0945 and g["gain_correct_accept"] == 0.2233 and g["source"] == "targets.yaml"
    assert load_gains(p, cli_r1=0.1)["gain_recall_at_1"] == 0.1 and load_gains(p, cli_r1=0.1)["source"] == "dòng lệnh"
    q = tmp_path / "u.yaml"
    q.write_text("a: 1\n")
    with pytest.raises(ValueError, match="gain"):
        load_gains(q)


def test_build_verdict_and_markdown_end_to_end(tmp_path):
    rd = tmp_path
    srcs = list(range(60))
    rng = np.random.default_rng(1)
    base = rng.uniform(0.3, 0.6, 60)
    def write(label, p_hit, tau=0.8):
        hit = {c: (rng.uniform(size=60) < p_hit).astype(float) for c in GROUP_OF}
        np.savez(rd / f"e5_robust_test_{label}_perquery.npz", **_pq(hit, srcs, with_unknown=rng.uniform(0.2, 0.9, 40)))
        groups = {g: {"recall_at_1": p_hit, "correct_accept_at_tau": p_hit * 0.9} for g in ("cut", "noise")}
        (rd / f"e5_robust_test_{label}.json").write_text(json.dumps({
            "tau": tau, "macro": {"recall_at_1": p_hit, "correct_accept_at_tau": p_hit * 0.9}, "groups": groups, "n_queries": 300}))
    write("m0", 0.01); write("m1", 0.4); write("m2", 0.7)
    for s in (42, 43, 44):
        write(f"m3_abl_noise_coherent_k2_seed{s}", 0.95)
    v = build_verdict(rd, {"gain_recall_at_1": 0.09, "gain_correct_accept": 0.2, "source": "test"}, GROUP_OF)
    assert v["verdict_primary"] == "ĐẠT" and v["macro_recall_at_1"]["ci95"][0] > 0
    assert set(v["models"]) >= {"m0", "m1", "m2"} and "fmr_ci95" in v["models"]["m1"]
    md = to_markdown(v, groups_order=("cut", "noise"))
    assert "**ĐẠT**" in md and "M3 (3 seed" in md and "macro Recall@1 (chính)" in md
