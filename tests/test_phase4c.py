"""Test 4C: phân tích lỗi (bảng cấu hình, chênh Val-Test, thể loại) và đo chi phí."""
import json

import numpy as np
import pytest

from src.evaluation.error_analysis import config_table, genre_table, to_markdown, val_test_gap
from src.evaluation.measure_cost import deep_nbytes, latency_stats, measure_model, to_markdown as cost_md

CFGS = {"cut_a": "cut", "noise_a": "noise"}
FINAL = "noise_coherent_k2"


def _j(r1, fmr=0.01):
    return {"configs": {c: {"recall_at_1": r1, "correct_accept_at_tau": r1 * 0.9, "fmr_at_tau": fmr} for c in CFGS}}


def _write(rd, split, label, r1):
    (rd / f"e5_robust_{split}_{label}.json").write_text(json.dumps(_j(r1)), encoding="utf-8")


def test_config_table_and_gap(tmp_path):
    for split, base in (("val", 0.95), ("test", 0.90)):
        _write(tmp_path, split, "m1", 0.5); _write(tmp_path, split, "m2", 0.7)
        for i, s in enumerate((42, 43, 44)):
            _write(tmp_path, split, f"m3_abl_{FINAL}_seed{s}", base + i * 0.01)
    t = config_table(tmp_path)
    assert t["cut_a"]["m1"]["recall_at_1"] == 0.5
    assert t["cut_a"]["m3"]["recall_at_1"]["mean"] == pytest.approx(0.91) and t["cut_a"]["m3"]["recall_at_1"]["std"] == pytest.approx(0.01)
    g = val_test_gap(tmp_path, flag=0.06)
    assert g["cut_a"]["delta"] == pytest.approx(-0.05) and g["cut_a"]["val"] == pytest.approx(0.96) and not g["cut_a"]["flag"]
    assert val_test_gap(tmp_path, flag=0.04)["noise_a"]["flag"]


def _pq(hits, srcs):
    out = {}
    for c, h in hits.items():
        out[f"{c}__hit1"], out[f"{c}__src"] = np.asarray(h, float), np.asarray(srcs)
    return out


def test_genre_table_small_n_flag_and_values():
    srcs = list(range(20))
    genre_of = {s: ("rock" if s < 16 else "jazz") for s in srcs}
    one, zero = {c: np.ones(20) for c in CFGS}, {c: np.zeros(20) for c in CFGS}
    labels = ["m3a", "m3b"]
    pqs = {"m1": _pq(zero, srcs), "m2": _pq(one, srcs), "m3a": _pq(one, srcs), "m3b": _pq(one, srcs)}
    rows = genre_table(pqs, genre_of, CFGS, labels)
    d = {r["genre"]: r for r in rows}
    assert d["rock"]["n_sources"] == 16 and not d["rock"]["small_n"] and d["jazz"]["small_n"]
    assert d["rock"]["m1"] == 0.0 and d["rock"]["m3"] == 1.0
    md = to_markdown({"cut_a": {"m1": {"recall_at_1": .5, "fmr": .01}, "m2": {"recall_at_1": .7, "fmr": .0},
                                "m3": {k: {"mean": .9, "std": .01} for k in ("recall_at_1", "correct_accept", "fmr")}}},
                     {"cut_a": {"val": .95, "test": .9, "delta": -.05, "flag": True}}, rows)
    assert "jazz (n nhỏ)" in md and "⚠" in md


def test_deep_nbytes_counts_shared_once():
    a = np.zeros(100, dtype=np.float32)
    class Box: pass
    b = Box(); b.x = a; b.y = [a, np.ones(10, dtype=np.float64)]; b.z = {"k": (a,)}
    assert deep_nbytes(b) == 400 + 80
    assert deep_nbytes(object()) == 0


def test_latency_and_measure_model():
    calls = {"n": 0}
    s = latency_stats(lambda: calls.__setitem__("n", calls["n"] + 1), n=10, warmup=2)
    assert calls["n"] == 12 and s["n"] == 10 and s["p95_ms"] >= s["median_ms"] >= 0

    class Dummy:
        index_size = 7
        def build_index(self, ids, loader):
            self.w = np.zeros((len(ids), 4), np.float32)
        def score_matrix(self, qs):
            return np.zeros((len(qs), len(self.w)), np.float32)
    r = measure_model(Dummy(), [1, 2, 3], lambda t: None, [np.zeros((4, 4))] * 5, n_single=5)
    assert r["index_bytes"] == 3 * 4 * 4 and r["n_index_tracks"] == 3 and r["n_batch_queries"] == 5
    md = cost_md({"models": {"dummy": r}, "machine": {"platform": "x", "cpu_count": 1}, "n_index_tracks": 3, "n_batch_queries": 5})
    assert "dummy" in md
