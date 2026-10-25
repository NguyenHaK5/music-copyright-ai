"""Chạy baseline M0/M1/M2 trên VAL với bộ truy vấn sạch dùng chung.

    python -m src.run_baselines --split val --models m0 m1 m2

Chỉ cho phép split=val. Test chỉ chạy MỘT lần ở Phase 4.
Thiếu đầu vào -> thoát mã 2, in 'CHƯA CHẠY'.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from src.common.config import load_config
from src.common.paths import resolve
from src.data.queries import QueryMels, sample_query_specs
from src.evaluation.runner import evaluate_scores
from src.features.cache import load_logmel
from src.models.m0_trivial import M0Random
from src.models.m1_classical import M1Classical, sample_train_features
from src.models.m2_fingerprint import M2Fingerprint


def build_model(name: str, cfg: dict, tracks: pd.DataFrame, cache_dir):
    sr, hop = cfg["sr"], cfg["hop_length"]
    if name == "m0":
        return M0Random(seed=cfg["queries"]["seed"])
    if name == "m1":
        c = cfg["m1"]
        win, hp = round(c["win_sec"] * sr / hop), round(c["hop_sec"] * sr / hop)
        model = M1Classical(win, hp, c["n_mfcc"])
        tr = tracks[tracks["split"] == "train"]["track_id"].astype(int)
        X = sample_train_features(tr, cache_dir, c["fit_tracks"], c["fit_windows_per_track"],
                                  win, c["n_mfcc"], c["seed"])
        return model.fit(X)
    if name == "m2":
        return M2Fingerprint(**cfg["m2"], frames_per_sec=sr / hop)
    if name == "m3":
        from src.models.m3_retrieval import build_m3
        return build_m3(cfg)
    raise ValueError(f"Mô hình chưa hỗ trợ: {name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--models", nargs="+", default=["m0", "m1"])
    ap.add_argument("--m3-ckpt", default=None, help="đường dẫn checkpoint M3 (best.pt) khi chạy m3")
    ap.add_argument("--tag", default=None, help="hậu tố tên file kết quả của m3 (vd seed42)")
    a = ap.parse_args(argv)
    if a.split != "val":
        print("Từ chối: Test chỉ chạy một lần ở Phase 4 (--split val).", file=sys.stderr)
        return 2
    cfg = load_config("baseline")
    if a.m3_ckpt:
        cfg["m3"] = {**(cfg.get("m3") or {}), "ckpt": a.m3_ckpt}
    tpath, cache_dir = resolve(cfg["tracks_path"]), resolve(cfg["logmel_cache_dir"])
    if not tpath.exists() or not cache_dir.exists():
        print(f"CHƯA CHẠY: thiếu {tpath} hoặc {cache_dir} (hoàn thành Phase 1 trước).", file=sys.stderr)
        return 2
    tracks = pd.read_parquet(tpath)
    if "track_id" not in tracks.columns:
        tracks = tracks.reset_index()

    q = cfg["queries"]
    qpath = resolve(q["dir"]) / f"queries_{a.split}_clean.parquet"
    if qpath.exists():
        specs = pd.read_parquet(qpath)
    else:
        specs = sample_query_specs(tracks, a.split, cache_dir, q["crop_sec"], cfg["sr"],
                                   cfg["hop_length"], q["n_per_known"], q["n_per_unknown"], q["seed"])
        qpath.parent.mkdir(parents=True, exist_ok=True)
        specs.to_parquet(qpath)
    idx = tracks[(tracks["split"] == a.split) & (tracks["role"] == "known")]
    index_ids = sorted(int(t) for t in idx["track_id"])
    queries = QueryMels(cache_dir, specs)
    o = cfg["openset"]
    out_dir = resolve(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in a.models:
        label = f"{name}_{a.tag}" if (a.tag and name == "m3") else name
        model = build_model(name, cfg, tracks, cache_dir)
        t0 = time.perf_counter()
        model.build_index(index_ids, lambda t: load_logmel(cache_dir, t))
        t1 = time.perf_counter()
        S = model.score_matrix(queries)
        t2 = time.perf_counter()
        res = evaluate_scores(S, index_ids, specs["source_track_id"], specs["is_known"],
                              o["max_fmr"], o["max_loss"], o["topk"], o["bootstrap"], q["seed"])
        res.update({
            "model": name, "label": label, "ckpt": a.m3_ckpt if name == "m3" else None,
            "split": a.split, "query_type": "clean_crop",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": q["seed"], "crop_sec": q["crop_sec"],
            "timing_sec": {"build_index": t1 - t0, "score_all_queries": t2 - t1,
                           "score_per_query_ms": 1000 * (t2 - t1) / len(specs)},
            "index_hashes": getattr(model, "index_size", None),
            "note": "Số trên Val dùng để chọn ngưỡng/mục tiêu; chưa phải kết quả cuối. Truy vấn là crop của clip đã index.",
        })
        p = out_dir / f"e1_{label}_{a.split}.json"
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        r1, r5 = res["recall_at_1"]["value"], res["recall_at_5"]["value"]
        print(f"[{label}] R@1={r1:.4f} R@5={r5:.4f} MRR={res['mrr']['value']:.4f} "
              f"AUC={res['roc_auc_known_vs_unknown']:.4f} EER={res['eer']['eer']:.4f} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
