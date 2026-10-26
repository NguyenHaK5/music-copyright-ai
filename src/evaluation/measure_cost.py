"""Độ trễ và dung lượng chỉ mục (Phase 4C, thí nghiệm E9). Đo trên Val (không chạm Test), chỉ mục ~800 bài.

Báo kèm cấu hình máy: số liệu là benchmark của đồ án trên máy này, không phải tốc độ cố định.
CLI: python -m src.evaluation.measure_cost [--m3-ckpt experiments/runs/m3_abl_noise_coherent_k2_seed42/best.pt]
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np


def deep_nbytes(obj, _seen=None, _depth: int = 0) -> int:
    """Tổng byte của mọi mảng numpy tới được từ obj (dict/list/tuple/đối tượng); mảng dùng chung chỉ tính một lần.
    Dùng để ước lượng dung lượng CHỈ MỤC (không tính trọng số mô hình torch)."""
    _seen = set() if _seen is None else _seen
    if id(obj) in _seen or _depth > 6:
        return 0
    _seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(deep_nbytes(v, _seen, _depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(deep_nbytes(v, _seen, _depth + 1) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return sum(deep_nbytes(v, _seen, _depth + 1) for v in vars(obj).values())
    return 0


def latency_stats(fn, n: int = 50, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ts)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean()), "n": int(n)}


def measure_model(model, index_ids, mel_loader, queries: list, n_single: int = 50) -> dict:
    t0 = time.perf_counter()
    model.build_index(index_ids, mel_loader)
    build = time.perf_counter() - t0
    i = {"k": 0}

    def one():
        model.score_matrix([queries[i["k"] % len(queries)]])
        i["k"] += 1

    single = latency_stats(one, n=n_single)
    t0 = time.perf_counter()
    model.score_matrix(queries)
    batch = time.perf_counter() - t0
    return {"build_index_sec": build, "single_query": single, "batch_ms_per_query": batch * 1000 / len(queries),
            "n_batch_queries": len(queries), "index_bytes": deep_nbytes(model), "index_size_units": getattr(model, "index_size", None),
            "n_index_tracks": len(index_ids)}


def machine_info() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version()}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available(), torch_threads=torch.get_num_threads())
    except Exception:
        pass
    return info


def to_markdown(res: dict) -> str:
    L = ["| Mô hình | dựng chỉ mục (s) | độ trễ 1 truy vấn, trung vị / p95 (ms) | ms/truy vấn khi chấm cả lô | chỉ mục (MB) |", "|---|---|---|---|---|"]
    for k, r in res["models"].items():
        L.append(f"| {k} | {r['build_index_sec']:.1f} | {r['single_query']['median_ms']:.1f} / {r['single_query']['p95_ms']:.1f} | "
                 f"{r['batch_ms_per_query']:.1f} | {r['index_bytes'] / 1e6:.2f} |")
    m = res["machine"]
    L += ["", f"Máy: {m.get('platform')}, {m.get('cpu_count')} luồng CPU, torch {m.get('torch')} (CUDA: {m.get('cuda')}); "
          f"chỉ mục {res['n_index_tracks']} bài Val, {res['n_batch_queries']} truy vấn crop 5 s."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    import argparse
    import pandas as pd
    from src.common.config import load_config
    from src.common.paths import resolve
    from src.features.cache import load_logmel
    from src.run_baselines import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--m3-ckpt", default="experiments/runs/m3_abl_noise_coherent_k2_seed42/best.pt")
    ap.add_argument("--models", nargs="+", default=["m1", "m2", "m3"])
    ap.add_argument("--n-queries", type=int, default=200)
    a = ap.parse_args(argv)
    cfg = load_config("baseline")
    cfg["m3"] = {**(cfg.get("m3") or {}), "ckpt": a.m3_ckpt}
    tp, rp = resolve(cfg["tracks_path"]), resolve(cfg["robust"]["dir"]) / "robust_val_crop_5s.npz"
    if not tp.exists() or not rp.exists():
        print("CHƯA CHẠY: thiếu tracks.parquet hoặc robust_val_crop_5s.npz", file=sys.stderr)
        return 2
    tracks = pd.read_parquet(tp)
    if "track_id" not in tracks.columns:
        tracks = tracks.reset_index()
    cache_dir = resolve(cfg["logmel_cache_dir"])
    index_ids = sorted(int(t) for t in tracks[(tracks["split"] == "val") & (tracks["role"] == "known")]["track_id"])
    z = np.load(rp)
    qs = [m.astype(np.float32) for m in z["mels"][z["is_known"].astype(bool)][:a.n_queries]]
    res = {"split": "val", "n_index_tracks": len(index_ids), "n_batch_queries": len(qs), "machine": machine_info(), "models": {}}
    for name in a.models:
        res["models"][name] = measure_model(build_model(name, cfg, tracks, cache_dir), index_ids,
                                            lambda t: load_logmel(cache_dir, t), qs)
        print(f"[{name}] build {res['models'][name]['build_index_sec']:.1f}s, 1 truy vấn trung vị "
              f"{res['models'][name]['single_query']['median_ms']:.1f} ms, chỉ mục {res['models'][name]['index_bytes'] / 1e6:.2f} MB")
    out = resolve("experiments/results/e9_cost.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = resolve("reports/tables/cost.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res) + f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
