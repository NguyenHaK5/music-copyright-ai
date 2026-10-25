"""Sinh và chấm bộ truy vấn robustness trên VAL.

    python -m src.run_robustness build --split val [--limit-sources 20]
    python -m src.run_robustness eval  --split val --models m0 m1 m2

Val: tự do. Test (Phase 4): chỉ với --final-test, sau khi đóng băng Phase 3 đạt; tau lấy từ Val; mỗi nhãn chỉ ghi
kết quả Test một lần (xem src/evaluation/test_gate.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.paths import resolve
from src.data.robust_queries import CONFIGS, GROUPS, build_robust_set, select_sources
from src.evaluation.robustness import evaluate_robust
from src.evaluation.test_gate import check_test_allowed, label_of, val_tau
from src.features.cache import load_logmel
from src.features.logmel import load_audio


def fma_audio_loader(audio_dir: Path, sr: int):
    def load(tid: int) -> np.ndarray:
        name = f"{int(tid):06d}"
        p = audio_dir / name[:3] / f"{name}.mp3"
        if not p.exists():
            hits = list(audio_dir.rglob(f"{name}.mp3"))
            if not hits:
                raise FileNotFoundError(f"Không thấy audio track {tid} dưới {audio_dir}")
            p = hits[0]
        return load_audio(p, sr=sr)
    return load


def _load_tracks(cfg):
    p = resolve(cfg["tracks_path"])
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df if "track_id" in df.columns else df.reset_index()


def cmd_build(a, cfg) -> int:
    audio_dir = resolve(cfg["audio_dir"])
    tracks = _load_tracks(cfg)
    if tracks is None or not audio_dir.exists():
        print(f"CHƯA CHẠY: thiếu tracks.parquet hoặc thư mục audio {audio_dir} "
              "(sửa `audio_dir` trong configs/baseline.yaml).", file=sys.stderr)
        return 2
    r = cfg["robust"]
    src = select_sources(tracks, a.split, a.limit_sources or r["n_sources"], r["seed"])
    rng = np.random.default_rng(r["seed"] + 1)
    train_ids = sorted(int(t) for t in tracks[tracks["split"] == "train"]["track_id"])
    itf = [int(t) for t in rng.choice(train_ids, size=min(r["n_interferers"], len(train_ids)), replace=False)]
    t0 = time.perf_counter()
    man = build_robust_set(src, itf, fma_audio_loader(audio_dir, cfg["sr"]), cfg["sr"], cfg["n_fft"],
                           cfg["hop_length"], cfg["n_mels"], resolve(r["dir"]), a.split, r["seed"])
    print(f"Xong {len(man['configs'])}/{len(CONFIGS)} cấu hình, {man['n_sources']} bài nguồn, "
          f"{time.perf_counter() - t0:.0f}s")
    for k, v in man["skipped"].items():
        print(f"BỎ QUA {k}: {v}")
    return 0 if not man["skipped"] else 3


def cmd_eval(a, cfg) -> int:
    from src.run_baselines import build_model
    tracks = _load_tracks(cfg)
    cache_dir = resolve(cfg["logmel_cache_dir"])
    rdir = resolve(cfg["robust"]["dir"])
    man_p = rdir / f"robust_{a.split}_manifest.json"
    if tracks is None or not man_p.exists():
        print(f"CHƯA CHẠY: thiếu tracks.parquet hoặc {man_p} (chạy `build` trước).", file=sys.stderr)
        return 2
    man = json.loads(man_p.read_text(encoding="utf-8"))
    index_ids = sorted(int(t) for t in tracks[(tracks["split"] == a.split) & (tracks["role"] == "known")]["track_id"])
    out_dir = resolve(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        label = label_of(name, a.tag)
        model = build_model(name, cfg, tracks, cache_dir)
        model.build_index(index_ids, lambda t: load_logmel(cache_dir, t))
        by_cfg, t0 = {}, time.perf_counter()
        for cname in man["configs"]:
            z = np.load(rdir / f"robust_{a.split}_{cname}.npz")
            qs = [m.astype(np.float32) for m in z["mels"]]
            by_cfg[cname] = {"S": model.score_matrix(qs), "index_ids": index_ids,
                             "source_ids": z["source_track_id"], "is_known": z["is_known"]}
        n_q = sum(len(d["source_ids"]) for d in by_cfg.values())
        o = cfg["openset"]
        fixed_tau = val_tau(out_dir, label) if a.split == "test" else None
        res = evaluate_robust(by_cfg, tau=fixed_tau, max_fmr=o["max_fmr"])
        pq = {}
        for cname, c in res["configs"].items():
            pq[f"{cname}__hit1"], pq[f"{cname}__ca"], pq[f"{cname}__src"] = c["_hit1"], c["_ca"], c["_known_src"]
            d = by_cfg[cname]
            unk = ~np.asarray(d["is_known"], bool)
            pq[f"{cname}__unk_top1"] = np.asarray(d["S"], float)[unk].max(1)       # để tính FMR kèm CI ở Phase 4
            pq[f"{cname}__unk_src"] = np.asarray(d["source_ids"], int)[unk]
        np.savez(out_dir / f"e5_robust_{a.split}_{label}_perquery.npz", **pq)
        clean = {n: {k: v for k, v in c.items() if not k.startswith("_")} for n, c in res["configs"].items()}
        payload = {**res, "configs": clean, "model": name, "label": label,
                   "ckpt": a.m3_ckpt if name == "m3" else None, "split": a.split,
                   "created_utc": datetime.now(timezone.utc).isoformat(), "seed": cfg["robust"]["seed"],
                   "n_index_tracks": len(index_ids), "n_queries": n_q,
                   "skipped_configs": man["skipped"], "max_fmr_target": o["max_fmr"],
                   "tau_source": f"val:e5_robust_val_{label}.json" if a.split == "test" else "chọn trên Val (unknown gộp)",
                   "timing_sec": {"score_all_queries": time.perf_counter() - t0},
                   "note": ("KẾT QUẢ TEST (một lần): tau cố định từ Val." if a.split == "test"
                            else "tau chọn trên Val từ unknown gộp mọi cấu hình; chưa phải kết quả cuối.")}
        p = out_dir / f"e5_robust_{a.split}_{label}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        m = res["macro"]
        print(f"[{label}] tau={res['tau']:.4f} macro R@1={m['recall_at_1']:.4f} "
              f"nhận đúng@FMR={m['correct_accept_at_tau']:.4f} ({m['n_groups']}/{len(GROUPS)} nhóm) -> {p}")
        print("   " + "  ".join(f"{g}={res['groups'][g]['recall_at_1']:.3f}" for g in res["groups"]))
        if res["missing_groups"]:
            print(f"   CẢNH BÁO: thiếu nhóm {res['missing_groups']} -> macro KHÔNG so sánh được")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", default="val")
    b.add_argument("--limit-sources", type=int, default=0)
    b.add_argument("--final-test", action="store_true", help="bắt buộc khi --split test")
    e = sub.add_parser("eval")
    e.add_argument("--split", default="val")
    e.add_argument("--models", nargs="+", default=["m0", "m1", "m2"])
    e.add_argument("--final-test", action="store_true", help="bắt buộc khi --split test")
    e.add_argument("--m3-ckpt", default=None, help="đường dẫn checkpoint M3 (best.pt) khi chạy m3")
    e.add_argument("--tag", default=None, help="hậu tố tên file kết quả của m3 (vd seed42)")
    a = ap.parse_args(argv)
    cfg = load_config("baseline")
    if a.split not in ("val", "test"):
        print(f"Từ chối: split {a.split!r} không hợp lệ (val hoặc test).", file=sys.stderr)
        return 2
    errs = check_test_allowed(a.split, a.cmd, getattr(a, "final_test", False), getattr(a, "models", []),
                              getattr(a, "tag", None), resolve(cfg["results_dir"]),
                              resolve("experiments/results/e3_final_freeze.json"),
                              runs_dir=None, config_dir=None, robust_dir=resolve(cfg["robust"]["dir"]))
    if errs:
        print("TỪ CHỐI chạy Test:\n - " + "\n - ".join(errs), file=sys.stderr)
        return 2
    if a.cmd == "eval" and a.m3_ckpt:
        cfg["m3"] = {**(cfg.get("m3") or {}), "ckpt": a.m3_ckpt}
    return cmd_build(a, cfg) if a.cmd == "build" else cmd_eval(a, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
