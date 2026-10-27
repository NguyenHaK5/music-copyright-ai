"""Hiệu chuẩn ngưỡng demo từ VAL (Phase 5A): tau_high, tau_low cho mô hình demo (cấu hình cuối, seed 42).

tau_high = tau Val đã dùng ở Phase 4 (unknown gộp mọi cấu hình, FMR gộp <= 1%), đọc từ e5_robust_val_<nhãn>.json.
tau_low  = phân vị 5% của điểm top-1 của các truy vấn known ĐƯỢC NHẬN ĐÚNG trên Val (thiết kế Phase 0 mục 9.2): <= 5% truy vấn đúng rơi dưới nó.
Vùng [tau_low, tau_high) -> KHÔNG_XÁC_ĐỊNH. Chấm điểm lại truy vấn robustness Val (không chạm Test).
CLI: python -m src.app.calibrate_demo
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np


def compute_thresholds(known_correct_top1: np.ndarray, unknown_top1: np.ndarray, tau_high: float, q: float = 5.0) -> dict:
    kc = np.asarray(known_correct_top1, float)
    tau_low = float(np.percentile(kc, q))
    note = None
    if tau_low > tau_high:
        tau_low, note = float(tau_high), "tau_low vượt tau_high; đặt bằng tau_high (không có vùng KHÔNG_XÁC_ĐỊNH)"
    un = np.asarray(unknown_top1, float)
    return {"tau_high": float(tau_high), "tau_low": tau_low, "note": note,
            "known_correct": {"n": int(len(kc)), "frac_ge_tau_high": float((kc >= tau_high).mean()),
                              "frac_in_uncertain": float(((kc >= tau_low) & (kc < tau_high)).mean()), "frac_lt_tau_low": float((kc < tau_low).mean())},
            "unknown": {"n": int(len(un)), "frac_ge_tau_high": float((un >= tau_high).mean()),
                        "frac_in_uncertain": float(((un >= tau_low) & (un < tau_high)).mean()), "frac_lt_tau_low": float((un < tau_low).mean())}}


def main(argv=None) -> int:
    import pandas as pd
    from src.app.engine import _sha
    from src.common.config import load_config
    from src.common.paths import resolve
    from src.data.robust_queries import CONFIGS
    from src.features.cache import load_logmel
    from src.models.m3_retrieval import build_m3

    demo, base = load_config("demo"), load_config("baseline")
    ck = resolve(demo["model"]["ckpt"])
    label = ck.parent.name                                   # m3_abl_noise_coherent_k2_seed42
    vp = resolve(base["results_dir"]) / f"e5_robust_val_{label}.json"
    rdir = resolve(base["robust"]["dir"])
    paths = {c.name: rdir / f"robust_val_{c.name}.npz" for c in CONFIGS}
    if not ck.exists() or not vp.exists() or not all(p.exists() for p in paths.values()):
        print("CHƯA CHẠY: thiếu checkpoint, kết quả Val hoặc bộ truy vấn robustness Val", file=sys.stderr)
        return 2
    tau_high = float(json.loads(vp.read_text(encoding="utf-8"))["tau"])
    tracks = pd.read_parquet(resolve(base["tracks_path"]))
    if "track_id" not in tracks.columns:
        tracks = tracks.reset_index()
    ids = sorted(int(t) for t in tracks[(tracks["split"] == "val") & (tracks["role"] == "known")]["track_id"])
    col = {t: i for i, t in enumerate(ids)}
    cache_dir = resolve(base["logmel_cache_dir"])
    retr = build_m3({"sr": base["sr"], "hop_length": base["hop_length"], "m1": base.get("m1", {}),
                     "m3": {"ckpt": str(ck), "device": demo["model"].get("device")}})
    retr.build_index(ids, lambda t: load_logmel(cache_dir, t))
    kc, un = [], []
    for name, p in paths.items():
        z = np.load(p)
        S = retr.score_matrix([m.astype(np.float32) for m in z["mels"]])
        known, src = z["is_known"].astype(bool), z["source_track_id"]
        top, arg = S.max(1), S.argmax(1)
        for i in range(len(S)):
            if known[i]:
                if arg[i] == col[int(src[i])]:
                    kc.append(top[i])
            else:
                un.append(top[i])
        print(f"{name}: xong", flush=True)
    res = compute_thresholds(np.array(kc), np.array(un), tau_high)
    res.update({"split": "val", "model_label": label, "ckpt": str(demo["model"]["ckpt"]), "ckpt_sha256": _sha(ck)[:12],
                "created_utc": datetime.now(timezone.utc).isoformat(), "n_configs": len(paths),
                "rule": "tau_high = tau Val (FMR gộp <= 1%); tau_low = phân vị 5% điểm top-1 của truy vấn known được nhận đúng"})
    out = resolve(demo["calibration_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
