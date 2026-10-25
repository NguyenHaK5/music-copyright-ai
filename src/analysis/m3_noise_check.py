"""Chẩn đoán độ khớp augmentation nhiễu (Phase 3E): nhiễu THẬT (miền audio) so với nhiễu TỔNG HỢP (miền log-mel).

M3 yếu ở SNR thấp dù huấn luyện với nhiễu SNR 3-25 dB. Hai nguyên nhân khác nhau cần phân biệt:
  (a) augmentation không giống nhiễu thật (hiệu chuẩn sai) -> sửa augmentation;
  (b) mô hình/độ khó bản chất -> cần sức chứa/huấn luyện khác.
Cách đo (chỉ truy vấn known, cùng chỉ mục): Recall@1 của truy vấn thật `snr_X` so với truy vấn sạch `crop_5s` cộng nhiễu
tổng hợp ở cùng SNR danh nghĩa; và độ nghiêng phổ (tilt = trung bình dB của 32 dải cao trừ 32 dải thấp), bất biến độ lợi,
để tìm SNR tổng hợp "tương đương" với SNR thật.
CLI: python -m src.analysis.m3_noise_check --ckpt <best.pt> [--tag seed42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from src.features.augment_mel import NOISE_MODES, add_noise_coherent


def tilt_db(mel: np.ndarray, n: int = 32) -> float:
    return float(mel[-n:].mean() - mel[:n].mean())


def _recall1(S: np.ndarray, true_cols: np.ndarray) -> float:
    return float((S.argmax(1) == true_cols).mean())


def _syn(clean: np.ndarray, snr: float, seed: int, mode: str = "additive", k: int = 1) -> np.ndarray:
    rng = np.random.default_rng([seed, int(round(snr * 10)) + 1000, int(k)])
    if mode == "coherent":
        return np.stack([add_noise_coherent(m, snr, rng, k=k) for m in clean])
    return np.stack([NOISE_MODES[mode](m, snr, rng) for m in clean])


def _equiv(real_r1: float, table: dict) -> float:
    """SNR trên lưới có Recall@1 gần Recall@1 thật nhất (hòa -> SNR cao hơn). Chỉ có nghĩa khi real_r1 < ~0,97."""
    return float(min(table, key=lambda s: (abs(table[s] - real_r1), -s)))


def run_check(retriever, clean: np.ndarray, true_cols: np.ndarray, real: dict[str, np.ndarray],
              snr_grid=tuple(range(-5, 31, 2)), tilt_sample: int = 60, seed: int = 0,
              recall_grid=tuple(range(-10, 26, 5)), k_candidates=()) -> dict:
    """clean: (N,F,T) truy vấn sạch known; real: {'snr_5db': (N,F,T), ...} (cùng thứ tự); true_cols: cột đúng trong chỉ mục.

    Với mỗi mô phỏng nhiễu (additive, coherent): Recall@1 ở SNR danh nghĩa, độ nghiêng phổ, và Recall@1 trên lưới
    `recall_grid` để tìm SNR tổng hợp có độ khó tương đương nhiễu thật (equivalent_snr_by_recall).
    """
    out: dict = {"n_queries": int(len(clean)),
                 "clean": {"recall_at_1": _recall1(retriever.score_matrix(list(clean)), true_cols),
                           "tilt_db": float(np.mean([tilt_db(m) for m in clean]))},
                 "recall_by_snr": {}, "conditions": {}}
    for mode in NOISE_MODES:
        out["recall_by_snr"][mode] = {str(s): _recall1(retriever.score_matrix(list(_syn(clean, s, seed, mode))),
                                                         true_cols) for s in recall_grid}
    sub = clean[:tilt_sample]
    grid_tilt = {s: float(np.mean([tilt_db(m) for m in _syn(sub, s, seed)])) for s in snr_grid}
    for name, mels in real.items():
        nominal = float(name.split("_")[1].rstrip("db"))
        real_tilt = float(np.mean([tilt_db(m) for m in mels]))
        real_r1 = _recall1(retriever.score_matrix(list(mels)), true_cols)
        eq_tilt = min(snr_grid, key=lambda s: abs(grid_tilt[s] - float(np.mean([tilt_db(m) for m in mels[:tilt_sample]]))))
        cond = {"nominal_snr_db": nominal, "real": {"recall_at_1": real_r1, "tilt_db": real_tilt},
                "equivalent_synthetic_snr_db_by_tilt": float(eq_tilt)}
        for mode in NOISE_MODES:
            syn = _syn(clean, nominal, seed, mode)
            key = "synthetic_same_nominal" if mode == "additive" else f"synthetic_{mode}_same_nominal"
            cond[key] = {"recall_at_1": _recall1(retriever.score_matrix(list(syn)), true_cols),
                         "tilt_db": float(np.mean([tilt_db(m) for m in syn]))}
            cond[f"equivalent_snr_by_recall_{mode}"] = _equiv(real_r1, {int(k): v for k, v in out["recall_by_snr"][mode].items()})
        out["conditions"][name] = cond
    if k_candidates:
        conds = out["conditions"]
        real_r1 = {n: c["real"]["recall_at_1"] for n, c in conds.items()}
        table = {"1": {n: c["synthetic_coherent_same_nominal"]["recall_at_1"] for n, c in conds.items()}}
        for k in k_candidates:
            table[str(int(k))] = {n: _recall1(retriever.score_matrix(list(_syn(clean, c["nominal_snr_db"], seed, "coherent", int(k)))),
                                              true_cols) for n, c in conds.items()}
        out["coherent_k"] = {k: {"recall_at_1": row, "l1_to_real": float(sum(abs(row[n] - real_r1[n]) for n in row))}
                             for k, row in table.items()}
        out["best_k_by_l1"] = int(min(out["coherent_k"], key=lambda k: out["coherent_k"][k]["l1_to_real"]))
    return out


def main(argv=None) -> int:
    from src.common.config import load_config
    from src.common.paths import resolve
    from src.features.cache import load_logmel
    from src.models.m3_retrieval import build_m3
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--k", type=int, nargs="*", default=[2, 4, 8, 16], help="các k của nhiễu coherent cần thử")
    a = ap.parse_args(argv)
    cfg = load_config("baseline")
    cfg["m3"] = {**(cfg.get("m3") or {}), "ckpt": a.ckpt}
    rdir = resolve(cfg["robust"]["dir"])
    names = ["crop_5s", "snr_5db", "snr_10db", "snr_20db"]
    paths = {n: rdir / f"robust_val_{n}.npz" for n in names}
    tpath, cache_dir = resolve(cfg["tracks_path"]), resolve(cfg["logmel_cache_dir"])
    if not all(p.exists() for p in paths.values()) or not tpath.exists():
        print("CHƯA CHẠY: thiếu bộ truy vấn robustness hoặc tracks.parquet", file=sys.stderr)
        return 2
    tracks = pd.read_parquet(tpath)
    if "track_id" not in tracks.columns:
        tracks = tracks.reset_index()
    index_ids = sorted(int(t) for t in tracks[(tracks["split"] == "val") & (tracks["role"] == "known")]["track_id"])
    z = {n: np.load(p) for n, p in paths.items()}
    known = z["crop_5s"]["is_known"].astype(bool)
    src = z["crop_5s"]["source_track_id"]
    for n in names[1:]:
        if not np.array_equal(z[n]["source_track_id"], src):
            print(f"Thứ tự bài nguồn của {n} khác crop_5s", file=sys.stderr)
            return 2
    col = {t: i for i, t in enumerate(index_ids)}
    true_cols = np.array([col[int(t)] for t in src[known]])
    get = lambda n: z[n]["mels"][known].astype(np.float32)
    model = build_m3(cfg)
    t0 = time.perf_counter()
    model.build_index(index_ids, lambda t: load_logmel(cache_dir, t))
    res = run_check(model, get("crop_5s"), true_cols, {n: get(n) for n in names[1:]}, k_candidates=tuple(a.k))
    res.update({"ckpt": a.ckpt, "tag": a.tag, "split": "val", "seconds": round(time.perf_counter() - t0, 1),
                "note": "Chỉ truy vấn known, chỉ mục 802 bài Val; chẩn đoán, không phải kết quả cuối."})
    out = resolve(f"experiments/results/e3_noise_check_{a.tag or 'm3'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"clean R@1 = {res['clean']['recall_at_1']:.3f}, tilt {res['clean']['tilt_db']:.2f} dB")
    for n, c in res["conditions"].items():
        print(f"{n}: thật R@1={c['real']['recall_at_1']:.3f} | cộng công suất R@1={c['synthetic_same_nominal']['recall_at_1']:.3f}"
              f" | kết hợp (coherent) R@1={c['synthetic_coherent_same_nominal']['recall_at_1']:.3f} | SNR tương đương theo R@1: "
              f"additive {c['equivalent_snr_by_recall_additive']:.0f} dB, coherent {c['equivalent_snr_by_recall_coherent']:.0f} dB")
    for mode, tab in res["recall_by_snr"].items():
        print(f"R@1 theo SNR tổng hợp [{mode}]: " + "  ".join(f"{k}dB={v:.2f}" for k, v in tab.items()))
    if "coherent_k" in res:
        for k, v in res["coherent_k"].items():
            print(f"coherent k={k}: " + "  ".join(f"{n}={r:.3f}" for n, r in v["recall_at_1"].items()) + f"  | L1 so với thật = {v['l1_to_real']:.3f}")
        print("k tốt nhất theo L1 (trên Val, chỉ là hiệu chuẩn mô phỏng):", res["best_k_by_l1"])
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
