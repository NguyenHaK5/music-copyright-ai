"""Kết luận Phase 4 trên Test: quy tắc ĐẠT/KHÔNG ĐẠT đặt trước, bootstrap cặp đôi theo bài nguồn, FMR kèm CI, bảng chính.

QUY TẮC (cố định trước khi chạm Test; xem mục 10 của PHASE2_INTEGRATION_GUIDE và mục 21 của PHASE3_INTEGRATION_GUIDE):
  M3 = trung bình của 3 seed cấu hình cuối; M1 = baseline. Với chỉ số macro (7 nhóm, mỗi nhóm là trung bình các cấu hình):
  ĐẠT khi (M3 - M1) >= gain trong targets.yaml VÀ cận dưới CI 95% của hiệu > 0 (bootstrap theo BÀI NGUỒN, cặp đôi,
  1.000 lần, seed 0). Chỉ số chính: macro Recall@1. Phụ: macro nhận đúng@FMR (trạng thái riêng, không quyết định kết luận chung).
  FMR trên Test (tau cố định từ Val) chỉ báo cáo kèm CI, không gán ĐẠT/KHÔNG ĐẠT. M0/M2 chỉ tham khảo.
  Báo thêm (không ảnh hưởng kết luận): từng seed riêng lẻ có đạt gain hay không.
CLI: python -m src.evaluation.test_verdict [--gain-r1 X --gain-ca Y]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

N_BOOT, BOOT_SEED = 1000, 0
FINAL = "noise_coherent_k2"
SEEDS = (42, 43, 44)


def per_source_macro(pq, kind: str, group_of: dict[str, str]) -> dict[int, float]:
    """kind: 'hit1' hoặc 'ca'. Trả {source_id: macro theo 7 nhóm của bài đó} (mỗi nhóm = trung bình các cấu hình)."""
    cfgs = sorted({k.split("__")[0] for k in pq if k.endswith(f"__{kind}")})
    by_cfg = {c: dict(zip(np.asarray(pq[f"{c}__src"]).astype(int).tolist(), np.asarray(pq[f"{c}__{kind}"], float).tolist())) for c in cfgs}
    sources = sorted(set.intersection(*[set(d) for d in by_cfg.values()]))
    groups = sorted({group_of[c] for c in cfgs})
    out = {}
    for s in sources:
        out[s] = float(np.mean([np.mean([by_cfg[c][s] for c in cfgs if group_of[c] == g]) for g in groups]))
    return out


def paired_bootstrap(diff: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)], "n_sources": int(len(diff)), "n_boot": int(n_boot)}


def _align(a: dict, b: dict) -> tuple[list, np.ndarray, np.ndarray]:
    keys = sorted(set(a) & set(b))
    return keys, np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def compare(m1_ps: dict, seed_ps: list[dict], gain: float) -> dict:
    """m1_ps, seed_ps[i]: {source: macro}. M3 = trung bình các seed theo từng bài nguồn."""
    common = sorted(set(m1_ps).intersection(*[set(d) for d in seed_ps]))
    m1 = np.array([m1_ps[k] for k in common])
    seeds = np.array([[d[k] for k in common] for d in seed_ps])
    m3 = seeds.mean(0)
    boot = paired_bootstrap(m3 - m1)
    per_seed = [{"macro": float(s.mean()), "diff_vs_m1": float(s.mean() - m1.mean()), "meets_gain": bool(s.mean() - m1.mean() >= gain)}
                for s in seeds]
    return {"m1_macro": float(m1.mean()), "m3_macro_mean_of_seeds": float(m3.mean()),
            "m3_seed_std": float(np.std([p["macro"] for p in per_seed], ddof=1)) if len(per_seed) > 1 else None,
            "diff": boot["mean"], "ci95": boot["ci95"], "n_sources": boot["n_sources"], "gain": float(gain),
            "passed": bool(boot["mean"] >= gain and boot["ci95"][0] > 0), "per_seed": per_seed}


def fmr_pooled(pq, tau: float, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    cfgs = sorted({k.split("__")[0] for k in pq if k.endswith("__unk_top1")})
    by_src: dict[int, list[float]] = {}
    per_cfg = {}
    for c in cfgs:
        top, src = np.asarray(pq[f"{c}__unk_top1"], float), np.asarray(pq[f"{c}__unk_src"]).astype(int)
        acc = (top >= tau).astype(float)
        per_cfg[c] = float(acc.mean())
        for s, a in zip(src.tolist(), acc.tolist()):
            by_src.setdefault(s, []).append(a)
    per_source = np.array([np.mean(v) for v in by_src.values()])
    rng = np.random.default_rng(seed)
    means = per_source[rng.integers(0, len(per_source), size=(n_boot, len(per_source)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"fmr": float(per_source.mean()), "ci95": [float(lo), float(hi)], "n_unknown_sources": int(len(per_source)),
            "per_config": per_cfg}


def load_gains(targets_path, cli_r1: float | None = None, cli_ca: float | None = None) -> dict:
    """Đọc gain từ targets.yaml bằng cách tìm khóa chứa 'gain'; khóa chứa 'recall' -> R@1, 'correct'/'accept'/'nhan' -> nhận đúng.
    Tham số dòng lệnh ghi đè. Nếu không tìm được, báo lỗi kèm danh sách khóa phẳng."""
    import yaml

    flat: dict[str, float] = {}

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flat[prefix.rstrip(".")] = float(node)

    walk(yaml.safe_load(Path(targets_path).read_text(encoding="utf-8")))
    gains = {k: v for k, v in flat.items() if "gain" in k.lower()}
    r1 = next((v for k, v in gains.items() if "recall" in k.lower()), None)
    ca = next((v for k, v in gains.items() if any(t in k.lower() for t in ("correct", "accept", "nhan"))), None)
    src = "targets.yaml"
    if cli_r1 is not None:
        r1, src = cli_r1, "dòng lệnh"
    if cli_ca is not None:
        ca, src = cli_ca, "dòng lệnh"
    if r1 is None or ca is None:
        raise ValueError(f"Không đọc được gain từ {targets_path}. Khóa số tìm thấy: {sorted(flat)}. "
                         "Truyền --gain-r1 và --gain-ca.")
    return {"gain_recall_at_1": r1, "gain_correct_accept": ca, "source": src, "flat_keys": sorted(flat)}


def build_verdict(results_dir, gains: dict, group_of: dict[str, str], seeds=SEEDS, final=FINAL) -> dict:
    rd = Path(results_dir)
    pqs = {lab: dict(np.load(rd / f"e5_robust_test_{lab}_perquery.npz")) for lab in
           ["m0", "m1", "m2"] + [f"m3_abl_{final}_seed{s}" for s in seeds]}
    seed_labels = [f"m3_abl_{final}_seed{s}" for s in seeds]
    out: dict = {"final_config": final, "seeds": list(seeds), "gains": gains, "rule": "mean-of-seeds, CI95 lower > 0, bootstrap by source"}
    for metric, kind, g in (("macro_recall_at_1", "hit1", gains["gain_recall_at_1"]),
                            ("macro_correct_accept", "ca", gains["gain_correct_accept"])):
        m1_ps = per_source_macro(pqs["m1"], kind, group_of)
        out[metric] = compare(m1_ps, [per_source_macro(pqs[l], kind, group_of) for l in seed_labels], g)
    out["verdict_primary"] = "ĐẠT" if out["macro_recall_at_1"]["passed"] else "KHÔNG ĐẠT"
    out["verdict_secondary"] = "ĐẠT" if out["macro_correct_accept"]["passed"] else "KHÔNG ĐẠT"
    tj = {lab: json.loads((rd / f"e5_robust_test_{lab}.json").read_text(encoding="utf-8")) for lab in pqs}
    out["models"] = {}
    for lab, j in tj.items():
        f = fmr_pooled(pqs[lab], j["tau"])
        out["models"][lab] = {"tau_from_val": j["tau"], "macro_recall_at_1": j["macro"]["recall_at_1"],
                              "macro_correct_accept": j["macro"]["correct_accept_at_tau"],
                              "group_recall_at_1": {k: v["recall_at_1"] for k, v in j["groups"].items()},
                              "group_correct_accept": {k: v["correct_accept_at_tau"] for k, v in j["groups"].items()},
                              "fmr_pooled": f["fmr"], "fmr_ci95": f["ci95"], "fmr_per_config": f["per_config"],
                              "n_queries": j["n_queries"]}
    return out


def to_markdown(v: dict, groups_order=("cut", "codec", "noise", "eq", "pitch", "tempo", "overlay")) -> str:
    seeds = [f"m3_abl_{v['final_config']}_seed{s}" for s in v["seeds"]]
    rows = []

    def row(name, m):
        return (f"| {name} | {m['macro_recall_at_1']:.4f} | {m['macro_correct_accept']:.4f} | "
                + " | ".join(f"{m['group_recall_at_1'].get(g, float('nan')):.3f}" for g in groups_order)
                + f" | {m['fmr_pooled']:.4f} [{m['fmr_ci95'][0]:.4f}, {m['fmr_ci95'][1]:.4f}] |")

    lines = ["| Mô hình (Test) | macro R@1 | macro nhận đúng@FMR | " + " | ".join(groups_order) + " | FMR gộp [CI95] |",
             "|---|---|---|" + "---|" * len(groups_order) + "---|"]
    for lab in ("m0", "m1", "m2"):
        lines.append(row(lab, v["models"][lab]))
    arr = {k: np.array([v["models"][l][k] for l in seeds]) for k in ("macro_recall_at_1", "macro_correct_accept", "fmr_pooled")}
    garr = {g: np.array([v["models"][l]["group_recall_at_1"][g] for l in seeds]) for g in groups_order}
    lines.append(f"| **M3 ({len(seeds)} seed, mean ± std)** | {arr['macro_recall_at_1'].mean():.4f} ± {arr['macro_recall_at_1'].std(ddof=1):.4f} | "
                 f"{arr['macro_correct_accept'].mean():.4f} ± {arr['macro_correct_accept'].std(ddof=1):.4f} | "
                 + " | ".join(f"{garr[g].mean():.3f}" for g in groups_order) + f" | {arr['fmr_pooled'].mean():.4f} |")
    for l in seeds:
        lines.append(row(l.replace("m3_abl_", "m3 "), v["models"][l]))
    lines += ["", "**Kết luận theo quy tắc đặt trước (M3 = trung bình 3 seed, so với M1)**", "",
              "| Chỉ số | M1 | M3 | Hiệu (CI95 bootstrap theo bài nguồn) | Gain yêu cầu | Kết luận |", "|---|---|---|---|---|---|"]
    for name, key, vd in (("macro Recall@1 (chính)", "macro_recall_at_1", v["verdict_primary"]),
                          ("macro nhận đúng@FMR (phụ)", "macro_correct_accept", v["verdict_secondary"])):
        c = v[key]
        lines.append(f"| {name} | {c['m1_macro']:.4f} | {c['m3_macro_mean_of_seeds']:.4f} | {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                     f"{c['gain']:.4f} | **{vd}** |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    import argparse
    from src.common.paths import resolve
    from src.data.robust_queries import CONFIGS

    ap = argparse.ArgumentParser()
    ap.add_argument("--gain-r1", type=float, default=None)
    ap.add_argument("--gain-ca", type=float, default=None)
    a = ap.parse_args(argv)
    rd = resolve("experiments/results")
    need = [rd / f"e5_robust_test_{l}_perquery.npz" for l in ["m0", "m1", "m2"] + [f"m3_abl_{FINAL}_seed{s}" for s in SEEDS]]
    miss = [p.name for p in need if not p.exists()]
    if miss:
        print("CHƯA CHẠY: thiếu kết quả Test:", ", ".join(miss), file=sys.stderr)
        return 2
    gains = load_gains(resolve("configs/targets.yaml"), a.gain_r1, a.gain_ca)
    v = build_verdict(rd, gains, {c.name: c.group for c in CONFIGS})
    (rd / "e4_test_verdict.json").write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
    md = resolve("reports/tables/test_main_table.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(to_markdown(v), encoding="utf-8")
    print(to_markdown(v))
    print(f"gain lấy từ: {gains['source']}  | -> {rd / 'e4_test_verdict.json'}  | -> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
