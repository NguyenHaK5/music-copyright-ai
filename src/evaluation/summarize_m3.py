"""Tổng hợp kết quả M3 trên Val (Phase 3E): trung bình ± độ lệch chuẩn qua seed và bảng ablation.

Đọc experiments/results/e5_robust_val_<label>.json. Quy ước nhãn:
  m3_seed<N>                  : lần chạy chính (cấu hình gốc)
  m3_abl_<tên>_seed<N>        : biến thể ablation/tuning (so với m3_seed<N> cùng seed)
  m1, m2                      : baseline (nếu có file) để tham chiếu
Số trên Val; chưa phải kết quả cuối. Độ lệch chuẩn qua seed chỉ phản ánh biến thiên huấn luyện (n nhỏ), không phải CI.
CLI: python -m src.evaluation.summarize_m3
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

GROUPS = ["cut", "codec", "noise", "eq", "pitch", "tempo", "overlay"]
_SEED = re.compile(r"^m3_seed(\d+)$")
_ABL = re.compile(r"^m3_abl_(.+)_seed(\d+)$")


def collect(results_dir, split: str = "val") -> dict[str, dict]:
    out = {}
    for p in sorted(Path(results_dir).glob(f"e5_robust_{split}_*.json")):
        label = p.stem[len(f"e5_robust_{split}_"):]
        if label.endswith("_perquery"):
            continue
        out[label] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _row(res: dict) -> dict:
    g = res["groups"]
    return {"macro_recall_at_1": res["macro"]["recall_at_1"],
            "macro_correct_accept": res["macro"]["correct_accept_at_tau"],
            "tau": res.get("tau"),
            "group_recall_at_1": {k: g[k]["recall_at_1"] for k in g},
            "group_correct_accept": {k: g[k]["correct_accept_at_tau"] for k in g}}


def _agg(rows: list[dict]) -> dict:
    def ms(vals):
        a = np.asarray(vals, float)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else None}

    out = {"n_seeds": len(rows),
           "macro_recall_at_1": ms([r["macro_recall_at_1"] for r in rows]),
           "macro_correct_accept": ms([r["macro_correct_accept"] for r in rows])}
    for key in ("group_recall_at_1", "group_correct_accept"):
        names = [g for g in GROUPS if all(g in r[key] for r in rows)]
        out[key] = {g: ms([r[key][g] for r in rows]) for g in names}
    return out


def summarize(results: dict[str, dict]) -> dict:
    seeds = {int(m.group(1)): _row(r) for k, r in results.items() if (m := _SEED.match(k))}
    summary: dict = {"baselines": {k: _row(results[k]) for k in ("m1", "m2") if k in results},
                     "m3_seeds": {str(s): seeds[s] for s in sorted(seeds)},
                     "m3_mean_std": _agg([seeds[s] for s in sorted(seeds)]) if seeds else None,
                     "ablations": {}}
    per_name: dict[str, dict[int, dict]] = {}
    for k, r in results.items():
        m = _ABL.match(k)
        if m:
            per_name.setdefault(m.group(1), {})[int(m.group(2))] = _row(r)

    def delta(row, b):
        return {"macro_recall_at_1": row["macro_recall_at_1"] - b["macro_recall_at_1"],
                "macro_correct_accept": row["macro_correct_accept"] - b["macro_correct_accept"],
                "group_recall_at_1": {g: row["group_recall_at_1"][g] - b["group_recall_at_1"][g]
                                      for g in row["group_recall_at_1"] if g in b["group_recall_at_1"]}}

    for name, by_seed in per_name.items():
        first = min(by_seed)
        entry = {"seed": first, **by_seed[first], "per_seed": {str(s): by_seed[s] for s in sorted(by_seed)}}
        if first in seeds:
            entry["delta_vs_seed_baseline"] = delta(by_seed[first], seeds[first])
        if len(by_seed) > 1:
            entry["mean_std"] = _agg([by_seed[s] for s in sorted(by_seed)])
            if len(seeds) > 1:       # so với trung bình của cấu hình gốc
                mean_row = {"macro_recall_at_1": entry["mean_std"]["macro_recall_at_1"]["mean"],
                            "macro_correct_accept": entry["mean_std"]["macro_correct_accept"]["mean"],
                            "group_recall_at_1": {g: v["mean"] for g, v in entry["mean_std"]["group_recall_at_1"].items()}}
                bm = summary["m3_mean_std"]
                base_row = {"macro_recall_at_1": bm["macro_recall_at_1"]["mean"],
                            "macro_correct_accept": bm["macro_correct_accept"]["mean"],
                            "group_recall_at_1": {g: v["mean"] for g, v in bm["group_recall_at_1"].items()}}
                entry["delta_mean_vs_baseline_mean"] = delta(mean_row, base_row)
        summary["ablations"][name] = entry
    return summary


def to_markdown(s: dict) -> str:
    def fmt(m):
        return f"{m['mean']:.4f}" + (f" ± {m['std']:.4f}" if m["std"] is not None else "")

    lines = ["| Mô hình | macro R@1 | macro nhận đúng@FMR | " + " | ".join(GROUPS) + " |",
             "|---|---|---|" + "---|" * len(GROUPS)]
    for k, r in s["baselines"].items():
        lines.append(f"| {k} | {r['macro_recall_at_1']:.4f} | {r['macro_correct_accept']:.4f} | " + " | ".join(
            f"{r['group_recall_at_1'].get(g, float('nan')):.3f}" for g in GROUPS) + " |")
    ms = s["m3_mean_std"]
    if ms:
        lines.append(f"| M3 (n={ms['n_seeds']} seed, mean ± std) | {fmt(ms['macro_recall_at_1'])} | "
                     f"{fmt(ms['macro_correct_accept'])} | " + " | ".join(
                         f"{ms['group_recall_at_1'][g]['mean']:.3f}" if g in ms["group_recall_at_1"] else "-"
                         for g in GROUPS) + " |")
    lines += ["", "**Ablation / tuning (Recall@1 theo nhóm; Δ so với `m3_seed` cùng seed)**", "",
              "| Biến thể | macro R@1 (Δ) | macro nhận đúng@FMR (Δ) | " + " | ".join(GROUPS) + " |",
              "|---|---|---|" + "---|" * len(GROUPS)]
    for name, e in sorted(s["ablations"].items()):
        d = e.get("delta_vs_seed_baseline")
        dm = f" ({d['macro_recall_at_1']:+.4f})" if d else ""
        dc = f" ({d['macro_correct_accept']:+.4f})" if d else ""
        cells = []
        for g in GROUPS:
            v = e["group_recall_at_1"].get(g)
            dv = d["group_recall_at_1"].get(g) if d else None
            cells.append("-" if v is None else f"{v:.3f}" + (f" ({dv:+.3f})" if dv is not None else ""))
        lines.append(f"| {name} (seed {e['seed']}) | {e['macro_recall_at_1']:.4f}{dm} | {e['macro_correct_accept']:.4f}{dc} | "
                     + " | ".join(cells) + " |")
        ms2 = e.get("mean_std")
        if ms2:
            dd = e.get("delta_mean_vs_baseline_mean")
            dm2 = f" (Δ {dd['macro_recall_at_1']:+.4f})" if dd else ""
            dc2 = f" (Δ {dd['macro_correct_accept']:+.4f})" if dd else ""
            cells2 = [f"{ms2['group_recall_at_1'][g]['mean']:.3f}" + (f" (Δ {dd['group_recall_at_1'][g]:+.3f})" if dd and g in dd["group_recall_at_1"] else "")
                      if g in ms2["group_recall_at_1"] else "-" for g in GROUPS]
            lines.append(f"| **{name} (n={ms2['n_seeds']} seed, mean ± std)** | {fmt(ms2['macro_recall_at_1'])}{dm2} | "
                         f"{fmt(ms2['macro_correct_accept'])}{dc2} | " + " | ".join(cells2) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from src.common.paths import resolve

    rdir = resolve("experiments/results")
    results = collect(rdir)
    if not any(k.startswith("m3_") for k in results):
        print("CHƯA CHẠY: chưa có e5_robust_val_m3_*.json", file=sys.stderr)
        return 2
    s = summarize(results)
    (rdir / "e3_summary_val.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    md = resolve("reports/tables/m3_val_summary.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(to_markdown(s), encoding="utf-8")
    print(to_markdown(s))
    print(f"-> {rdir / 'e3_summary_val.json'}\n-> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
