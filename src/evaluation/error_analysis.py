"""Phân tích lỗi trên Test (Phase 4C) từ kết quả ĐÃ CÓ: không chấm điểm lại, không chạm thêm Test.

Bảng: (1) theo cấu hình (R@1, nhận đúng@tau, FMR) cho M1/M2/M3 (mean ± std qua seed); (2) chênh lệch Val -> Test của M3 theo cấu hình;
(3) theo THỂ LOẠI (macro theo bài nguồn); (4) các cấu hình yếu nhất. Thể loại ít bài (n < 15) được gắn cờ: không diễn giải.
Giới hạn: file perquery không lưu ID dự đoán nên không có "ví dụ sai cụ thể" trên Test; muốn xem ví dụ sai hãy dùng Val.
CLI: python -m src.evaluation.error_analysis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.evaluation.test_verdict import FINAL, SEEDS, per_source_macro

MIN_GENRE_N = 15


def _load(rd: Path, split: str, label: str) -> dict:
    return json.loads((rd / f"e5_robust_{split}_{label}.json").read_text(encoding="utf-8"))


def _ms(vals):
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else None}


def config_table(rd, split="test", seeds=SEEDS, final=FINAL) -> dict:
    rd = Path(rd)
    m3 = [_load(rd, split, f"m3_abl_{final}_seed{s}") for s in seeds]
    out = {}
    for c in m3[0]["configs"]:
        row = {}
        for lab in ("m1", "m2"):
            d = _load(rd, split, lab)["configs"][c]
            row[lab] = {"recall_at_1": d["recall_at_1"], "correct_accept": d["correct_accept_at_tau"], "fmr": d["fmr_at_tau"]}
        row["m3"] = {"recall_at_1": _ms([j["configs"][c]["recall_at_1"] for j in m3]),
                     "correct_accept": _ms([j["configs"][c]["correct_accept_at_tau"] for j in m3]),
                     "fmr": _ms([j["configs"][c]["fmr_at_tau"] for j in m3])}
        out[c] = row
    return out


def val_test_gap(rd, seeds=SEEDS, final=FINAL, flag: float = 0.05) -> dict:
    rd = Path(rd)
    val = [_load(rd, "val", f"m3_abl_{final}_seed{s}") for s in seeds]
    test = [_load(rd, "test", f"m3_abl_{final}_seed{s}") for s in seeds]
    out = {}
    for c in test[0]["configs"]:
        v = float(np.mean([j["configs"][c]["recall_at_1"] for j in val]))
        t = float(np.mean([j["configs"][c]["recall_at_1"] for j in test]))
        out[c] = {"val": v, "test": t, "delta": t - v, "flag": bool(abs(t - v) > flag)}
    return out


def genre_table(pqs: dict[str, dict], genre_of: dict[int, str], group_of: dict[str, str], m3_labels: list[str]) -> list[dict]:
    """pqs: {nhãn: npz-dict}. Macro hit theo từng bài nguồn -> trung bình theo thể loại; M3 = trung bình các seed theo bài."""
    ps = {lab: per_source_macro(pqs[lab], "hit1", group_of) for lab in ["m1", "m2"] + m3_labels}
    srcs = sorted(set.intersection(*[set(d) for d in ps.values()]))
    m3_by_src = {s: float(np.mean([ps[l][s] for l in m3_labels])) for s in srcs}
    rows = []
    for g in sorted({genre_of.get(s, "unknown") for s in srcs}):
        sel = [s for s in srcs if genre_of.get(s, "unknown") == g]
        rows.append({"genre": g, "n_sources": len(sel), "small_n": len(sel) < MIN_GENRE_N,
                     "m1": float(np.mean([ps["m1"][s] for s in sel])), "m2": float(np.mean([ps["m2"][s] for s in sel])),
                     "m3": float(np.mean([m3_by_src[s] for s in sel]))})
    return rows


def to_markdown(cfgs: dict, gap: dict, genres: list[dict], worst_k: int = 5) -> str:
    L = ["**Theo cấu hình (Test; M3 = mean ± std qua seed)**", "",
         "| Cấu hình | M1 R@1 | M2 R@1 | M3 R@1 | M3 nhận đúng@tau | M3 FMR | M1 FMR | M2 FMR |", "|---|---|---|---|---|---|---|---|"]
    for c, r in cfgs.items():
        f = lambda m: f"{m['mean']:.3f}" + (f" ± {m['std']:.3f}" if m["std"] is not None else "")
        L.append(f"| {c} | {r['m1']['recall_at_1']:.3f} | {r['m2']['recall_at_1']:.3f} | {f(r['m3']['recall_at_1'])} | "
                 f"{f(r['m3']['correct_accept'])} | {f(r['m3']['fmr'])} | {r['m1']['fmr']:.3f} | {r['m2']['fmr']:.3f} |")
    worst = sorted(cfgs, key=lambda c: cfgs[c]["m3"]["recall_at_1"]["mean"])[:worst_k]
    L += ["", f"**{worst_k} cấu hình yếu nhất của M3 (R@1):** " + ", ".join(f"{c} ({cfgs[c]['m3']['recall_at_1']['mean']:.3f})" for c in worst),
          "", "**Chênh lệch Val -> Test của M3 (R@1 theo cấu hình, trung bình seed; cờ khi |Δ| > 0,05)**", "",
          "| Cấu hình | Val | Test | Δ | Cờ |", "|---|---|---|---|---|"]
    for c, g in gap.items():
        L.append(f"| {c} | {g['val']:.3f} | {g['test']:.3f} | {g['delta']:+.3f} | {'⚠' if g['flag'] else ''} |")
    L += ["", f"**Theo thể loại (macro R@1 theo bài nguồn; n < {MIN_GENRE_N} bị gắn cờ, không diễn giải)**", "",
          "| Thể loại | n bài | M1 | M2 | M3 |", "|---|---|---|---|---|"]
    for r in genres:
        L.append(f"| {r['genre']}{' (n nhỏ)' if r['small_n'] else ''} | {r['n_sources']} | {r['m1']:.3f} | {r['m2']:.3f} | {r['m3']:.3f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    import pandas as pd
    from src.common.config import load_config
    from src.common.paths import resolve
    from src.data.robust_queries import CONFIGS

    rd = resolve("experiments/results")
    labels = [f"m3_abl_{FINAL}_seed{s}" for s in SEEDS]
    need = [rd / f"e5_robust_test_{l}_perquery.npz" for l in ["m1", "m2"] + labels]
    miss = [p.name for p in need if not p.exists()]
    if miss:
        print("CHƯA CHẠY: thiếu kết quả Test:", ", ".join(miss), file=sys.stderr)
        return 2
    cfg = load_config("baseline")
    tracks = pd.read_parquet(resolve(cfg["tracks_path"]))
    if "track_id" not in tracks.columns:
        tracks = tracks.reset_index()
    genre_of = {int(t): str(g) for t, g in zip(tracks["track_id"], tracks["genre"])}
    pqs = {l: dict(np.load(rd / f"e5_robust_test_{l}_perquery.npz")) for l in ["m1", "m2"] + labels}
    cfgs, gap = config_table(rd), val_test_gap(rd)
    genres = genre_table(pqs, genre_of, {c.name: c.group for c in CONFIGS}, labels)
    (rd / "e4_error_analysis.json").write_text(json.dumps({"configs": cfgs, "val_test_gap": gap, "genres": genres},
                                                          ensure_ascii=False, indent=2), encoding="utf-8")
    md = resolve("reports/tables/test_error_analysis.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(to_markdown(cfgs, gap, genres), encoding="utf-8")
    print(to_markdown(cfgs, gap, genres))
    print(f"-> {rd / 'e4_error_analysis.json'}\n-> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
