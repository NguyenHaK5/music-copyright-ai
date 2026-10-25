"""Sinh configs/targets.yaml từ kết quả baseline robustness trên VAL, rồi KHÓA (frozen).

    python -m src.evaluation.make_targets [--force]

Quy tắc (cố định trước khi xem Test): mức tăng tối thiểu của M3 so với M1 trên mỗi chỉ số macro
    gain = max(0.02, 0.25 * (1 - giá_trị_M1_trên_Val))   (thu hẹp 1/4 khoảng cách còn lại, tối thiểu 2 điểm)
cộng điều kiện: khoảng tin cậy bootstrap cặp đôi của hiệu có cận dưới > 0.
M2 chỉ để tham khảo (fingerprint có điểm mù pitch/tempo); M3 không bắt buộc vượt M2.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import yaml

from src.common.paths import CONFIG_DIR, resolve
from src.data.robust_queries import GROUPS

METRICS = ("recall_at_1", "correct_accept_at_tau")


def _gain(v: float) -> float:
    return round(max(0.02, 0.25 * (1.0 - v)), 4)


def build_targets(results: dict[str, dict], max_fmr: float) -> dict:
    for name, r in results.items():
        if r["macro"]["n_groups"] != len(GROUPS) or r["missing_groups"]:
            raise ValueError(f"{name}: macro thiếu nhóm {r['missing_groups']}; không chốt mục tiêu")
    m1 = results["m1"]["macro"]
    gains = {k: _gain(m1[k]) for k in METRICS}
    return {
        "frozen": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "macro_recall_at_1_robust",
        "rule": "gain = max(0.02, 0.25*(1 - M1_val)); CI cặp đôi (bootstrap theo bài nguồn) của hiệu M3-M1 có cận dưới > 0",
        "baselines_val_macro": {n: {k: round(r["macro"][k], 4) for k in METRICS} for n, r in results.items()},
        "tau_val": {n: round(r["tau"], 6) for n, r in results.items()},
        "targets": {
            "m3_vs_m1_min_gain": {"macro_recall_at_1": gains["recall_at_1"],
                                  "macro_correct_accept_at_fmr": gains["correct_accept_at_tau"]},
            "paired_ci_lower_bound_gt": 0.0,
            "max_fmr_at_operating_point": max_fmr,
            "min_macro_correct_accept_at_operating_point":
                round(m1["correct_accept_at_tau"] + gains["correct_accept_at_tau"], 4),
            "min_macro_recall_at_1": round(m1["recall_at_1"] + gains["recall_at_1"], 4),
        },
        "status_labels": ["DAT", "KHONG_DAT", "CHUA_CHAY"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    path = CONFIG_DIR / "targets.yaml"
    if path.exists() and not a.force:
        cur = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if cur.get("frozen") is True:
            print("targets.yaml đã khóa (frozen). Dùng --force chỉ khi chưa chạm Test.", file=sys.stderr)
            return 2
    res_dir = resolve("experiments/results")
    results = {}
    for n in ("m0", "m1", "m2"):
        p = res_dir / f"e5_robust_val_{n}.json"
        if not p.exists():
            print(f"CHƯA CHẠY: thiếu {p}", file=sys.stderr)
            return 2
        results[n] = json.loads(p.read_text(encoding="utf-8"))
    cfg_max_fmr = results["m1"]["max_fmr_target"]
    try:
        t = build_targets(results, cfg_max_fmr)
    except ValueError as e:
        print(f"Từ chối: {e}", file=sys.stderr)
        return 2
    path.write_text(yaml.safe_dump(t, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Đã ghi và khóa {path}")
    print(yaml.safe_dump(t["targets"], allow_unicode=True, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
