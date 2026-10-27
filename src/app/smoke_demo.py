"""Smoke test demo (Phase 5): chạy pipeline thật trên bộ tệp mẫu, so với kỳ vọng, ghi e5_demo_smoke.json.

Kỳ vọng "A|B" nghĩa là kết quả thuộc một trong các giá trị. Mã lỗi đầu vào so với trường 'error'.
CLI: python -m src.app.smoke_demo     (cần: calibrate_demo và make_demo_samples đã chạy)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def run_smoke(engine, samples_dir, analyze=None) -> dict:
    from src.app.pipeline import analyze_file
    analyze = analyze or analyze_file
    sd = Path(samples_dir)
    expected = json.loads((sd / "expected.json").read_text(encoding="utf-8"))
    rows, ok_all = [], True
    for name, exp in expected.items():
        p = sd / name
        res = analyze(p, name, p.stat().st_size, engine)
        got = res["label"] if res["ok"] else res["error"]
        ok = got in exp.split("|")
        ok_all &= ok
        rows.append({"file": name, "expected": exp, "got": got, "pass": ok,
                     "top1": (res["top"][0] if res["ok"] else None), "timing_ms": res.get("timing_ms"),
                     "license": (res["rights"] or {}).get("license") if res["ok"] else None})
    return {"all_pass": bool(ok_all), "rows": rows}


def main(argv=None) -> int:
    from src.app.engine import build_engine
    from src.common.config import load_config
    from src.common.paths import resolve

    demo = load_config("demo")
    sd = resolve(demo["samples_dir"])
    if not (sd / "expected.json").exists():
        print("CHƯA CHẠY: chạy `python -m src.app.make_demo_samples` trước", file=sys.stderr)
        return 2
    res = run_smoke(build_engine(), sd)
    for r in res["rows"]:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['file']:32s} kỳ vọng {r['expected']:32s} nhận {r['got']}"
              + (f"  (điểm {r['top1']['score']:.3f})" if r["top1"] else ""))
    out = resolve("experiments/results/e5_demo_smoke.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'SMOKE ĐẠT' if res['all_pass'] else 'SMOKE CHƯA ĐẠT'} -> {out}")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
