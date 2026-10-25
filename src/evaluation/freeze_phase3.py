"""Kiểm tra và đóng băng Phase 3 trước khi chạm Test (Phase 3F).

Chạy:  python -m src.evaluation.freeze_phase3 [--runs-dir experiments/runs]
Kiểm tra (mỗi mục PASS/FAIL), ghi experiments/results/e3_final_freeze.json gồm SHA256 của cấu hình và checkpoint,
rồi Phase 4 gọi verify_freeze() để chắc chắn không có gì đổi sau khi đóng băng. Exit 1 nếu có FAIL.

Cấu hình cuối: noise_coherent_k2 (nhiễu coherent, k=2), 3 seed 42/43/44, chọn `best.pt` theo loss Val.
Không đọc Test; chỉ đọc kết quả Val đã có.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FINAL = "noise_coherent_k2"
SEEDS = (42, 43, 44)
EXPECTED_NOISE = {"mode": "coherent", "coherent_k": 2}
VAL_REFERENCE = {"macro_recall_at_1": 0.7166, "macro_correct_accept": 0.3301}   # M1_Val + gain (mục 9-10 guide Phase 2)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _chk(checks: list, name: str, ok: bool, detail: str = "") -> bool:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def freeze(results_dir, runs_dir, config_dir, seeds=SEEDS, final=FINAL) -> dict:
    import torch
    from src.evaluation.summarize_m3 import collect, summarize

    results_dir, runs_dir, config_dir = Path(results_dir), Path(runs_dir), Path(config_dir)
    checks: list[dict] = []
    per_seed: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    n_queries, norm_stds = set(), set()

    for seed in seeds:
        label = f"m3_abl_{final}_seed{seed}"
        tj, ej = results_dir / f"e3_train_abl_{final}_seed{seed}.json", results_dir / f"e5_robust_val_{label}.json"
        ck_path = runs_dir / f"m3_abl_{final}_seed{seed}" / "best.pt"
        info: dict = {}
        if _chk(checks, f"seed{seed}: có {tj.name}", tj.exists(), str(tj)):
            t = json.loads(tj.read_text(encoding="utf-8"))
            _chk(checks, f"seed{seed}: huấn luyện chỉ dùng train/val, test_touched=False",
                 t.get("split_used") == ["train", "val"] and t.get("test_touched") is False)
            _chk(checks, f"seed{seed}: chạy đủ epoch", t.get("epochs_run") == t.get("epochs_planned"),
                 f"{t.get('epochs_run')}/{t.get('epochs_planned')}, best_epoch={t.get('best_epoch')}, stopped_early={t.get('stopped_early')}")
            _chk(checks, f"seed{seed}: đúng biến thể {final}", (t.get("ablation") or {}).get("name") == final)
            info["best_epoch"], info["best_val_loss"] = t.get("best_epoch"), t.get("best_val_loss")
        if _chk(checks, f"seed{seed}: có {ej.name}", ej.exists(), str(ej)):
            e = json.loads(ej.read_text(encoding="utf-8"))
            _chk(checks, f"seed{seed}: đủ 7 nhóm, không bỏ cấu hình",
                 e.get("missing_groups") == [] and e.get("skipped_configs") == {} and e["macro"]["n_groups"] == 7)
            n_queries.add(e.get("n_queries"))
            info.update(tau=e.get("tau"), macro_recall_at_1=e["macro"]["recall_at_1"],
                        macro_correct_accept=e["macro"]["correct_accept_at_tau"])
        if _chk(checks, f"seed{seed}: có checkpoint {ck_path.name} ở {ck_path.parent.name}", ck_path.exists(), str(ck_path)):
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            noise = (ck["cfg"].get("augment") or {}).get("noise", {})
            _chk(checks, f"seed{seed}: cấu hình trong checkpoint là nhiễu {EXPECTED_NOISE}",
                 all(noise.get(k) == v for k, v in EXPECTED_NOISE.items()), str(noise))
            _chk(checks, f"seed{seed}: seed trong checkpoint khớp", int(ck.get("seed", -1)) == int(seed))
            norm_stds.add(round(float(ck["norm_std"]), 9))
            hashes[f"ckpt_seed{seed}"] = sha256_file(ck_path)
        per_seed[str(seed)] = info
    _chk(checks, "mọi seed đánh giá trên cùng số truy vấn Val", len(n_queries) == 1, str(sorted(map(str, n_queries))))
    _chk(checks, "mọi checkpoint dùng cùng norm_std", len(norm_stds) == 1, str(sorted(norm_stds)))

    res = collect(results_dir)
    summ = summarize(res)
    ent = summ["ablations"].get(final)
    ms = ent.get("mean_std") if ent else None
    _chk(checks, "bộ tổng hợp thấy đủ 3 seed của cấu hình cuối", bool(ms) and ms["n_seeds"] == len(seeds),
         f"n_seeds={ms['n_seeds'] if ms else 0}")
    if ms:
        ref_ok = (ms["macro_recall_at_1"]["mean"] >= VAL_REFERENCE["macro_recall_at_1"]
                  and ms["macro_correct_accept"]["mean"] >= VAL_REFERENCE["macro_correct_accept"])
        _chk(checks, "trung bình Val vượt mức tham chiếu (thông tin, không phải ĐẠT/KHÔNG ĐẠT ở Test)", ref_ok,
             f"R@1 {ms['macro_recall_at_1']['mean']:.4f}, nhận đúng@FMR {ms['macro_correct_accept']['mean']:.4f}")
    test_files = [p.name for p in results_dir.glob("*test*.json")]
    _chk(checks, "chưa có kết quả Test nào trong experiments/results", not test_files, ", ".join(test_files))

    for name in ("targets.yaml", "m3.yaml", "m3_ablations.yaml", "baseline.yaml", "features.yaml"):
        p = config_dir / name
        if _chk(checks, f"có configs/{name}", p.exists(), str(p)):
            hashes[f"config_{name}"] = sha256_file(p)

    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "final_config": final, "seeds": list(seeds),
                "ok": all(c["ok"] for c in checks), "checks": checks, "sha256": hashes, "per_seed_val": per_seed,
                "val_mean_std": ms, "val_reference": VAL_REFERENCE,
                "paths": {"results_dir": str(results_dir), "runs_dir": str(runs_dir), "config_dir": str(config_dir)}}
    return manifest


def verify_freeze(manifest_path, runs_dir=None, config_dir=None) -> list[str]:
    """Băm lại các file trong manifest; trả danh sách sai khác (rỗng = không có gì đổi)."""
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not m.get("ok"):
        return ["manifest không ở trạng thái ok: đóng băng chưa đạt"]
    rd = Path(runs_dir or m["paths"]["runs_dir"])
    cd = Path(config_dir or m["paths"]["config_dir"])
    bad = []
    for key, want in m["sha256"].items():
        kind, rest = key.split("_", 1)
        if kind == "ckpt":
            p = rd / f"m3_abl_{m['final_config']}_{rest}" / "best.pt"
        else:
            p = cd / rest
        if not p.exists():
            bad.append(f"thiếu {p}")
        elif sha256_file(p) != want:
            bad.append(f"đã đổi: {p}")
    return bad


def main(argv=None) -> int:
    import argparse
    from src.common.paths import resolve

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=None, help="thư mục chứa m3_abl_noise_coherent_k2_seed*/best.pt (mặc định experiments/runs)")
    a = ap.parse_args(argv)
    m = freeze(resolve("experiments/results"), a.runs_dir or resolve("experiments/runs"), resolve("configs"))
    for c in m["checks"]:
        print(("PASS " if c["ok"] else "FAIL ") + c["name"] + (f"  [{c['detail']}]" if c["detail"] and not c["ok"] else ""))
    if m.get("val_mean_std"):
        ms = m["val_mean_std"]
        print(f"\nVal, {ms['n_seeds']} seed: macro R@1 {ms['macro_recall_at_1']['mean']:.4f} ± {ms['macro_recall_at_1']['std'] or 0:.4f}; "
              f"nhận đúng@FMR {ms['macro_correct_accept']['mean']:.4f} ± {ms['macro_correct_accept']['std'] or 0:.4f}")
    out = resolve("experiments/results/e3_final_freeze.json")
    out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'ĐÓNG BĂNG ĐẠT' if m['ok'] else 'ĐÓNG BĂNG CHƯA ĐẠT'} -> {out}")
    return 0 if m["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
