"""Cổng bảo vệ lần chạy Test duy nhất (Phase 4A). Hàm thuần, có test.

check_test_allowed trả danh sách lý do TỪ CHỐI (rỗng = được chạy). Chỉ áp cho split='test'.
Điều kiện: có cờ --final-test; đóng băng Phase 3 đạt và không file nào đổi (verify_freeze); mô hình thuộc danh sách cho phép;
m3 chỉ cho cấu hình cuối (thẻ abl_noise_coherent_k2_seed<N>); mỗi nhãn chỉ ghi kết quả Test một lần (không ghi đè); tau Val có sẵn.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.freeze_phase3 import FINAL, verify_freeze

ALLOWED_MODELS = ("m0", "m1", "m2", "m3")
FINAL_TAG_PREFIX = f"abl_{FINAL}_seed"


def label_of(model: str, tag: str | None) -> str:
    return f"{model}_{tag}" if (tag and model == "m3") else model


def check_test_allowed(split: str, cmd: str, final_flag: bool, models, tag: str | None, results_dir, manifest_path,
                       runs_dir=None, config_dir=None, robust_dir=None) -> list[str]:
    if split != "test":
        return []
    errs: list[str] = []
    if not final_flag:
        errs.append("Test cần cờ --final-test (chạy chính thức một lần sau khi đóng băng Phase 3).")
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        errs.append(f"Chưa có {manifest_path.name}: chạy freeze_phase3 trước.")
    else:
        errs += [f"Đóng băng: {x}" for x in verify_freeze(manifest_path, runs_dir, config_dir)]
    results_dir = Path(results_dir)
    if cmd == "build":
        if robust_dir is not None and (Path(robust_dir) / "robust_test_manifest.json").exists():
            errs.append("Bộ truy vấn Test đã được sinh (robust_test_manifest.json tồn tại); không sinh lại.")
        return errs
    for m in models:
        if m not in ALLOWED_MODELS:
            errs.append(f"Mô hình {m!r} không nằm trong danh sách chạy Test {ALLOWED_MODELS}.")
    if "m3" in models:
        if len([m for m in models if m == "m3"]) != 1 or not tag or not tag.startswith(FINAL_TAG_PREFIX):
            errs.append(f"m3 chỉ được chạy với cấu hình cuối: --tag {FINAL_TAG_PREFIX}<N> (nhận {tag!r}).")
    for m in models:
        lab = label_of(m, tag)
        if (results_dir / f"e5_robust_test_{lab}.json").exists():
            errs.append(f"{lab}: kết quả Test đã tồn tại; Test chỉ chạy đúng một lần (không ghi đè).")
        vp = results_dir / f"e5_robust_val_{lab}.json"
        if not vp.exists():
            errs.append(f"{lab}: thiếu {vp.name} (cần tau từ Val).")
        else:
            try:
                float(json.loads(vp.read_text(encoding="utf-8"))["tau"])
            except Exception:
                errs.append(f"{lab}: {vp.name} không có tau hợp lệ.")
    return errs


def val_tau(results_dir, label: str) -> float:
    return float(json.loads((Path(results_dir) / f"e5_robust_val_{label}.json").read_text(encoding="utf-8"))["tau"])
