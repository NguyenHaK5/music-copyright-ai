"""Lớp luật giấy phép T2 (Phase 4D): từ chuỗi giấy phép của metadata FMA -> mô tả giấy phép mở hoặc UNKNOWN.

Nguyên tắc: luật tường minh (không học máy); không bao giờ xuất "không có bản quyền"/"Hợp lệ"/"được phép dùng";
chỉ áp cho nhãn THUỘC_DANH_MỤC; không nhận ra hoặc mâu thuẫn -> UNKNOWN. Cố ý để UNKNOWN: Sampling Plus (giấy phép CC đã ngừng, điều khoản khác CC thường), "Music Sharing" trơn, thiếu/None.
Nhận cả dạng văn bản ("Attribution-NonCommercial 3.0
International") lẫn dạng URL (creativecommons.org/licenses/by-nc/3.0/). Không sửa src/data/license_norm.py (giữ nguyên test cũ).
CLI kiểm tra độ phủ trên dữ liệu thật:  python -m src.rights.license_rules --coverage
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

STATUS_IN = "THUỘC_DANH_MỤC"
_URL = re.compile(r"creativecommons\.org/licenses/([a-z\-]+)/([0-9.]+)")
_VER = re.compile(r"(\d\.\d)")


def parse_license(raw) -> dict | None:
    """Trả {'kind': 'cc'|'cc0'|'pdm'|'pd_claim'|'free_art', 'by','nc','nd','sa': bool, 'version': str|None} hoặc None nếu không nhận ra."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return None
    s = str(raw).strip().lower()
    if not s or s in ("nan", "none", "unknown"):
        return None
    if "publicdomain/zero" in s or "cc0" in s or "public domain dedication" in s or ("zero" in s and "universal" in s):
        return {"kind": "cc0", "by": False, "nc": False, "nd": False, "sa": False, "version": None}
    if "public domain mark" in s or "publicdomain/mark" in s:
        return {"kind": "pdm", "by": False, "nc": False, "nd": False, "sa": False, "version": None}
    if "free art" in s or "art libre" in s:
        return {"kind": "free_art", "by": True, "nc": False, "nd": False, "sa": True, "version": None}
    if "public domain" in s:                                  # "Public Domain" trơn: khai báo của metadata, không phải CC0/PDM
        return {"kind": "pd_claim", "by": False, "nc": False, "nd": False, "sa": False, "version": None}
    m = _URL.search(s)
    if m:
        parts = m.group(1).split("-")
        return {"kind": "cc", "by": "by" in parts, "nc": "nc" in parts, "nd": "nd" in parts, "sa": "sa" in parts, "version": m.group(2)}
    by = "attribution" in s
    nc = bool(re.search(r"non[\s\-]?commercial", s))
    nd = bool(re.search(r"no[\s\-]?deriv", s))
    sa = bool(re.search(r"share[\s\-]?alike", s))
    if not by:
        return None                 # CC (trừ CC0) luôn có Attribution; không có thì không dám suy diễn
    v = _VER.search(s)
    return {"kind": "cc", "by": True, "nc": nc, "nd": nd, "sa": sa, "version": v.group(1) if v else None}


def _label(p: dict, rules: dict) -> str:
    if p["kind"] in rules["special"]:
        return rules["special"][p["kind"]]["label"]
    code = "CC BY" + "".join(f"-{k.upper()}" for k in ("nc", "nd", "sa") if p[k])
    return code + (f" {p['version']}" if p["version"] else "")


def describe(raw, status: str, rules: dict) -> dict | None:
    """Mô tả giấy phép cho một kết quả nhận diện. Trả None nếu nhãn không phải THUỘC_DANH_MỤC (không xuất thông tin giấy phép)."""
    if status != STATUS_IN:
        return None
    unknown = {"result": rules["result_unknown"], "license": None, "conditions": [], "commercial_use": None,
               "derivatives": None, "disclaimer": rules["disclaimer"], "rules_version": rules["version"], "raw": None if raw is None else str(raw)}
    p = parse_license(raw)
    if p is None or (p["kind"] == "cc" and p["nd"] and p["sa"]):      # không nhận ra hoặc mâu thuẫn (ND và SA không đi cùng)
        return unknown
    if p["kind"] in rules["special"]:
        conds = list(rules["special"][p["kind"]]["conditions"])
        com, der = rules["commercial_use"]["allowed"], rules["derivatives"]["share_alike" if "Chia sẻ" in " ".join(conds) or p["sa"] else "allowed"]
    else:
        conds = [rules["conditions"][k] for k in ("by", "nc", "nd", "sa") if p[k]]
        com = rules["commercial_use"]["not_allowed" if p["nc"] else "allowed"]
        der = rules["derivatives"]["not_allowed" if p["nd"] else ("share_alike" if p["sa"] else "allowed")]
    return {**unknown, "result": rules["result_open"], "license": _label(p, rules), "conditions": conds,
            "commercial_use": com, "derivatives": der}


def coverage(raw_values, rules: dict | None = None) -> dict:
    c = Counter(str(v) for v in raw_values)
    fam, unmatched = Counter(), Counter()
    for v, n in c.items():
        p = parse_license(v)
        if p is None or (p["kind"] == "cc" and p["nd"] and p["sa"]):
            fam["UNKNOWN"] += n
            unmatched[v] += n
        else:
            fam[_label(p, rules) if rules else p["kind"]] += n
    tot = sum(c.values())
    return {"n_tracks": tot, "families": dict(fam.most_common()), "unknown_fraction": fam["UNKNOWN"] / tot if tot else None,
            "top_unmatched": unmatched.most_common(15)}


def main(argv=None) -> int:
    import pandas as pd
    from src.common.config import load_config
    from src.common.paths import resolve

    p = resolve("data/processed/tracks.parquet")
    if not p.exists():
        print("CHƯA CHẠY: thiếu tracks.parquet", file=sys.stderr)
        return 2
    df = pd.read_parquet(p)
    rules = load_config("license_rules")
    res = coverage(df["license_raw"].tolist(), rules)
    out = resolve("experiments/results/e8_license_coverage.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
