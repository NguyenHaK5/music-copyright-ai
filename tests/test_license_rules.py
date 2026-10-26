"""Test 4D / E8: lớp luật giấy phép T2 (luật tường minh, UNKNOWN, không dùng từ cấm)."""
import json

import pytest
import yaml

from src.rights.license_rules import STATUS_IN, coverage, describe, parse_license

RULES = yaml.safe_load(open("configs/license_rules.yaml", encoding="utf-8"))
FORBIDDEN = ["không có bản quyền", "hợp lệ", "được phép dùng", "miễn phí bản quyền"]


@pytest.mark.parametrize("raw,label,nc,nd,sa", [
    ("Attribution-NonCommercial-ShareAlike 3.0 International", "CC BY-NC-SA 3.0", True, False, True),
    ("Attribution 3.0 International", "CC BY 3.0", False, False, False),
    ("Attribution-NonCommercial-NoDerivatives (aka Music Sharing) 3.0 International", "CC BY-NC-ND 3.0", True, True, False),
    ("Attribution-ShareAlike 4.0 International", "CC BY-SA 4.0", False, False, True),
    ("Attribution-NoDerivatives 3.0 International", "CC BY-ND 3.0", False, True, False),
    ("Attribution-Non-Commercial 3.0", "CC BY-NC 3.0", True, False, False),
    ("http://creativecommons.org/licenses/by-nc-sa/3.0/", "CC BY-NC-SA 3.0", True, False, True),
    ("https://creativecommons.org/licenses/by/4.0/", "CC BY 4.0", False, False, False),
])
def test_cc_parsing_and_description(raw, label, nc, nd, sa):
    p = parse_license(raw)
    assert (p["nc"], p["nd"], p["sa"], p["by"]) == (nc, nd, sa, True)
    d = describe(raw, STATUS_IN, RULES)
    assert d["result"] == RULES["result_open"] and d["license"] == label
    assert ("Chỉ dùng phi thương mại (NonCommercial)" in d["conditions"]) == nc
    assert (d["commercial_use"] == RULES["commercial_use"]["not_allowed"]) == nc
    assert (d["derivatives"] == RULES["derivatives"]["not_allowed"]) == nd
    if sa:
        assert d["derivatives"] == RULES["derivatives"]["share_alike"]


@pytest.mark.parametrize("raw", ["CC0 1.0 Universal", "http://creativecommons.org/publicdomain/zero/1.0/", "Public Domain Dedication (CC0)"])
def test_cc0(raw):
    d = describe(raw, STATUS_IN, RULES)
    assert d["result"] == RULES["result_open"] and d["license"].startswith("CC0") and d["conditions"] == []


def test_pdm_and_free_art():
    assert describe("Public Domain Mark 1.0", STATUS_IN, RULES)["license"].startswith("Public Domain Mark")
    fa = describe("Free Art License", STATUS_IN, RULES)
    assert fa["license"] == "Free Art License" and any("ShareAlike" in c for c in fa["conditions"])


@pytest.mark.parametrize("raw", [None, float("nan"), "", "unknown", "All rights reserved", "Custom license", "Non-Commercial only",
                                 "Attribution-NoDerivatives-ShareAlike 3.0"])        # không nhận ra / thiếu Attribution / ND+SA mâu thuẫn
def test_unknown_cases(raw):
    d = describe(raw, STATUS_IN, RULES)
    assert d["result"] == RULES["result_unknown"] and d["license"] is None and d["commercial_use"] is None


@pytest.mark.parametrize("status", ["NGOÀI_DANH_MỤC", "KHÔNG_XÁC_ĐỊNH", "", "other"])
def test_no_license_info_when_not_in_catalog(status):
    assert describe("Attribution 3.0 International", status, RULES) is None


def test_no_forbidden_phrases_in_any_output():
    raws = ["Attribution-NonCommercial-ShareAlike 3.0", "CC0 1.0 Universal", "Public Domain Mark 1.0", None, "weird"]
    text = json.dumps([describe(r, STATUS_IN, RULES) for r in raws], ensure_ascii=False).lower()
    text += RULES["disclaimer"].lower() + RULES["result_open"].lower() + RULES["result_unknown"].lower()
    assert not [w for w in FORBIDDEN if w in text]
    assert "không phải kết luận pháp lý" in RULES["disclaimer"].lower() and RULES["version"] == 2


def test_coverage_counts_and_unmatched():
    vals = ["Attribution 3.0 International"] * 3 + ["Attribution-NonCommercial-NoDerivatives 3.0"] * 2 + ["weird thing"] * 4
    c = coverage(vals, RULES)
    assert c["n_tracks"] == 9 and c["families"]["UNKNOWN"] == 4 and c["unknown_fraction"] == pytest.approx(4 / 9)
    assert c["top_unmatched"][0] == ("weird thing", 4) and c["families"]["CC BY 3.0"] == 3


@pytest.mark.parametrize("raw,label_start", [("Public Domain", "Phạm vi công cộng"), ("Art Libre", "Free Art License")])
def test_public_domain_claim_and_art_libre(raw, label_start):
    d = describe(raw, STATUS_IN, RULES)
    assert d["result"] == RULES["result_open"] and d["license"].startswith(label_start)


@pytest.mark.parametrize("raw", ["Noncommercial Sampling Plus", "Sampling Plus", "Music Sharing", "None", "Music Sharing License"])
def test_deliberately_unknown(raw):
    assert describe(raw, STATUS_IN, RULES)["result"] == RULES["result_unknown"]
