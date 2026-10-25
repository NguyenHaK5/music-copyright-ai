"""Chuẩn hóa chuỗi giấy phép FMA (URL Creative Commons) về 'họ' giấy phép."""
from __future__ import annotations

import re

_CC_RE = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)", re.IGNORECASE)


def license_family(raw: object) -> str:
    """'http://creativecommons.org/licenses/by-nc-sa/3.0/' -> 'by-nc-sa'.

    publicdomain/zero -> 'cc0'; thiếu/không nhận ra -> 'unknown'.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "unknown"
    m = _CC_RE.search(raw)
    if not m:
        return "unknown"
    kind, code = m.group(1).lower(), m.group(2).lower()
    if kind == "publicdomain":
        return "cc0" if code in ("zero", "cc0") else "publicdomain"
    return code
