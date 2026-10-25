"""Open-set: ngưỡng từ chối, FMR, EER, ROC-AUC, nhãn 3 trạng thái.

Định nghĩa (ISSUE-08), điểm cao hơn = giống hơn, chấp nhận khi score >= tau:
  Known  : correct_accept (>= tau và đúng ID), wrong_id (>= tau, sai ID), rejected (< tau)
  Unknown: false_accept (>= tau) -> FMR = tỷ lệ false_accept
  FNMR   = 1 - correct_accept_rate (gồm cả rejected và wrong_id)
Ngưỡng CHỈ được chọn trên Val.
"""
from __future__ import annotations

import numpy as np

LABEL_IN = "THUỘC_DANH_MỤC"
LABEL_OUT = "NGOÀI_DANH_MỤC"
LABEL_UNK = "KHÔNG_XÁC_ĐỊNH"


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("Cần mảng 1 chiều không rỗng")
    if not np.all(np.isfinite(a)):
        raise ValueError("Điểm chứa NaN/inf")
    return a


def fmr(unknown_scores, tau: float) -> float:
    return float((_arr(unknown_scores) >= tau).mean())


def known_outcomes(known_scores, known_correct, tau: float) -> dict[str, float]:
    """Tỷ lệ 3 kết quả của truy vấn known tại ngưỡng tau.
    known_correct: top-1 có đúng ID không (bool)."""
    s = _arr(known_scores)
    c = np.asarray(known_correct, dtype=bool)
    if c.shape != s.shape:
        raise ValueError("known_scores và known_correct khác kích thước")
    acc = s >= tau
    return {
        "correct_accept": float((acc & c).mean()),
        "wrong_id": float((acc & ~c).mean()),
        "rejected": float((~acc).mean()),
    }


def roc_auc(known_scores, unknown_scores) -> float:
    """P(điểm known > điểm unknown), hòa tính 0,5 (Mann-Whitney)."""
    k, u = _arr(known_scores), _arr(unknown_scores)
    allv = np.concatenate([k, u])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size, dtype=float)
    sorted_v = allv[order]
    i = 0
    while i < sorted_v.size:
        j = i
        while j + 1 < sorted_v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_known = ranks[: k.size].sum()
    return float((r_known - k.size * (k.size + 1) / 2.0) / (k.size * u.size))


def eer(known_scores, known_correct, unknown_scores) -> dict[str, float]:
    """EER: điểm FMR(tau) ≈ FNMR(tau) với FNMR = 1 - correct_accept_rate.
    Quét mọi ngưỡng ứng viên; trả điểm có |FMR-FNMR| nhỏ nhất (nội suy không dùng)."""
    ks, kc, us = _arr(known_scores), np.asarray(known_correct, dtype=bool), _arr(unknown_scores)
    cands = np.unique(np.concatenate([ks, us]))
    cands = np.concatenate([cands, [np.nextafter(cands.max(), np.inf)]])
    best = None
    for t in cands:
        f = float((us >= t).mean())
        n = 1.0 - float(((ks >= t) & kc).mean())
        gap = abs(f - n)
        if best is None or gap < best[0]:
            best = (gap, t, f, n)
    _, t, f, n = best
    return {"eer": (f + n) / 2.0, "tau": float(t), "fmr": f, "fnmr": n}


def choose_tau_high(unknown_scores, max_fmr: float) -> float:
    """Ngưỡng nhỏ nhất sao cho FMR trên Val <= max_fmr (chấp nhận khi score >= tau)."""
    if not 0.0 <= max_fmr <= 1.0:
        raise ValueError("max_fmr phải trong [0,1]")
    u = np.sort(_arr(unknown_scores))[::-1]
    k = int(np.floor(max_fmr * u.size + 1e-9))
    if k >= u.size:
        return float("-inf")
    return float(np.nextafter(u[k], np.inf))


def choose_tau_low(known_correct_scores, max_loss: float) -> float:
    """Ngưỡng sao cho <= max_loss số known-đúng trên Val có điểm < tau_low."""
    if not 0.0 <= max_loss <= 1.0:
        raise ValueError("max_loss phải trong [0,1]")
    s = _arr(known_correct_scores)
    return float(np.quantile(s, max_loss, method="lower"))


def decide_labels(scores, tau_low: float, tau_high: float) -> list[str]:
    """>= tau_high: THUỘC; < tau_low: NGOÀI; giữa hai ngưỡng: KHÔNG_XÁC_ĐỊNH."""
    if tau_low > tau_high:
        raise ValueError("tau_low phải <= tau_high")
    out = []
    for s in _arr(scores):
        out.append(LABEL_IN if s >= tau_high else LABEL_OUT if s < tau_low else LABEL_UNK)
    return out
