"""Chấm điểm một ma trận điểm (n_queries x n_tracks) theo giao thức Phase 0.

Xếp hạng bảo thủ khi hòa điểm: hạng của bài đúng = số bài có điểm >= điểm bài đúng
(bài đúng xếp SAU mọi bài hòa điểm). Mô hình trả điểm hằng số không được lợi từ hòa điểm.
Ngưỡng chọn trên chính tập đang chấm (chỉ dùng cho Val); số trên Val KHÔNG dùng để báo cáo kết quả cuối.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.evaluation import metrics as M
from src.evaluation import openset as O
from src.evaluation.bootstrap import bootstrap_ci


def conservative_ranks(S: np.ndarray, true_cols: np.ndarray) -> np.ndarray:
    """Hạng (từ 1) của cột đúng trong từng hàng của S, hòa điểm tính bất lợi."""
    s_true = S[np.arange(S.shape[0]), true_cols][:, None]
    return (S >= s_true).sum(1).astype(int)


def evaluate_scores(S: np.ndarray, index_ids: Sequence[int], source_ids: Sequence[int],
                    is_known: Sequence[bool], max_fmr: float, max_loss: float,
                    topk: Sequence[int] = (1, 5, 10), n_boot: int = 1000, seed: int = 42) -> dict:
    S = np.asarray(S, dtype=float)
    index_ids = np.asarray(index_ids, dtype=int)
    source_ids = np.asarray(source_ids, dtype=int)
    known = np.asarray(is_known, dtype=bool)
    if S.shape != (len(source_ids), len(index_ids)) or known.shape != source_ids.shape:
        raise ValueError("Kích thước S / index_ids / source_ids / is_known không khớp")
    if not np.all(np.isfinite(S)):
        raise ValueError("S chứa NaN/inf")
    if known.all() or (~known).all():
        raise ValueError("Cần cả truy vấn known và unknown")
    col = {int(t): i for i, t in enumerate(index_ids)}
    if len(col) != len(index_ids):
        raise ValueError("index_ids có phần tử trùng")
    if any(int(t) not in col for t in source_ids[known]):
        raise ValueError("Có bài known không nằm trong chỉ mục")
    if any(int(t) in col for t in source_ids[~known]):
        raise ValueError("Bài unknown nằm trong chỉ mục (rò rỉ giao thức)")

    Sk, Su = S[known], S[~known]
    true_cols = np.array([col[int(t)] for t in source_ids[known]])
    rk = conservative_ranks(Sk, true_cols)
    rank_list = rk.tolist()
    res: dict = {"n_known_queries": int(known.sum()), "n_unknown_queries": int((~known).sum()),
                 "n_index_tracks": int(len(index_ids))}
    clusters = source_ids[known]
    for k in topk:
        h = M.hit_indicators(rank_list, k)
        m, lo, hi = bootstrap_ci(h, clusters, n_boot=n_boot, seed=seed)
        res[f"recall_at_{k}"] = {"value": m, "ci95": [lo, hi]}
    rr = M.reciprocal_ranks(rank_list)
    m, lo, hi = bootstrap_ci(rr, clusters, n_boot=n_boot, seed=seed)
    res["mrr"] = {"value": m, "ci95": [lo, hi]}
    res["chance_recall_at_1"] = 1.0 / len(index_ids)

    k_scores, u_scores = Sk.max(1), Su.max(1)
    k_correct = rk == 1
    res["roc_auc_known_vs_unknown"] = O.roc_auc(k_scores, u_scores)
    res["eer"] = O.eer(k_scores, k_correct, u_scores)
    tau_high = O.choose_tau_high(u_scores, max_fmr)
    if k_correct.any():
        tau_low = min(O.choose_tau_low(k_scores[k_correct], max_loss), tau_high)
    else:
        tau_low = tau_high
    res["operating_point"] = {
        "max_fmr_target": max_fmr, "max_loss_target": max_loss,
        "tau_high": float(tau_high), "tau_low": float(tau_low),
        "fmr_at_tau_high": O.fmr(u_scores, tau_high),
        "known_outcomes_at_tau_high": O.known_outcomes(k_scores, k_correct, tau_high),
        "labels_known": _count(O.decide_labels(k_scores, tau_low, tau_high)),
        "labels_unknown": _count(O.decide_labels(u_scores, tau_low, tau_high)),
    }
    return res


def _count(labels: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for l in labels:
        out[l] = out.get(l, 0) + 1
    return out
