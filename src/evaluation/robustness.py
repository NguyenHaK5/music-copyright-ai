"""Chấm điểm bộ truy vấn robustness: một ngưỡng tau CỐ ĐỊNH cho mỗi mô hình.

Val: tau_high chọn từ điểm unknown GỘP mọi cấu hình sao cho FMR gộp <= max_fmr.
Test (Phase 4): truyền tau đã chọn trên Val; không chọn lại.
Số chính: macro (trung bình theo nhóm, rồi theo 7 nhóm) của Recall@1 và của tỷ lệ nhận đúng tại tau.
"""
from __future__ import annotations

import numpy as np

from src.data.robust_queries import CONFIGS, GROUPS
from src.evaluation import metrics as M
from src.evaluation import openset as O
from src.evaluation.runner import conservative_ranks


def choose_tau_pooled(scores_by_cfg: dict, max_fmr: float) -> float:
    """scores_by_cfg[name] = dict(S, source_ids, is_known, index_ids). Gộp top-1 của mọi truy vấn unknown."""
    pooled = np.concatenate([d["S"][~np.asarray(d["is_known"], bool)].max(1) for d in scores_by_cfg.values()])
    return O.choose_tau_high(pooled, max_fmr)


def evaluate_config(S, index_ids, source_ids, is_known, tau: float) -> dict:
    S = np.asarray(S, float)
    known = np.asarray(is_known, bool)
    col = {int(t): i for i, t in enumerate(index_ids)}
    src = np.asarray(source_ids, int)
    if any(int(t) in col for t in src[~known]):
        raise ValueError("Bài unknown nằm trong chỉ mục (rò rỉ giao thức)")
    true_cols = np.array([col[int(t)] for t in src[known]])
    rk = conservative_ranks(S[known], true_cols)
    ks, us = S[known].max(1), S[~known].max(1)
    hit1 = (rk == 1).astype(float)
    ca = ((ks >= tau) & (rk == 1)).astype(float)
    return {"n_known": int(known.sum()), "n_unknown": int((~known).sum()),
            "recall_at_1": float(hit1.mean()), "mrr": float(M.reciprocal_ranks(rk.tolist()).mean()),
            "auc": O.roc_auc(ks, us), "eer": O.eer(ks, rk == 1, us)["eer"],
            "correct_accept_at_tau": float(ca.mean()), "fmr_at_tau": O.fmr(us, tau),
            "_hit1": hit1, "_ca": ca, "_known_src": src[known]}


def evaluate_robust(scores_by_cfg: dict, tau: float | None = None, max_fmr: float = 0.01) -> dict:
    """Trả {'tau', 'configs': {...}, 'groups': {...}, 'macro': {...}}; các khóa bắt đầu '_' là vector
    từng truy vấn (dùng cho bootstrap cặp đôi), không đưa vào JSON."""
    if tau is None:
        tau = choose_tau_pooled(scores_by_cfg, max_fmr)
    cfgs = {n: evaluate_config(d["S"], d["index_ids"], d["source_ids"], d["is_known"], tau)
            for n, d in scores_by_cfg.items()}
    group_of = {c.name: c.group for c in CONFIGS}
    groups = {}
    for g in GROUPS:
        names = [n for n in cfgs if group_of[n] == g]
        if names:
            groups[g] = {k: float(np.mean([cfgs[n][k] for n in names]))
                         for k in ("recall_at_1", "mrr", "auc", "correct_accept_at_tau")}
            groups[g]["n_configs"] = len(names)
    macro = {k: float(np.mean([groups[g][k] for g in groups])) for k in
             ("recall_at_1", "mrr", "auc", "correct_accept_at_tau")}
    macro["n_groups"] = len(groups)
    return {"tau": float(tau), "configs": cfgs, "groups": groups, "macro": macro,
            "missing_groups": [g for g in GROUPS if g not in groups]}
