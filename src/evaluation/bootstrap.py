"""Bootstrap theo cụm (bài nguồn) cho metric dạng trung bình của chỉ báo từng truy vấn.

Lấy mẫu lại theo bài nguồn (không theo truy vấn) vì các truy vấn cùng bài không độc lập.
"""
from __future__ import annotations

import numpy as np


def _cluster_stats(values, clusters):
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    if v.shape != c.shape or v.ndim != 1 or v.size == 0:
        raise ValueError("values và clusters phải là mảng 1 chiều cùng độ dài, không rỗng")
    uniq, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=uniq.size)
    cnts = np.bincount(inv, minlength=uniq.size).astype(float)
    return uniq, sums, cnts


def bootstrap_ci(values, clusters, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    """CI percentile của trung bình. Trả (mean, lo, hi)."""
    _, sums, cnts = _cluster_stats(values, clusters)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, sums.size, size=(n_boot, sums.size))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return (float(sums.sum() / cnts.sum()),
            float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def bootstrap_diff_ci(values_a, values_b, clusters, n_boot: int = 1000,
                      alpha: float = 0.05, seed: int = 42):
    """CI cặp đôi của mean(a) - mean(b) trên CÙNG truy vấn, cùng cụm lấy mẫu.
    Trả (diff, lo, hi). 'A vượt B' chỉ khi lo > 0."""
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    if a.shape != b.shape:
        raise ValueError("values_a và values_b phải cùng kích thước")
    _, sa, cnt = _cluster_stats(a, clusters)
    _, sb, _ = _cluster_stats(b, clusters)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, sa.size, size=(n_boot, sa.size))
    n = cnt[idx].sum(1)
    d = (sa[idx].sum(1) - sb[idx].sum(1)) / n
    return (float((sa.sum() - sb.sum()) / cnt.sum()),
            float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2)))
