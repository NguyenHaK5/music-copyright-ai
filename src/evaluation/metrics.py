"""Metric truy hồi cho truy vấn KNOWN (không cần ngưỡng).

Quy ước: mỗi truy vấn có một danh sách xếp hạng track_id (điểm giảm dần)
và một track_id đúng. Thứ hạng bắt đầu từ 1.
"""
from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np


def rank_of_true(ranked_ids: Sequence[Hashable], true_id: Hashable) -> int | None:
    """Thứ hạng (từ 1) của true_id trong danh sách; None nếu không có."""
    for i, tid in enumerate(ranked_ids, start=1):
        if tid == true_id:
            return i
    return None


def ranks(ranked_lists: Sequence[Sequence[Hashable]], true_ids: Sequence[Hashable]) -> list[int | None]:
    if len(ranked_lists) != len(true_ids):
        raise ValueError("ranked_lists và true_ids phải cùng độ dài")
    return [rank_of_true(r, t) for r, t in zip(ranked_lists, true_ids)]


def hit_indicators(rank_list: Sequence[int | None], k: int) -> np.ndarray:
    """Vector 0/1 từng truy vấn: true_id nằm trong top-k. Dùng cho bootstrap."""
    if k < 1:
        raise ValueError("k phải >= 1")
    return np.array([1.0 if (r is not None and r <= k) else 0.0 for r in rank_list])


def recall_at_k(rank_list: Sequence[int | None], k: int) -> float:
    if len(rank_list) == 0:
        raise ValueError("Không có truy vấn")
    return float(hit_indicators(rank_list, k).mean())


def reciprocal_ranks(rank_list: Sequence[int | None]) -> np.ndarray:
    """1/hạng; 0 nếu không có trong danh sách trả về."""
    return np.array([0.0 if r is None else 1.0 / r for r in rank_list])


def mrr(rank_list: Sequence[int | None]) -> float:
    if len(rank_list) == 0:
        raise ValueError("Không có truy vấn")
    return float(reciprocal_ranks(rank_list).mean())
