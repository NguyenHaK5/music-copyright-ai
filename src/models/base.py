"""Giao diện chung cho mọi mô hình truy hồi (M0-M3).

build_index(track_ids, mel_loader): dựng chỉ mục cho các bài known (toàn bộ clip).
score_matrix(queries) -> (n_queries, n_tracks): điểm CAO hơn = giống hơn; thứ tự cột = thứ tự
track_ids lúc dựng chỉ mục. Mọi mô hình dùng cùng bộ truy vấn và cùng bộ metric.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

MelLoader = Callable[[int], np.ndarray]


class Retriever(Protocol):
    name: str
    track_ids: list[int]

    def build_index(self, track_ids: Sequence[int], mel_loader: MelLoader) -> None: ...

    def score_matrix(self, queries: Sequence[np.ndarray]) -> np.ndarray: ...
