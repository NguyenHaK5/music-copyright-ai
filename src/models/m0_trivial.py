"""M0: điểm ngẫu nhiên đều (mốc tối thiểu; Recall@k kỳ vọng ~ k / số bài trong chỉ mục)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.models.base import MelLoader


class M0Random:
    name = "m0"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.track_ids: list[int] = []

    def build_index(self, track_ids: Sequence[int], mel_loader: MelLoader | None = None) -> None:
        self.track_ids = [int(t) for t in track_ids]

    def score_matrix(self, queries: Sequence) -> np.ndarray:
        if not self.track_ids:
            raise RuntimeError("Chưa build_index")
        rng = np.random.default_rng(self.seed)
        return rng.random((len(queries), len(self.track_ids)))
