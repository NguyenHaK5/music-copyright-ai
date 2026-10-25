"""Chuẩn hóa đầu vào dùng CHUNG cho huấn luyện và suy luận của M3 (3A, dùng lại ở 3D)."""
from __future__ import annotations

import numpy as np


def normalize_crop(mel: np.ndarray, std: float) -> np.ndarray:
    """(mel - trung bình của chính crop) / std toàn cục. Triệt độ lợi cố định (gain) theo thiết kế.

    mel: (n_mels, n_frames) dB. Trả float32 cùng shape.
    """
    if std <= 0:
        raise ValueError("std phải > 0")
    m = np.asarray(mel, dtype=np.float32)
    return ((m - m.mean()) / np.float32(std)).astype(np.float32)
