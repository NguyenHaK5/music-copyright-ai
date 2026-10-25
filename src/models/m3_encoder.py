"""Bộ mã hóa M3 cho suy luận (Phase 3D-1): nạp checkpoint -> embedding L2 của crop/cửa sổ log-mel.

Độc lập với giao diện build_index/score_matrix (adapter sẽ viết ở 3D-2). Dùng đúng chuẩn hóa của huấn luyện
(m3_input.normalize_crop với norm_std lưu trong checkpoint). Độ dài vào tùy ý (mô hình lấy trung bình theo thời gian).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.features.m3_input import normalize_crop
from src.models.m3_cnn import build_model


def window_starts(n_frames: int, window: int, hop: int) -> list[int]:
    """Điểm bắt đầu cửa sổ trượt: 0, hop, 2*hop, ... sao cho cửa sổ nằm trọn trong clip; clip ngắn hơn cửa sổ -> [0]."""
    if n_frames <= window:
        return [0]
    return list(range(0, n_frames - window + 1, hop))


class M3Encoder:
    def __init__(self, model: torch.nn.Module, norm_std: float, device: str | torch.device = "cpu",
                 batch_size: int = 256):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.norm_std = float(norm_std)
        self.batch_size = int(batch_size)
        self.n_mels = int(getattr(model, "n_mels", 0)) or None

    @classmethod
    def from_checkpoint(cls, path, device: str | None = None, batch_size: int = 256) -> "M3Encoder":
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(Path(path), map_location=dev, weights_only=False)
        model = build_model(ck["cfg"])
        model.load_state_dict(ck["model"])
        enc = cls(model, ck["norm_std"], dev, batch_size)
        enc.n_mels = int(ck["cfg"]["data"]["n_mels"])
        enc.data_cfg = dict(ck["cfg"]["data"])
        enc.meta = {"seed": ck.get("seed"), "epoch": ck.get("epoch"), "best_epoch": ck.get("best_epoch")}
        return enc

    def _check(self, mel: np.ndarray) -> None:
        if mel.ndim != 2:
            raise ValueError(f"mel phải là (n_mels, T), nhận shape {mel.shape}")
        if self.n_mels is not None and mel.shape[0] != self.n_mels:
            raise ValueError(f"mel có {mel.shape[0]} dải, mô hình cần {self.n_mels}")

    @torch.no_grad()
    def embed_crops(self, crops: np.ndarray) -> np.ndarray:
        """crops: (N, n_mels, T) CÙNG T, giá trị dB chưa chuẩn hóa -> (N, d) float32, chuẩn L2 = 1."""
        if crops.ndim != 3:
            raise ValueError(f"crops phải là (N, n_mels, T), nhận {crops.shape}")
        if len(crops) == 0:
            return np.zeros((0, self.model.embed_dim), np.float32)
        self._check(crops[0])
        out = []
        for i in range(0, len(crops), self.batch_size):
            x = np.stack([normalize_crop(c, self.norm_std) for c in crops[i:i + self.batch_size]])[:, None]
            out.append(self.model(torch.from_numpy(x).to(self.device)).float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.model.embed_dim), np.float32)

    def embed_mel(self, mel: np.ndarray) -> np.ndarray:
        """Một truy vấn/crop có độ dài bất kỳ (n_mels, T) -> (d,)."""
        mel = np.asarray(mel, dtype=np.float32)
        self._check(mel)
        return self.embed_crops(mel[None])[0]

    def window_embeddings(self, mel: np.ndarray, window_frames: int, hop_frames: int) -> np.ndarray:
        """Embedding của các cửa sổ trượt của một clip -> (W, d). Clip ngắn hơn cửa sổ được đệm cạnh (edge)."""
        mel = np.asarray(mel, dtype=np.float32)
        self._check(mel)
        if mel.shape[1] < window_frames:
            mel = np.pad(mel, ((0, 0), (0, window_frames - mel.shape[1])), mode="edge")
        starts = window_starts(mel.shape[1], window_frames, hop_frames)
        return self.embed_crops(np.stack([mel[:, s:s + window_frames] for s in starts]))
