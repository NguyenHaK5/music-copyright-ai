"""ResNet nhỏ trên log-mel -> embedding L2 (Phase 3B).

Vào: (B, 1, n_mels, T) đã chuẩn hóa (m3_input.normalize_crop). Ra: (B, embed_dim) có chuẩn L2 = 1.
Giữ trục tần số trước khi chiếu (pool="freq_flatten") vì VỊ TRÍ tần số mang thông tin nhận diện; trục thời gian
được lấy trung bình để bất biến dịch thời gian. pool="gap" (trung bình cả hai trục) dùng cho ablation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.short = None
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.short is None else self.short(x)), inplace=True)


class M3Net(nn.Module):
    def __init__(self, n_mels: int = 128, channels=(16, 32, 64, 128), blocks_per_stage: int = 1,
                 embed_dim: int = 128, pool: str = "freq_flatten", dropout: float = 0.0):
        super().__init__()
        if pool not in ("freq_flatten", "gap"):
            raise ValueError(f"pool phải là freq_flatten hoặc gap, nhận {pool!r}")
        channels = list(channels)
        self.pool, self.embed_dim, self.n_mels = pool, int(embed_dim), int(n_mels)
        self.stem = nn.Sequential(nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        layers, c_prev = [], channels[0]
        for i, c in enumerate(channels):
            for b in range(blocks_per_stage):
                stride = 2 if (i > 0 and b == 0) else 1
                layers.append(ResBlock(c_prev, c, stride))
                c_prev = c
        self.stages = nn.Sequential(*layers)
        f_out = int(n_mels)
        for _ in range(len(channels) - 1):
            f_out = math.floor((f_out + 2 - 3) / 2) + 1      # conv 3x3 stride 2 pad 1
        in_feat = c_prev * f_out if pool == "freq_flatten" else c_prev
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(in_feat, self.embed_dim)

    def forward(self, x):
        h = self.stages(self.stem(x))                        # (B, C, F', T')
        if self.pool == "freq_flatten":
            h = h.mean(dim=3).flatten(1)                     # (B, C*F')
        else:
            h = h.mean(dim=(2, 3))                           # (B, C)
        return F.normalize(self.proj(self.drop(h)), dim=1)


def build_model(cfg: dict) -> M3Net:
    m, d = cfg["model"], cfg["data"]
    return M3Net(n_mels=int(d["n_mels"]), channels=m["channels"], blocks_per_stage=int(m.get("blocks_per_stage", 1)),
                 embed_dim=int(m["embed_dim"]), pool=m.get("pool", "freq_flatten"),
                 dropout=float(m.get("dropout", 0.0)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
