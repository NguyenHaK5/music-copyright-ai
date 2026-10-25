"""Loss tương phản NT-Xent (Phase 3B)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_NEG = -1e9


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1, track_ids: torch.Tensor | None = None):
    """NT-Xent đối xứng. z1[i], z2[i] là cặp dương (hai view cùng bài); mọi mẫu khác trong batch là âm.

    z1, z2: (B, d), tự chuẩn hóa L2 lại cho an toàn. track_ids (B,): nếu có, mẫu khác cùng bài với anchor KHÔNG
    bị coi là âm (che khỏi mẫu số), tránh âm giả khi một bài xuất hiện hai lần trong batch.
    Trả (loss, top1_acc): top1_acc = tỷ lệ anchor có cặp dương là mẫu giống nhất (chỉ để theo dõi).
    """
    if z1.shape != z2.shape or z1.dim() != 2:
        raise ValueError(f"z1, z2 phải cùng shape (B, d); nhận {tuple(z1.shape)} và {tuple(z2.shape)}")
    if temperature <= 0:
        raise ValueError("temperature phải > 0")
    b = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0).float(), dim=1)           # (2B, d), luôn tính ở float32
    logits = z @ z.t() / temperature
    idx = torch.arange(2 * b, device=z.device)
    pos = (idx + b) % (2 * b)
    mask = torch.eye(2 * b, dtype=torch.bool, device=z.device)           # loại chính nó
    if track_ids is not None:
        t = torch.cat([track_ids, track_ids]).to(z.device)
        same = t[:, None] == t[None, :]
        same[idx, pos] = False                                           # giữ cặp dương
        mask = mask | same
    logits = logits.masked_fill(mask, _NEG)
    loss = F.cross_entropy(logits, pos)
    acc = (logits.argmax(dim=1) == pos).float().mean()
    return loss, acc
