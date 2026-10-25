"""M3 theo giao diện Retriever (Phase 3D-2): build_index / score_matrix như M1.

Chỉ mục: mỗi bài = embedding của các cửa sổ trượt (win_frames, hop_frames) trên toàn clip.
Điểm bài = max cosine giữa embedding truy vấn và các cửa sổ của bài (điểm cao = giống hơn; cột theo track_ids).
query_mode: "whole" (mặc định) = embed cả truy vấn với độ dài thật;
            "windows" = truy vấn dài hơn cửa sổ được cắt thành cửa sổ rồi lấy trung bình embedding (ablation).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from src.models.base import MelLoader
from src.models.m3_encoder import M3Encoder, window_starts


class M3Retriever:
    name = "m3"

    def __init__(self, encoder: M3Encoder, win_frames: int, hop_frames: int, query_mode: str = "whole"):
        if query_mode not in ("whole", "windows"):
            raise ValueError(f"query_mode phải là whole/windows, nhận {query_mode!r}")
        self.enc, self.win, self.hop, self.query_mode = encoder, int(win_frames), int(hop_frames), query_mode
        self.track_ids: list[int] = []
        self.index_size: int | None = None
        self._W: np.ndarray | None = None
        self._offsets: np.ndarray | None = None

    def build_index(self, track_ids: Sequence[int], mel_loader: MelLoader) -> None:
        self.track_ids = [int(t) for t in track_ids]
        embs, offsets, n = [], [], 0
        for tid in self.track_ids:
            w = self.enc.window_embeddings(np.asarray(mel_loader(tid), dtype=np.float32), self.win, self.hop)
            offsets.append(n)
            embs.append(w)
            n += len(w)
        self._W = np.concatenate(embs).astype(np.float32)
        self._offsets = np.array(offsets)
        self.index_size = int(n)

    def _embed_one(self, q: np.ndarray) -> np.ndarray:
        if self.query_mode == "windows" and q.shape[1] > self.win:
            z = self.enc.window_embeddings(q, self.win, self.hop).mean(0)
            return z / max(float(np.linalg.norm(z)), 1e-9)
        return self.enc.embed_mel(q)

    def _embed_queries(self, queries: Iterable[np.ndarray], chunk: int = 512) -> np.ndarray:
        """Duyệt truy vấn theo luồng (không giữ toàn bộ log-mel trong RAM); gom theo độ dài trong mỗi chunk."""
        out, buf = [], []

        def flush():
            if not buf:
                return
            res = [None] * len(buf)
            groups: dict[int, list[int]] = {}
            for i, q in enumerate(buf):
                groups.setdefault(q.shape[1], []).append(i)
            for _, idx in groups.items():
                if self.query_mode == "whole":
                    z = self.enc.embed_crops(np.stack([buf[i] for i in idx]))
                    for i, zi in zip(idx, z):
                        res[i] = zi
                else:
                    for i in idx:
                        res[i] = self._embed_one(buf[i])
            out.append(np.stack(res))
            buf.clear()

        for q in queries:
            buf.append(np.asarray(q, dtype=np.float32))
            if len(buf) >= chunk:
                flush()
        flush()
        return np.concatenate(out) if out else np.zeros((0, self._W.shape[1]), np.float32)

    def score_matrix(self, queries: Iterable[np.ndarray], chunk: int = 512) -> np.ndarray:
        if self._W is None:
            raise RuntimeError("Chưa build_index")
        Q = self._embed_queries(queries, chunk).astype(np.float32)
        out = np.empty((Q.shape[0], len(self.track_ids)), dtype=np.float32)
        for i in range(0, Q.shape[0], chunk):
            out[i:i + chunk] = np.maximum.reduceat(Q[i:i + chunk] @ self._W.T, self._offsets, axis=1)
        return out


def build_m3(cfg: dict) -> M3Retriever:
    """Dựng M3Retriever từ cấu hình baseline (cfg['m3']['ckpt'] bắt buộc). Cửa sổ chỉ mục mặc định = cửa sổ M1."""
    c = dict(cfg.get("m3") or {})
    ck = c.get("ckpt")
    if not ck:
        raise ValueError("Thiếu checkpoint M3: truyền --m3-ckpt hoặc đặt m3.ckpt trong cấu hình.")
    sr, hop = int(cfg["sr"]), int(cfg["hop_length"])
    m1 = cfg.get("m1", {})
    win_sec, hop_sec = c.get("win_sec", m1.get("win_sec", 5.0)), c.get("hop_sec", m1.get("hop_sec", 2.5))
    enc = M3Encoder.from_checkpoint(ck, device=c.get("device"), batch_size=int(c.get("batch_size", 256)))
    d = enc.data_cfg
    if int(d["sr"]) != sr or int(d["hop_length"]) != hop:
        raise ValueError(f"Checkpoint huấn luyện với sr={d['sr']}, hop={d['hop_length']} nhưng đánh giá dùng sr={sr}, hop={hop}")
    return M3Retriever(enc, round(win_sec * sr / hop), round(hop_sec * sr / hop), c.get("query_mode", "whole"))
