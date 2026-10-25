"""Sinh bộ truy vấn robustness (17 cấu hình, 7 nhóm) từ audio của các bài Val/Test.

Mỗi bài nguồn: giải mã clip một lần, chọn đoạn gốc bắt đầu s0 (ngẫu nhiên, theo seed); mọi cấu hình
dùng cùng đoạn gốc 5 s (trừ nhóm cắt 3/5/10 s) nên so sánh giữa cấu hình là cặp đôi.
Mỗi truy vấn được peak-normalize riêng rồi tính log-mel (đúng như hệ thống làm khi nhận truy vấn).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data import transforms as T
from src.features.logmel import logmel, peak_normalize


@dataclass(frozen=True)
class Cfg:
    name: str
    group: str
    kind: str
    param: float


CONFIGS: list[Cfg] = [
    Cfg("crop_3s", "cut", "crop", 3), Cfg("crop_5s", "cut", "crop", 5), Cfg("crop_10s", "cut", "crop", 10),
    Cfg("mp3_64k", "codec", "mp3", 64), Cfg("mp3_128k", "codec", "mp3", 128),
    Cfg("snr_5db", "noise", "noise", 5), Cfg("snr_10db", "noise", "noise", 10),
    Cfg("snr_20db", "noise", "noise", 20),
    Cfg("eq_lowpass_3k", "eq", "lowpass", 3000), Cfg("eq_highpass_500", "eq", "highpass", 500),
    Cfg("pitch_p1", "pitch", "pitch", 1), Cfg("pitch_m1", "pitch", "pitch", -1),
    Cfg("pitch_p2", "pitch", "pitch", 2), Cfg("pitch_m2", "pitch", "pitch", -2),
    Cfg("tempo_0.90", "tempo", "tempo", 0.90), Cfg("tempo_1.10", "tempo", "tempo", 1.10),
    Cfg("overlay_5db", "overlay", "overlay", 5),
]
GROUPS = ["cut", "codec", "noise", "eq", "pitch", "tempo", "overlay"]
BASE_SEC = 5.0
MAX_SEC = 10.0


def apply_cfg(cfg: Cfg, seg5: np.ndarray, seg_long: np.ndarray, sr: int,
              rng: np.random.Generator, interferer: np.ndarray) -> np.ndarray:
    """Trả audio đã biến đổi (chưa chuẩn hóa). seg5 = đoạn gốc 5 s; seg_long = 10 s từ cùng s0."""
    if cfg.kind == "crop":
        return seg_long[: int(cfg.param * sr)]
    if cfg.kind == "mp3":
        return T.mp3_roundtrip(seg5, sr, int(cfg.param))
    if cfg.kind == "noise":
        return T.add_noise_snr(seg5, cfg.param, rng)
    if cfg.kind == "lowpass":
        return T.lowpass(seg5, sr, cfg.param)
    if cfg.kind == "highpass":
        return T.highpass(seg5, sr, cfg.param)
    if cfg.kind == "pitch":
        return T.pitch_shift(seg5, sr, cfg.param)
    if cfg.kind == "tempo":
        return T.time_stretch(seg5, cfg.param)
    if cfg.kind == "overlay":
        return T.overlay_snr(seg5, interferer[: seg5.size], cfg.param)
    raise ValueError(cfg.kind)


def select_sources(tracks: pd.DataFrame, split: str, n_sources: int, seed: int) -> pd.DataFrame:
    """Chọn ngẫu nhiên (theo seed) tối đa n_sources bài known và n_sources bài unknown."""
    rng = np.random.default_rng(seed)
    parts = []
    for role in ("known", "unknown"):
        ids = np.array(sorted(tracks[(tracks["split"] == split) & (tracks["role"] == role)]["track_id"].astype(int)))
        if ids.size == 0:
            raise ValueError(f"Không có bài {role} trong split={split}")
        pick = np.sort(rng.choice(ids, size=min(n_sources, ids.size), replace=False))
        parts.append(pd.DataFrame({"track_id": pick, "is_known": role == "known"}))
    return pd.concat(parts, ignore_index=True)


def build_robust_set(sources: pd.DataFrame, interferer_ids: list[int],
                     audio_loader: Callable[[int], np.ndarray], sr: int, n_fft: int, hop: int,
                     n_mels: int, out_dir: str | Path, split: str, seed: int,
                     configs: list[Cfg] | None = None, log=print) -> dict:
    """Sinh và lưu `<out_dir>/robust_<split>_<cfg>.npz` (mels float16, source_track_id, is_known)
    và manifest JSON. Cấu hình lỗi hạ tầng (vd thiếu ffmpeg) bị bỏ qua và ghi lý do, KHÔNG sinh dữ liệu giả."""
    configs = configs or CONFIGS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped: dict[str, str] = {}
    active = []
    for c in configs:
        if c.kind == "mp3" and not T.ffmpeg_available():
            skipped[c.name] = "thiếu ffmpeg trong PATH"
        else:
            active.append(c)
    store = {c.name: {"mels": [], "tid": [], "known": []} for c in active}
    rng_if = np.random.default_rng([seed, 999])
    n5, n10 = int(BASE_SEC * sr), int(MAX_SEC * sr)
    for k, (tid, known) in enumerate(zip(sources["track_id"], sources["is_known"])):
        y = audio_loader(int(tid))
        if y.size < n10:
            raise ValueError(f"track {tid}: audio {y.size / sr:.1f}s < {MAX_SEC}s")
        s0 = int(np.random.default_rng([seed, int(tid)]).integers(0, y.size - n10 + 1))
        seg_long, seg5 = y[s0:s0 + n10], y[s0:s0 + n5]
        itf = audio_loader(int(interferer_ids[rng_if.integers(0, len(interferer_ids))]))
        itf = itf[int(rng_if.integers(0, max(1, itf.size - n5))):][:n5]
        for ci, c in enumerate(active):
            rng = np.random.default_rng([seed, int(tid), ci])
            try:
                z = apply_cfg(c, seg5, seg_long, sr, rng, itf)
            except Exception as e:  # lỗi biến đổi từng bài: dừng hẳn để không có tập thiếu lặng lẽ
                raise RuntimeError(f"{c.name} track {tid}: {e}") from e
            m = logmel(peak_normalize(z), sr, n_fft, hop, n_mels)
            store[c.name]["mels"].append(m.astype(np.float16))
            store[c.name]["tid"].append(int(tid))
            store[c.name]["known"].append(bool(known))
        if (k + 1) % 50 == 0:
            log(f"  {k + 1}/{len(sources)} bài nguồn")
    manifest = {"split": split, "seed": seed, "n_sources": int(len(sources)),
                "n_known_sources": int(sources["is_known"].sum()), "configs": {}, "skipped": skipped}
    for c in active:
        d = store[c.name]
        shapes = {m.shape for m in d["mels"]}
        if len(shapes) != 1:
            raise RuntimeError(f"{c.name}: số khung không đồng nhất {shapes}")
        p = out_dir / f"robust_{split}_{c.name}.npz"
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez(tmp, mels=np.stack(d["mels"]), source_track_id=np.array(d["tid"]),
                 is_known=np.array(d["known"]))
        tmp.replace(p)
        manifest["configs"][c.name] = {"group": c.group, "n_queries": len(d["tid"]),
                                       "n_frames": int(next(iter(shapes))[1])}
    (out_dir / f"robust_{split}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
