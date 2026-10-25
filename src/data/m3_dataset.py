"""Dataset huấn luyện tương phản cho M3 (Phase 3A).

Mỗi mục = hai view (hai crop log-mel 5 s khác nhau + augmentation khác nhau) của CÙNG một bài Train.
Thuần numpy (không import torch): torch DataLoader tự gộp mảng numpy thành tensor.
Ngẫu nhiên xác định theo (seed, epoch, chỉ số): kết quả không phụ thuộc số worker.
Lưu ý: gọi set_epoch(e) ở tiến trình chính TRƯỚC khi tạo iterator; dùng persistent_workers=False.

CLI:  python -m src.data.m3_dataset [--split train] [--limit N] [--preview] [--recompute-norm]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.features.augment_mel import MelAugmenter
from src.features.m3_input import normalize_crop


def logmel_path(logmel_dir, track_id) -> Path:
    return Path(logmel_dir) / f"{int(track_id):06d}.npy"


def crop_frames_from_cfg(d: dict) -> int:
    return int(round(float(d["crop_seconds"]) * int(d["sr"]) / int(d["hop_length"])))


class PairDataset:
    def __init__(self, track_ids, logmel_dir, augmenter: MelAugmenter, crop_frames: int, seed: int = 0,
                 norm_std: float = 1.0, repeats: int = 1, min_start_distance: int = 0, preload: bool = False):
        self.track_ids = [int(t) for t in track_ids]
        if not self.track_ids:
            raise ValueError("Danh sách track_ids rỗng")
        self.logmel_dir = Path(logmel_dir)
        self.aug = augmenter
        self.crop_frames = int(crop_frames)
        self.seed = int(seed)
        self.norm_std = float(norm_std)
        self.repeats = int(repeats)
        self.min_start_distance = int(min_start_distance)
        self.epoch = 0
        self._cache: dict[int, np.ndarray] | None = None
        if preload:
            self._cache = {t: np.load(logmel_path(self.logmel_dir, t)) for t in self.track_ids}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.track_ids) * self.repeats

    def load_mel(self, track_id: int) -> np.ndarray:
        """Trả mảng (n_mels, T) (float16 đã nạp hoặc memmap)."""
        if self._cache is not None:
            return self._cache[track_id]
        return np.load(logmel_path(self.logmel_dir, track_id), mmap_mode="r")

    def sample_starts(self, rng: np.random.Generator, t_total: int, src_frames: int) -> tuple[int, int]:
        """Hai điểm bắt đầu của hai view. Nếu min_start_distance>0, thử tối đa 20 lần để đạt khoảng cách."""
        max_start = t_total - src_frames
        if max_start <= 0:
            return 0, 0
        s1 = int(rng.integers(0, max_start + 1))
        s2 = int(rng.integers(0, max_start + 1))
        if self.min_start_distance > 0:
            for _ in range(20):
                if abs(s1 - s2) >= self.min_start_distance:
                    break
                s2 = int(rng.integers(0, max_start + 1))
        return s1, s2

    def _view(self, mel, start: int, src_frames: int, rng: np.random.Generator):
        raw = np.asarray(mel[:, start:start + src_frames], dtype=np.float32)
        if raw.shape[1] < src_frames:
            raw = np.pad(raw, ((0, 0), (0, src_frames - raw.shape[1])), mode="edge")
        x, applied = self.aug(raw, self.crop_frames, rng)
        return normalize_crop(x, self.norm_std)[None, :, :], applied

    def get_item(self, i: int):
        """(x1, x2, track_id, applied1, applied2); x: (1, n_mels, crop_frames) float32."""
        tid = self.track_ids[i % len(self.track_ids)]
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])
        mel = self.load_mel(tid)
        src = self.aug.source_frames(self.crop_frames)
        s1, s2 = self.sample_starts(rng, mel.shape[1], src)
        x1, a1 = self._view(mel, s1, src, rng)
        x2, a2 = self._view(mel, s2, src, rng)
        return x1, x2, tid, a1, a2

    def __getitem__(self, i: int):
        x1, x2, tid, _, _ = self.get_item(i)
        return x1, x2, tid


# ---------------------------------------------------------------- thống kê chuẩn hóa
def compute_norm_stats(load_fn, track_ids, crop_frames: int, seed: int, max_tracks: int = 500,
                       crops_per_track: int = 4) -> dict:
    """std của (crop - trung bình crop) trên crop SẠCH ngẫu nhiên của bài Train, kèm phân vị giá trị thô."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(int(t) for t in track_ids))
    if len(ids) > max_tracks:
        ids = np.sort(rng.choice(ids, size=max_tracks, replace=False))
    ss, n, raws = 0.0, 0, []
    for tid in ids:
        m = load_fn(int(tid))
        t_total = m.shape[1]
        for _ in range(crops_per_track):
            st = int(rng.integers(0, max(1, t_total - crop_frames + 1)))
            c = np.asarray(m[:, st:st + crop_frames], dtype=np.float64)
            d = c - c.mean()
            ss += float((d ** 2).sum())
            n += d.size
            raws.append(c.ravel()[::17])
    raw = np.concatenate(raws)
    return {
        "std_after_crop_mean": float(np.sqrt(ss / n)),
        "raw_min": float(raw.min()), "raw_p01": float(np.percentile(raw, 1)),
        "raw_p50": float(np.percentile(raw, 50)), "raw_p99": float(np.percentile(raw, 99)),
        "raw_max": float(raw.max()),
        "n_tracks": int(len(ids)), "n_crops": int(len(ids) * crops_per_track),
        "seed": int(seed), "crop_frames": int(crop_frames),
    }


def load_or_compute_norm(path, load_fn, track_ids, crop_frames: int, seed: int, max_tracks: int = 500,
                         crops_per_track: int = 4, force: bool = False) -> dict:
    p = Path(path)
    if p.exists() and not force:
        stats = json.loads(p.read_text(encoding="utf-8"))
        if int(stats.get("crop_frames", -1)) == int(crop_frames):
            return stats
    stats = compute_norm_stats(load_fn, track_ids, crop_frames, seed, max_tracks, crops_per_track)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# ---------------------------------------------------------------- dựng dataset từ cấu hình
def split_track_ids(tracks_df, split: str) -> list[int]:
    df = tracks_df
    ids = df["track_id"] if "track_id" in df.columns else df.index.to_series()
    return sorted(int(t) for t in ids[df["split"] == split])


def build_pair_dataset(cfg: dict, split: str = "train", *, tracks_df=None, logmel_dir=None, norm_path=None,
                       force_norm: bool = False, augment: bool = True, norm_ids=None):
    """Dựng PairDataset cho một split. Thống kê chuẩn hóa luôn tính từ bài Train (norm_ids mặc định = Train).

    Trả (dataset, norm_stats).
    """
    import pandas as pd
    from src.common.paths import resolve

    d = cfg["data"]
    if tracks_df is None:
        tracks_df = pd.read_parquet(resolve(d["tracks_path"]))
    logmel_dir = Path(logmel_dir) if logmel_dir is not None else resolve(d["logmel_dir"])
    norm_path = Path(norm_path) if norm_path is not None else resolve(d["norm_path"])
    crop_frames = crop_frames_from_cfg(d)
    seed = int(cfg.get("seed", 0))

    train_ids = split_track_ids(tracks_df, "train") if norm_ids is None else list(norm_ids)
    load_fn = lambda t: np.load(logmel_path(logmel_dir, t), mmap_mode="r")
    ns = d.get("norm_stats", {})
    norm_seed = int(d.get("norm_seed", seed))      # cố định giữa các seed huấn luyện để chuẩn hóa giống hệt nhau
    stats = load_or_compute_norm(norm_path, load_fn, train_ids, crop_frames, norm_seed,
                                 int(ns.get("max_tracks", 500)), int(ns.get("crops_per_track", 4)), force_norm)

    aug = MelAugmenter(cfg.get("augment") if augment else {}, sr=int(d["sr"]), n_mels=int(d["n_mels"]))
    ds = PairDataset(split_track_ids(tracks_df, split), logmel_dir, aug, crop_frames, seed=seed,
                     norm_std=stats["std_after_crop_mean"], repeats=int(d.get("repeats_per_epoch", 1)),
                     min_start_distance=int(d.get("min_start_distance_frames", 0)),
                     preload=bool(d.get("preload", False)))
    return ds, stats


# ---------------------------------------------------------------- xem trước và kiểm tra
def save_preview(ds: PairDataset, path, track_index: int = 0, n_aug: int = 5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tid = ds.track_ids[track_index]
    mel = ds.load_mel(tid)
    clean = np.asarray(mel[:, :ds.crop_frames], dtype=np.float32)
    panels = [("sạch", normalize_crop(clean, ds.norm_std))]
    for k in range(n_aug):
        x1, _, _, a1, _ = ds.get_item(track_index + k * len(ds.track_ids))
        panels.append((", ".join(a1) or "không biến đổi", x1[0]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (title, img) in zip(axes.ravel(), panels):
        ax.imshow(img, origin="lower", aspect="auto", vmin=-3, vmax=3, cmap="magma")
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"Augmentation log-mel (track {tid}); thang chuẩn hóa")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from src.common.config import load_config
    from src.common.log import get_logger
    from src.common.paths import resolve
    from src.common.seed import set_seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=200, help="số mục lấy mẫu để đo thời gian/kiểm tra")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--recompute-norm", action="store_true")
    args = ap.parse_args(argv)
    log = get_logger("m3_dataset")

    cfg = load_config("m3")
    set_seed(int(cfg.get("seed", 0)))
    tracks, mel_dir = resolve(cfg["data"]["tracks_path"]), resolve(cfg["data"]["logmel_dir"])
    if not tracks.exists() or not mel_dir.exists():
        log.error("CHƯA CHẠY: thiếu %s hoặc %s", tracks, mel_dir)
        return 2
    ds, stats = build_pair_dataset(cfg, args.split, force_norm=args.recompute_norm)
    n = min(args.limit, len(ds))
    t0 = time.perf_counter()
    means, stds, freq = [], [], {}
    for i in range(n):
        x1, x2, tid, a1, a2 = ds.get_item(i)
        assert x1.shape == (1, int(cfg["data"]["n_mels"]), ds.crop_frames), x1.shape
        assert np.isfinite(x1).all() and np.isfinite(x2).all()
        means.append(float(x1.mean())); stds.append(float(x1.std()))
        for op in a1 + a2:
            freq[op] = freq.get(op, 0) + 1
    ms = (time.perf_counter() - t0) * 1000 / n
    out = {
        "split": args.split, "n_tracks": len(ds.track_ids), "crop_frames": ds.crop_frames,
        "active_ops": ds.aug.active_ops, "norm_stats": stats, "n_checked": n,
        "ms_per_item": round(ms, 2),
        "view_mean": round(float(np.mean(means)), 4), "view_std_mean": round(float(np.mean(stds)), 4),
        "view_std_range": [round(min(stds), 4), round(max(stds), 4)],
        "op_frequency_per_view": {k: round(v / (2 * n), 3) for k, v in sorted(freq.items())},
    }
    res = resolve("experiments/results/e3_data_check.json")
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Đã ghi %s", res)
    log.info(json.dumps(out, indent=2, ensure_ascii=False))
    if args.preview:
        pv = resolve("reports/figures/m3_aug_preview.png")
        save_preview(ds, pv)
        log.info("Đã ghi %s", pv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
