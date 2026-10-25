"""Huấn luyện M3 (Phase 3C): CNN embedding + NT-Xent, checkpoint/resume, learning curve.

Chỉ dùng split train (huấn luyện) và val (theo dõi loss, chọn epoch tốt nhất). KHÔNG đọc Test.
CLI: python -m src.train_m3 --seed 42 [--out-dir DIR] [--resume] [--epochs N] [--max-steps N]
                            [--limit-tracks N] [--preload]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.m3_dataset import build_pair_dataset
from src.models.losses import nt_xent
from src.models.m3_cnn import build_model, count_params


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_ratio: float = 0.01) -> float:
    """Khởi động tuyến tính rồi giảm cosine về base_lr*min_ratio."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog)))


class EarlyStopper:
    """Theo dõi loss Val. step() -> (là_tốt_nhất, nên_dừng). patience<=0: không bao giờ dừng."""

    def __init__(self, patience: int, min_delta: float = 1e-6, best: float = float("inf"), bad: int = 0):
        self.patience, self.min_delta, self.best, self.bad = int(patience), float(min_delta), float(best), int(bad)

    def step(self, val_loss: float) -> tuple[bool, bool]:
        if val_loss < self.best - self.min_delta:
            self.best, self.bad = float(val_loss), 0
            return True, False
        self.bad += 1
        return False, self.patience > 0 and self.bad >= self.patience


def plot_history(history: list[dict], path, best_epoch: int | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] + 1 for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("NT-Xent loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val")
    ax[1].set_title("top-1 trong batch (theo dõi)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    if best_epoch is not None:
        for a in ax:
            a.axvline(best_epoch + 1, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _run_val(model, loader, device, temperature, use_amp) -> tuple[float, float]:
    model.eval()
    ls, ac, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x1, x2, tid in loader:
            if x1.shape[0] < 2:
                continue
            x = torch.cat([x1, x2]).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                z = model(x)
            z1, z2 = z.chunk(2)
            loss, acc = nt_xent(z1, z2, temperature, tid.to(device))
            b = x1.shape[0]
            ls += loss.item() * b; ac += acc.item() * b; n += b
    return ls / max(n, 1), ac / max(n, 1)


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    """Áp 'a.b.c=giá_trị' (giá trị đọc bằng YAML) lên bản sao của cfg. Khóa phải tồn tại sẵn (chống gõ nhầm)."""
    import yaml

    out = copy.deepcopy(cfg)
    for item in sets:
        if "=" not in item:
            raise ValueError(f"--set phải có dạng khóa=giá_trị, nhận {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for k in parts[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise KeyError(f"Không có khóa cấu hình {key!r} (thiếu {k!r})")
            node = node[k]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise KeyError(f"Không có khóa cấu hình {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_ablation(name: str) -> dict:
    """Đọc configs/m3_ablations.yaml; trả {'desc', 'set': [...], 'kind'}."""
    from src.common.config import load_config

    ab = load_config("m3_ablations")["ablations"]
    if name not in ab:
        raise KeyError(f"Không có ablation {name!r}. Có: {sorted(ab)}")
    return ab[name]


def train(cfg: dict, seed: int, out_dir, *, resume: bool = False, epochs: int | None = None,
          max_steps_per_epoch: int | None = None, stop_after: int | None = None, tracks_df=None, logmel_dir=None, norm_path=None,
          results_path=None, device: str | None = None, extra: dict | None = None, log=print) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = int(seed)
    tc = cfg["train"]
    n_epochs = int(epochs if epochs is not None else tc["epochs"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(tc.get("amp", True)) and dev.type == "cuda"
    temperature = float(cfg["loss"]["temperature"])
    bs, nw = int(tc["batch_size"]), int(tc.get("num_workers", 0))

    kw = dict(tracks_df=tracks_df, logmel_dir=logmel_dir, norm_path=norm_path)
    train_ds, stats = build_pair_dataset(cfg, "train", **kw)
    val_ds, _ = build_pair_dataset(cfg, "val", augment=bool(tc.get("val_augment", True)), **kw)
    val_ds.set_epoch(0)
    norm_std = float(stats["std_after_crop_mean"])
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False,
                            pin_memory=dev.type == "cuda")

    model = build_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc.get("weight_decay", 0.0)))
    scaler = (torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=use_amp))
    steps_per_epoch = len(train_ds) // bs
    if steps_per_epoch < 1:
        raise ValueError(f"Chỉ có {len(train_ds)} mục < batch_size {bs}")
    if max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, int(max_steps_per_epoch))
    total_steps = n_epochs * steps_per_epoch
    warm = int(tc.get("warmup_epochs", 0)) * steps_per_epoch

    history: list[dict] = []
    stopper = EarlyStopper(int(tc.get("early_stop_patience", 0)))
    best_epoch, start_epoch, elapsed_before = -1, 0, 0.0
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    meta = {"cfg": {k: cfg[k] for k in ("data", "model", "loss", "train", "augment", "seed")},
            "norm_std": norm_std, "seed": int(seed)}
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        if ck["cfg"]["model"] != cfg["model"] or int(ck["seed"]) != int(seed):
            raise RuntimeError("Checkpoint không khớp cấu hình model/seed hiện tại; không resume.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        history, best_epoch, start_epoch = ck["history"], ck["best_epoch"], ck["epoch"] + 1
        stopper = EarlyStopper(int(tc.get("early_stop_patience", 0)), best=ck["best_val"], bad=ck["bad_epochs"])
        elapsed_before = float(ck.get("elapsed", 0.0))
        log(f"Resume từ epoch {start_epoch}/{n_epochs}")

    t_start, stopped_early = time.time(), False
    for epoch in range(start_epoch, n_epochs):
        if stop_after is not None and epoch >= int(stop_after):
            break          # dừng sớm có chủ đích (lịch lr vẫn theo n_epochs); chạy lại với --resume để tiếp tục
        t_ep = time.time()
        train_ds.set_epoch(epoch)
        g = torch.Generator(); g.manual_seed(int(seed) * 1000 + epoch)
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw, generator=g,
                            pin_memory=dev.type == "cuda")
        model.train()
        ls = ac = 0.0
        n_steps = 0
        for step, (x1, x2, tid) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            lr = cosine_lr(epoch * steps_per_epoch + step, total_steps, warm, float(tc["lr"]))
            for pg in opt.param_groups:
                pg["lr"] = lr
            x = torch.cat([x1, x2]).to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                z = model(x)
            z1, z2 = z.chunk(2)
            loss, acc = nt_xent(z1, z2, temperature, tid.to(dev))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss không hữu hạn tại epoch {epoch}, step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if float(tc.get("grad_clip", 0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
            scaler.step(opt); scaler.update()
            ls += loss.item(); ac += acc.item(); n_steps += 1
        val_loss, val_acc = _run_val(model, val_loader, dev, temperature, use_amp)
        entry = {"epoch": epoch, "train_loss": ls / n_steps, "train_acc": ac / n_steps, "val_loss": val_loss,
                 "val_acc": val_acc, "lr": lr, "seconds": round(time.time() - t_ep, 1)}
        history.append(entry)
        is_best, stop = stopper.step(val_loss)
        if is_best:
            best_epoch = epoch
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
              "history": history, "best_val": stopper.best, "best_epoch": best_epoch, "bad_epochs": stopper.bad,
              "elapsed": elapsed_before + time.time() - t_start, **meta}
        _atomic_save(ck, last_path)
        if is_best:
            _atomic_save(ck, best_path)
        _atomic_json(history, out_dir / "history.json")
        plot_history(history, out_dir / "curve.png", best_epoch)
        log(f"epoch {epoch + 1:3d}/{n_epochs} train {entry['train_loss']:.4f}/{entry['train_acc']:.3f} "
            f"val {val_loss:.4f}/{val_acc:.3f} lr {lr:.2e} {entry['seconds']}s{'  *best' if is_best else ''}")
        if stop:
            stopped_early = True
            log(f"Dừng sớm: {stopper.bad} epoch liên tiếp không cải thiện loss Val")
            break

    bh = history[best_epoch] if best_epoch >= 0 else {}
    summary = {
        "seed": int(seed), "epochs_planned": n_epochs, "epochs_run": len(history), "stopped_early": stopped_early,
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None, "best_val_loss": bh.get("val_loss"),
        "best_val_acc": bh.get("val_acc"), "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "n_params": count_params(model), "norm_std": norm_std, "batch_size": bs, "steps_per_epoch": steps_per_epoch,
        "device": dev.type, "cuda_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "torch": torch.__version__, "amp": use_amp,
        "total_seconds": round(elapsed_before + time.time() - t_start, 1),
        "split_used": ["train", "val"], "test_touched": False,
        "model": cfg["model"], "loss": cfg["loss"], "train": tc, "augment": cfg.get("augment"),
        **(extra or {}),
    }
    _atomic_json(summary, out_dir / "summary.json")
    if results_path is not None:
        _atomic_json(summary, Path(results_path))
    return summary


def main(argv=None) -> int:
    from src.common.config import load_config
    from src.common.paths import resolve

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stop-after", type=int, default=None, help="chỉ chạy tới epoch này (chia phiên Colab); sau đó --resume")
    ap.add_argument("--max-steps", type=int, default=None, help="giới hạn số step mỗi epoch (chạy thử)")
    ap.add_argument("--limit-tracks", type=int, default=None, help="chỉ dùng N bài Train (và N//3 bài Val) để chạy thử")
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--ablation", default=None, help="tên trong configs/m3_ablations.yaml")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="ghi đè cấu hình, vd augment.noise.p=0")
    args = ap.parse_args(argv)

    cfg = load_config("m3")
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    sets, extra, tag = list(args.sets), {}, None
    if args.ablation:
        ab = load_ablation(args.ablation)
        sets = list(ab["set"]) + sets
        tag = f"abl_{args.ablation}"
        extra["ablation"] = {"name": args.ablation, "kind": ab.get("kind", "ablation"), "desc": ab.get("desc")}
    if sets:
        cfg = apply_overrides(cfg, sets)
        extra["overrides"] = sets
    stem = f"{tag}_seed{seed}" if tag else f"seed{seed}"
    if args.preload:
        cfg["data"]["preload"] = True
    tracks_p, mel_d = resolve(cfg["data"]["tracks_path"]), resolve(cfg["data"]["logmel_dir"])
    if not tracks_p.exists() or not mel_d.exists():
        print(f"CHƯA CHẠY: thiếu {tracks_p} hoặc {mel_d}")
        return 2
    import pandas as pd
    df = pd.read_parquet(tracks_p)
    if args.limit_tracks:
        parts = [df[df["split"] == "train"].head(args.limit_tracks), df[df["split"] == "val"].head(max(2, args.limit_tracks // 3))]
        df = pd.concat(parts)
    out_dir = Path(args.out_dir) if args.out_dir else resolve(cfg["train"]["checkpoint_dir"]) / f"m3_{stem}"
    summary = train(cfg, seed, out_dir, resume=args.resume, epochs=args.epochs, max_steps_per_epoch=args.max_steps, stop_after=args.stop_after,
                    tracks_df=df, results_path=resolve(f"experiments/results/e3_train_{stem}.json"), extra=extra)
    print(json.dumps({k: summary[k] for k in ("seed", "epochs_run", "best_epoch", "best_val_loss", "best_val_acc",
                                              "final_train_loss", "total_seconds", "device")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
