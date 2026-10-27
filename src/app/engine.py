"""Dựng Engine demo từ cấu hình + checkpoint (Phase 5). Chỉ mục lưu .npz khóa theo SHA256 checkpoint để khởi động nhanh."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.app.pipeline import DemoConfig, Engine


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_calibration(path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Chưa có {p}: chạy `python -m src.app.calibrate_demo` trước (ngưỡng demo lấy từ Val).")
    c = json.loads(p.read_text(encoding="utf-8"))
    if c.get("split") != "val":
        raise ValueError("Hiệu chuẩn ngưỡng demo phải từ split val")
    return c


def get_or_build_index(retriever, index_ids, mel_loader, cache_path, ckpt_sha: str, log=print) -> str:
    """Nạp chỉ mục từ cache nếu khớp checkpoint và danh sách bài; ngược lại dựng và lưu. Trả 'cache' hoặc 'built'."""
    cp = Path(cache_path)
    if cp.exists():
        try:
            d = np.load(cp, allow_pickle=False)
            if str(d["ckpt_sha"]) == ckpt_sha and np.array_equal(d["track_ids"], np.asarray(index_ids, dtype=np.int64)):
                retriever.import_index(d)
                return "cache"
        except Exception as e:                      # cache hỏng/khác phiên bản: dựng lại
            log(f"Bỏ qua cache chỉ mục ({e})")
    retriever.build_index(index_ids, mel_loader)
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cp, ckpt_sha=np.asarray(ckpt_sha), **retriever.export_index())
    return "built"


def load_meta(tracks_clean_path, tracks_path, ids) -> dict:
    """track_id -> {title, artist, genre, license_raw}; thiếu cột nào thì để None (không bịa)."""
    import pandas as pd

    path = Path(tracks_clean_path) if Path(tracks_clean_path).exists() else Path(tracks_path)
    df = pd.read_parquet(path)
    if "track_id" in df.columns:
        df = df.set_index("track_id")
    want = {"title": "title", "artist": "artist_name", "genre": "genre", "license_raw": "license_raw"}
    out = {}
    for t in ids:
        if int(t) in df.index:
            row = df.loc[int(t)]
            out[int(t)] = {k: (None if c not in df.columns or row[c] is None or (isinstance(row[c], float) and row[c] != row[c])
                               else str(row[c])) for k, c in want.items()}
    return out


def build_engine(log=print) -> Engine:
    import pandas as pd
    from src.common.config import load_config
    from src.common.paths import resolve
    from src.features.cache import load_logmel
    from src.models.m3_retrieval import build_m3

    demo, base, feat = load_config("demo"), load_config("baseline"), load_config("features")
    rules = load_config("license_rules")
    ck = resolve(demo["model"]["ckpt"])
    if not ck.exists():
        raise FileNotFoundError(f"Thiếu checkpoint {ck}")
    cal = load_calibration(resolve(demo["calibration_path"]))
    up = demo["upload"]
    cfg = DemoConfig(sr=int(feat["sample_rate"]), n_fft=int(feat["n_fft"]), hop_length=int(feat["hop_length"]),
                     n_mels=int(feat["n_mels"]), max_mb=float(up["max_mb"]), allowed_ext=tuple(up["allowed_ext"]),
                     min_seconds=float(up["min_seconds"]), query_seconds=float(up["query_seconds"]),
                     silence_peak=float(up["silence_peak"]), top_k=int(demo["decision"]["top_k"]))
    retr = build_m3({"sr": base["sr"], "hop_length": base["hop_length"], "m1": base.get("m1", {}),
                     "m3": {"ckpt": str(ck), "device": demo["model"].get("device")}})
    tracks = pd.read_parquet(resolve(base["tracks_path"]))
    if "track_id" not in tracks.columns:
        tracks = tracks.reset_index()
    ix = demo["index"]
    ids = sorted(int(t) for t in tracks[(tracks["split"] == ix["split"]) & (tracks["role"] == ix["role"])]["track_id"])
    cache_dir = resolve(base["logmel_cache_dir"])
    sha = _sha(ck)
    how = get_or_build_index(retr, ids, lambda t: load_logmel(cache_dir, t), resolve(ix["cache_path"]), sha, log)
    meta = load_meta(resolve("data/interim/tracks_clean.parquet"), resolve(base["tracks_path"]), ids)
    info = {"ckpt": str(demo["model"]["ckpt"]), "ckpt_sha256": sha[:12], "n_index_tracks": len(ids), "index_windows": retr.index_size,
            "index_source": how, "calibration": {k: cal[k] for k in ("tau_high", "tau_low", "ckpt_sha256", "created_utc") if k in cal},
            "meta_has_titles": any(m.get("title") for m in meta.values())}
    return Engine(retr, cfg, rules, float(cal["tau_high"]), float(cal["tau_low"]), meta, info)
