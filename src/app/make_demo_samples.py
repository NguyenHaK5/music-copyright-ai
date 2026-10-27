"""Sinh bộ tệp mẫu cho demo/smoke test (Phase 5): ca THUỘC, ca nhiễu, ca NGOÀI, và các ca lỗi đầu vào.

Chỉ dùng bài Test (known/unknown) để minh họa; không phải phép đo. CLI: python -m src.app.make_demo_samples [--audio-dir DIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

IN, OUT, UNSURE = "THUỘC_DANH_MỤC", "NGOÀI_DANH_MỤC", "KHÔNG_XÁC_ĐỊNH"


def _crop(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(seconds * sr)
    s = max(0, (len(y) - n) // 2)
    return y[s:s + n]


def _noisy(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = float(np.mean(y.astype(np.float64) ** 2))
    return (y + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), size=y.shape)).astype(np.float32)


def write_samples(out_dir, known: tuple[int, np.ndarray], unknown: tuple[int, np.ndarray], sr: int = 22050) -> dict:
    """known/unknown = (track_id, audio mono ở sr). Trả {tên tệp: kỳ vọng} và ghi expected.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kt, ky = known
    ut, uy = unknown
    k10 = _crop(ky, sr, 10.0)
    exp = {}

    def w(name, y, expected):
        sf.write(out / name, y, sr)
        exp[name] = expected

    w(f"known_{kt}.wav", k10, IN)
    w(f"known_{kt}_snr10.wav", _noisy(k10, 10.0, 0), f"{IN}|{UNSURE}")
    w(f"unknown_{ut}.wav", _crop(uy, sr, 10.0), f"{OUT}|{UNSURE}")
    w("too_short.wav", k10[:sr], "too_short")
    w("silent.wav", np.zeros(5 * sr, np.float32), "silent")
    (out / "broken.mp3").write_bytes(np.random.default_rng(0).integers(0, 256, 4096, dtype=np.uint8).tobytes())
    exp["broken.mp3"] = "decode_error|too_short"
    (out / "empty.wav").write_bytes(b"")
    exp["empty.wav"] = "empty"
    (out / "expected.json").write_text(json.dumps(exp, ensure_ascii=False, indent=2), encoding="utf-8")
    return exp


def main(argv=None) -> int:
    import argparse
    import pandas as pd
    from src.common.config import load_config
    from src.common.paths import resolve
    from src.data.clean import check_audio, fma_audio_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default=None)
    a = ap.parse_args(argv)
    demo, base = load_config("demo"), load_config("baseline")
    audio_dir = resolve(a.audio_dir or base.get("audio_dir", "data/raw/fma_small"))
    tracks = pd.read_parquet(resolve(base["tracks_path"]))
    if "track_id" not in tracks.columns:
        tracks = tracks.reset_index()
    test = tracks[tracks["split"] == "test"]

    def load(role):
        for t in sorted(int(x) for x in test[test["role"] == role]["track_id"]):
            code, _, y = check_audio(fma_audio_path(audio_dir, t), 22050, 20.0, 30.0, 1e-4)
            if code is None:
                return t, y
        raise SystemExit(f"CHƯA CHẠY: không đọc được audio Test role={role} trong {audio_dir}")

    exp = write_samples(resolve(demo["samples_dir"]), load("known"), load("unknown"))
    print(json.dumps(exp, ensure_ascii=False, indent=2))
    print(f"-> {resolve(demo['samples_dir'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
