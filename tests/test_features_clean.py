import numpy as np
import pandas as pd
import soundfile as sf

from src.data.build_clean_cache import build
from src.data.clean import check_audio, fma_audio_path
from src.features.cache import has_logmel, load_logmel, save_logmel
from src.features.logmel import logmel, peak_normalize

SR = 22050
FCFG = {"sample_rate": SR, "n_fft": 2048, "hop_length": 512, "n_mels": 128, "cache_dtype": "float16",
        "clean": {"min_seconds": 2.0, "max_seconds": 3.0, "silence_peak_threshold": 1e-4,
                  "checkpoint_every": 2}}


def _tone(sec, amp=0.5):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_fma_audio_path():
    assert fma_audio_path("a", 2).as_posix() == "a/000/000002.mp3"
    assert fma_audio_path("a", 123456).as_posix() == "a/123/123456.mp3"


def test_logmel_shape_and_norm():
    y = peak_normalize(_tone(3.0, amp=0.1))
    assert abs(float(np.max(np.abs(y))) - 1.0) < 1e-5
    m = logmel(y)
    assert m.shape == (128, 1 + len(y) // 512) and m.dtype == np.float32
    assert np.isfinite(m).all()
    assert np.array_equal(peak_normalize(np.zeros(10, np.float32)), np.zeros(10, np.float32))


def test_check_audio_reasons(tmp_path):
    ok, short, silent = tmp_path / "ok.wav", tmp_path / "short.wav", tmp_path / "sil.wav"
    sf.write(ok, _tone(5), SR); sf.write(short, _tone(0.5), SR); sf.write(silent, np.zeros(SR * 3), SR)
    bad = tmp_path / "bad.wav"; bad.write_bytes(b"not audio")
    args = (SR, 2.0, 3.0, 1e-4)
    r, dur, y = check_audio(ok, *args)
    assert r is None and abs(dur - 3.0) < 0.05 and y is not None  # cắt còn 3 s
    assert check_audio(short, *args)[0] == "too_short"
    assert check_audio(silent, *args)[0] == "silent"
    assert check_audio(bad, *args)[0] == "decode_error"
    assert check_audio(tmp_path / "nope.wav", *args)[0] == "missing_file"


def test_cache_roundtrip(tmp_path):
    m = np.random.randn(128, 10).astype(np.float32)
    save_logmel(tmp_path, 7, m)
    assert has_logmel(tmp_path, 7) and not has_logmel(tmp_path, 8)
    assert np.allclose(load_logmel(tmp_path, 7), m, atol=0.05)  # float16


def test_build_resume(tmp_path):
    for tid in (1, 2, 3):
        sf.write(tmp_path / f"{tid}.wav", _tone(3), SR)
    sf.write(tmp_path / "4.wav", _tone(0.3), SR)
    pf = lambda t: tmp_path / f"{t}.wav"
    cache, rep_path = tmp_path / "c", tmp_path / "r.csv"
    rep = build([1, 2, 3, 4, 5], pf, cache, rep_path, FCFG, limit=2)
    assert len(rep) == 2                       # chạy dở
    rep = build([1, 2, 3, 4, 5], pf, cache, rep_path, FCFG)
    assert len(rep) == 5 and rep["track_id"].is_unique   # tiếp tục, không lặp
    r = rep.set_index("track_id")
    assert r.loc[1, "file_ok"] and has_logmel(cache, 1)
    assert r.loc[4, "drop_reason"] == "too_short" and not has_logmel(cache, 4)
    assert r.loc[5, "drop_reason"] == "missing_file"
