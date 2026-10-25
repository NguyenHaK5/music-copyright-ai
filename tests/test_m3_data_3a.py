"""Test 3A: augmentation log-mel, dataset cặp view, thống kê chuẩn hóa. Dữ liệu tổng hợp, không cần FMA."""
import numpy as np
import pytest

from src.data.m3_dataset import (PairDataset, build_pair_dataset, compute_norm_stats, crop_frames_from_cfg,
                                 load_or_compute_norm, logmel_path)
from src.features import augment_mel as am
from src.features.m3_input import normalize_crop

SR, F, L = 22050, 128, 215
FULL = {k: dict(v, p=1.0) for k, v in am.DEFAULTS.items()}


def _mel(rng, t=900):
    return (rng.normal(-30, 12, size=(F, t))).astype(np.float32)


def test_held_out_and_unknown_ops_rejected():
    for bad in ("mix", "overlay", "codec"):
        with pytest.raises(ValueError, match="giữ ra"):
            am.MelAugmenter({bad: {"p": 1.0}})
    with pytest.raises(ValueError):
        am.MelAugmenter({"pitch_magic": {"p": 1.0}})


def test_time_warp_identity_and_rate():
    x = np.tile(np.arange(100, dtype=np.float32), (F, 1))
    assert np.allclose(am.time_warp(x, 1.0, 100), x)
    y = am.time_warp(x, 2.0, 40)
    assert np.allclose(y[0], np.arange(40) * 2.0)


def test_freq_warp_identity_and_peak_shift():
    rng = np.random.default_rng(0)
    x = _mel(rng, 50)
    assert np.allclose(am.freq_warp(x, 0.0, SR), x, atol=1e-3)
    pk = np.full((F, 20), -80.0, dtype=np.float32)
    pk[40] = 0.0
    out = am.freq_warp(pk, 12.0, SR)          # lên 1 quãng tám: tần số gấp đôi
    f = am.mel_bin_freqs(F, SR)
    expected = int(np.argmin(np.abs(f - 2 * f[40])))
    assert abs(int(np.argmax(out[:, 0])) - expected) <= 1
    down = am.freq_warp(pk, -12.0, SR)         # xuống: dải cao nhất không có nguồn -> điền min
    assert down[-1, 0] == pk.min()


def test_eq_lowpass_and_highpass():
    f = am.mel_bin_freqs(F, SR)
    lp = am.eq_gain_db(f, "lowpass", 3000.0)
    hp = am.eq_gain_db(f, "highpass", 500.0)
    assert lp[f < 500].max() <= 0 and lp[f < 500].min() > -0.5
    assert lp[f > 9000].max() < -30
    assert hp[f > 3000].min() > -0.5 and hp[f < 100].max() < -20
    assert lp.min() >= -60.0
    with pytest.raises(ValueError):
        am.eq_gain_db(f, "bandpass", 1000.0)


def test_noise_snr_matches_target():
    rng = np.random.default_rng(1)
    x = _mel(rng, 400)
    p = 10 ** (x.astype(np.float64) / 10)
    p2 = 10 ** (am.add_noise(x, 10.0, np.random.default_rng(2)).astype(np.float64) / 10)
    ratio = (p2 - p).mean() / p.mean()
    assert 0.08 < ratio < 0.12      # 10 dB -> 0,1


def test_gain_ramp_zero_mean_shift_and_span():
    x = np.zeros((F, 101), dtype=np.float32)
    y = am.gain_ramp(x, 6.0)
    assert abs(y.mean()) < 1e-5 and abs(y[0, -1] - y[0, 0] - 6.0) < 1e-4


def test_spec_mask_fills_with_mean_and_bounds():
    rng = np.random.default_rng(3)
    x = _mel(rng, 200)
    y = am.spec_mask(x, np.random.default_rng(4), 2, 12, 2, 20)
    assert (y == x.mean()).any() and (y != x).mean() < 0.5
    assert y.shape == x.shape


def test_augmenter_deterministic_and_shapes():
    aug = am.MelAugmenter(FULL, SR, F)
    src = aug.source_frames(L)
    assert src == int(np.ceil(L * 1.15)) + 2
    x = _mel(np.random.default_rng(5), src + 10)
    a, ops_a = aug(x, L, np.random.default_rng(7))
    b, ops_b = aug(x, L, np.random.default_rng(7))
    assert a.shape == (F, L) and a.dtype == np.float32 and np.isfinite(a).all()
    assert np.array_equal(a, b) and ops_a == ops_b
    assert set(ops_a) <= set(am.ALLOWED_OPS)


def test_augmenter_all_off_is_plain_crop():
    aug = am.MelAugmenter({}, SR, F)
    assert aug.active_ops == [] and aug.source_frames(L) == L
    x = _mel(np.random.default_rng(6), 300)
    out, ops = aug(x, L, np.random.default_rng(0))
    assert ops == [] and np.array_equal(out, x[:, :L])
    with pytest.raises(ValueError):
        aug(x[:, :100], L, np.random.default_rng(0))


def test_op_probability_roughly_respected():
    cfg = {"noise": {"p": 0.3}}
    aug = am.MelAugmenter(cfg, SR, F)
    x = _mel(np.random.default_rng(8), L)
    rng = np.random.default_rng(9)
    hits = sum("noise" in aug(x, L, rng)[1] for _ in range(400)) / 400
    assert 0.22 < hits < 0.38


def test_normalize_crop():
    x = _mel(np.random.default_rng(10), L) + 17.0
    y = normalize_crop(x, 5.0)
    assert abs(y.mean()) < 1e-4
    assert np.allclose(normalize_crop(x, 5.0), normalize_crop(x + 9.0, 5.0), atol=1e-4)   # bất biến độ lợi
    with pytest.raises(ValueError):
        normalize_crop(x, 0.0)


def _write_tracks(tmp_path, ids, t=900, short=None):
    rng = np.random.default_rng(11)
    d = tmp_path / "logmel"
    d.mkdir()
    for i in ids:
        tt = 120 if i == short else t
        np.save(logmel_path(d, i), _mel(rng, tt).astype(np.float16))
    return d


def _ds(tmp_path, ids=(1, 2, 3, 4), cfg=FULL, **kw):
    d = _write_tracks(tmp_path, ids, short=kw.pop("short", None))
    return PairDataset(list(ids), d, am.MelAugmenter(cfg, SR, F), L, seed=3, norm_std=8.0, **kw)


def test_dataset_item_shapes_and_determinism(tmp_path):
    ds = _ds(tmp_path)
    x1, x2, tid = ds[2]
    assert x1.shape == (1, F, L) and x1.dtype == np.float32 and tid == 3
    assert not np.array_equal(x1, x2)
    y1, y2, _ = ds[2]
    assert np.array_equal(x1, y1) and np.array_equal(x2, y2)
    ds.set_epoch(1)
    z1, _, _ = ds[2]
    assert not np.array_equal(x1, z1)
    assert abs(float(x1.mean())) < 1e-4


def test_dataset_repeats_and_ids(tmp_path):
    ds = _ds(tmp_path, repeats=3)
    assert len(ds) == 12 and [ds[i][2] for i in range(5)] == [1, 2, 3, 4, 1]
    with pytest.raises(ValueError):
        PairDataset([], tmp_path, am.MelAugmenter({}, SR, F), L)


def test_dataset_preload_equals_mmap(tmp_path):
    a = _ds(tmp_path)
    b = PairDataset(a.track_ids, a.logmel_dir, a.aug, L, seed=3, norm_std=8.0, preload=True)
    assert np.array_equal(a[1][0], b[1][0])


def test_dataset_pads_short_clip(tmp_path):
    ds = _ds(tmp_path, short=2)
    x1, _, tid = ds[1]
    assert tid == 2 and x1.shape == (1, F, L) and np.isfinite(x1).all()


def test_min_start_distance_honoured(tmp_path):
    ds = _ds(tmp_path, min_start_distance=215)
    rng = np.random.default_rng(0)
    for _ in range(50):
        s1, s2 = ds.sample_starts(rng, 900, 215)
        assert abs(s1 - s2) >= 215
    assert ds.sample_starts(rng, 100, 215) == (0, 0)


def test_norm_stats_deterministic_and_cached(tmp_path):
    d = _write_tracks(tmp_path, range(1, 7))
    load = lambda t: np.load(logmel_path(d, t), mmap_mode="r")
    a = compute_norm_stats(load, range(1, 7), L, 5, max_tracks=4, crops_per_track=3)
    b = compute_norm_stats(load, range(1, 7), L, 5, max_tracks=4, crops_per_track=3)
    assert a == b and 8 < a["std_after_crop_mean"] < 16 and a["n_tracks"] == 4
    p = tmp_path / "norm.json"
    s1 = load_or_compute_norm(p, load, range(1, 7), L, 5)
    assert p.exists()
    s2 = load_or_compute_norm(p, lambda t: (_ for _ in ()).throw(AssertionError("không được tính lại")),
                              range(1, 7), L, 5)
    assert s1 == s2
    s3 = load_or_compute_norm(p, load, range(1, 7), 100, 5)      # crop_frames đổi -> tính lại
    assert s3["crop_frames"] == 100


def test_build_pair_dataset_from_cfg(tmp_path):
    import pandas as pd
    d = _write_tracks(tmp_path, range(1, 9))
    df = pd.DataFrame({"track_id": range(1, 9), "split": ["train"] * 5 + ["val"] * 2 + ["test"]})
    cfg = {"seed": 1,
           "data": {"sr": SR, "hop_length": 512, "n_mels": F, "crop_seconds": 5.0, "repeats_per_epoch": 2,
                    "norm_stats": {"max_tracks": 3, "crops_per_track": 2}, "preload": False},
           "augment": {"noise": {"p": 1.0}}}
    assert crop_frames_from_cfg(cfg["data"]) == 215
    tr, stats = build_pair_dataset(cfg, "train", tracks_df=df, logmel_dir=d, norm_path=tmp_path / "n.json")
    va, _ = build_pair_dataset(cfg, "val", tracks_df=df, logmel_dir=d, norm_path=tmp_path / "n.json", augment=False)
    assert tr.track_ids == [1, 2, 3, 4, 5] and len(tr) == 10 and va.track_ids == [6, 7]
    assert tr.aug.active_ops == ["noise"] and va.aug.active_ops == []
    assert stats["std_after_crop_mean"] > 0
    bad = dict(cfg, augment={"codec": {"p": 1.0}})
    with pytest.raises(ValueError):
        build_pair_dataset(bad, "train", tracks_df=df, logmel_dir=d, norm_path=tmp_path / "n2.json")


def test_dataloader_collate_if_torch_available(tmp_path):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    ds = _ds(tmp_path)
    x1, x2, tid = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    assert x1.shape == (4, 1, F, L) and tid.tolist() == [1, 2, 3, 4]
