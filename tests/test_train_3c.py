"""Test 3C: lịch lr, early stopping, vòng huấn luyện, checkpoint, resume (CPU, dữ liệu tổng hợp nhỏ)."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.data.m3_dataset import logmel_path
from src.train_m3 import EarlyStopper, cosine_lr, plot_history, train

F = 128


def _cfg():
    return {"seed": 0,
            "data": {"sr": 22050, "hop_length": 512, "n_mels": F, "crop_seconds": 1.0, "repeats_per_epoch": 1,
                     "norm_stats": {"max_tracks": 4, "crops_per_track": 2}, "norm_seed": 1, "preload": False},
            "augment": {"noise": {"p": 0.5}, "spec_mask": {"p": 0.5}, "time_warp": {"p": 0.5}},
            "model": {"channels": [4, 8], "blocks_per_stage": 1, "embed_dim": 16, "pool": "freq_flatten",
                      "dropout": 0.0},
            "loss": {"name": "nt_xent", "temperature": 0.2},
            "train": {"epochs": 3, "batch_size": 4, "lr": 3e-3, "weight_decay": 0.0, "warmup_epochs": 1,
                      "grad_clip": 5.0, "amp": False, "num_workers": 0, "val_augment": True,
                      "early_stop_patience": 0, "checkpoint_dir": "x"}}


@pytest.fixture()
def data(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "logmel"; d.mkdir()
    ids = list(range(1, 15))
    for i in ids:
        np.save(logmel_path(d, i), (rng.normal(-30, 10, (F, 200)) + np.linspace(0, 20, F)[:, None]).astype(np.float16))
    df = pd.DataFrame({"track_id": ids, "split": ["train"] * 8 + ["val"] * 4 + ["test"] * 2})
    return df, d, tmp_path / "norm.json"


def _run(cfg, data, out, **kw):
    df, d, n = data
    return train(cfg, 0, out, tracks_df=df, logmel_dir=d, norm_path=n, log=lambda *_: None, **kw)


def test_cosine_lr_shape():
    lrs = [cosine_lr(s, 100, 10, 1.0) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:]))
    assert lrs[-1] < 0.02
    assert cosine_lr(0, 10, 0, 2.0) == pytest.approx(2.0)


def test_early_stopper():
    es = EarlyStopper(patience=2)
    assert es.step(1.0) == (True, False)
    assert es.step(1.0) == (False, False)
    assert es.step(0.5) == (True, False) and es.bad == 0
    assert es.step(0.6) == (False, False)
    assert es.step(0.7) == (False, True)
    never = EarlyStopper(patience=0)
    never.step(1.0)
    assert all(never.step(2.0) == (False, False) for _ in range(5))


def test_train_produces_artifacts_and_finite_history(data, tmp_path):
    out = tmp_path / "run"
    s = _run(_cfg(), data, out, results_path=tmp_path / "res.json")
    h = json.loads((out / "history.json").read_text())
    assert len(h) == 3 and s["epochs_run"] == 3 and s["test_touched"] is False
    assert all(np.isfinite([e["train_loss"], e["val_loss"]]).all() for e in h)
    for f in ("last.pt", "best.pt", "curve.png", "summary.json"):
        assert (out / f).exists(), f
    assert (tmp_path / "res.json").exists()
    ck = torch.load(out / "best.pt", weights_only=False)
    assert ck["norm_std"] == pytest.approx(s["norm_std"]) and ck["cfg"]["model"]["embed_dim"] == 16
    assert s["best_epoch"] in (1, 2, 3) and s["split_used"] == ["train", "val"]


def test_resume_matches_uninterrupted_run(data, tmp_path):
    straight = _run(_cfg(), data, tmp_path / "a", epochs=4)
    c2 = _cfg()
    _run(c2, data, tmp_path / "b", epochs=4, stop_after=2)      # lịch lr vẫn theo 4 epoch
    _run(c2, data, tmp_path / "b", epochs=4, resume=True)
    ha = json.loads((tmp_path / "a" / "history.json").read_text())
    hb = json.loads((tmp_path / "b" / "history.json").read_text())
    assert len(ha) == len(hb) == 4 and straight["epochs_run"] == 4
    for x, y in zip(ha, hb):
        assert x["train_loss"] == pytest.approx(y["train_loss"], abs=1e-5)
    pa = torch.load(tmp_path / "a" / "last.pt", weights_only=False)["model"]
    pb = torch.load(tmp_path / "b" / "last.pt", weights_only=False)["model"]
    assert all(torch.allclose(pa[k], pb[k], atol=1e-5) for k in pa)


def test_resume_rejects_mismatched_model(data, tmp_path):
    _run(_cfg(), data, tmp_path / "r", epochs=1)
    bad = copy.deepcopy(_cfg()); bad["model"]["embed_dim"] = 32
    with pytest.raises(RuntimeError, match="không khớp"):
        _run(bad, data, tmp_path / "r", epochs=2, resume=True)


def test_never_uses_test_split(data, tmp_path, monkeypatch):
    seen = []
    import src.train_m3 as tm
    orig = tm.build_pair_dataset
    def spy(cfg, split, **kw):
        seen.append(split); return orig(cfg, split, **kw)
    monkeypatch.setattr(tm, "build_pair_dataset", spy)
    _run(_cfg(), data, tmp_path / "s", epochs=1)
    assert set(seen) == {"train", "val"}


def test_too_small_dataset_and_plot(data, tmp_path):
    c = _cfg(); c["train"]["batch_size"] = 64
    with pytest.raises(ValueError):
        _run(c, data, tmp_path / "t", epochs=1)
    plot_history([{"epoch": 0, "train_loss": 1, "val_loss": 2, "train_acc": .1, "val_acc": .2}], tmp_path / "p.png", 0)
    assert (tmp_path / "p.png").exists()


def test_max_steps_limits_steps(data, tmp_path):
    s = _run(_cfg(), data, tmp_path / "m", epochs=1, max_steps_per_epoch=1)
    assert s["steps_per_epoch"] == 1
