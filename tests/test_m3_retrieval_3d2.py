"""Test 3D-2: M3Retriever (giao diện build_index/score_matrix) và build_m3."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.m3_cnn import build_model
from src.models.m3_encoder import M3Encoder, window_starts
from src.models.m3_retrieval import M3Retriever, build_m3

F, WIN, HOP = 128, 215, 107
CFG = {"data": {"n_mels": F, "sr": 22050, "hop_length": 512},
       "model": {"channels": [4, 8], "blocks_per_stage": 1, "embed_dim": 16, "pool": "freq_flatten", "dropout": 0.0}}


def _tracks(n=6, t=700):
    out = {}
    for i in range(n):
        rng = np.random.default_rng(100 + i)
        base = np.linspace(-10 - 5 * i, -60 + 4 * i, F)[:, None] + 8 * np.sin(np.arange(t) / (5 + i))[None, :]
        out[10 + i] = (base + rng.normal(0, 4, (F, t))).astype(np.float32)
    return out


@pytest.fixture()
def ckpt(tmp_path):
    torch.manual_seed(0)
    m = build_model(CFG).train()
    for _ in range(4):
        m(torch.randn(8, 1, F, 80))
    p = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "cfg": CFG, "norm_std": 12.0, "seed": 1, "epoch": 2, "best_epoch": 2}, p)
    return p


def _ret(ckpt, mode="whole"):
    enc = M3Encoder.from_checkpoint(ckpt, device="cpu")
    r = M3Retriever(enc, WIN, HOP, mode)
    tr = _tracks()
    r.build_index(list(tr), lambda t: tr[t])
    return r, tr


def test_shape_column_order_and_exact_window_hits(ckpt):
    r, tr = _ret(ckpt)
    assert r.track_ids == list(tr) and r.index_size == sum(len(window_starts(700, WIN, HOP)) for _ in tr)
    starts = window_starts(700, WIN, HOP)
    qs = [tr[t][:, starts[2]:starts[2] + WIN] for t in tr]
    S = r.score_matrix(qs)
    assert S.shape == (6, 6) and S.dtype == np.float32
    assert (S.argmax(1) == np.arange(6)).all()
    assert np.allclose(np.diag(S), 1.0, atol=1e-4)


def test_streaming_iterable_without_len_and_mixed_lengths(ckpt):
    r, tr = _ret(ckpt)
    qs = [tr[10][:, :129], tr[11][:, 50:265], tr[12][:, 10:441], tr[13][:, :129]]
    S_list = r.score_matrix(qs)
    S_gen = r.score_matrix(q for q in qs)
    S_chunk = r.score_matrix(qs, chunk=2)
    one = np.stack([r.score_matrix([q])[0] for q in qs])
    assert np.allclose(S_list, S_gen) and np.allclose(S_list, S_chunk, atol=1e-5) and np.allclose(S_list, one, atol=1e-5)


def test_windows_mode_matches_whole_for_short_and_differs_for_long(ckpt):
    rw, tr = _ret(ckpt, "windows")
    rh, _ = _ret(ckpt, "whole")
    q5, q10 = tr[10][:, :WIN], tr[10][:, :431]
    assert np.allclose(rw.score_matrix([q5]), rh.score_matrix([q5]), atol=1e-5)
    s = rw.score_matrix([q10])
    assert s.shape == (1, 6) and np.isfinite(s).all()
    with pytest.raises(ValueError):
        M3Retriever(rw.enc, WIN, HOP, "bogus")


def test_errors_and_reindex(ckpt):
    enc = M3Encoder.from_checkpoint(ckpt, device="cpu")
    r = M3Retriever(enc, WIN, HOP)
    with pytest.raises(RuntimeError):
        r.score_matrix([np.zeros((F, 100), np.float32)])
    tr = _tracks()
    r.build_index([10, 11], lambda t: tr[t])
    n2 = r.index_size
    r.build_index(list(tr), lambda t: tr[t])
    assert r.index_size > n2 and len(r.track_ids) == 6


def test_build_m3_from_cfg(ckpt):
    base = {"sr": 22050, "hop_length": 512, "m1": {"win_sec": 5.0, "hop_sec": 2.5}}
    with pytest.raises(ValueError, match="checkpoint"):
        build_m3(base)
    r = build_m3({**base, "m3": {"ckpt": str(ckpt), "device": "cpu"}})
    assert r.win == 215 and r.hop == round(2.5 * 22050 / 512)
    assert r.query_mode == "whole"
    with pytest.raises(ValueError, match="sr="):
        build_m3({**base, "sr": 16000, "m3": {"ckpt": str(ckpt), "device": "cpu"}})
