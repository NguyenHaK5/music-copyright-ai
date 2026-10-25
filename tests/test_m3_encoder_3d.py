"""Test 3D-1: M3Encoder (nạp checkpoint, embedding, cửa sổ trượt)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.m3_cnn import build_model
from src.models.m3_encoder import M3Encoder, window_starts

F = 128
CFG = {"data": {"n_mels": F},
       "model": {"channels": [4, 8], "blocks_per_stage": 1, "embed_dim": 16, "pool": "freq_flatten", "dropout": 0.0}}


@pytest.fixture()
def ckpt(tmp_path):
    torch.manual_seed(0)
    model = build_model(CFG)
    model.train()
    for _ in range(3):                                   # cập nhật thống kê BatchNorm để eval khác khởi tạo
        model(torch.randn(8, 1, F, 60))
    p = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": CFG, "norm_std": 12.0, "seed": 5, "epoch": 3,
                "best_epoch": 3}, p)
    return p, model.eval()


def _mel(seed=0, t=300):
    return np.random.default_rng(seed).normal(-30, 12, (F, t)).astype(np.float32)


def test_window_starts():
    assert window_starts(100, 215, 107) == [0]
    assert window_starts(215, 215, 107) == [0]
    assert window_starts(1292, 215, 107) == list(range(0, 1078, 107))


def test_roundtrip_matches_direct_model(ckpt):
    p, model = ckpt
    enc = M3Encoder.from_checkpoint(p, device="cpu")
    mel = _mel()
    from src.features.m3_input import normalize_crop
    with torch.no_grad():
        ref = model(torch.from_numpy(normalize_crop(mel, 12.0))[None, None]).numpy()[0]
    z = enc.embed_mel(mel)
    assert np.allclose(z, ref, atol=1e-5) and abs(np.linalg.norm(z) - 1) < 1e-5
    assert enc.meta["seed"] == 5 and enc.n_mels == F


def test_variable_length_and_gain_invariance(ckpt):
    enc = M3Encoder.from_checkpoint(ckpt[0], device="cpu")
    for t in (129, 215, 431):
        assert enc.embed_mel(_mel(1, t)).shape == (16,)
    m = _mel(2)
    assert np.allclose(enc.embed_mel(m), enc.embed_mel(m + 9.0), atol=1e-4)


def test_embed_crops_batching_consistent(ckpt):
    enc = M3Encoder.from_checkpoint(ckpt[0], device="cpu", batch_size=3)
    crops = np.stack([_mel(i, 215) for i in range(7)])
    z = enc.embed_crops(crops)
    assert z.shape == (7, 16)
    assert np.allclose(z[4], enc.embed_mel(crops[4]), atol=1e-5)
    assert enc.embed_crops(crops[:0].reshape(0, F, 215)).shape[0] == 0


def test_window_embeddings_count_and_padding(ckpt):
    enc = M3Encoder.from_checkpoint(ckpt[0], device="cpu")
    w = enc.window_embeddings(_mel(3, 1292), 215, 107)
    assert w.shape == (len(window_starts(1292, 215, 107)), 16)
    assert enc.window_embeddings(_mel(4, 100), 215, 107).shape == (1, 16)


def test_wrong_shapes_rejected(ckpt):
    enc = M3Encoder.from_checkpoint(ckpt[0], device="cpu")
    with pytest.raises(ValueError):
        enc.embed_mel(np.zeros((64, 200), np.float32))
    with pytest.raises(ValueError):
        enc.embed_mel(np.zeros((F,), np.float32))
    with pytest.raises(ValueError):
        enc.embed_crops(np.zeros((F, 200), np.float32))
