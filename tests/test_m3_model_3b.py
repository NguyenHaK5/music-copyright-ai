"""Test 3B: M3Net và NT-Xent (cần torch)."""
import math

import pytest

torch = pytest.importorskip("torch")

from src.models.losses import nt_xent
from src.models.m3_cnn import M3Net, build_model, count_params

CFG = {"data": {"n_mels": 128},
       "model": {"channels": [16, 32, 64, 128], "blocks_per_stage": 1, "embed_dim": 128,
                 "pool": "freq_flatten", "dropout": 0.0}}


def test_output_shape_unit_norm_and_variable_length():
    net = build_model(CFG).eval()
    for t in (215, 150, 300):
        z = net(torch.randn(3, 1, 128, t))
        assert z.shape == (3, 128)
        assert torch.allclose(z.norm(dim=1), torch.ones(3), atol=1e-5)


def test_gap_variant_and_bad_pool():
    net = M3Net(pool="gap").eval()
    assert net(torch.randn(2, 1, 128, 215)).shape == (2, 128)
    with pytest.raises(ValueError):
        M3Net(pool="max")


def test_param_count_reasonable_and_deterministic_eval():
    net = build_model(CFG).eval()
    assert 3e5 < count_params(net) < 1.5e6
    x = torch.randn(2, 1, 128, 215)
    assert torch.equal(net(x), net(x))


def test_backward_gives_finite_grads():
    net = build_model(CFG).train()
    x1, x2 = torch.randn(8, 1, 128, 215), torch.randn(8, 1, 128, 215)
    loss, _ = nt_xent(net(x1), net(x2), 0.1)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_nt_xent_hand_value_two_orthogonal_pairs():
    # B=2, z1==z2 và hai bài trực giao. Anchor 0: dương sim 1, âm sim 0 (x2) và mình (bị che).
    z = torch.eye(2)
    loss, acc = nt_xent(z, z, temperature=1.0)
    expected = -math.log(math.e / (math.e + 1 + 1))
    assert abs(loss.item() - expected) < 1e-5 and acc.item() == 1.0


def test_nt_xent_low_for_aligned_high_for_random():
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(32, 64), dim=1)
    good, acc_good = nt_xent(z, z + 0.01 * torch.randn_like(z), 0.1)
    bad, _ = nt_xent(z, torch.nn.functional.normalize(torch.randn(32, 64), dim=1), 0.1)
    assert good.item() < 0.1 < bad.item() and acc_good.item() == 1.0


def test_nt_xent_masks_same_track_false_negatives():
    torch.manual_seed(1)
    a = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)
    z1 = torch.cat([a, a, torch.nn.functional.normalize(torch.randn(2, 16), dim=1)])
    z2 = z1.clone()
    ids = torch.tensor([7, 7, 8, 9])
    unmasked, _ = nt_xent(z1, z2, 0.1)
    masked, _ = nt_xent(z1, z2, 0.1, track_ids=ids)
    assert masked.item() < unmasked.item()       # bản sao cùng bài không còn bị phạt như âm
    distinct, _ = nt_xent(z1, z2, 0.1, track_ids=torch.tensor([1, 2, 3, 4]))
    assert abs(distinct.item() - unmasked.item()) < 1e-6


def test_nt_xent_input_validation():
    with pytest.raises(ValueError):
        nt_xent(torch.randn(4, 8), torch.randn(3, 8))
    with pytest.raises(ValueError):
        nt_xent(torch.randn(4, 8), torch.randn(4, 8), temperature=0.0)


def test_tiny_overfit_loss_decreases():
    torch.manual_seed(0)
    net = M3Net(channels=(8, 16), embed_dim=32).train()
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    base = torch.randn(8, 1, 128, 64)
    losses = []
    for _ in range(25):
        x1 = base + 0.3 * torch.randn_like(base)
        x2 = base + 0.3 * torch.randn_like(base)
        loss, _ = nt_xent(net(x1), net(x2), 0.1)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8
