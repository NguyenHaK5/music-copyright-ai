import numpy as np
import pytest

from src.common.config import load_config
from src.common.seed import set_seed


def test_load_config_data():
    cfg = load_config("data")
    assert cfg["dataset"]["subset"] == "small"
    assert cfg["seed"] == 42


def test_load_config_missing():
    with pytest.raises(FileNotFoundError):
        load_config("khong_ton_tai")


def test_set_seed_reproducible():
    set_seed(7)
    a = np.random.rand(3)
    set_seed(7)
    b = np.random.rand(3)
    assert np.allclose(a, b)
