"""Đọc file cấu hình YAML trong configs/."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.common.paths import CONFIG_DIR


def load_config(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Đọc configs/<name>.yaml. Ném FileNotFoundError nếu thiếu."""
    path = (config_dir or CONFIG_DIR) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Cấu hình {path} phải là một mapping YAML")
    return data
