"""Đường dẫn gốc của dự án (chạy được trên Windows/Linux/Colab)."""
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"


def resolve(rel: str | Path) -> Path:
    """Chuyển đường dẫn tương đối (so với gốc dự án) thành tuyệt đối."""
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p
