"""Tải file lớn có thể chạy lại + kiểm SHA1.

Dùng cho metadata/audio FMA từ nguồn chính thức. Nếu mạng bị chặn, tải thủ công
rồi đặt file vào data/raw/ và chạy verify_sha1().
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

from src.common.log import get_logger

log = get_logger(__name__)
_CHUNK = 1 << 20  # 1 MiB


def sha1_file(path: str | Path) -> str:
    """SHA1 (hex) của một file, đọc theo khối."""
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def verify_sha1(path: str | Path, expected: str | None) -> bool:
    """True nếu khớp. expected=None -> cảnh báo và trả True (không kiểm được)."""
    if expected is None:
        log.warning("Chưa có SHA1 tham chiếu cho %s - bỏ qua kiểm tra.", path)
        return True
    actual = sha1_file(path)
    ok = actual.lower() == expected.lower()
    if not ok:
        log.error("SHA1 KHÔNG khớp %s: nhận %s, mong đợi %s", path, actual, expected)
    return ok


def download_file(url: str, dest: str | Path, timeout: int = 60) -> Path:
    """Tải url -> dest, tiếp tục từ phần đã tải nếu server hỗ trợ Range."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        if r.status_code == 416:  # đã tải đủ
            return dest
        r.raise_for_status()
        resuming = r.status_code == 206
        if done and not resuming:
            done = 0  # server không hỗ trợ Range -> tải lại từ đầu
        total = int(r.headers.get("Content-Length", 0)) + done
        with dest.open("ab" if resuming else "wb") as f, tqdm(
            total=total or None, initial=done, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for block in r.iter_content(_CHUNK):
                f.write(block)
                bar.update(len(block))
    return dest
