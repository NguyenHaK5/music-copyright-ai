"""Tải fma_small.zip, kiểm SHA1, giải nén vào data/raw/.

Chạy:  python -m src.data.fetch_audio [--no-download] [--no-extract]
Nếu mạng chặn: tải thủ công về data/raw/fma_small.zip rồi chạy với --no-download.
Cần ~7 GB cho zip + ~7 GB giải nén (xóa zip sau khi xác minh nếu thiếu ổ).
"""
from __future__ import annotations

import argparse
import zipfile

from src.common.config import load_config
from src.common.log import get_logger
from src.common.paths import resolve
from src.data.download import download_file, verify_sha1

log = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    cfg = load_config("data")
    raw = resolve(cfg["paths"]["raw_dir"])
    zip_path = raw / "fma_small.zip"
    if not args.no_download:
        download_file(cfg["dataset"]["audio_url"], zip_path)
    if not zip_path.exists():
        log.error("CHƯA CHẠY: không có %s", zip_path)
        return 2
    if not verify_sha1(zip_path, cfg["dataset"]["audio_sha1"]):
        log.error("SHA1 sai - dừng, không giải nén.")
        return 3
    if not args.no_extract:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw)
        log.info("Đã giải nén vào %s", raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
