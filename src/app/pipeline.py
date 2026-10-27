"""Lõi demo (Phase 5): kiểm tra file -> log-mel -> M3 -> ngưỡng -> nhãn 3 trạng thái -> luật giấy phép -> kết quả.

Không import streamlit: kiểm thử được độc lập. Nhãn: THUỘC_DANH_MỤC / NGOÀI_DANH_MỤC / KHÔNG_XÁC_ĐỊNH (+ mã lỗi đầu vào).
Không bao giờ xuất "hợp lệ", "không có bản quyền", "được phép dùng". Không lưu audio người dùng (xóa tệp tạm ngay).
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.data.clean import check_audio
from src.features.logmel import logmel, peak_normalize
from src.rights.license_rules import STATUS_IN, describe

IN, OUT, UNSURE = "THUỘC_DANH_MỤC", "NGOÀI_DANH_MỤC", "KHÔNG_XÁC_ĐỊNH"
assert IN == STATUS_IN
ERROR_MESSAGES = {
    "too_large": "Tệp quá lớn (giới hạn {max_mb} MB).",
    "bad_ext": "Định dạng không được hỗ trợ (chỉ nhận: {exts}).",
    "empty": "Tệp rỗng.",
    "missing_file": "Không đọc được tệp tải lên.",
    "decode_error": "Không giải mã được âm thanh: tệp hỏng hoặc không phải audio hợp lệ.",
    "too_short": "Đoạn âm thanh quá ngắn (cần tối thiểu {min_s:.0f} giây).",
    "silent": "Đoạn âm thanh gần như im lặng, không đủ thông tin để nhận diện.",
    "nonfinite": "Dữ liệu âm thanh chứa giá trị không hợp lệ.",
}


@dataclass
class DemoConfig:
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    max_mb: float = 20.0
    allowed_ext: tuple = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
    min_seconds: float = 3.0
    query_seconds: float = 10.0
    silence_peak: float = 1e-4
    top_k: int = 3


@dataclass
class Engine:
    retriever: object                 # M3Retriever (đã có chỉ mục)
    cfg: DemoConfig
    rules: dict                       # configs/license_rules.yaml
    tau_high: float
    tau_low: float
    meta: dict = field(default_factory=dict)       # track_id -> {title, artist, genre, license_raw}
    info: dict = field(default_factory=dict)


def validate_upload(name: str, size_bytes: int, cfg: DemoConfig) -> str | None:
    """Trả mã lỗi hoặc None. Kiểm tra rẻ trước khi giải mã; loại file thật được kiểm tra khi giải mã."""
    if size_bytes <= 0:
        return "empty"
    if size_bytes > cfg.max_mb * 1024 * 1024:
        return "too_large"
    if Path(name).suffix.lower() not in cfg.allowed_ext:
        return "bad_ext"
    return None


def error_text(code: str, cfg: DemoConfig) -> str:
    return ERROR_MESSAGES.get(code, "Lỗi đầu vào không xác định.").format(
        max_mb=int(cfg.max_mb), exts=", ".join(cfg.allowed_ext), min_s=cfg.min_seconds)


def decide(score: float, tau_low: float, tau_high: float) -> str:
    if score >= tau_high:
        return IN
    if score < tau_low:
        return OUT
    return UNSURE


def crop_query(y: np.ndarray, cfg: DemoConfig) -> np.ndarray:
    """Tối đa query_seconds: file dài hơn lấy đoạn GIỮA (độ dài dài nhất đã đánh giá là 10 s)."""
    n = int(cfg.query_seconds * cfg.sr)
    if len(y) <= n:
        return y
    s = (len(y) - n) // 2
    return y[s:s + n]


def prepare_audio(path, cfg: DemoConfig):
    """-> (error_code | None, y | None, duration_s). Giải mã thật bằng check_audio (đúng bộ kiểm tra của pipeline dữ liệu)."""
    code, dur, y = check_audio(path, cfg.sr, cfg.min_seconds, 600.0, cfg.silence_peak)
    return code, y, dur


def analyze_audio(y: np.ndarray, engine: Engine) -> dict:
    """y: mono float32 ở cfg.sr đã qua kiểm tra. Trả kết quả đầy đủ (không có trường lỗi)."""
    cfg = engine.cfg
    t0 = time.perf_counter()
    q = crop_query(y, cfg)
    mel = logmel(peak_normalize(q), cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels)
    t1 = time.perf_counter()
    scores = engine.retriever.score_matrix([mel])[0]
    t2 = time.perf_counter()
    order = np.argsort(-scores)[:cfg.top_k]
    ids = engine.retriever.track_ids
    top = []
    for r, j in enumerate(order, 1):
        tid = int(ids[j])
        m = engine.meta.get(tid, {})
        top.append({"rank": r, "track_id": tid, "score": float(scores[j]), "title": m.get("title"), "artist": m.get("artist"),
                    "genre": m.get("genre"), "license_raw": m.get("license_raw")})
    label = decide(top[0]["score"], engine.tau_low, engine.tau_high)
    rights = describe(top[0]["license_raw"], label, engine.rules)      # chỉ khi THUỘC_DANH_MỤC; ngược lại None
    return {"ok": True, "label": label, "top": top, "tau_high": engine.tau_high, "tau_low": engine.tau_low,
            "analyzed_seconds": len(q) / cfg.sr, "rights": rights, "disclaimer": engine.rules["disclaimer"],
            "timing_ms": {"logmel": (t1 - t0) * 1000, "match": (t2 - t1) * 1000}}


def analyze_file(path, name: str, size_bytes: int, engine: Engine) -> dict:
    """Điểm vào đầy đủ cho một tệp đã nằm trên đĩa: kiểm tra -> giải mã -> phân tích. Lỗi trả {'ok': False, 'error': mã, 'message': ...}."""
    cfg = engine.cfg
    err = validate_upload(name, size_bytes, cfg)
    if err:
        return {"ok": False, "error": err, "message": error_text(err, cfg), "disclaimer": engine.rules["disclaimer"]}
    t0 = time.perf_counter()
    code, y, dur = prepare_audio(path, cfg)
    decode_ms = (time.perf_counter() - t0) * 1000
    if code:
        return {"ok": False, "error": code, "message": error_text(code, cfg), "disclaimer": engine.rules["disclaimer"]}
    res = analyze_audio(y, engine)
    res["timing_ms"]["decode"] = decode_ms
    res["duration_s"] = dur
    return res


def analyze_bytes(data: bytes, name: str, engine: Engine) -> dict:
    """Cho giao diện: ghi tệp tạm (đuôi lấy từ danh sách cho phép), phân tích, XÓA tệp tạm trong finally; không lưu audio."""
    err = validate_upload(name, len(data), engine.cfg)
    if err:
        return {"ok": False, "error": err, "message": error_text(err, engine.cfg), "disclaimer": engine.rules["disclaimer"]}
    suffix = Path(name).suffix.lower()
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return analyze_file(tmp, name, len(data), engine)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
