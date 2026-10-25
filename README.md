# music-copyright-ai
Đồ án ML: nhận diện open-set trong danh mục tham chiếu (THUỘC_DANH_MỤC / NGOÀI_DANH_MỤC / KHÔNG_XÁC_ĐỊNH) + lớp luật giấy phép.
Đây là phân loại kỹ thuật, KHÔNG phải kết luận pháp lý.

## Cài đặt (Windows, PowerShell)
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    # ffmpeg cần cho Phase 4 (biến đổi codec): cài và thêm vào PATH

## Chạy test
    pytest -q

## Phase 1A - smoke check metadata
    python -m src.data.smoke_check

## Phase 1B - tải audio, làm sạch, cache log-mel
    python -m src.data.fetch_audio
    python -m src.data.build_clean_cache --limit 50

## Phase 1C - khử trùng, EDA, Data Card
    python -m src.data.dedup
    python -m src.data.eda
    python -m src.data.data_card

## Phase 1D - chia tập + kiểm tra rò rỉ
    python -m src.data.split
    python -m src.data.leakage_check
    pytest tests/test_split_leakage.py -rs

Tiến độ: xem phase-tracker.md. Hướng dẫn tích hợp: PHASE1_INTEGRATION_GUIDE.md.
