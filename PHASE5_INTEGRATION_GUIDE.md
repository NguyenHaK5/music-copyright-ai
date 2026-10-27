# PHASE 5 - INTEGRATION GUIDE (Demo Streamlit)
Cập nhật: 2026-10-30 · **Phase 5 ĐANG LÀM**: code + test xong (13 test mới), CHƯA chạy trên dữ liệu thật (hiệu chuẩn ngưỡng, tệp mẫu, smoke test, chạy giao diện).

## 1. Kế hoạch chia phần
| Phần | Nội dung | Trạng thái |
|---|---|---|
| 5A | Lõi demo: kiểm tra tệp, log-mel, M3, ngưỡng, nhãn 3 trạng thái, luật giấy phép; dựng engine, lưu chỉ mục; hiệu chuẩn ngưỡng từ Val | Code + test xong |
| 5B | Giao diện Streamlit: Upload -> Kết quả -> Bằng chứng -> Disclaimer | Code + test xong (khởi động và hiển thị kiểm bằng `AppTest`) |
| 5C | Tệp mẫu, smoke test ba nhánh + ca lỗi, ảnh chụp cho báo cáo | Code + test xong; chờ chạy |

## 2. Quyết định thiết kế (bám Phase 0 mục 11 và mục 15 prompt)
- **Mô hình demo:** cấu hình cuối `noise_coherent_k2`, **seed 43** = seed tốt nhất trên VAL (macro R@1 0,9636; nhận đúng@FMR 0,8992; seed 42: 0,9580/0,8754) theo đúng nguyên tắc Phase 0 "mô hình tốt nhất trên Val", không chọn theo Test. Đổi bằng `model.ckpt` trong `configs/demo.yaml`.
- **Chỉ mục demo:** 800 bài known của Test (Phase 0). Demo không phải phép đo nên dùng Test để minh họa là hợp lệ.
- **Ngưỡng (từ VAL, không dùng Test):** `tau_high` = tau Val đã dùng ở Phase 4 (FMR gộp <= 1%); `tau_low` = phân vị 5% điểm top-1 của truy vấn known được nhận đúng trên Val (Phase 0 mục 9.2). Điểm top-1 >= tau_high -> `THUỘC_DANH_MỤC`; < tau_low -> `NGOÀI_DANH_MỤC`; ở giữa -> `KHÔNG_XÁC_ĐỊNH`.
- **Đầu vào:** tối đa 20 MB; đuôi cho phép mp3/wav/flac/ogg/m4a; tối thiểu 3 s; chỉ phân tích tối đa 10 s (đoạn giữa nếu dài hơn) vì 3-10 s là dải độ dài đã đánh giá; kiểm tra loại tệp thật bằng giải mã thử (`check_audio`), không chỉ nhìn đuôi.
- **Giấy phép (T2):** chỉ hiện khi nhãn là `THUỘC_DANH_MỤC`; cảnh báo nổi bật khi giấy phép có NC ("nhận diện đúng bài KHÔNG có nghĩa là được dùng tự do": khoảng 82% bài trong danh mục có NC).
- **Riêng tư/bảo mật:** tệp tạm xóa trong `finally` (kể cả khi lỗi); không lưu audio; không có khóa/tài khoản; không dịch vụ ngoài; disclaimer hiện ở đầu và cuối trang và trong mọi kết quả; không dùng từ "hợp lệ", "không có bản quyền", "được phép dùng" (có test).

## 3. Những gì đã có
| File | Vai trò |
|---|---|
| `configs/demo.yaml`, `.streamlit/config.toml` | Cấu hình demo (checkpoint, chỉ mục, giới hạn tải lên) và giới hạn upload 20 MB của Streamlit |
| `src/app/pipeline.py` | `validate_upload`, `prepare_audio`, `analyze_audio`, `analyze_file`, `analyze_bytes`, `decide`, `crop_query`; lỗi trả `{ok: False, error, message}` |
| `src/app/engine.py` | `build_engine()` dựng M3 + chỉ mục (nạp từ `data/interim/demo_index.npz` nếu khớp SHA256 checkpoint, ngược lại dựng và lưu), nạp metadata (tên bài/nghệ sĩ nếu có) và ngưỡng |
| `src/app/calibrate_demo.py` | Sinh `experiments/results/e5_demo_calibration.json` từ Val (chấm lại 6.800 truy vấn robustness Val) |
| `src/app/render.py` | Chuỗi hiển thị thuần Python (nhãn, giải thích, giấy phép, bảng top-3) |
| `src/app/streamlit_app.py` | Giao diện; thiếu checkpoint/hiệu chuẩn thì báo lỗi rõ, không crash |
| `src/app/make_demo_samples.py`, `src/app/smoke_demo.py` | Tệp mẫu (THUỘC, nhiễu SNR 10 dB, NGOÀI, quá ngắn, im lặng, hỏng, rỗng) và smoke test so với kỳ vọng |
| `src/models/m3_retrieval.py` | Thêm `export_index`/`import_index` (nạp chỉ mục nhanh) |
| `tests/test_app_pipeline.py` | 13 test |

## 4. Cách chạy (PowerShell; thứ tự quan trọng)
```
pip install streamlit        # rồi ghim phiên bản vào requirements.txt (pip show streamlit); soundfile đã có trong danh sách phụ thuộc
pytest -q                    # local mong đợi 195 passed (182 + 13); nếu chưa cài streamlit: 193 passed, 2 skipped
python -m src.app.calibrate_demo       # vài phút CPU: dựng chỉ mục Val (khoảng 70-80 s trên máy trước) rồi chấm 6.800 truy vấn
python -m src.app.make_demo_samples    # cần audio FMA (như khi dựng bộ truy vấn robustness)
python -m src.app.smoke_demo
streamlit run src/app/streamlit_app.py
```
Kết quả mong đợi: `calibrate_demo` in `tau_high`, `tau_low` và, cho truy vấn known-đúng và unknown của Val, tỷ lệ >= tau_high, trong vùng không xác định, < tau_low (mình KHÔNG dự đoán số); `smoke_demo` in PASS/FAIL từng tệp rồi `SMOKE ĐẠT`; lần mở ứng dụng đầu mất khoảng 1 phút để dựng chỉ mục (800 bài) rồi lưu cache, các lần sau nhanh.
Kỳ vọng smoke: tệp known -> `THUỘC_DANH_MỤC`; known nhiễu 10 dB -> THUỘC hoặc KHÔNG_XÁC_ĐỊNH; unknown -> NGOÀI hoặc KHÔNG_XÁC_ĐỊNH (không được THUỘC); `too_short`, `silent`, `empty` -> đúng mã lỗi; `broken.mp3` -> `decode_error` (hoặc `too_short` nếu bộ giải mã không ném lỗi).
Dán lại: log `calibrate_demo`, log `smoke_demo` và `e5_demo_calibration.json`; chụp ảnh giao diện cho báo cáo: (1) ca THUỘC (có khung giấy phép), (2) ca NGOÀI hoặc KHÔNG_XÁC_ĐỊNH, (3) ca lỗi (tệp hỏng/quá ngắn), (4) phần "Thông tin hệ thống". Không để lộ đường dẫn cá nhân hay tên người dùng trong ảnh.

## 5. Cách đọc kết quả hiệu chuẩn (quyết định trước khi chạy)
- Cần: `known_correct.frac_ge_tau_high` đủ lớn (Val: nhóm truy vấn khó như nhiễu 5 dB và overlay kéo xuống) và `unknown.frac_ge_tau_high` gần 1% (đúng FMR gộp Val).
- **Rủi ro thiết kế cần theo dõi:** `tau_low` là phân vị 5% của các truy vấn known khó nên có thể thấp; khi đó bài lạ sạch có điểm giữa `tau_low` và `tau_high` sẽ hiện "KHÔNG_XÁC_ĐỊNH" thay vì "NGOÀI_DANH_MỤC". Nếu `unknown.frac_lt_tau_low` thấp (ví dụ dưới một nửa), báo mình trước khi đổi gì: đây là ngưỡng chọn trên Val, đổi quy tắc phải được ghi lại như một quyết định thiết kế, và không được chỉnh theo kết quả Test.

## 6. Rủi ro 5
- Chưa chạy `calibrate_demo`, `make_demo_samples`, `smoke_demo` và giao diện trên dữ liệu/checkpoint thật; test dùng mô hình nhỏ ngẫu nhiên, âm thanh tổng hợp và `AppTest` (khởi động và hiển thị; chưa mô phỏng thao tác tải tệp).
- Phiên bản Streamlit/torch: máy bạn dùng torch 2.2.2 CPU; sandbox mình dùng Streamlit 1.66. Nếu `streamlit run` báo lỗi API (ví dụ tham số hiển thị đã đổi), dán nguyên văn.
- Độ trễ đầu-cuối chưa đo (Phase 4 chỉ đo so khớp 13 ms); giao diện hiển thị thời gian giải mã, log-mel, so khớp thực tế. File mp3 dài giải mã có thể chậm hơn truy vấn đã là log-mel.
- Tệp ngoài miền (nhạc thương mại, live, cover) nằm ngoài những gì đã đánh giá: mô hình có thể cho điểm thấp-trung bình; đó là hạn chế đã nêu trong báo cáo.
- Metadata tên bài/nghệ sĩ lấy từ `data/interim/tracks_clean.parquet` nếu có; nếu thiếu cột thì hiện "track <id>" (không bịa).
- Còn lại: chạy 5C, chốt Phase 5 (tag `phase5-final`), rồi Phase 6 (báo cáo, slide, README, bộ câu hỏi hội đồng).
