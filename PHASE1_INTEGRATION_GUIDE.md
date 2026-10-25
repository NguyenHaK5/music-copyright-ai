# PHASE 1 - INTEGRATION GUIDE (Data Foundation)
Cập nhật: 2026-10-20 · **Phase 1 XONG**: 1A-1D đã chạy trên FMA small thật (mục 11). Bản chốt.

## 1. Phần 1A: những gì đã có
| File | Vai trò | Yêu cầu GV |
|---|---|---|
| `requirements.txt` | Ghim phiên bản (Python ≥ 3.10) | 4, tái lập |
| `configs/data.yaml` | Nguồn FMA, đường dẫn, ngưỡng smoke gate | 1 |
| `configs/features.yaml` | Tham số log-mel khởi đầu (22.050 Hz, 128 mel, n_fft 2048, hop 512) | 1 |
| `configs/targets.yaml` | Khung mục tiêu; **TBD** đến sau baseline Val | 3 |
| `src/common/{paths,config,seed,log}.py` | Hạ tầng dùng chung | 4 |
| `src/data/download.py` | Tải có resume + `sha1_file`/`verify_sha1` | 1 |
| `src/data/license_norm.py` | URL CC -> họ giấy phép (`by`, `by-nc-sa`, `cc0`, `unknown`) | 1 |
| `src/data/smoke_check.py` | Đọc `tracks.csv`, tính 5 cổng, ghi `experiments/results/e0_smoke_check.json` | 1 |
| `tests/test_*.py` | 16 test (config, seed, license, gates, loader, sha1) | 4 |

## 2. Điểm nối (API nội bộ)
- `load_config(name) -> dict`; `set_seed(seed)`; `get_logger(name)`; `resolve(rel_path) -> Path`.
- `load_fma_tracks(path, subset) -> DataFrame` cột: `artist_id, artist_name, album_id, title, genre, license_raw, duration, subset, license_family`, index `track_id`. **1C và 1D dùng lại hàm này.**
- `evaluate_gates(df, gates_cfg, free_disk_gb) -> list[dict]` (hàm thuần, đã test).

## 3. Cách chạy (Windows PowerShell)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```
Kết quả mong đợi: `16 passed`. (Trong sandbox tôi chạy với phiên bản mới nhất có sẵn, không phải đúng bản ghim; nếu `pip install` báo xung đột phiên bản trên máy bạn, dán lỗi lên.)

Tải metadata (thủ công hoặc dùng `download_file`): lấy `fma_metadata.zip` theo `configs/data.yaml`, giải nén sao cho có `data/raw/fma_metadata/tracks.csv`, rồi:
```
python -m src.data.smoke_check
```
Kết quả mong đợi: log PASS/FAIL cho 5 cổng và file `e0_smoke_check.json`. Chưa có `tracks.csv` thì lệnh thoát mã 2 với thông báo "CHƯA CHẠY" (đã kiểm).

## 4. Cách đọc kết quả smoke check
| Cổng FAIL | Hành động |
|---|---|
| `artist_id_missing_rate` | Nhóm theo `artist_name`, ghi Data Card |
| `n_artists` | Thu nhỏ known/unknown hoặc thêm nguồn |
| `min_artists_per_genre` | Gộp/bỏ thể loại ít nghệ sĩ |
| `license_families_ge_min_tracks` | Bỏ T3 (dự án vẫn tiếp tục) |
| `free_disk_gb` | Chỉ cache crop hoặc dọn ổ |

## 5. Đóng góp cho báo cáo
Chương 2 (Dữ liệu): mục nguồn dữ liệu và Data Card dùng số trong `e0_smoke_check.json`. Chưa có số nào để đưa vào báo cáo trước khi bạn chạy.

## 6. Rủi ro và việc còn lại
- Chưa kiểm cấu trúc `tracks.csv` thật (header 2 tầng, giá trị `subset == 'small'`): loader sẽ báo `KeyError` rõ ràng nếu khác.
- URL/SHA1 FMA chưa xác minh.
- Cả Phase 1 chưa chạy trên dữ liệu thật.

## 7. Phần 1B: tải audio, làm sạch, cache log-mel
| File | Vai trò |
|---|---|
| `src/data/fetch_audio.py` | Tải `fma_small.zip` (resume), kiểm SHA1, giải nén vào `data/raw/` |
| `src/data/clean.py` | `fma_audio_path(audio_dir, track_id)`; `check_audio(...) -> (drop_reason, duration, y)` |
| `src/features/logmel.py` | `load_audio`, `peak_normalize`, `logmel` (dB, ref=1.0) |
| `src/features/cache.py` | `save_logmel`/`load_logmel`/`has_logmel`, float16, ghi nguyên tử |
| `src/data/build_clean_cache.py` | Chạy toàn bộ; ghi `data/interim/clean_report.csv`; chạy lại bỏ qua file đã xử lý |
| `configs/features.yaml` | Thêm `clean:` (min 20 s, max 30 s, ngưỡng im lặng, checkpoint) và `paths:` |

**Điểm nối cho 1C/1D:** `clean_report.csv` cột `track_id, file_ok, drop_reason, duration_s`; cache `data/interim/logmel/<track_id 6 chữ số>.npy` shape (128, ~1292) cho clip 30 s. 1C dùng cache để tính chữ ký phổ trung bình; 1D lọc theo `file_ok` rồi mới chia tập. `drop_reason` ∈ missing_file, decode_error, too_short, silent, nonfinite.

**Cách chạy (PowerShell):**
```
pytest -q                                   # mong đợi: 21 passed
python -m src.data.fetch_audio              # ~7 GB tải + ~7 GB giải nén; hoặc đặt zip thủ công và thêm --no-download
python -m src.data.build_clean_cache --limit 50   # chạy thử
python -m src.data.build_clean_cache        # chạy đủ (có thể ngắt rồi chạy lại)
```
Kết quả mong đợi: log "Tổng N file; đạt M" và bảng lý do loại. Ước tính cache ≈ 2,6 GB cho 8.000 clip (ước lượng từ kích thước, chưa đo). Thiếu `tracks.csv` hoặc thư mục audio thì lệnh thoát mã 2 "CHƯA CHẠY".

**Rủi ro 1B:** test dùng audio tổng hợp (wav); chưa thử giải mã .mp3 FMA thật. Nếu `decode_error` xuất hiện hàng loạt trên Windows, nguyên nhân thường là thiếu hỗ trợ mp3 (cần soundfile/libsndfile đủ mới hoặc ffmpeg trong PATH) - dán log để xử lý. Ngưỡng `clean:` là đề xuất.

**Đóng góp báo cáo:** Chương 2, mục tiền xử lý: bảng tham số (22.050 Hz, mono, peak norm, log-mel 128/2048/512) và thống kê file loại theo `drop_reason` lấy từ `clean_report.csv` (chưa có số).

## 8. Phần 1C: khử trùng, EDA, Data Card
| File | Vai trò |
|---|---|
| `src/data/dedup.py` | SHA1 file -> trùng chính xác (giữ `track_id` nhỏ nhất, loại bản còn lại, `drop_reason='duplicate_exact'`). Chữ ký phổ (mean log-mel, trừ trung bình toàn tập, L2) -> cặp nghi trùng gần (cosine >= `dedup.near_dup_cosine`) -> union-find thành `dup_group`. Nghi trùng gần chỉ GẮN CỜ, không xóa |
| `src/data/eda.py` | `eda_summary`, `make_figures`; ghi `experiments/results/e0_eda.json`, `reports/figures/eda_*.png`, `reports/tables/eda_*.csv` |
| `src/data/data_card.py` | Sinh `reports/report/data_card.md` từ hai file JSON; thiếu số thì ghi "CHƯA CHẠY" |
| `notebooks/01_eda.ipynb` | Gọi hàm EDA, xem histogram `nn_cosine`, ô nhận xét để điền tay |

**Điểm nối cho 1D (quan trọng):** `data/interim/tracks_clean.parquet` cột gồm metadata + `file_ok` (cuối cùng, đã tính cả trùng chính xác), `drop_reason`, `duration_s`, `file_sha1`, `exact_dup_of`, `dup_group`, `nn_cosine`. 1D chỉ chia các dòng `file_ok == True`, chia theo `artist_id`, và phải giữ mọi `dup_group` trong cùng một tập. `file_sha1` là cơ sở cho kiểm tra leakage theo `fingerprint_hash` ở test.

**Cách chạy (sau khi 1B chạy đủ):**
```
pytest -q                          # mong đợi: 26 passed
python -m src.data.dedup           # ghi tracks_clean.parquet + near_dup_pairs.csv
python -m src.data.eda             # ghi e0_eda.json, hình, bảng
python -m src.data.data_card       # ghi data_card.md
jupyter notebook notebooks/01_eda.ipynb   # cần cài thêm jupyter (không nằm trong requirements)
```
Thiếu đầu vào thì mỗi lệnh thoát mã 2 với "CHƯA CHẠY" (đã kiểm với dữ liệu trống).

**Rủi ro 1C:**
- Chữ ký phổ trung bình là đặc trưng thô: có thể báo nhầm hai bài khác nhau cùng phong cách và bỏ sót bản trùng bị đổi mạnh (codec, pitch). Vì thế chỉ gắn cờ; hãy kiểm tay các cặp cosine cao nhất, nhất là cặp khác nghệ sĩ. Ngưỡng 0,995 chưa hiệu chỉnh.
- Cùng bản thu đăng dưới hai nghệ sĩ khác nhau vẫn có thể lọt qua chia theo nghệ sĩ nếu chữ ký phổ không bắt được; `dup_group` giảm rủi ro này, không loại bỏ hoàn toàn.
- `nn_cosine` tính O(n²) theo khối; với 8.000 bài là vài giây đến vài chục giây (ước lượng, chưa đo).
- Notebook chưa được thực thi (không có dữ liệu).

**Đóng góp báo cáo:** Chương 2 (Dữ liệu): Data Card, bảng/hình EDA, thống kê file loại và nghi trùng. Chưa có số liệu để trích.

## 9. Phần 1D: chia tập, kiểm tra rò rỉ, `tracks.parquet`
| File | Vai trò |
|---|---|
| `src/data/split.py` | Chỉ lấy `file_ok`. Đơn vị chia = thành phần liên thông theo (cùng nghệ sĩ) hoặc (cùng `dup_group`), nên nhóm trùng chéo nghệ sĩ không bị tách. Phân tầng theo thể loại chủ đạo của đơn vị; chia tham lam theo số bài (đơn vị lớn trước). Train/Val/Test = 0,6/0,2/0,2; trong Val/Test chia tiếp known/unknown 0,5/0,5 theo đơn vị. Xác định theo seed. Ghi `data/processed/tracks.parquet` CHỈ khi check pass |
| `src/data/leakage_check.py` | `check_split(df)` / `assert_no_leakage(df)`: kiểm không giao nhau theo `artist_key`, `album_id`, `fingerprint_hash` (SHA1 file), `dup_group` giữa các split; known/unknown/train không chung nghệ sĩ hay nhóm; chỉ `file_ok` nằm trong split; nhãn split-role nhất quán |
| `tests/test_split_leakage.py` | 6 test dữ liệu tổng hợp (tỷ lệ, nhóm trùng chéo nghệ sĩ, xác định, unknown tách biệt, phát hiện 5 loại vi phạm cố ý) + 1 test dữ liệu thật (SKIP nếu chưa có `tracks.parquet`) |
| `configs/data.yaml` | Khối `split:` |

**Schema `data/processed/tracks.parquet`:** `track_id, artist_id, artist_key, album_id, genre, license_raw, license_family, duration, fingerprint_hash, file_ok, drop_reason, dup_group, split, role, source`. `split` ∈ train/val/test/excluded; `role` ∈ train/known/unknown/excluded; `source` = FMA. **Đây là điểm nối cho Phase 2:** chỉ mục Val dựng từ `split==val & role==known`, chỉ mục Test từ `split==test & role==known`, truy vấn unknown từ `role==unknown`; M1/M3 chỉ huấn luyện trên `split==train`. Cache log-mel đọc qua `track_id` (mục 7).

**Quy ước:** `artist_key` = `id:<artist_id>`, nếu thiếu thì `name:<tên>`, nếu thiếu nữa thì `track:<id>`. Tập Val/Test 800+800 chỉ là xấp xỉ; số bài thật phụ thuộc kết quả 1B/1C.

## 10. Thứ tự chạy toàn bộ Phase 1 (PowerShell)
```
pytest -q                                   # 31 passed, 1 skipped (trước khi có dữ liệu)
python -m src.data.smoke_check              # 1A  -> e0_smoke_check.json
python -m src.data.fetch_audio              # 1B
python -m src.data.build_clean_cache        # 1B  -> clean_report.csv, cache log-mel
python -m src.data.dedup                    # 1C  -> tracks_clean.parquet
python -m src.data.eda                      # 1C
python -m src.data.data_card                # 1C
python -m src.data.split                    # 1D  -> tracks.parquet, e0_split.json
python -m src.data.leakage_check            # 1D  -> e0_leakage.json
pytest tests/test_split_leakage.py -rs      # test cuối phải PASSED, không SKIPPED
```
**Điều kiện hoàn thành Phase 1:** dòng cuối pass không skip, và notebook EDA chạy được. Đến lúc đó mới chuyển `phase-tracker.md` sang "XONG".

**Rủi ro 1D:**
- Chia theo số bài nên tỷ lệ chỉ xấp xỉ; nghệ sĩ có rất nhiều bài làm lệch tỷ lệ hoặc thể loại giữa các tập. Xem `genre_by_split` trong `e0_split.json`; nếu lệch nhiều thì báo để chỉnh.
- Unknown "cùng thể loại với known" chỉ được đảm bảo ở mức phân tầng, chưa có kiểm tra riêng.
- Rò rỉ mà chữ ký phổ và SHA1 không bắt được (cùng bài đăng dưới nghệ sĩ khác, bị đổi mã hóa mạnh) vẫn có thể còn; nêu trong phần hạn chế của báo cáo.
- Chia theo nghệ sĩ không loại bỏ ảnh hưởng của thể loại/phong cách chung.

**Đóng góp báo cáo:** Chương 2: giao thức chia tập, bảng đếm theo split/role từ `e0_split.json`, kết quả `e0_leakage.json`. Chưa có số liệu để trích.

## 11. Kết quả chạy thật (chốt Phase 1)
Nguồn: `experiments/results/` do người dùng chạy và dán lại (2026-10-19).

| Kiểm tra | Kết quả |
|---|---|
| `pytest -q` | 32 passed, 0 skipped |
| `pytest tests/test_split_leakage.py -rs` | 6 passed (có test trên `tracks.parquet` thật) |
| Notebook `01_eda.ipynb` | Chạy được (cần `ipython<9.0`, `matplotlib-inline<0.2.0`) |
| Smoke check | 4/5 cổng PASS; FAIL: `license_families_ge_min_tracks` (value 0) |
| Clean | 7992/8000 giữ; too_short 3, decode_error 3, silent 2 |
| Dedup | 47 `dup_group` > 1; 77 cặp nghi trùng gần (60 cùng nghệ sĩ) |
| Split | train 4794; val 1601 (802 known / 799 unknown); test 1597 (800 / 797) |
| Leakage | passed, violations: [] |

**Ba phát hiện khi chạy dữ liệu thật**
1. `configs/data.yaml`: tên miền tải sai, đã sửa; SHA1 `fma_metadata.zip` và `fma_small.zip` đã điền.
2. `license_norm.py`: cột license của `tracks.csv` là tên văn bản, không phải URL, nên cả 8000 bài thành `unknown`. Hệ quả: T3 mất cơ sở; T2 cần sửa bộ chuẩn hóa trước khi dùng. CHƯA sửa.
3. `split.py`: thêm `album_id` vào các cột nối đơn vị chia (album tổng hợp nhiều nghệ sĩ gây trùng album giữa các tập). Sau sửa: không còn vi phạm.

**Điểm nối cho Phase 2 (không đổi):** `data/processed/tracks.parquet` (schema mục 9) và cache log-mel `data/interim/logmel/<track_id 6 chữ số>.npy` (mục 7). Chỉ mục Val: `split==val & role==known`; chỉ mục Test: `split==test & role==known`; unknown: `role==unknown`; M1/M3 chỉ huấn luyện trên `split==train`.

**Hạn chế còn lại (đưa vào báo cáo):** ngưỡng nghi trùng 0,995 chưa kiểm tay; chữ ký phổ trung bình là đặc trưng thô (median `nn_cosine` ≈ 0,926 nên nhiều bài khác nhau vẫn khá giống nhau theo chữ ký này); rò rỉ do cùng bản thu đăng dưới nghệ sĩ khác hoặc bị mã hóa lại mạnh có thể còn sót; chia theo nghệ sĩ không loại bỏ ảnh hưởng của phong cách chung.
