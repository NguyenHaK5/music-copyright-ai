# Music Copyright AI — Nhận diện nhạc thuộc danh mục tham chiếu (open-set)

Đồ án môn Học máy: **"Xây dựng mô hình AI phân loại nhạc bản quyền và không bản quyền"**.

Hệ thống nhận một đoạn audio 3–10 giây và trả về một trong ba nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| `THUỘC_DANH_MỤC` | Đoạn nhạc khớp một bài trong danh mục tham chiếu |
| `NGOÀI_DANH_MỤC` | Không bài nào trong danh mục đủ giống |
| `KHÔNG_XÁC_ĐỊNH` | Bằng chứng yếu hoặc file lỗi, cần người xem xét |

Nếu nhãn là `THUỘC_DANH_MỤC`, hệ thống tra thêm **giấy phép Creative Commons** của bản ghi khớp (lớp luật tường minh, tách khỏi mô hình AI).

> **Disclaimer.** "Danh mục tham chiếu" ở đây là một danh mục **mô phỏng có kiểm soát**, xây từ FMA small (nhạc giấy phép Creative Commons), không phải dữ liệu bản quyền thương mại. Kết quả là phân loại kỹ thuật, **không phải kết luận pháp lý**. Nhận diện đúng bài không có nghĩa là được dùng tự do: khoảng 82% bài trong danh mục có điều kiện phi thương mại (NC).

---

## 1. Tóm tắt kỹ thuật

| Mã | Mô hình | Vai trò |
|---|---|---|
| M0 | Điểm ngẫu nhiên | Mốc tối thiểu |
| M1 | MFCC / thống kê mel + kNN cosine | Baseline học máy cổ điển |
| M2 | Fingerprint tự cài (peak → cặp neo/đích → hash → bỏ phiếu theo độ lệch thời gian) | Baseline mạnh, tự xây dựng |
| **M3** | **ResNet nhỏ (569.520 tham số) → embedding 128-d chuẩn hóa L2, học tương phản NT-Xent** | **Mô hình học sâu chính, tự huấn luyện** |

Cấu hình M3 cuối: `noise_coherent_k2` (nhiễu "coherent" k=2), 3 seed (42, 43, 44).

### Kết quả trên Test (chạy một lần, ngưỡng cố định từ Val)

Nguồn: `experiments/results/e4_test_verdict.json`, `reports/tables/test_main_table.md`.

| Mô hình | macro Recall@1 | macro nhận đúng@FMR≤1% |
|---|---|---|
| M0 | 0,0000 | 0,0000 |
| M1 | 0,5939 | 0,0965 |
| M2 | 0,7883 | 0,7137 |
| **M3 (3 seed)** | **0,9635 ± 0,0037** | **0,8832 ± 0,0129** |

Theo quy tắc đặt trước (`configs/targets.yaml`), M3 **ĐẠT** cả chỉ số chính và phụ (hiệu M3−M1 = +0,3696, CI95 [+0,3495; +0,3883]).

**Hạn chế chính (đọc trước khi trích số liệu):**
- Truy vấn là crop của clip đã nằm trong chỉ mục (in-index) và các biến đổi (nhiễu, codec, pitch, tempo, chồng âm) là nhân tạo; không có cover/live thật.
- Test dùng 200 bài known + 200 bài unknown làm nguồn (lệch kế hoạch 400+400).
- Chỉ 4 nhóm biến đổi M3 thắng M2 rõ ràng; ở `overlay` (chồng âm) M3 0,787 so với M2 0,980.
- FMR trên Test chỉ gần mục tiêu 1% (M1 vượt 1,76%).
- Tham số nhiễu `k` được chọn bằng Val; độ bền nhiễu chỉ kết luận trên Test.
- Toàn bộ nhạc là Creative Commons, clip 30 s, nhạc quốc tế.

---

## 2. Yêu cầu hệ thống

| Mục | Yêu cầu |
|---|---|
| Python | 3.10 trở lên |
| ffmpeg | Bắt buộc trong `PATH` (tạo biến đổi codec MP3; kiểm tra: `ffmpeg -version`) |
| Ổ đĩa trống | ≥ 20 GB nếu chạy từ đầu (FMA small ~7 GB tải + ~7 GB giải nén + cache log-mel ~2,6 GB + bộ truy vấn robustness) |
| RAM | ≥ 8 GB (16 GB thoải mái) |
| GPU | Không bắt buộc để chạy demo/đánh giá. **Huấn luyện M3 nên dùng GPU** (Colab T4: ~19–30 phút/seed) |
| HĐH | Windows, Linux, Google Colab |

---

## 3. Cài đặt

```bash
git clone <<XÁC NHẬN: URL repo>>
cd <<XÁC NHẬN: tên thư mục>>

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Kiểm tra cài đặt (không cần dữ liệu):

```bash
pytest -q
```

Kết quả mong đợi khi chưa có dữ liệu: phần lớn test pass (test tổng hợp); test cần `tracks.parquet` thật sẽ `SKIP`. Khi đã có đủ dữ liệu và `streamlit`, mong đợi **195 passed**.

Ghi chú cài đặt đã gặp:
- Chạy notebook EDA cần `ipython<9.0` và `matplotlib-inline<0.2.0`; Jupyter không nằm trong `requirements.txt`: `pip install jupyter`.
- PyTorch: nếu `pip install` không đúng bản CPU/GPU, cài trước theo hướng dẫn của https://pytorch.org rồi mới `pip install -r requirements.txt`.
- Cảnh báo `libmpg123: dequantization failed` khi giải mã một vài file mp3 FMA là khung mp3 hỏng, không dừng chương trình.

---

## 4. Cấu trúc thư mục

```text
├── configs/          data.yaml, features.yaml, baseline.yaml, m3.yaml, m3_ablations.yaml,
│                     targets.yaml (đã khóa), license_rules.yaml, demo.yaml
├── data/             (không commit audio; xem mục 5)
│   ├── raw/          fma_metadata/, fma_small/
│   ├── interim/      logmel/ (cache), robust/ (bộ truy vấn), tracks_clean.parquet, ...
│   └── processed/    tracks.parquet (kết quả chia tập)
├── src/
│   ├── common/       config, seed, log, paths
│   ├── data/         tải, làm sạch, khử trùng, EDA, chia tập, kiểm tra rò rỉ, truy vấn robustness
│   ├── features/     log-mel, cache, augmentation, peaks (M2), đặc trưng M1
│   ├── models/       m0, m1, m2, m3_cnn, losses, m3_encoder, m3_retrieval
│   ├── evaluation/   metrics, openset, bootstrap, robustness, test_gate, test_verdict, ...
│   ├── rights/       license_rules (lớp luật giấy phép)
│   ├── app/          demo Streamlit
│   ├── run_baselines.py, run_robustness.py, train_m3.py
├── experiments/      results/ (JSON, NPZ kết quả), runs/ (checkpoint M3)
├── reports/          figures/, tables/, report/
├── notebooks/        01_eda.ipynb, 03_train_m3_colab.ipynb
├── tests/            pytest
├── PHASE*_INTEGRATION_GUIDE.md, phase-tracker.md   (nhật ký chi tiết từng phase)
└── requirements.txt
```

---

## 5. Hai cách sử dụng

Repo trên GitHub **không chứa** audio FMA (~7 GB), cache log-mel và (tuỳ cấu hình `.gitignore`) checkpoint. Chọn một trong hai cách:

### Cách A — Chạy lại toàn bộ từ đầu (tái lập đầy đủ)

Thực hiện lần lượt mục 6 → 9. Dữ liệu được tải từ nguồn gốc, mọi bước đều xác định theo seed.

### Cách B — Chỉ chạy demo / kiểm tra kết quả với checkpoint có sẵn

Cần các file sau (tải riêng, đặt đúng đường dẫn): `<<XÁC NHẬN: nơi lưu — GitHub Release / Google Drive / Git LFS>>`

| File | Đặt tại |
|---|---|
| Checkpoint M3 (3 seed) | `experiments/runs/m3_abl_noise_coherent_k2_seed{42,43,44}/best.pt` |
| `m3_norm.json` | `data/interim/m3_norm.json` |
| `tracks.parquet`, `tracks_clean.parquet` | `data/processed/`, `data/interim/` |
| Cache log-mel của các bài dùng trong chỉ mục demo (800 bài known của Test) | `data/interim/logmel/<track_id 6 chữ số>.npy` |
| `e5_demo_calibration.json` | `experiments/results/` |

Sau đó nhảy tới mục 9 (demo).

---

## 6. Phase 1 — Dữ liệu (FMA small)

```bash
python -m src.data.smoke_check           # kiểm tra metadata (cần data/raw/fma_metadata/tracks.csv)
python -m src.data.fetch_audio           # tải + kiểm SHA1 + giải nén fma_small (~7 GB)
python -m src.data.build_clean_cache     # làm sạch + cache log-mel (có thể ngắt rồi chạy lại)
python -m src.data.dedup                 # khử trùng -> data/interim/tracks_clean.parquet
python -m src.data.eda
python -m src.data.data_card             # -> reports/report/data_card.md
python -m src.data.split                 # -> data/processed/tracks.parquet
python -m src.data.leakage_check         # -> experiments/results/e0_leakage.json
pytest tests/test_split_leakage.py -rs   # phải PASSED, không SKIPPED
```

Metadata: tải `fma_metadata.zip`, giải nén để có `data/raw/fma_metadata/tracks.csv`. URL và SHA1 nằm trong `configs/data.yaml`. Nếu tải tự động chậm, tải thủ công rồi chạy `fetch_audio --no-download`.

Kết quả mong đợi (số thật đã đo): 7.992/8.000 file giữ lại; split train 4.794 / val 1.601 / test 1.597; leakage `violations: []`.

Giao thức chia tập: chia theo **nhóm nghệ sĩ** (cộng album và nhóm trùng), phân tầng thể loại; trong Val/Test chia tiếp known/unknown theo nghệ sĩ. Thứ tự: khử trùng → chia → cắt crop → augment.

---

## 7. Phase 2 — Baseline và bộ truy vấn robustness

```bash
python -m src.run_baselines --split val --models m0 m1 m2

# Bộ truy vấn biến đổi (17 cấu hình / 7 nhóm); cần ffmpeg
python -m src.run_robustness build --split val
python -m src.run_robustness eval  --split val --models m0 m1 m2
python -m src.evaluation.make_targets    # CHỈ chạy lần đầu; file targets.yaml đã khóa trong repo
```

> `configs/targets.yaml` đã được khóa trước khi chạm Test. **Không chạy lại / không sửa** nếu chỉ tái lập kết quả; `make_targets` sẽ từ chối khi file đã khóa.

Bảy nhóm biến đổi: cắt đoạn (3/5/10 s), codec MP3 (64/128 kbps), nhiễu (SNR 5/10/20 dB), EQ (lowpass/highpass), pitch (±1, ±2 bán cung), tempo (0,90/1,10), chồng âm (5 dB).

---

## 8. Phase 3 — Huấn luyện M3

Khuyến nghị dùng **Google Colab (GPU T4)** với `notebooks/03_train_m3_colab.ipynb` (có resume khi Colab ngắt kết nối). Lệnh tương đương:

```bash
python -m src.train_m3 --seed 42 --ablation noise_coherent_k2 --preload --resume \
    --out-dir experiments/runs/m3_abl_noise_coherent_k2_seed42
# lặp lại với --seed 43 và --seed 44 (đổi --out-dir tương ứng)
```

<<XÁC NHẬN: kiểm tra `python -m src.train_m3 --help` để chắc chắn tên cờ `--out-dir` và quy ước thư mục.>>

- Mỗi seed: 60 epoch, ~19–30 phút trên T4. Checkpoint `last.pt` mỗi epoch, `best.pt` chọn theo loss Val. Test không bị đọc khi huấn luyện.
- Đánh giá Val cho từng seed:

```bash
python -m src.run_robustness eval --split val --models m3 \
    --m3-ckpt experiments/runs/m3_abl_noise_coherent_k2_seed42/best.pt \
    --tag abl_noise_coherent_k2_seed42
python -m src.evaluation.summarize_m3          # -> reports/tables/m3_val_summary.md
python -m src.evaluation.freeze_phase3         # phải in "ĐÓNG BĂNG ĐẠT"
```

Ablation (một seed): `no_noise`, `no_pitch_tempo`, `gap`, `no_overlap` — định nghĩa trong `configs/m3_ablations.yaml`.

---

## 9. Phase 4 — Đánh giá trên Test (chỉ một lần)

Test được bảo vệ bởi `src/evaluation/test_gate.py`: yêu cầu `--final-test`, `freeze_phase3` đạt, các file cấu hình/checkpoint không đổi so với lúc đóng băng, và **không ghi đè** kết quả Test đã có.

Kết quả Test của dự án đã có sẵn trong `experiments/results/e5_robust_test_*.json`. Để xem lại kết luận mà không chạm Test:

```bash
python -m src.evaluation.test_verdict          # -> e4_test_verdict.json, reports/tables/test_main_table.md
python -m src.evaluation.error_analysis        # đọc kết quả Test có sẵn
python -m src.evaluation.measure_cost          # đo độ trễ / dung lượng chỉ mục (trên Val)
python -m src.rights.license_rules             # độ phủ lớp luật giấy phép
```

Nếu bạn tái lập từ đầu trên bản sao mới (chưa có file kết quả Test), quy trình đúng như `PHASE4_INTEGRATION_GUIDE.md` mục 5: dựng bộ truy vấn Test, rồi `eval --split test --final-test` cho M0/M1/M2 và 3 seed M3, cuối cùng `test_verdict`. **Mỗi nhãn chỉ chạy một lần.**

---

## 10. Phase 5 — Demo Streamlit

Mô hình demo: M3 `noise_coherent_k2`, **seed 43** (tốt nhất trên Val). Chỉ mục: 800 bài known của Test. Ngưỡng lấy từ Val.

```bash
pip install streamlit
python -m src.app.calibrate_demo        # hiệu chuẩn ngưỡng từ Val -> e5_demo_calibration.json (vài phút CPU)
python -m src.app.make_demo_samples     # tạo demo_samples/ (cần audio FMA)
python -m src.app.smoke_demo            # kiểm tra ba nhánh + ca lỗi, mong đợi "SMOKE ĐẠT"
streamlit run src/app/streamlit_app.py
```

Lần mở đầu mất khoảng 1 phút để dựng chỉ mục (lưu cache `data/interim/demo_index.npz`), các lần sau nhanh.

Giới hạn đầu vào: tối đa 20 MB; mp3/wav/flac/ogg/m4a; tối thiểu 3 s; chỉ phân tích tối đa 10 s (đoạn giữa). Tệp tạm được xóa sau khi xử lý; không lưu audio; không dùng dịch vụ ngoài hay khóa API.

Đổi mô hình demo: sửa `model.ckpt` trong `configs/demo.yaml`.

---

## 11. Xử lý sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Lệnh thoát mã 2 với "CHƯA CHẠY" | Thiếu đầu vào của bước trước; chạy lại đúng thứ tự mục 6–8 |
| `build` thoát mã 3 | Có cấu hình biến đổi bị bỏ qua (thường do thiếu `ffmpeg`); cài ffmpeg rồi chạy lại |
| `decode_error` hàng loạt khi cache | Thiếu hỗ trợ mp3: cập nhật `soundfile`/libsndfile hoặc đưa ffmpeg vào `PATH` |
| Hết bộ nhớ GPU khi huấn luyện | Giảm `train.batch_size` trong `configs/m3.yaml` (ví dụ 64) |
| Colab hết RAM | Đặt `num_workers: 0` hoặc `preload: false` |
| `freeze_phase3`/`test_gate` báo file đã đổi | Bạn đã sửa `targets.yaml`, `m3.yaml`, `baseline.yaml` hoặc checkpoint; khôi phục bản gốc |
| `streamlit run` báo lỗi API | Phiên bản Streamlit khác bản đã dùng; dán nguyên văn lỗi để xử lý |

---

## 12. Nguyên tắc khoa học của dự án

- Chia tập theo nhóm nghệ sĩ/album, kiểm tra rò rỉ bằng test tự động.
- Siêu tham số và ngưỡng chỉ chọn trên Val; mục tiêu thành công khóa trong `targets.yaml` trước khi chạm Test; Test chạy đúng một lần.
- Cùng split, cùng metric, cùng bộ truy vấn cho mọi mô hình.
- Mọi con số trong báo cáo lấy từ `experiments/results/`; phần chưa chạy ghi "CHƯA CHẠY".
- Nhãn đầu ra không bao giờ là "hợp lệ", "không có bản quyền" hay "được phép dùng".

Nhật ký chi tiết từng phase: `PHASE0_Project_Review_Final_Architecture.md`, `PHASE1…5_INTEGRATION_GUIDE.md`, `phase-tracker.md`.

---

## 13. Dữ liệu, giấy phép và trích dẫn

- Dữ liệu: **FMA — Free Music Archive** (`fma_small`, `fma_metadata`), Defferrard et al., *FMA: A Dataset for Music Analysis*, ISMIR 2017. Mỗi bài có giấy phép Creative Commons riêng; xem điều khoản của FMA khi tái phân phối.
- Mã nguồn: `<<XÁC NHẬN: giấy phép của repo, ví dụ MIT>>`.
- Tác giả: `<<XÁC NHẬN: tên / nhóm / trường / năm>>`.
