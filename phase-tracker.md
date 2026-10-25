# PHASE TRACKER
Cập nhật lần cuối: 2026-10-27 · Trạng thái: **Phase 0-3 XONG (freeze ĐẠT, tag `phase3-final`). Phase 4 ĐANG LÀM: 4A/4B code+test xong, CHƯA chạm Test. `configs/targets.yaml` đã khóa**

Quy ước: `XONG` = code + test pass (và đã chạy dữ liệu thật nếu cần); `CHƯA CHẠY` = chưa chạy trên dữ liệu thật; số liệu chỉ ghi khi có file trong `experiments/results/`.

| Phase | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| 0 | Review & thiết kế cuối | XONG | `PHASE0_Project_Review_Final_Architecture.md` |
| 1 | Data Foundation | **XONG** | 32 passed, 0 skipped; leakage check PASS; notebook EDA đã chạy |
| 2 | Baselines (M0, M1, M2) + truy vấn robustness + `targets.yaml` | **XONG** | 75 passed; M0/M1/M2 đã chạy Val (sạch và 17 cấu hình); targets đã khóa |
| 3 | Deep model (M3) | **XONG** | Cấu hình cuối `noise_coherent_k2`, 3 seed; Val macro R@1 0,9615 ± 0,0030, nhận đúng@FMR 0,8845 ± 0,0129; `freeze_phase3` ĐẠT; tag `phase3-final` |
| 4 | Evaluation & Robustness (Test một lần) | **ĐANG LÀM** | 4A/4B code+test xong, chưa chạy Test; 4C (lỗi, độ trễ), 4D (T2) chưa làm. Chi tiết: `PHASE4_INTEGRATION_GUIDE.md` |
| 5 | Demo Streamlit | CHƯA BẮT ĐẦU | Bắt buộc (đã xác nhận) |
| 6 | Báo cáo & bảo vệ | CHƯA BẮT ĐẦU | |

## Phase 1 (đã xong) - số liệu thật, nguồn: `experiments/results/`
| Mục | Giá trị |
|---|---|
| File loại (clean_report) | 7992/8000 giữ; too_short 3, decode_error 3, silent 2 |
| Nghệ sĩ / album (sau lọc) | 2307 / 2462; median 1 bài/nghệ sĩ, max 208 |
| Nhóm trùng (dup_group > 1) / cặp nghi trùng gần | 47 / 77 (60 cùng nghệ sĩ, 17 khác nghệ sĩ) |
| Split | train 4794; val 1601 (known 802, unknown 799); test 1597 (known 800, unknown 797) |
| Leakage | passed, violations: [] |
| Smoke gates | 4/5 PASS; `license_families_ge_min_tracks` FAIL (license_family = unknown cho cả 8000 bài; cột license là tên văn bản) |

Thay đổi so với thiết kế ban đầu: `split.py` nối đơn vị chia thêm theo `album_id`; SHA1 và URL FMA đã sửa trong `configs/data.yaml`; cần `ipython<9.0`, `matplotlib-inline<0.2.0` để chạy notebook.

## Phase 2 (đã xong)
| Phần | Nội dung | Test | Chạy dữ liệu thật |
|---|---|---|---|
| 2A | `evaluation/{metrics,openset,bootstrap}.py` | 47 passed | Không cần |
| 2B | truy vấn sạch, M0, M1, `runner.py`, `run_baselines.py` | 56 passed | Val: xong |
| 2C | `peaks.py`, `m2_fingerprint.py` | 62 passed | Val: xong |
| 2D | `transforms.py`, `robust_queries.py`, `robustness.py`, `make_targets.py`, `run_robustness.py` | 75 passed | Val: xong; `targets.yaml` khóa |

### Baseline trên Val, truy vấn sạch (crop 5 s; `e1_*_val.json`)
| Mô hình | R@1 | R@5 | MRR | AUC | EER | Nhận đúng tại FMR<=1% |
|---|---|---|---|---|---|---|
| M0 ngẫu nhiên | 0,0017 (ngẫu nhiên 0,0012) | 0,0096 | 0,0105 | 0,506 | 0,998 | 0,0% |
| M1 MFCC/mel cosine | 0,9967 | 1,0000 | 0,9982 | 0,919 | 0,167 | 34,6% |
| M2 fingerprint | 1,0000 | 1,0000 | 1,0000 | 1,000 | 0,000 | 100% |
M2 bão hòa trên truy vấn sạch nên bảng này KHÔNG dùng để đặt mục tiêu.

### Baseline trên Val, truy vấn biến đổi (6.800 truy vấn = 17 cấu hình x 200 known + 200 unknown; chỉ mục 802 bài; `e5_robust_val_*.json`)
Recall@1 theo nhóm (một tau cố định mỗi mô hình, FMR gộp = 1,0%):
| Mô hình | cut | codec | noise | eq | pitch | tempo | overlay | **macro R@1** | **macro nhận đúng@FMR** | tau |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 1,0000 |
| M1 | 0,987 | 0,980 | 0,183 | 0,165 | 0,533 | 0,968 | 0,540 | **0,6221** | **0,1068** | 0,9442 |
| M2 | 1,000 | 1,000 | 0,995 | 1,000 | 0,056 | 0,508 | 0,970 | **0,7898** | **0,7093** | 0,0341 |
Điểm yếu: M1 sụp ở nhiễu và EQ; M2 sụp ở pitch (0,056) và tempo (0,508), đúng điểm mù đã dự báo ở Phase 0. M3 cần cải thiện đúng các nhóm M1 yếu (noise, eq, pitch, overlay) mà không mất cut/codec/tempo.

### `configs/targets.yaml` (đã khóa, tạo 2026-10-20)
| Mục | Giá trị |
|---|---|
| Quy tắc | gain = max(0,02; 0,25 x (1 - M1_Val)); CI cặp đôi (bootstrap theo bài nguồn) của hiệu M3-M1 có cận dưới > 0 |
| gain macro Recall@1 | 0,0945 (tham chiếu Val: M1 + gain = 0,7166) |
| gain macro nhận đúng@FMR | 0,2233 (tham chiếu Val: M1 + gain = 0,3301) |
| FMR vận hành | 0,01 |
Cách xét ĐẠT/KHÔNG ĐẠT ở Phase 4 xem mục 10 của `PHASE2_INTEGRATION_GUIDE.md`. M2 chỉ để tham khảo.

## Phase 3 (đang làm) - chi tiết ở `PHASE3_INTEGRATION_GUIDE.md`
| Phần | Nội dung | Test | Chạy dữ liệu thật |
|---|---|---|---|
| 3A | Augmentation log-mel, dataset cặp view, thống kê chuẩn hóa, CLI kiểm tra | local 93 passed khi đó | **Xong**: std crop 15,19; 13,0 ms/mục; tần suất op khớp cấu hình |
| 3B | ResNet nhỏ (569.520 tham số, embedding 128-d L2), NT-Xent mặt nạ cùng bài | 28 passed trên Colab (3A+3B) | **Xong** |
| 3C | Huấn luyện (AdamW + cosine, AMP, checkpoint/resume, dừng sớm, learning curve), notebook Colab | local 111 passed | **Xong**: seed 42, 60 epoch, T4 1.121 s; loss train/Val 1,103/1,396 |
| 3D-1 | `m3_encoder.py` (nạp checkpoint, embedding crop/cửa sổ) | local 117 passed | Dùng trong 3D-2 |
| 3D-2 | `m3_retrieval.py`, vá `run_baselines.py`/`run_robustness.py` (`--m3-ckpt`, `--tag`) | local 122 passed | **Xong** (Val, seed 42): sạch R@1 1,000, EER 0,0025; robust macro R@1 **0,9095**, nhận đúng@FMR **0,8102** (tau 0,7933); yếu ở noise (0,635) và overlay (0,750) |
| 3E | `--ablation`/`--set`, `m3_ablations.yaml`, `summarize_m3.py`, `analysis/m3_noise_check.py`, notebook Ô 12-15 | local 127 passed | Chẩn đoán nhiễu **xong**; seed 43/44 và 4 ablation đang chạy Colab |
| 3E-2 | `add_noise_coherent`, `noise.mode`, chẩn đoán mở rộng (lưới R@1 theo SNR) | local 128 passed | **Xong** (Ô 16): nhiễu thật nằm giữa additive (quá dễ) và coherent k=1 (quá khó) |
| 3E-3 | `noise.coherent_k`, `--k` trong `m3_noise_check`, ablation `noise_coherent_k{2,4,8}` | local 130 passed | **Xong**: k=2 khớp nhiễu thật nhất; `noise_coherent_k2` seed 42: noise 0,673 -> 0,953, macro R@1 0,9580, nhận đúng@FMR 0,8754 |
| 3F | Bộ tổng hợp nhiều seed; Ô 20-21; `evaluation/freeze_phase3.py` (kiểm tra + SHA256 + `verify_freeze`) | local 135 passed | **Xong**: 3 seed, `freeze_phase3` ĐẠT (36/37 dòng log dán, `ok=true`); Val 3 seed macro R@1 0,9615 ± 0,0030 |

Chẩn đoán nhiễu v2 (seed 42, `e3_noise_check_seed42_v2.json`): R@1 nhiễu thật 0,290 / 0,650 / 0,965 (SNR 5/10/20 dB); nhiễu tổng hợp cộng công suất 1,000 ở cả ba mức (tương đương SNR thật khoảng -10/-5/0 dB, tức dưới dải huấn luyện 3-25 dB); nhiễu coherent k=1: 0,155 / 0,245 / 0,620 (quá khó). Nhiễu thật nằm giữa hai mô phỏng; giả thuyết giao thoa pha được ủng hộ định tính, chưa chứng minh.

### M3 trên Val (17 cấu hình / 7 nhóm, một tau; nguồn `m3_val_summary.md`)
| Mô hình | cut | codec | noise | eq | pitch | tempo | overlay | macro R@1 | macro nhận đúng@FMR |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 0,987 | 0,980 | 0,183 | 0,165 | 0,532 | 0,968 | 0,540 | 0,6221 | 0,1068 |
| M2 | 1,000 | 1,000 | 0,995 | 1,000 | 0,056 | 0,508 | 0,970 | 0,7898 | 0,7093 |
| M3 gốc, 3 seed (mean ± std) | 1,000 | 1,000 | 0,673 | 0,992 | 0,996 | 0,998 | 0,752 | 0,9158 ± 0,0056 | 0,8243 ± 0,0122 |
| M3 `noise_coherent_k2`, seed 42 (cấu hình cuối, còn 2 seed) | 1,000 | 1,000 | 0,953 | 0,985 | 0,998 | 1,000 | 0,770 | 0,9580 | 0,8754 |
Chỉ là Val, 200 bài nguồn mỗi loại. k chọn theo truy vấn nhiễu thật của Val: kết luận về độ bền nhiễu chỉ dựa trên Test. Mức tham chiếu ĐẠT ở Test: macro R@1 >= 0,7166 và nhận đúng@FMR >= 0,3301, cộng CI cặp đôi cận dưới > 0 (mục 10 `PHASE2_INTEGRATION_GUIDE.md`).

Ablation (seed 42, Δ macro R@1 so với `m3_seed42`): `no_noise` -0,041 (noise -0,250); `no_pitch_tempo` -0,031 (pitch -0,585, tempo không đổi); `gap` -0,025 (nhận đúng@FMR -0,155); `no_overlap` +0,0065 (không khác biệt).

Quyết định ở 3E: ablation ưu tiên `no_noise`, `no_pitch_tempo`, `gap`, `no_overlap` (seed 42, 60 epoch); `noise_harder` chỉ thử sau chẩn đoán nhiễu.
Quyết định ở 3C: Val dùng cùng augmentation (epoch cố định); chọn `best.pt` theo loss Val; batch 128, lr 1e-3, 60 epoch.
Quyết định ở 3B: `pool: freq_flatten`; nhiệt độ NT-Xent 0,1.
Quyết định ở 3A: `gain_ramp` thay `gain` cố định; chuẩn hóa = trừ trung bình crop, chia std toàn cục (từ bài Train).

## Lịch rút gọn (còn 2 tuần)
| Ngày | Nội dung | Trạng thái |
|---|---|---|
| 1-3 | Phase 2 (2A-2D) | XONG |
| 4-8 | Phase 3: M3 (cổng learning curve đã qua) | ĐANG LÀM (3A-3D xong; 3E chờ Colab) |
| 9-10 | Phase 4: Test một lần, robustness, bootstrap, T2 nếu kịp | |
| 11-12 | Phase 5: Streamlit tối thiểu | |
| 13-14 | Phase 6: báo cáo, slide, README | |
Cắt ngay: T3 (cổng đã FAIL), M4 (không dùng pretrained), M5, FastAPI. T2 giữ nếu sửa được `license_norm` (có bảng giá trị thật, việc nhỏ).

## Việc đang chờ bạn
- Chép gói Phase 4A/4B (chép ĐÈ `src/run_robustness.py`); bước 0 trong `PHASE4_INTEGRATION_GUIDE.md`: `pytest -q` (mong đợi 145 passed), kiểm `load_gains` in 0,0945 và 0,2233, `freeze_phase3` vẫn ĐẠT; rồi `git commit`.
- Sau đó chạy Test đúng một lần theo mục 5 của guide Phase 4 (build, eval m0/m1/m2, eval m3 ba seed, `test_verdict`) và dán kết quả.
- Không sửa `targets.yaml`. Sửa `license_norm.py` (T2) thuộc 4D.

## Vấn đề mở
- Cổng `license_families_ge_min_tracks` FAIL do lỗi nhận dạng văn bản, không phải do dữ liệu thiếu giấy phép; chưa biết số nhóm thật.
- Ngưỡng `dedup.near_dup_cosine` = 0,995: giữ. 15 cặp khác nghệ sĩ đều nằm trong `dup_group` nên split an toàn; cặp 110736-119979 (cosine 0,9998) có vẻ là cùng một bản thu nhưng chưa nghe kiểm.
- M1 chỉ dùng đặc trưng từ log-mel; chroma/spectral contrast/ZCR của Phase 0 bị bỏ (cần audio). Có thể thêm ở ablation.
- M2 chạy trên log-mel (không dựng lại STFT); tham số `m2:` là đề xuất, chưa tinh chỉnh.
- FMR gộp = 1% nhưng KHÔNG đều giữa các cấu hình (M2: 11,5% ở crop 3 s, 0% ở crop 10 s; M1: 4,5% ở SNR 5 dB). Điểm M2 phụ thuộc độ dài truy vấn. Phải báo FMR từng cấu hình và lưu ý khi làm demo.
- Chỉ 200 bài unknown làm nguồn (17 truy vấn/bài, không độc lập): ước lượng FMR 1% nhiễu; Phase 4 báo FMR kèm CI.
- Truy vấn là crop của clip đã index (giao thức in-index): M2 rất cao là đúng định nghĩa bài toán, không phải hiệu năng thực tế.
- M0 là MỘT lần rút ngẫu nhiên lặp lại cho cả 17 cấu hình (cùng seed, cùng kích thước): chỉ dùng làm mốc, không đọc từng cấu hình.
- Tiến độ 2 tuần chặt; Phase 3 là điểm rủi ro lớn nhất (GPU MX230 yếu, dùng Colab).
- Nhiễu: nguyên nhân đã rõ (nhiễu cộng công suất quá dễ so với nhiễu thật); giải pháp nhiễu coherent k=2 nâng nhóm noise trên Val từ 0,673 lên 0,953 (một seed). k chọn theo Val, nên chỉ Test mới là bằng chứng độc lập.
- `overlay` còn 0,75-0,77 (M2 0,970): hạn chế đã biết, họ giữ ra khỏi huấn luyện; M5 (M2+M3) tùy chọn.
- Bốn vòng tuning trên Val; đây là vòng cuối, không tuning thêm trước Test.
- Val chỉ 200 bài nguồn mỗi loại và một seed: đừng diễn giải chênh lệch vài điểm % giữa các cấu hình riêng lẻ.
