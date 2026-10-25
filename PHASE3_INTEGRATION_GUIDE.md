# PHASE 3 - INTEGRATION GUIDE (M3: CNN embedding + học tương phản)
Cập nhật: 2026-10-27 · **Phase 3 XONG**: `freeze_phase3` ĐẠT, cấu hình cuối `noise_coherent_k2` 3 seed (Val macro R@1 0,9615 ± 0,0030; nhận đúng@FMR 0,8845 ± 0,0129); tag `phase3-final`. Phase 4 bắt đầu ở `PHASE4_INTEGRATION_GUIDE.md`.

## 1. Kế hoạch chia phần
| Phần | Nội dung | Trạng thái |
|---|---|---|
| 3A | Dữ liệu huấn luyện: augmentation log-mel, dataset cặp view, thống kê chuẩn hóa | **XONG** (mục 9) |
| 3B | ResNet nhỏ -> embedding 128-d L2; NT-Xent có mặt nạ cùng bài | **XONG** (mục 10; pass local và Colab T4) |
| 3C | Vòng huấn luyện, checkpoint theo epoch, learning curve, notebook Colab | **XONG** (seed 42 chạy thật trên T4) |
| 3D-1 | `M3Encoder`: nạp checkpoint, embedding crop/cửa sổ trượt | **XONG** (117 passed local) |
| 3D-2 | Lớp `M3Retriever`, nối vào `run_baselines`/`run_robustness`, chạy trên Val | **XONG** (122 passed local; đã chạy Val seed 42) |
| 3E | 3 seed, ablation, chẩn đoán nhiễu, chốt Phase 3 | Code + test xong; chẩn đoán nhiễu đã chạy; seed/ablation đang chạy Colab |
| 3E-2 | Mô phỏng nhiễu kết hợp (`noise.mode: coherent`) và chẩn đoán mở rộng | **XONG** (128 passed local; Ô 16 đã chạy) |
| 3E-3 | Hiệu chuẩn `noise.coherent_k` theo nhiễu thật, huấn luyện biến thể | **XONG** (130 passed local; Ô 18-19 đã chạy): k=2 được chọn |
| 3F | Chốt cấu hình cuối: 3 seed của `noise_coherent_k2`, tổng hợp, đóng băng | **XONG** (135 passed local; `freeze_phase3` ĐẠT; tag `phase3-final`) |

Cổng: có learning curve đầu tiên vào ngày 6 của lịch rút gọn; không có thì cắt mọi thứ phụ để dồn cho M3. Bốn quyết định thiết kế (mục 11 của `PHASE2_INTEGRATION_GUIDE.md`) đã được bạn xác nhận.

## 2. Phần 3A: những gì đã có
| File | Vai trò | Yêu cầu GV |
|---|---|---|
| `configs/m3.yaml` | Khối `data:` (đường dẫn, sr/hop/n_mels, crop 5 s = 215 khung, preload, norm) và `augment:` (xác suất và dải tham số từng op) | 1, 2 |
| `src/features/augment_mel.py` | `time_warp`, `freq_warp`, `eq_gain_db`, `add_noise`, `gain_ramp`, `spec_mask`; `MelAugmenter` (thứ tự cố định, tự chặn `mix`/`overlay`/`codec`) | 2 |
| `src/features/m3_input.py` | `normalize_crop(mel, std)`: trừ trung bình crop, chia std toàn cục. Dùng chung huấn luyện và suy luận (3D) | 2 |
| `src/data/m3_dataset.py` | `PairDataset` (2 view khác nhau của cùng bài Train), `compute_norm_stats`/`load_or_compute_norm`, `build_pair_dataset(cfg, split)`, `save_preview`, CLI kiểm tra | 1, 2, 4 |
| `tests/test_m3_data_3a.py` | 19 test (tổng hợp) | 4 |

## 3. Điểm nối (API nội bộ)
- `build_pair_dataset(cfg, split="train"|"val", augment=True) -> (dataset, norm_stats)`. `dataset[i] -> (x1, x2, track_id)`, `x*`: float32 `(1, 128, 215)`; DataLoader tự gộp thành `(B, 1, 128, 215)` và `track_id` thành tensor int.
- `dataset.set_epoch(e)`: gọi ở tiến trình chính trước mỗi epoch, `persistent_workers=False`. Ngẫu nhiên xác định theo `(seed, epoch, chỉ số)`, không phụ thuộc số worker.
- `get_item(i)` trả thêm danh sách op đã áp dụng cho từng view (dùng cho phân tích/ablation).
- Thống kê chuẩn hóa tính một lần từ bài Train, lưu `data/interim/m3_norm.json`; 3C phải sao chép `std_after_crop_mean` vào checkpoint để 3D dùng đúng cùng giá trị.
- `repeats_per_epoch > 1` có thể đưa cùng một bài vào một batch hai lần: 3B sẽ cài mặt nạ theo `track_id` cho NT-Xent (không coi cùng bài là âm).
- Val: `build_pair_dataset(cfg, "val", augment=False)` chỉ để theo dõi loss; chuẩn hóa vẫn dùng std của Train.

## 4. Cách chạy (PowerShell)
Chép 5 file code + `tests/` + `configs/m3.yaml` vào repo (giữ nguyên đường dẫn), rồi:
```
pytest -q
python -m src.data.m3_dataset --preview
```
Kết quả mong đợi:
- `pytest -q`: **93 passed, 1 skipped** (75 cũ + 18 mới; test DataLoader bị skip nếu máy chưa cài torch; có torch thì 94 passed).
- CLI: ghi `data/interim/m3_norm.json`, `experiments/results/e3_data_check.json`, `reports/figures/m3_aug_preview.png`; thoát mã 2 "CHƯA CHẠY" nếu thiếu `tracks.parquet` hoặc cache log-mel.
- Trong sandbox với 30 bài tổng hợp: ~1,8 ms/mục, `view_mean` = 0,0, `view_std_mean` ≈ 0,9. Số này KHÔNG đại diện FMA thật; mình không dự đoán số thật.
- Hãy xem ảnh preview: ô "sạch" và 5 ô augment phải trông như cùng một bản nhạc bị biến đổi, không có vùng NaN/đen hoàn toàn.

Dán lại giúp mình: `e3_data_check.json` và ảnh preview.

## 5. Cách đọc `e3_data_check.json`
| Trường | Bình thường | Nếu khác |
|---|---|---|
| `norm_stats.raw_*` | Khoảng dB hợp lý (ví dụ min khoảng -80, max khoảng 0 hoặc hơn chút) | Giá trị lạ (ví dụ dương rất lớn, hoặc thang biên độ thay vì công suất) -> báo để chỉnh `add_noise` |
| `std_after_crop_mean` | Vài đến vài chục dB | Gần 0 hoặc rất lớn -> cache có vấn đề |
| `view_std_range` | Quanh 1, không có 0 | Có 0 -> view phẳng (clip im lặng sót lại) |
| `op_frequency_per_view` | Gần `p` trong `m3.yaml` | Lệch nhiều -> kiểm tra cấu hình |
| `ms_per_item` | Vài ms (ổ SSD) | Hàng chục ms -> bật `preload: true` hoặc dùng nhiều worker |

## 6. Quyết định thiết kế và lệch so với Phase 2
1. **`gain` cố định được thay bằng `gain_ramp`** (độ lợi biến thiên tuyến tính theo thời gian). Lý do: chuẩn hóa crop trừ trung bình nên độ lợi cố định bị triệt tiêu, thêm vào chỉ tốn công. Ưu điểm phụ: M3 bất biến độ lợi theo cấu trúc, phù hợp việc truy vấn được peak-normalize riêng từng truy vấn còn chỉ mục chuẩn hóa theo cả clip.
2. **Họ giữ ra (ISSUE-05):** `mix`/`overlay`/`codec` không có trong augmentation; thêm vào `m3.yaml` sẽ báo lỗi (có test). Báo cáo tách nhóm "đã thấy" (cut, noise, eq, pitch, tempo) và "chưa thấy" (overlay, codec).
3. **Pitch/tempo/EQ là xấp xỉ trên log-mel:** `freq_warp` co giãn trục mel (không giữ được formant như pitch shift thật), `time_warp` nội suy khung, `eq` cộng độ lợi Butterworth theo dải mel. Đánh giá dùng biến đổi THẬT (bộ truy vấn 2D) nên độ lệch xấp xỉ-thật được đo trung thực. Nhóm pitch/tempo/EQ nằm trong họ augmentation nên là "đã thấy" ở mức họ, không phải đúng tham số.
4. **Hai view có thể chồng nhau** (mặc định `min_start_distance_frames: 0`), nhất quán với giao thức in-index (truy vấn là crop của clip đã index). Đặt >= 215 để buộc không chồng: đưa vào ablation 3E.
5. Thang mel trong augmentation dùng công thức HTK, còn cache có thể dựng bằng thang Slaney của librosa; sai khác nhỏ, chấp nhận cho augmentation.

## 7. Đóng góp báo cáo
Chương 2 (mô hình/huấn luyện): bảng op augmentation với xác suất và dải tham số (từ `m3.yaml`), lý do họ giữ ra, lý do chuẩn hóa theo crop, ảnh `m3_aug_preview.png`. Chưa có số liệu hiệu năng.

## 8. Rủi ro và việc còn lại
- Chưa chạy trên cache thật; mình chưa xem `logmel.py` nên giả định cache là dB công suất (10*log10), `float16`, shape `(128, T)`. `add_noise` sai tỷ lệ nếu cache là dB biên độ (`e3_data_check.json` sẽ lộ ra).
- `m3.yaml` (`sr`, `hop_length`, `n_mels`) phải khớp `features.yaml`; mình chưa thấy file đó. `n_mels` sai sẽ làm CLI báo lỗi shape; `sr` sai thì không báo lỗi (chỉ lệch tần số trong EQ/pitch).
- `load_config("m3")` giả định hàm nhận tên không đuôi như ở các phần trước; nếu khác, báo lỗi để chỉnh.
- Tham số augmentation là đề xuất, chưa tinh chỉnh; chỉ chỉnh dựa trên Val ở 3E.
- `preload: true` cần khoảng 1,6 GB RAM (Train, float16): bật trên Colab; trên máy 16 GB cũng ổn.

## 9. Kết quả chạy thật 3A (nguồn: `e3_data_check.json`, ảnh preview, do bạn chạy 2026-10-21)
| Mục | Giá trị |
|---|---|
| `pytest -q` | 93 passed, 1 skipped |
| Train | 4.794 bài, crop 215 khung |
| Cache log-mel (dB công suất) | min -100 (sàn `amin=1e-10`), p01 -63,3, p50 -8,7, p99 23,7, max 39,6 |
| `std_after_crop_mean` | 15,19 (500 bài x 4 crop) |
| View chuẩn hóa | mean 0,0; std trung bình 0,729; khoảng 0,307-2,152 |
| Tần suất op / view | eq 0,378; freq_warp 0,477; gain_ramp 0,310; noise 0,575; spec_mask 0,780; time_warp 0,500 (khớp `p` trong cấu hình) |
| Tốc độ | 13,0 ms/mục (ổ đĩa, memmap, chưa preload) |

Nhận xét:
- Ba giả định của 3A đã được xác nhận: cache là dB công suất float16 `(128, T)`; `features.yaml` khớp `m3.yaml`; `load_config("m3")` đúng chữ ký. Ảnh preview khớp kỳ vọng (không NaN, các biến đổi nhìn hợp lý; nhiễu SNR thấp làm lộ rõ dải cao).
- Sàn -100 dB tồn tại thật (vùng im lặng số): `freq_warp` điền bằng giá trị nhỏ nhất của crop nên các dải trống có thể sâu hơn nội dung xung quanh; chấp nhận, vì chuẩn hóa theo trung bình crop.
- 13 ms/mục x 4.794 mục = khoảng 62 s/epoch nếu chỉ một tiến trình: bật `preload: true` và `num_workers` >= 2 trên Colab để GPU không phải chờ dữ liệu.
- Bản `m3.yaml` bạn dán trong ghi chú thiếu một số dòng (`p` và `semitone_min` của `freq_warp`, `lowpass_hz` của `eq`) so với bản mình gửi. Số liệu tần suất cho thấy các op đó đang chạy đúng, nên nhiều khả năng chỉ là hao dòng khi dán; dù vậy, bản mới của `m3.yaml` trong gói 3B ghi đè toàn bộ file, hãy chép đè nguyên file.

## 10. Phần 3B: mô hình và loss
| File | Vai trò | Yêu cầu GV |
|---|---|---|
| `configs/m3.yaml` | Thêm khối `model:` (channels, embed_dim, pool, dropout) và `loss:` (temperature) | 2 |
| `src/models/m3_cnn.py` | `M3Net`, `ResBlock`, `build_model(cfg)`, `count_params` | 2 |
| `src/models/losses.py` | `nt_xent(z1, z2, temperature, track_ids=None) -> (loss, top1_acc)` | 2 |
| `tests/test_m3_model_3b.py` | 9 test (cần torch) | 4 |

**Kiến trúc:** stem conv 3x3 (16) -> 4 stage ResBlock (16, 32, 64, 128 kênh; stage 2-4 stride 2) -> bản đồ `(128, 16, 27)` cho đầu vào `(1, 128, 215)` -> trung bình theo thời gian, giữ trục tần số, flatten 128x16 = 2.048 -> Linear -> 128 -> chuẩn hóa L2. Tổng 569.520 tham số.

**Vì sao giữ trục tần số (`freq_flatten`):** vị trí tần số mang thông tin nhận diện (và nhạy với pitch); GAP cả hai trục làm mất nó. `pool: gap` giữ cho ablation 3E. Trục thời gian được lấy trung bình nên bất biến dịch thời gian và dùng được cho truy vấn 3/5/10 s mà không đổi mô hình.

**Loss:** NT-Xent đối xứng, nhiệt độ 0,1 (đề xuất, chưa tinh chỉnh). Mọi mẫu khác trong batch là âm; nếu truyền `track_ids`, mẫu cùng bài (ngoài cặp dương) bị che khỏi mẫu số để tránh âm giả khi `repeats_per_epoch > 1`. Logits luôn tính ở float32 (an toàn khi dùng AMP ở 3C). `top1_acc` chỉ để theo dõi.

**Điểm nối cho 3C/3D:** `net = build_model(cfg)`; `z = net(x)` với `x` từ DataLoader `(B, 1, 128, 215)`; `loss, acc = nt_xent(net(x1), net(x2), cfg["loss"]["temperature"], track_ids)`. Khi suy luận, đặt `net.eval()` (BatchNorm dùng thống kê chạy). Batch càng lớn càng nhiều âm: kích thước batch sẽ chốt ở 3C theo bộ nhớ T4.

**Cách chạy:**
Máy local của bạn chưa cài torch (test DataLoader của 3A bị skip ở lần chạy trước), nên test 3B sẽ bị skip ở local.
```
pytest -q                                                       # local: mong đợi 93 passed, 2 skipped
pytest tests/test_m3_data_3a.py tests/test_m3_model_3b.py -q    # trên Colab (có torch): mong đợi 28 passed
```
Sandbox của mình (torch 2.14 CPU): tập `tests/` của 3A+3B, 28 test, pass hết. Trên CPU, forward+backward batch 64 mất khoảng 6 s (Colab T4 sẽ nhanh hơn nhiều; chưa đo).

**Rủi ro 3B:**
- Chưa huấn luyện trên dữ liệu thật: nhiệt độ 0,1, kích thước mô hình và việc chọn `freq_flatten` là đề xuất; chỉ chỉnh dựa trên Val ở 3C/3E.
- Với 4.794 bài và crop chồng nhau, M3 có thể học nhanh "nhận ra bài" mà chưa bền với nhiễu/pitch: đó là điều bảng robustness Val sẽ cho biết ở 3D.
- BatchNorm với augmentation mạnh có thể khác thống kê giữa train và eval; nếu thấy chênh lớn ở 3D, xem xét GroupNorm (chưa làm, YAGNI).
- Còn lại: 3C (huấn luyện), 3D (tích hợp đánh giá), 3E (3 seed + ablation).

## 11. Kết quả 3B (do bạn chạy) và Phần 3C: huấn luyện
**3B đã xác nhận:** local `pytest -q` = 103 passed; Colab (torch 2.11.0+cu130, Tesla T4) 3A+3B = 28 passed. Máy local có torch (mình đã đoán sai ở lượt trước).

| File | Vai trò | Yêu cầu GV |
|---|---|---|
| `src/train_m3.py` | `train(cfg, seed, out_dir, ...)`, `cosine_lr`, `EarlyStopper`, `plot_history`, CLI | 2, 3, 4 |
| `configs/m3.yaml` | Thêm khối `train:` và `data.norm_seed` (CHÉP ĐÈ nguyên file) | 2 |
| `src/data/m3_dataset.py` | Sửa nhỏ: thống kê chuẩn hóa dùng `data.norm_seed` (cố định giữa các seed). Giá trị đã cache của bạn không đổi (seed 42 như cũ) | 2 |
| `notebooks/03_train_m3_colab.ipynb` | Quy trình Colab: mount, giải nén, test, chạy thử, huấn luyện có resume, chép kết quả | 2, 4 |
| `tests/test_train_3c.py` | 8 test (CPU, dữ liệu tổng hợp nhỏ) | 4 |

**Thiết kế huấn luyện:**
- Dữ liệu: Train 4.794 bài, mỗi mục là hai view của một bài; hai view được ghép chung một lượt forward (BatchNorm thấy cả hai) rồi tách đôi để tính NT-Xent. Batch 128 cặp (256 mẫu), `drop_last`.
- Tối ưu: AdamW, lr 1e-3 (khởi động tuyến tính 2 epoch rồi cosine về 1%), weight decay 1e-4, cắt gradient 5, AMP float16 khi có GPU. Tất cả là đề xuất chưa tinh chỉnh.
- Theo dõi: mỗi epoch tính loss/top-1-trong-batch trên Val với cùng augmentation, cố định epoch=0 (xác định, nên so sánh được giữa các epoch). Lệch nhẹ so với mục 3 của guide ("Val augment=False"): mình đổi vì loss Val có augmentation đo độ bền, có ích hơn. Tắt bằng `val_augment: false`.
- Chọn epoch: `best.pt` = epoch có loss Val thấp nhất (dùng Val để chọn là hợp lệ). Dừng sớm sau 12 epoch không cải thiện. **Test không bị đọc** (có test kiểm tra, và `summary.json` ghi `test_touched: false`).
- Checkpoint: `last.pt` mỗi epoch (mô hình, optimizer, scaler, lịch sử, thống kê early stopping), ghi nguyên tử; `best.pt`; `history.json`; `curve.png`; `summary.json`. Mỗi checkpoint lưu cấu hình `model`/`data` và `norm_std` để 3D dựng lại đúng mô hình và chuẩn hóa.
- Resume: `--resume` tiếp tục từ `last.pt`; từ chối nếu `model`/`seed` khác. Có test chứng minh kết quả resume (dừng ở epoch 2 rồi chạy tiếp) khớp lần chạy liền mạch (loss và trọng số, sai số 1e-5, trên CPU). Trên GPU kết quả không bảo đảm khớp từng bit.
- `--stop-after N` dừng sau epoch N nhưng lịch lr vẫn theo `epochs`, dùng để chia nhỏ phiên Colab.

**Cách chạy:**
Local (kiểm tra): `pytest -q` -> mong đợi **111 passed** (103 + 8).
Colab: (1) chép gói 3C vào repo local, tar lại `configs/` và `src/` thành `code.tar`, upload lên Drive; (2) mở `notebooks/03_train_m3_colab.ipynb`, sửa ô 1 (đường dẫn Drive là giả định); (3) chạy tuần tự. Ô 5 mong đợi **36 passed**. Ô 6 là chạy thử 5 step: dán lại dòng log (cột `seconds`) để mình ước lượng thời gian cho 60 epoch x 3 seed. Ô 7 chạy thật, lặp lại ô này nếu Colab ngắt.
Lệnh tương đương không dùng notebook:
```
python -m src.train_m3 --seed 42 --preload --resume --out-dir <thư mục checkpoint>
```
Kết quả mong đợi: mỗi epoch một dòng `train loss/acc, val loss/acc, lr, seconds`; cuối cùng `summary.json` và `experiments/results/e3_train_seed42.json`. Mình KHÔNG dự đoán số loss thật. Điều cần thấy để qua cổng ngày 6: loss train giảm rõ rệt, loss Val giảm và không bật tăng sớm.
Sandbox (CPU, dữ liệu tổng hợp, batch 16): CLI chạy được, `--stop-after` rồi `--resume` hoạt động. Batch 128 + 2 worker trong sandbox bị hệ điều hành kill vì hết RAM; đây là giới hạn sandbox. Ước lượng thô của mình là bộ nhớ kích hoạt khoảng 20 MB/mẫu fp32 (~5 GB cho 256 mẫu, khoảng một nửa với AMP) nên T4 15 GB dự kiến vừa; chưa đo. Nếu hết bộ nhớ GPU: giảm `batch_size` xuống 64 (NT-Xent có ít mẫu âm hơn).

**Cách đọc learning curve:**
| Hiện tượng | Ý nghĩa | Hành động |
|---|---|---|
| Train và Val loss cùng giảm | Bình thường | Tiếp tục |
| Train giảm, Val phẳng/tăng sớm | Quá khớp | Tăng augmentation hoặc `weight_decay`, giảm epoch (chỉnh trên Val) |
| Cả hai loss gần `ln(255)` ~ 5,54 và không giảm | Không học (lr, dữ liệu) | Gửi log cho mình |
| `val_acc` rất cao ngay epoch đầu | Tác vụ quá dễ (hai view chồng nhau) | Thử `min_start_distance_frames: 215` ở ablation |

**Rủi ro 3C:**
- Chưa chạy thật: siêu tham số (lr, batch, nhiệt độ, 60 epoch) là đề xuất; thời gian một epoch trên T4 chưa đo (dữ liệu: 13 ms/mục chưa preload; ước lượng thô, chưa phải số đo).
- Top-1 trong batch không phải Recall@1 của bài toán: số đo thật chỉ có ở 3D trên bộ truy vấn robustness (một ngưỡng chung, 802 bài chỉ mục).
- `num_workers: 2` + `preload: true` nhân đôi RAM (mỗi worker fork chia sẻ trang chỉ đọc; thường ổn nhưng nếu Colab báo hết RAM, đặt `num_workers: 0` hoặc `preload: false`).
- Early stopping dựa loss Val có augmentation cố định; chọn `best.pt` theo loss chứ không theo Recall@1, nên epoch tốt nhất theo loss có thể không phải epoch tốt nhất theo Recall@1 (xem lại ở 3D).
- Đường dẫn trong notebook (tên tar, thư mục Drive) là giả định.
- Còn lại: 3D (lớp `M3`, tích hợp `run_baselines`/`run_robustness`, đánh giá Val), 3E (3 seed + ablation).

## 12. Kết quả chạy thật 3C (nguồn: `summary.json`, `curve.png`, log Colab T4, seed 42)
| Mục | Giá trị |
|---|---|
| Môi trường | Tesla T4, torch 2.11.0+cu130, AMP |
| Epoch chạy | 60/60, không dừng sớm; `best_epoch` = 60 |
| Thời gian | 1.121 s (khoảng 18,7 phút), khoảng 18,2 s/epoch |
| Loss train / Val (epoch cuối) | 1,103 / 1,396 |
| Top-1 trong batch train / Val | 0,845 / 0,753 |
| Loss Val epoch 1 -> 60 | 4,52 -> 1,396 (ngẫu nhiên ~ ln(255) = 5,54) |
| Tham số | 569.520 |
| `test_touched` | false |

Nhận xét (đây là Val có augmentation, 127 âm trong batch; CHƯA phải Recall@1 của bài toán):
- Cả hai loss giảm đều, không có dấu hiệu quá khớp phân kỳ: khoảng cách train-Val ổn định ở mức 0,25-0,30 loss. Qua cổng "có learning curve vào ngày 6".
- `best_epoch = 60` vì lịch cosine hạ lr về 1e-5; Val gần như phẳng ở 5 epoch cuối (1,4006 -> 1,3955). Early stopping không kích hoạt, nên việc chọn epoch chưa được kiểm chứng trong thực tế; vấn đề "loss tốt nhất khác Recall@1 tốt nhất" để xem ở 3D-2.
- Val còn giảm nhẹ ở cuối, nên có thể thêm epoch hoặc sức chứa mô hình sẽ cải thiện được; chỉ quyết định sau khi có Recall@1/robustness trên Val (3D-2), không dựa riêng vào loss trong batch.
- Ngân sách thời gian: khoảng 19 phút/seed -> 3 seed khoảng 1 giờ; mỗi biến thể ablation đầy đủ cũng khoảng 19 phút (có thể ablation 30 epoch để rẻ hơn, quyết ở 3E).
- Log "Ô 6" (chạy thử 5 step) của bạn hiển thị `epoch x/60` và 60 epoch (khoảng 7,5 s/epoch, tổng 480 s): tức lệnh đã chạy mà không có `--epochs 1` (hoặc cờ không được truyền). Không ảnh hưởng kết quả (thư mục khác, `--resume` ở Ô 7 không đọc nó) nhưng thời gian 7,5 s/epoch chủ yếu là chi phí đánh giá Val, không phải thời gian 5 step. Số đáng tin để ước lượng là 18,2 s/epoch của lần chạy thật. Lưu ý Ô 6 cũng đã ghi `experiments/results/e3_train_seed42.json`; lần chạy thật ghi đè đúng file này, đúng như mong muốn.

## 13. Phần 3D-1: bộ mã hóa suy luận
| File | Vai trò |
|---|---|
| `src/models/m3_encoder.py` | `M3Encoder.from_checkpoint(path)`; `embed_crops((N,F,T))`, `embed_mel((F,T))` (độ dài bất kỳ), `window_embeddings(mel, window, hop)`; `window_starts` |
| `src/models/m3_cnn.py` | Sửa nhỏ: `M3Net` lưu thêm thuộc tính `n_mels` |
| `tests/test_m3_encoder_3d.py` | 6 test: khớp mô hình gốc, chuẩn L2, độ dài biến thiên, bất biến độ lợi, batch nhất quán, cửa sổ/đệm, từ chối shape sai |

Điểm nối cho 3D-2: checkpoint (`best.pt`) chứa `cfg` và `norm_std`; encoder tự dựng mô hình và chuẩn hóa. Điểm bài = max cosine giữa embedding truy vấn và các embedding cửa sổ 5 s (bước 2,5 s) của bài đó (cùng quy ước M1; điểm cao = giống hơn). Truy vấn 3 s và 10 s chạy trực tiếp với độ dài thật (mô hình huấn luyện ở 5 s nên đây là kiểm tra khái quát hóa về độ dài); phương án thay thế "cắt truy vấn dài thành cửa sổ 5 s rồi lấy trung bình" để ablation.

Cách chạy: `pytest -q` mong đợi **117 passed** (111 + 6). Sandbox: tập 3A-3D-1 = 42 passed.

**3D-2 cần xem code hiện có** (mình chưa thấy các file này nên không viết mù để tránh sai giao diện): `src/models/base.py`, `src/models/m1_classical.py`, `src/data/queries.py` (lớp `QueryMels`), `src/evaluation/robustness.py`, `src/run_baselines.py` (hàm `build_model`), `src/run_robustness.py`, và định dạng file `data/interim/robust/robust_val_<cfg>.npz` (khóa, shape, có phải log-mel chuẩn hóa đỉnh cả clip hay theo từng truy vấn). Dán nội dung hoặc đính kèm các file đó.

## 14. Phần 3D-2: M3 vào pipeline đánh giá Val
Kết quả 3D-1 (bạn chạy): local `pytest -q` = 117 passed.

| File | Vai trò | Yêu cầu GV |
|---|---|---|
| `src/models/m3_retrieval.py` | `M3Retriever` (`build_index`/`score_matrix` đúng giao diện `Retriever`), `build_m3(cfg)` | 2, 3 |
| `src/models/m3_encoder.py` | Sửa nhỏ: encoder giữ `data_cfg` của checkpoint (để kiểm tra `sr`/`hop` khớp) | 3 |
| `src/run_baselines.py` | Thêm nhánh `m3` trong `build_model`, cờ `--m3-ckpt`, `--tag`; tên file theo `label` (`e1_m3_seed42_val.json`) | 3, 4 |
| `src/run_robustness.py` | Như trên cho `eval`: `e5_robust_val_m3_seed42.json` và `..._perquery.npz` | 3, 4 |
| `notebooks/03_train_m3_colab.ipynb` | Thêm Ô 10-11: chạy đánh giá Val và chép kết quả | 4 |
| `tests/test_m3_retrieval_3d2.py` | 5 test | 4 |

`run_baselines.py` và `run_robustness.py` là bản vá TRÊN đúng nội dung bạn dán (đã kiểm các điểm neo khớp và cú pháp); nếu bạn đã sửa hai file này sau lần dán, hãy áp lại tay 3 thay đổi (nhánh `m3`, hai cờ, `label` trong tên file) thay vì chép đè.

**Thiết kế:**
- Chỉ mục: cửa sổ 5 s, bước 2,5 s, lấy từ `m1.win_sec`/`m1.hop_sec` của `baseline.yaml` (cùng cấu hình với M1; hàm cắt cửa sổ của M3 là `window_starts` riêng trong `m3_encoder.py`, chỉ giống M1 ở mức cửa sổ trượt nằm trọn trong clip, chưa đối chiếu từng khung với `handcrafted.window_starts`).
- Điểm bài = max cosine giữa embedding truy vấn và các cửa sổ của bài. Truy vấn 3 s/5 s/10 s được embed trực tiếp với độ dài thật (mô hình huấn luyện ở 5 s, nên nhóm `cut` đo khả năng khái quát hóa về độ dài). Chế độ `m3.query_mode: windows` (cắt truy vấn dài thành cửa sổ 5 s rồi lấy trung bình) dành cho ablation.
- Truy vấn được duyệt theo luồng (chunk 512), không giữ toàn bộ log-mel trong RAM.
- `tau` vẫn chọn từ điểm unknown gộp mọi cấu hình trên Val (một ngưỡng cho cả hệ thống), đúng như M0/M1/M2: số M3 so sánh được với bảng ở mục 9 của `PHASE2_INTEGRATION_GUIDE.md`. Đây vẫn là số Val, chưa phải kết quả cuối.

**Cách chạy** (Colab, dùng `best.pt` trên Drive; hoặc local nếu bạn tải `best.pt` về):
```
pytest -q                                   # local mong đợi: 122 passed (117 + 5)
python -m src.run_baselines --split val --models m3 --m3-ckpt <đường dẫn>/best.pt --tag seed42
python -m src.run_robustness eval --split val --models m3 --m3-ckpt <đường dẫn>/best.pt --tag seed42
```
Trên Colab: tar lại `configs/` + `src/` thành `code.tar` mới, chạy lại Ô 3, rồi Ô 10-11. Cần `data/interim/robust/` và cache log-mel (đã có trong tar dữ liệu); đánh giá KHÔNG cần audio. Trên CPU cũng chạy được (khoảng 8,8 nghìn cửa sổ chỉ mục + 11,6 nghìn truy vấn; chưa đo thời gian thật).
Kết quả mong đợi: dòng `[m3_seed42] R@1=... AUC=... EER=...` và dòng `[m3_seed42] tau=... macro R@1=... nhận đúng@FMR=... (7/7 nhóm)` kèm điểm 7 nhóm. Mình KHÔNG dự đoán số.

**Cách đọc để quyết bước tiếp theo** (so với bảng Val của baseline; chỉ là Val, dùng chọn hướng, chưa kết luận ĐẠT/KHÔNG ĐẠT):
| Tham chiếu Val | M1 | M2 | Mức tham chiếu để M3 "ĐẠT" ở Test (M1_Val + gain) |
|---|---|---|---|
| macro Recall@1 | 0,6221 | 0,7898 | 0,7166 |
| macro nhận đúng@FMR | 0,1068 | 0,7093 | 0,3301 |
- Nhìn từng nhóm: M3 có cải thiện đúng nhóm M1 yếu (noise, eq, pitch, overlay) mà không mất cut/codec/tempo không? `overlay` và `codec` là họ giữ ra khỏi huấn luyện nên là phép thử khái quát hóa thật.
- Nhóm `cut`: so crop_3s / crop_5s / crop_10s để thấy mô hình nhạy với độ dài truy vấn thế nào.
- FMR từng cấu hình có đều không (với M1/M2 từng không đều); báo cáo kèm FMR từng cấu hình.

**Rủi ro 3D-2:**
- Chưa chạy trên checkpoint và dữ liệu thật; chỉ test bằng mô hình nhỏ ngẫu nhiên và log-mel tổng hợp. Kiểm tra tích hợp `run_*` mới ở mức: nhánh `build_model("m3")`, thiếu `--m3-ckpt` báo lỗi rõ, cú pháp; chưa chạy toàn bộ CLI (các module `cache`, `runner`, ... mình chưa thấy).
- Chỉ mục dùng cửa sổ chồng 50% từ clip, còn truy vấn là crop ngẫu nhiên của clip đó (in-index): truy vấn sạch có thể đạt điểm rất cao; sức phân biệt thật nằm ở truy vấn biến đổi và unknown.
- Truy vấn robustness chuẩn hóa đỉnh theo từng truy vấn, chỉ mục theo cả clip: `normalize_crop` trừ trung bình crop nên triệt phần lớn độ lệch này, nhưng không phải mọi tác động (ví dụ sàn -100 dB khác nhau).
- Mô hình chọn theo loss Val: `best.pt` của seed 42 trùng epoch 60 (cuối), nên vấn đề "epoch tốt nhất theo loss khác theo Recall@1" chưa được kiểm tra; nếu muốn, có thể đánh giá thêm `last.pt` (cùng file ở seed 42).
- Còn lại: 3E (3 seed, ablation, chốt Phase 3).

## 15. Kết quả chạy thật 3D-2 (M3 seed 42, `best.pt`, Val; nguồn `e1_m3_seed42_val.json`, `e5_robust_val_m3_seed42.json`)
`pytest -q` local = 122 passed. `window_starts` của M1 và M3 đã được bạn đối chiếu và khớp ở n_frames = 215, 646, 1292 (giải quyết rủi ro đã nêu ở mục 14).

**Truy vấn sạch (crop 5 s, chỉ mục 802 bài, 4.803 truy vấn):** Recall@1 = Recall@5 = MRR = 1,000; AUC 0,99995; EER 0,0025. Tại FMR <= 1% (tau_high 0,808, FMR thực 0,96%): nhận đúng 99,92%, nhận nhầm ID 0%, bị từ chối 0,08% (M1: 34,6%). Truy vấn sạch là crop của clip đã index (in-index) nên đây KHÔNG phải hiệu năng thực tế.

**Truy vấn biến đổi, macro 7 nhóm (6.800 truy vấn, một tau = 0,7933):**
| Mô hình | cut | codec | noise | eq | pitch | tempo | overlay | macro R@1 | macro nhận đúng@FMR |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 0,987 | 0,980 | 0,183 | 0,165 | 0,533 | 0,968 | 0,540 | 0,6221 | 0,1068 |
| M2 | 1,000 | 1,000 | 0,995 | 1,000 | 0,056 | 0,508 | 0,970 | 0,7898 | 0,7093 |
| **M3 seed 42** | 1,000 | 1,000 | 0,635 | 0,988 | 0,996 | 0,998 | 0,750 | **0,9095** | **0,8102** |

**Đọc kết quả (Val, một seed, 200 bài nguồn mỗi loại: sai số chuẩn của một tỷ lệ gần 0,9 khoảng 2 điểm %, nên đừng đọc chênh lệch vài điểm giữa các cấu hình riêng lẻ):**
- M3 vượt mức tham chiếu suy từ M1 (macro R@1 0,7166; nhận đúng@FMR 0,3301) với biên rất lớn trên Val. Đây là tín hiệu tốt cho Phase 4, chưa phải kết luận ĐẠT (cần Test + CI bootstrap cặp đôi theo mục 10 của `PHASE2_INTEGRATION_GUIDE.md`).
- M3 thắng M2 ở pitch (0,996 so với 0,056) và tempo (0,998 so với 0,508), đúng hai điểm mù của fingerprint; nhóm `codec` (họ giữ ra khỏi huấn luyện) đạt 1,000, và `cut` kể cả 3 s/10 s (độ dài khác lúc huấn luyện) đều 1,000, FMR đều.
- M3 thua M2 ở nhiễu (0,635 so với 0,995) và overlay (0,750 so với 0,970), thấp nhất ở SNR 5 dB (R@1 0,29, nhận đúng 7%) và 10 dB (0,65 / 28%). Ở SNR 20 dB: 0,965. M3 không bắt buộc vượt M2 (mục 10), nhưng đây là điểm yếu cần trình bày trung thực trong báo cáo.
- Điều bất thường cần giải thích: augmentation nhiễu huấn luyện ở SNR 3-25 dB mà mô hình vẫn yếu ở 5-10 dB thật. Có hai khả năng: (a) nhiễu tổng hợp trên log-mel không giống nhiễu trắng thật (hiệu chuẩn SNR sai), hoặc (b) giới hạn của mô hình. Phần 3E bên dưới có phép đo để phân biệt (a) và (b); mình chưa kết luận.
- Nhóm `overlay` (trộn nhạc khác ở 5 dB) cũng là họ giữ ra khỏi huấn luyện: 0,750 là phép thử khái quát hóa thật.

## 16. Phần 3E: nhiều seed, ablation, chẩn đoán nhiễu
| File | Vai trò | Yêu cầu GV |
|---|---|---|
| `src/train_m3.py` | Thêm `--ablation <tên>`, `--set khóa=giá_trị` (nhiều lần), `apply_overrides`, `load_ablation`; ghi `ablation`/`overrides`/`augment` vào `summary.json`; tên chạy `m3_abl_<tên>_seed<N>` | 2, 3 |
| `configs/m3_ablations.yaml` | 7 biến thể (4 ưu tiên): `no_noise`, `no_pitch_tempo`, `gap`, `no_overlap`; thêm `no_eq`, `aug_none`, và `noise_harder` (loại `tuning`) | 3 |
| `src/evaluation/summarize_m3.py` | Gom `e5_robust_val_*.json`: trung bình ± std qua seed, bảng ablation (Δ so với seed 42), xuất `e3_summary_val.json` và `reports/tables/m3_val_summary.md` | 3 |
| `src/analysis/m3_noise_check.py` | So truy vấn nhiễu THẬT (`snr_5/10/20db`) với nhiễu TỔNG HỢP trên log-mel sạch (`crop_5s`) cùng SNR danh nghĩa; tìm SNR tổng hợp tương đương theo độ nghiêng phổ | 3 |
| `notebooks/03_train_m3_colab.ipynb` | Thêm Ô 12-15 | 4 |
| `tests/test_m3_3e.py` | 5 test | 4 |

**Quy ước tên:** huấn luyện `--tag` trong lệnh đánh giá phải khớp: `m3_seed43` <-> `--tag seed43`; `m3_abl_no_noise_seed42` <-> `--tag abl_no_noise_seed42`. `summarize_m3` dựa vào quy ước này.

**Cách chạy:** local `pytest -q` mong đợi **127 passed** (122 + 5; sandbox: 52). Rồi tar lại `configs/` + `src/` thành `code.tar`, chạy Ô 3 trên Colab, sau đó:
- Ô 12: seed 43, 44 (khoảng 40 phút) + đánh giá Val từng seed.
- Ô 13: chẩn đoán nhiễu (vài phút). Dán lại log và `e3_noise_check_seed42.json`. **Nên chạy ô này trước Ô 14**: kết quả quyết định có thử `noise_harder` hay sửa `add_noise`.
- Ô 14: 4 ablation ưu tiên (khoảng 80 phút; mỗi biến thể 60 epoch giống baseline để so sánh công bằng).
- Ô 15: `summarize_m3` và chép kết quả về Drive.
Tổng khoảng 2 giờ GPU; nếu Colab hết hạn mức, ưu tiên Ô 13, rồi seed 43/44, rồi `no_noise` và `no_pitch_tempo`.

**Cách đọc kết quả mong đợi (mình KHÔNG dự đoán số):**
- Seed: std qua 3 seed là biến thiên huấn luyện, chưa phải CI; báo cáo kèm 3 giá trị. Nếu macro R@1 dao động trên 2 điểm %, ghi vào hạn chế.
- Noise check: so `real.recall_at_1` với `synthetic_same_nominal.recall_at_1`. Nếu tổng hợp cao hơn nhiều so với thật -> augmentation chưa đủ giống nhiễu thật (hướng (a): chỉnh `add_noise` hoặc `noise_harder`). Nếu hai số gần nhau -> hướng (b). `equivalent_synthetic_snr_db_by_tilt` cho biết SNR thật tương ứng với SNR tổng hợp nào về độ nghiêng phổ; nếu lệch xa danh nghĩa thì hiệu chuẩn SNR là vấn đề.
- Ablation: `no_noise` nên làm nhóm noise tụt nếu augmentation nhiễu có tác dụng; `no_pitch_tempo` nên làm pitch/tempo tụt nếu `freq_warp`/`time_warp` là lý do M3 thắng M2 ở đó. Nếu không tụt, kết luận "nhờ augmentation" không đúng và phải nói rõ trong báo cáo.

**Rủi ro 3E:**
- Mọi thứ trên chưa chạy thật; test dùng mô hình nhỏ ngẫu nhiên và dữ liệu tổng hợp. Chẩn đoán nhiễu dựa trên độ nghiêng phổ (dải cao trừ dải thấp), một thống kê thô, có thể không bắt được hết khác biệt giữa nhiễu thật và tổng hợp.
- Chọn `best.pt` theo loss Val chứ không theo Recall@1; với seed 42 trùng epoch cuối nên chưa kiểm chứng.
- Ablation chạy một seed (42); chênh lệch nhỏ hơn biến thiên giữa seed (xem std ở Ô 12) không được diễn giải.
- `noise_harder` là tuning trên Val: nếu dùng làm cấu hình cuối thì phải huấn luyện lại 3 seed và ghi rõ đã chọn trên Val.
- Chưa làm: M2+M3 fusion (M5). Dữ liệu Val cho thấy hai mô hình bổ trợ nhau (M2 mạnh nhiễu/overlay, M3 mạnh pitch/tempo); nếu muốn một hệ thống mạnh nhất cho demo thì đây là hướng tự nhiên, nhưng không bắt buộc cho tiêu chí yêu cầu 2 và nằm trong danh sách "có thể cắt".
- Còn lại: chạy các ô trên, chốt Phase 3, sang Phase 4 (Test một lần).

## 17. Kết quả chẩn đoán nhiễu (3E, seed 42, Val; nguồn `e3_noise_check_seed42.json`) và Phần 3E-2
`pytest -q` local = 127 passed. Chẩn đoán dùng 200 truy vấn known, chỉ mục 802 bài, 124 s.

| Điều kiện | R@1 nhiễu THẬT | R@1 nhiễu tổng hợp (cộng công suất, cùng SNR) | SNR tổng hợp tương đương theo độ nghiêng phổ |
|---|---|---|---|
| SNR 5 dB | 0,290 | 1,000 | 7 dB |
| SNR 10 dB | 0,650 | 1,000 | 13 dB |
| SNR 20 dB | 0,965 | 1,000 | 23 dB |
Truy vấn sạch: R@1 = 1,000.

**Đọc kết quả (đây là điều đo được; phần giải thích bên dưới là giả thuyết chưa kiểm chứng):**
- Mô hình bất biến HOÀN TOÀN với nhiễu tổng hợp (R@1 = 1,000 kể cả 5 dB) nhưng không với nhiễu thật cùng SNR danh nghĩa. Nghĩa là M3 đã học tính bất biến với bộ MÔ PHỎNG nhiễu của mình, chứ chưa phải với nhiễu thật: giữa augmentation và dữ liệu đánh giá có chênh lệch phân phối.
- Không phải lệch SNR thô: theo độ nghiêng phổ, nhiễu tổng hợp chỉ nhẹ hơn nhiễu thật khoảng 2-3 dB (tương đương 7/13/23 dB cho 5/10/20 dB). Huấn luyện đã bao phủ SNR 3-25 dB, nên lệch 2-3 dB khó giải thích khoảng cách 1,000 so với 0,290; mình chưa thử lệch này bằng thực nghiệm. Lưu ý độ nghiêng phổ là thống kê thô, không bắt được cấu trúc dao động từng ô.
- Giả thuyết (chưa kiểm chứng): nhiễu thật cộng ở miền audio nên có giao thoa pha với tín hiệu, làm công suất các ô tín hiệu mạnh dao động quanh giá trị gốc (có thể tăng hoặc giảm); mô phỏng cộng công suất `P + N` chỉ nâng sàn, không có dao động này.

**3E-2: kiểm chứng trước khi huấn luyện lại.**
| File | Thay đổi |
|---|---|
| `src/features/augment_mel.py` | Thêm `add_noise_coherent` (mỗi ô mel là số phức, `P' = |sqrt(P) + sqrt(N)*z|^2`, cùng công suất kỳ vọng với cộng công suất) và tham số `augment.noise.mode` (`additive` mặc định / `coherent`) |
| `configs/m3.yaml` | Thêm `noise.mode: additive` (chép ĐÈ nguyên file) |
| `configs/m3_ablations.yaml` | Thêm `noise_coherent`, `noise_coherent_harder` (loại `tuning`) |
| `src/analysis/m3_noise_check.py` | Đánh giá cả hai mô phỏng; thêm lưới R@1 theo SNR tổng hợp và `equivalent_snr_by_recall_{additive,coherent}` |
| `tests/test_m3_3e.py` | Thêm 1 test (tính chất nhiễu kết hợp: cùng công suất kỳ vọng, dao động ô mạnh lớn hơn nhiều, có thể giảm công suất) |
| notebook | Ô 16-17 |

**Quy tắc quyết định sau Ô 16** (chạy lại chẩn đoán với checkpoint seed 42 hiện có, không cần huấn luyện; dùng `--tag seed42_v2` để không ghi đè kết quả cũ):
- Nếu R@1 của nhiễu `coherent` ở SNR danh nghĩa GẦN nhiễu thật (khoảng 0,29 / 0,65 / 0,965; chênh dưới khoảng 0,1) thì mô phỏng này là xấp xỉ tốt -> chạy Ô 17 (`noise_coherent`, khoảng 19 phút), rồi so nhóm noise với `m3_seed42` và so với biến thiên giữa seed.
- Nếu `coherent` vẫn cho R@1 gần 1,000 thì giả thuyết sai hoặc không đủ: mô phỏng ở miền log-mel không thể tái tạo nhiễu thật. Hướng thay thế (3F, chưa viết): tạo sẵn bản log-mel có nhiễu từ AUDIO cho các bài Train (giải mã lại 4.794 clip, thêm nhiễu trắng theo SNR, tính log-mel, lưu cache; mình ước khoảng 4,7 GB đĩa cho 3 biến thể và 15-25 phút CPU, chưa đo). Nếu cả hai hướng không cải thiện, ghi nhận trung thực: M3 yếu ở nhiễu thấp SNR, M2 bổ trợ (hướng M5).
- `equivalent_snr_by_recall_*` cho biết SNR tổng hợp nào có độ khó tương đương nhiễu thật (chỉ có nghĩa khi R@1 thật dưới khoảng 0,97). Nếu kết quả rơi ngoài dải huấn luyện 3-25 dB thì dải SNR huấn luyện cũng là một nguyên nhân.

**Cách chạy:** local `pytest -q` mong đợi **128 passed** (127 + 1; sandbox: 53). Đợi Ô 12 và Ô 14 đang chạy xong rồi mới tar lại `configs/`+`src/`, upload, chạy lại Ô 3; Ô 16 (khoảng 4 phút), dán log và `e3_noise_check_seed42_v2.json`.

**Rủi ro 3E-2:**
- `add_noise_coherent` là xấp xỉ mức một ô, có thể dao động mạnh hơn thực tế ở ô mel rộng (trung bình nhiều bin FFT); đó là lý do phải kiểm chứng bằng Ô 16 trước khi tin.
- Đổi cấu hình augmentation là tuning trên Val: nếu dùng làm cấu hình cuối thì phải huấn luyện lại 3 seed và ghi rõ đã chọn trên Val.
- Nhóm `overlay` (trộn nhạc khác, 0,750) không được xử lý bởi 3E-2 và vẫn là họ giữ ra khỏi huấn luyện.

## 18. Kết quả chẩn đoán nhiễu v2 (3E-2, `e3_noise_check_seed42_v2.json`) và Phần 3E-3
`pytest -q` local = 128 passed. 200 truy vấn known, chỉ mục 802 bài, 236 s.

**R@1 của chính checkpoint seed 42 (huấn luyện bằng nhiễu cộng công suất):**
| SNR | Nhiễu THẬT | Tổng hợp cộng công suất (additive) | Tổng hợp coherent (k = 1) |
|---|---|---|---|
| 5 dB | 0,290 | 1,000 | 0,155 |
| 10 dB | 0,650 | 1,000 | 0,245 |
| 20 dB | 0,965 | 1,000 | 0,620 |

**R@1 theo SNR tổng hợp (lưới):** additive: -10 dB 0,425; -5 dB 0,865; 0 dB 0,98; từ 5 dB trở lên 1,000. Coherent: -10 dB 0,025; 0 dB 0,06; 10 dB 0,245; 20 dB 0,62; 25 dB 0,785.
**SNR tổng hợp tương đương theo R@1 thật (5/10/20 dB):** additive -10 / -5 / 0 dB; coherent 10 / 20 / 25 dB.

**Đọc kết quả:**
- Nhiễu cộng công suất quá dễ so với nhiễu thật: R@1 thật ở 5 dB (0,29) chỉ ngang additive khoảng -10 dB, tức khoảng 15 dB dưới dải huấn luyện (3-25 dB; ở mức đó additive cho R@1 = 1,000 nên mô hình hầu như không bị thử thách). Đây là lời giải thích trực tiếp vì sao mô hình yếu ở nhiễu thật dù "có augmentation nhiễu".
- Nhiễu coherent (k = 1) khó hơn nhiễu thật: R@1 thật ngang coherent ở SNR cao hơn 5-10 dB (hoặc hơn, ở 20 dB). Nghĩa là nhiễu thật nằm GIỮA hai mô phỏng.
- Giả thuyết giao thoa pha được ủng hộ ở mức định tính: thêm dao động pha làm R@1 của cùng mô hình sụp từ 1,000 xuống 0,155-0,62, kiểu hỏng mà nhiễu cộng thuần không gây ra. Đây vẫn chưa chứng minh giao thoa pha là nguyên nhân duy nhất; chỉ cho thấy mô hình rất nhạy với dao động mức ô, và `k = 1` (mỗi ô mel là một bin FFT) vượt quá thực tế vì ô mel trung bình nhiều bin, đúng như đã nêu ở rủi ro 3E-2.
- Quy tắc ở mục 17 không rơi vào nhánh nào nguyên văn (coherent không gần thật, nhưng cũng không ≈ 1,000): kết luận hợp lý là cần mô phỏng trung gian.

**3E-3: hiệu chuẩn `k`.**
| File | Thay đổi |
|---|---|
| `src/features/augment_mel.py` | `add_noise_coherent(..., k=1)`: mỗi ô mel là trung bình k số phức; `augment.noise.coherent_k` (k lớn -> gần additive, k=1 -> coherent mạnh nhất) |
| `configs/m3.yaml` | Thêm `noise.coherent_k: 1` (chép ĐÈ nguyên file) |
| `configs/m3_ablations.yaml` | Thêm `noise_coherent_k2`, `_k4`, `_k8` (loại `tuning`) |
| `src/analysis/m3_noise_check.py` | `--k 2 4 8 16`: R@1 của coherent với từng k ở SNR 5/10/20 dB, `l1_to_real` và `best_k_by_l1` |
| `tests/test_m3_3e.py` | Thêm 2 test (k nội suy về additive, giữ công suất kỳ vọng; k_candidates) |
| notebook | Ô 18-19 |

**Cách chạy:** local `pytest -q` mong đợi **130 passed** (128 + 2; sandbox 55). Colab: tar lại `configs/`+`src/`, Ô 3, Ô 18 (dán log và `e3_noise_check_seed42_v3.json`; mình ước thêm vài phút so với Ô 16, chưa đo), rồi Ô 19 với K theo dòng "k tốt nhất" (khoảng 19 phút huấn luyện + vài phút đánh giá).

**Lưu ý phương pháp:** hiệu chuẩn k dùng truy vấn nhiễu thật của Val, nên số nhóm `noise` của biến thể đã chọn k sẽ không còn là bằng chứng độc lập hoàn toàn trên Val (chỉ Test mới độc lập). Báo cáo phải ghi rõ: k chọn theo Val; kết luận về độ bền nhiễu chỉ dựa trên Test.

**Quy tắc chốt cấu hình cuối (để Phase 3 không kéo dài):** biến thể mới được chọn nếu (1) nhóm noise tăng rõ rệt (vượt biến thiên giữa seed của cấu hình gốc) và (2) macro R@1, macro nhận đúng@FMR và các nhóm cut/codec/pitch/tempo không giảm quá biến thiên đó. Nếu được chọn: huấn luyện thêm seed 43 và 44 cho cấu hình mới (khoảng 40 phút) để đủ 3 seed trước Phase 4. Nếu không: giữ cấu hình gốc (đã có 3 seed) và ghi nhận hạn chế ở nhiễu thấp SNR. Mỗi vòng tuning thêm trên Val làm tăng nguy cơ chọn theo nhiễu của Val; đây là vòng cuối.

**Bảng tổng hợp Ô 15 chưa dùng được:** log của bạn chỉ có `n=1 seed` và bảng ablation rỗng, dù danh sách file trên Drive có đủ `e5_robust_val_m3_seed43/44` và 4 file ablation. Mình chưa biết nguyên nhân; khả năng cao là Ô 15 chạy trong phiên Colab mà thư mục `experiments/results` cục bộ chưa có các file đó (dòng m1, m2 có thể đến từ tar dữ liệu), còn `cp` ở cuối ô chép các file có sẵn trên Drive. Cách xử lý: tải thư mục `results/` trên Drive về `experiments/results/` ở máy local, chạy `python -m src.evaluation.summarize_m3` rồi dán `reports/tables/m3_val_summary.md`. Nếu lúc đó vẫn là `n=1`, có thể lỗi ở mình: gửi các file `e5_robust_val_m3_*.json`.

## 19. Kết quả 3E (seed, ablation) và 3E-3 (hiệu chuẩn k), quyết định cấu hình cuối
Nguồn: `m3_val_summary.md`, `e3_noise_check_seed42_v3.json`, log Ô 19 (do bạn chạy; local `pytest -q` = 130 passed). Mọi số là Val (200 bài nguồn mỗi loại), chưa phải kết quả cuối.

### 19.1. Ba seed của cấu hình gốc
macro Recall@1 = **0,9158 ± 0,0056**; macro nhận đúng@FMR = **0,8243 ± 0,0122** (n = 3; độ lệch chuẩn là biến thiên huấn luyện, chưa phải CI). Theo nhóm (trung bình): cut 1,000; codec 1,000; noise 0,673; eq 0,992; pitch 0,996; tempo 0,998; overlay 0,752. Biến thiên giữa seed nhỏ, nên chênh lệch macro trên khoảng 0,02 giữa hai biến thể có ý nghĩa; chênh lệch vài điểm % ở một nhóm riêng lẻ (n = 200 nguồn) thì chưa.

### 19.2. Ablation (một seed 42; Δ so với `m3_seed42`, macro R@1 0,9095 / nhận đúng@FMR 0,8102)
| Biến thể | macro R@1 (Δ) | macro nhận đúng@FMR (Δ) | Đọc kết quả |
|---|---|---|---|
| `no_noise` | 0,8684 (-0,041) | 0,7213 (-0,089) | Nhóm noise 0,635 -> 0,385 (-0,250): augmentation nhiễu cộng công suất vẫn có đóng góp thật, dù hiệu chuẩn chưa đúng |
| `no_pitch_tempo` | 0,8783 (-0,031) | 0,7236 (-0,087) | pitch 0,996 -> 0,411 (-0,585): `freq_warp` là lý do M3 bền với pitch. **tempo vẫn 1,000** khi bỏ `time_warp`: bền với tempo không đến từ augmentation (có thể do trung bình theo thời gian và dải tempo nhỏ 0,90-1,10). Đổi lại, không có hai augmentation này thì noise +0,222 và overlay +0,130: giữa pitch và nhiễu/overlay có sự đánh đổi sức chứa |
| `gap` | 0,8840 (-0,025) | 0,6557 (-0,155) | Trung bình cả hai trục kém hơn giữ trục tần số (overlay -0,115; nhận đúng@FMR -0,155): ủng hộ thiết kế `freq_flatten` |
| `no_overlap` | 0,9160 (+0,0065) | 0,8055 (-0,0047) | Hai view chồng nhau hay không không tạo khác biệt vượt biến thiên giữa seed (0,0056): không có bằng chứng cho việc đổi |

Điều chỉnh một nhận định trước đó: mục 16 nêu "nếu `no_pitch_tempo` không làm pitch/tempo tụt thì kết luận 'nhờ augmentation' sai". Kết quả: đúng với pitch, sai với tempo. Báo cáo chỉ được viết "bền với pitch nhờ augmentation `freq_warp`"; không viết rằng `time_warp` giúp tempo (chưa có bằng chứng, `time_warp` có thể bỏ). Ablation chỉ một seed.

### 19.3. Hiệu chuẩn k (Ô 18; `e3_noise_check_seed42_v3.json`, checkpoint gốc seed 42)
R@1 của truy vấn nhiễu thật: 0,29 / 0,65 / 0,965 (SNR 5/10/20 dB). Nhiễu coherent theo k (cùng SNR):
| k | 5 dB | 10 dB | 20 dB | L1 so với thật |
|---|---|---|---|---|
| 1 | 0,175 | 0,225 | 0,615 | 0,890 |
| **2** | 0,400 | 0,745 | 0,975 | **0,215** |
| 4 | 0,125 | 0,440 | 0,930 | khoảng 0,41 (mình tính từ ba R@1; dòng này bị cắt trong log) |
| 8 | 0,045 | 0,280 | 0,905 | 0,675 |
| 16 | 0,025 | 0,240 | 0,880 | 0,760 |
Nhiễu thật nằm giữa k = 2 (hơi dễ hơn thật) và k = 4 (khó hơn thật); k = 2 khớp nhất. Nhiễu cộng công suất thuần (R@1 = 1,000 từ 5 dB) vẫn quá dễ. Không thử k = 3: đã quyết định đây là vòng tuning cuối.

### 19.4. `noise_coherent_k2` (seed 42, Ô 19): đối chiếu quy tắc chốt cấu hình đã đặt trước ở mục 18
Huấn luyện 60 epoch, 1.831 s (khoảng 30 s/epoch, chậm hơn 18 s của cấu hình gốc vì sinh nhiễu coherent tốn CPU); loss Val cuối 1,487 (không so được trực tiếp với 1,396 của cấu hình gốc vì nhiễu huấn luyện khác nhau).
| | cut | codec | noise | eq | pitch | tempo | overlay | macro R@1 | macro nhận đúng@FMR |
|---|---|---|---|---|---|---|---|---|---|
| gốc, trung bình 3 seed | 1,000 | 1,000 | 0,673 | 0,992 | 0,996 | 0,998 | 0,752 | 0,9158 ± 0,0056 | 0,8243 ± 0,0122 |
| `noise_coherent_k2`, seed 42 | 1,000 | 1,000 | **0,953** | 0,985 | 0,998 | 1,000 | 0,770 | **0,9580** | **0,8754** |
- Điều kiện (1) nhóm noise tăng rõ rệt: 0,673 -> 0,953 (+0,28; so với seed 42 gốc +0,318). **Thỏa.**
- Điều kiện (2) không nhóm nào giảm quá biến thiên: macro R@1 +0,042 (khoảng 7,5 độ lệch chuẩn của 3 seed), nhận đúng@FMR +0,051 (khoảng 4,2 độ lệch chuẩn); cut/codec/pitch/tempo không giảm; eq 0,992 -> 0,985 (chênh khoảng 3 truy vấn trên 400) là không đáng kể. **Thỏa.**
- Quyết định (**P3-01**): chọn `noise_coherent_k2` làm cấu hình cuối. Cần thêm seed 43, 44 để có 3 seed và thấy biến thiên của chính cấu hình này; kết luận cuối dựa trên 3 seed, không trên một seed.
- Bằng chứng ủng hộ giả thuyết giao thoa pha: huấn luyện chỉ bằng nhiễu TỔNG HỢP coherent, không dùng nhiễu thật nào, đã nâng R@1 trên nhiễu THẬT của Val từ 0,635 lên 0,953. Giới hạn: scalar k được chọn bằng truy vấn nhiễu thật của Val (xem P3-02).

### 19.5. Danh sách ISSUE ghi nhận ở cuối Phase 3
| # | Vấn đề | Mức | Quyết định |
|---|---|---|---|
| P3-01 | Augmentation nhiễu cộng công suất không giống nhiễu thật (R@1 thật 0,29-0,965 so với 1,000), mô hình không bền với nhiễu thật | High | Đã xử lý: nhiễu coherent k=2, noise 0,673 -> 0,953 trên Val |
| P3-02 | k được chọn trên truy vấn nhiễu thật của Val; bốn vòng tuning trên Val (ablation, k, biến thể) | Medium | Chỉ chọn scalar k, vòng cuối. Kết luận về độ bền nhiễu chỉ dựa trên Test (Phase 4), nêu rõ trong báo cáo |
| P3-03 | `overlay` (trộn nhạc khác ở 5 dB) chỉ 0,75-0,77, M2 đạt 0,970 | Medium | Giữ ra khỏi huấn luyện đúng thiết kế; báo cáo như hạn chế. M5 (M2+M3) là hướng tùy chọn, chưa làm |
| P3-04 | Nhận định "time_warp giúp tempo" không được ablation ủng hộ | Low | Đã chỉnh cách viết (mục 19.2) |
| P3-05 | Ablation chỉ một seed; đọc theo biến thiên giữa seed (0,0056 macro) | Low | Chỉ diễn giải chênh lệch lớn |
| P3-06 | Bảng tổng hợp Ô 15 trước đó trống do thiếu file cục bộ | Low | Đã xử lý: đồng bộ `results/` về local rồi chạy lại |

### 19.6. Phần 3F: bộ tổng hợp nhiều seed cho biến thể và Ô 20
| File | Thay đổi |
|---|---|
| `src/evaluation/summarize_m3.py` | Biến thể có nhiều seed không còn ghi đè nhau: lưu `per_seed`, `mean_std`, `delta_mean_vs_baseline_mean`; bảng markdown thêm dòng `name (n=… seed, mean ± std)` |
| `tests/test_m3_3e.py` | Thêm 1 test (nhiều seed của biến thể) |
| notebook | Ô 20: seed 43, 44 của `noise_coherent_k2` (huấn luyện + đánh giá Val + chép kết quả) |

**Cách chạy:** local `pytest -q` mong đợi **131 passed** (130 + 1; sandbox: 56 test của các file 3A-3F pass). Colab: chép gói 3F, tar lại `src/`, Ô 3, rồi Ô 20 (khoảng 60 phút huấn luyện cộng vài phút đánh giá). Sau đó tải `results/` về local, chạy `python -m src.evaluation.summarize_m3`, dán `reports/tables/m3_val_summary.md`. Mình KHÔNG dự đoán số của seed 43, 44.
**Điều kiện đóng Phase 3:** có đủ 3 seed của `noise_coherent_k2`; macro R@1 và nhận đúng@FMR trung bình ± độ lệch chuẩn; xác nhận lại nhóm noise và rằng không nhóm nào tụt. Nếu seed 43/44 không tái lập mức cải thiện (ví dụ macro R@1 của một seed thấp hơn trung bình gốc), dừng lại và báo mình trước khi sang Phase 4.

### 19.7. Phase 4 (xem trước; chưa viết code)
Cấu hình cuối cố định = `noise_coherent_k2`, `best.pt` của 3 seed; `targets.yaml` giữ nguyên. Việc chính: sinh bộ truy vấn robustness cho Test (hiện hai CLI từ chối `--split test`, sẽ cần một cờ mở có kèm khóa chạy một lần), đánh giá M0/M1/M2/M3 trên Test với tau lấy từ Val, bootstrap cặp đôi theo bài nguồn cho hiệu M3 - M1, báo FMR từng cấu hình kèm CI, phân tích lỗi, độ trễ và dung lượng chỉ mục. Test chỉ chạy đúng một lần, sau khi cấu hình cuối được đóng băng.

## 20. Kiểm tra trước khi chốt Phase 3 (kết quả đối chiếu ghi chú 3F)
`pytest -q` local = 131 passed. Ghi chú của bạn nêu "hoàn tất 2 seed (43, 44)" và "tất cả điều kiện đã đạt". Mình đối chiếu với bảng tổng hợp và thấy MỘT điểm chưa khớp nên chưa chốt:

**Bảng chỉ có n = 2 seed, không phải 3.** Bằng chứng từ chính bảng của bạn:
- Dòng đơn lẻ của biến thể ghi `noise_coherent_k2 (seed 43)`; bộ tổng hợp lấy seed NHỎ NHẤT có mặt làm dòng đại diện, nên seed 42 không có trong thư mục `experiments/results/` của bạn.
- Trung bình n = 2 là macro R@1 0,9632 với seed 43 là 0,9636, suy ra seed còn lại khoảng 0,9628. Nếu hai seed là 42 và 43 thì trung bình phải là (0,9580 + 0,9636)/2 = 0,9608, không phải 0,9632. Vậy hai seed trong bảng là 43 và 44.
- Nguyên nhân khả dĩ nhất (mình suy luận, chưa kiểm chứng): file `e5_robust_val_m3_abl_noise_coherent_k2_seed42.json` được tạo ở Ô 19 trên Colab nhưng Ô 19 (do mình viết) không có lệnh `cp` về Drive; khi phiên Colab đổi, file mất. Lỗi nằm ở notebook của mình. Checkpoint seed 42 vẫn nằm trên Drive (`--out-dir` trỏ vào Drive), nên chỉ cần đánh giá lại.

**Những nhận định khác trong ghi chú, đã đối chiếu:**
| Nhận định | Kết quả |
|---|---|
| Seed 43, 44 tái lập cải thiện của seed 42 | Đúng: seed 43 macro R@1 0,9636, nhận đúng@FMR 0,8992, noise 0,980; seed 44 khoảng 0,9628 / 0,879 / 0,970 (mình suy từ trung bình n=2, chưa phải số đọc trực tiếp) |
| Không nhóm cut/codec/pitch/tempo nào sụt | Đúng theo bảng: 1,000 / 1,000 / 0,998 / 1,000 (trung bình n=2) |
| eq và overlay | eq 0,993 (Δ +0,001), overlay 0,778 (Δ +0,026): không tụt |
| Đủ 3 seed để chốt | **Chưa đủ**: thiếu seed 42 trong bảng |

**Ước tính (chỉ để biết trước, KHÔNG dùng làm số báo cáo):** nếu thêm seed 42 (0,9580 / 0,8754), trung bình 3 seed khoảng macro R@1 0,9615 ± 0,0030 và nhận đúng@FMR 0,8845 ± 0,0128. Số chính thức lấy từ `summarize_m3` sau Ô 21.

**Việc cần làm để chốt (theo thứ tự):**
1. Chép gói 3F-2 vào repo, `pytest -q` (mong đợi **135 passed**; sandbox: 60 test của các file 3A-3F pass).
2. Colab: tar lại `src/`, Ô 3, rồi **Ô 21** (đánh giá lại seed 42 từ `best.pt` trên Drive, chép đủ file về Drive).
3. Tải `results/` về `experiments/results/` (local); đặt ba checkpoint vào `experiments/runs/m3_abl_noise_coherent_k2_seed{42,43,44}/best.pt` (chép từ Drive; mỗi file vài MB).
4. Chạy `python -m src.evaluation.summarize_m3` (kỳ vọng dòng `noise_coherent_k2 (n=3 seed …)`) và `python -m src.evaluation.freeze_phase3`.
5. Dán log của bước 4. Phase 3 chỉ chuyển XONG khi `freeze_phase3` in `ĐÓNG BĂNG ĐẠT`.

**`freeze_phase3` kiểm tra gì (mỗi mục PASS/FAIL, ghi `e3_final_freeze.json`):** với từng seed: huấn luyện chỉ dùng train/val (`test_touched=False`), chạy đủ epoch, đúng biến thể; đánh giá Val đủ 7 nhóm, không bỏ cấu hình; checkpoint có mặt, cấu hình nhiễu bên trong là `mode=coherent, coherent_k=2`, seed khớp, cùng `norm_std`. Chung: cùng số truy vấn Val, đủ 3 seed trong bộ tổng hợp, trung bình Val vượt mức tham chiếu (chỉ thông tin), chưa có file kết quả Test nào, và ghi SHA256 của `targets.yaml`, `m3.yaml`, `m3_ablations.yaml`, `baseline.yaml`, `features.yaml` và ba `best.pt`. Phase 4 sẽ gọi `verify_freeze()` để dừng nếu bất kỳ file nào đổi sau đóng băng (kể cả `targets.yaml`).
Giới hạn: công cụ chỉ chứng minh các file hiện tại khớp lúc đóng băng, không chứng minh `targets.yaml` giống hệt bản đã khóa ở Phase 2 (mình không có bản đó); hãy `git commit` và gắn tag `phase3-final` ngay sau khi đóng băng đạt.

**Quyết định nhất quán cho báo cáo (không đổi):** cấu hình cuối chọn theo quy tắc đặt trước; k chọn bằng truy vấn nhiễu thật của Val nên độ bền nhiễu chỉ kết luận trên Test; `overlay` (0,75-0,78) là hạn chế đã biết.

## 21. Chốt Phase 3 (kết quả đóng băng)
Nguồn: log `freeze_phase3` và `m3_val_summary.md` do bạn chạy. `pytest -q` local = 135 passed.

**Đóng băng đạt:** mọi mục PASS, `e3_final_freeze.json` ghi `ok = true`. Một chi tiết nhỏ: log bạn dán có 36 dòng PASS, còn công cụ tạo ra 37 mục kiểm tra; thiếu dòng "seed43: đủ 7 nhóm, không bỏ cấu hình" (seed 42 và 44 đều có). Vì `ok` chỉ bằng true khi mọi mục đạt và bộ tổng hợp thấy đủ 3 seed nên mình xem đây là dòng bị rơi khi dán, không phải kết quả thiếu; nếu muốn chắc, đếm `len(checks)` trong `e3_final_freeze.json` (mong đợi 37, tất cả `ok: true`).

**Cấu hình cuối trên Val (3 seed, mean ± std; seed 42/43/44):**
| | cut | codec | noise | eq | pitch | tempo | overlay | macro R@1 | macro nhận đúng@FMR |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 0,987 | 0,980 | 0,183 | 0,165 | 0,532 | 0,968 | 0,540 | 0,6221 | 0,1068 |
| M2 | 1,000 | 1,000 | 0,995 | 1,000 | 0,056 | 0,508 | 0,970 | 0,7898 | 0,7093 |
| M3 gốc (3 seed) | 1,000 | 1,000 | 0,673 | 0,992 | 0,996 | 0,998 | 0,752 | 0,9158 ± 0,0056 | 0,8243 ± 0,0122 |
| **M3 `noise_coherent_k2` (3 seed)** | 1,000 | 1,000 | **0,968** | 0,990 | 0,998 | 1,000 | 0,775 | **0,9615 ± 0,0030** | **0,8845 ± 0,0129** |
Số chính thức khớp ước tính trước đó (0,9615 ± 0,0030; 0,8845 ± 0,0128). Mức cải thiện so với cấu hình gốc: macro R@1 +0,046, nhận đúng@FMR +0,060, noise +0,295; không nhóm nào tụt đáng kể (eq -0,002).

**Hạn chế đưa vào báo cáo:** k chọn bằng truy vấn nhiễu thật của Val (độ bền nhiễu chỉ kết luận trên Test); `overlay` 0,775 so với M2 0,970; bốn vòng tuning trên Val; ablation một seed; Val chỉ 200 bài nguồn mỗi loại. Bài học đáng ghi: cấu hình augmentation nhiễu cộng công suất cho R@1 = 1,000 trên nhiễu tổng hợp nhưng 0,29-0,965 trên nhiễu thật; phải kiểm chứng augmentation bằng dữ liệu biến đổi thật.
