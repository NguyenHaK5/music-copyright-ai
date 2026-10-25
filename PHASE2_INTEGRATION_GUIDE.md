# PHASE 2 - INTEGRATION GUIDE (Baselines)
Cập nhật: 2026-10-20 · **Phase 2 XONG**: 2A-2D đã chạy trên Val thật, `targets.yaml` đã khóa (mục 9-11). Bản chốt. Tài liệu cập nhật sau mỗi phần; bản chốt khi xong 2D.

## 1. Phần 2A: metric, open-set, bootstrap
| File | Vai trò | Yêu cầu GV |
|---|---|---|
| `src/evaluation/metrics.py` | `rank_of_true`, `ranks`, `hit_indicators`, `recall_at_k`, `reciprocal_ranks`, `mrr` | 3 |
| `src/evaluation/openset.py` | `fmr`, `known_outcomes` (đúng ID / sai ID / bị từ chối), `roc_auc`, `eer`, `choose_tau_high`, `choose_tau_low`, `decide_labels`, hằng số nhãn 3 trạng thái | 3 |
| `src/evaluation/bootstrap.py` | `bootstrap_ci`, `bootstrap_diff_ci` (bootstrap theo cụm = bài nguồn, CI cặp đôi) | 3 |
| `tests/test_metrics.py`, `tests/test_openset.py` | 15 test, đáp án tính tay | 4 |

Nếu `src/evaluation/__init__.py` chưa có trong repo, chép file rỗng đi kèm.

## 2. Điểm nối (API nội bộ)
- Chấp nhận khi `score >= tau`. Điểm cao hơn = giống hơn (mọi mô hình phải trả điểm theo chiều này).
- Ngưỡng: `tau_high = choose_tau_high(unknown_scores_val, max_fmr)`; `tau_low = choose_tau_low(known_correct_scores_val, max_loss)`; `decide_labels(scores, tau_low, tau_high)`.
- Metric known/unknown không phụ thuộc tỷ lệ lớp: FMR, EER, ROC-AUC. F1 không là số chính.
- Bootstrap chỉ cho metric dạng trung bình của chỉ báo từng truy vấn (`hit_indicators`, `reciprocal_ranks`). Cụm = `track_id` của bài nguồn.
- `eer` dùng FNMR = 1 - tỷ lệ nhận đúng ID (gồm cả bị từ chối và nhận nhầm ID); EER trả điểm quét có |FMR-FNMR| nhỏ nhất, không nội suy.

## 3. Cách chạy (PowerShell)
```
pytest -q
```
Kết quả đã xác nhận trên máy bạn: **75 passed** (32 Phase 1 + 15 + 9 + 6 + 13), ffmpeg có trong PATH nên không có test bị skip.

## 4. Đóng góp báo cáo
Chương 2/3 (giao thức đánh giá): định nghĩa FMR, FNMR, ba kết quả known, quy tắc chọn ngưỡng trên Val. Chưa có số liệu.

## 5. Rủi ro và việc còn lại
- 2A là hàm thuần, chưa đụng dữ liệu thật; lỗi tích hợp (định dạng điểm, thứ tự truy vấn) chỉ lộ ra ở 2B.
- `choose_tau_low` dùng quantile `lower`; với ít mẫu known-đúng, tỷ lệ rơi dưới ngưỡng có thể nhỏ hơn `max_loss` (an toàn, không vượt).
- Mức FMR 1% và tỷ lệ mất 5% là đề xuất, chốt ở 2D sau khi có M0/M1 trên Val.
- Còn lại: 2B (truy vấn sạch + khung truy hồi + M0 + M1), 2C (M2), 2D (bảng baseline + `targets.yaml`).

## 6. Phần 2B: truy vấn sạch, M0, M1, bộ chấm điểm
| File | Vai trò | Yêu cầu GV |
|---|---|---|
| `configs/baseline.yaml` | Tham số truy vấn, M1, ngưỡng (`sr`, `hop_length` phải khớp `features.yaml`) | 1 |
| `src/data/queries.py` | `sample_query_specs` (đặc tả theo khung và giây, xác định theo seed), `QueryMels` (nạp lười từ cache) | 1, 3 |
| `src/features/handcrafted.py` | `mel_stats_features` (MFCC, delta, thống kê dải mel; 336 chiều), `window_starts` | 2 |
| `src/models/base.py`, `m0_trivial.py`, `m1_classical.py` | Giao diện chung; M0 điểm ngẫu nhiên; M1 cosine trên đặc trưng chuẩn hóa, điểm bài = max qua các cửa sổ 5 s (bước 2,5 s), scaler fit trên Train | 2 |
| `src/evaluation/runner.py` | `evaluate_scores`: Recall@k, MRR (CI bootstrap theo bài), AUC, EER, ngưỡng, 3 kết quả known, nhãn 3 trạng thái | 3 |
| `src/run_baselines.py` | CLI; **từ chối `--split test`**; ghi `experiments/results/e1_<model>_val.json` | 3, 4 |
| `tests/test_baselines_2b.py` | 9 test (tổng hợp): xác định, không rò rỉ giao thức, hòa điểm bảo thủ, M0 gần ngẫu nhiên, M1 phân biệt được | 4 |

Nếu `src/models/__init__.py` chưa có, chép file rỗng đi kèm.

**Điểm nối cho 2C/2D/3:**
- Mọi mô hình triển khai `build_index(track_ids, mel_loader)` và `score_matrix(queries) -> (n_queries, n_tracks)`, điểm cao = giống hơn, cột theo thứ tự `track_ids`.
- Bộ truy vấn sạch dùng chung lưu ở `data/interim/queries_val_clean.parquet` (cột `query_id, source_track_id, is_known, start_frame, n_frames, start_sec, dur_sec`). M2 dùng `start_sec`, `dur_sec` để cắt audio; M3 dùng `start_frame`, `n_frames`. Xóa file này để sinh lại; đổi seed/tham số thì phải xóa.
- Hòa điểm tính bất lợi cho bài đúng (hạng = số bài có điểm >= điểm bài đúng), nên fingerprint trả điểm 0 hàng loạt không được lợi.
- Thêm mô hình mới: viết lớp theo giao diện, thêm nhánh vào `build_model` trong `run_baselines.py`.

**Cách chạy (PowerShell):**
```
pytest -q                                              # 75 passed
python -m src.run_baselines --split val --models m0 m1
```
Kết quả mong đợi: mỗi mô hình một dòng `R@1, R@5, MRR, AUC, EER` và file JSON. M0 phải gần ngẫu nhiên (`chance_recall_at_1` = 1/số bài chỉ mục, khoảng 0,0012 với 802 bài). Mình KHÔNG dự đoán số của M1 trên dữ liệu thật. Thiếu `tracks.parquet` hoặc cache thì lệnh thoát mã 2 với "CHƯA CHẠY"; `--split test` thoát mã 2 (đã kiểm). Trong sandbox mình chạy được CLI end-to-end trên dữ liệu tổng hợp, không phải FMA thật.

**Quyết định lệch Phase 0:** M1 chỉ dùng đặc trưng suy ra từ log-mel đã cache (không có chroma, spectral contrast, ZCR) để tránh giải mã lại ~8.000 file và để M1 cùng đầu vào với M3. Chroma có thể thêm làm ablation.

**Đóng góp báo cáo:** Chương 2: giao thức truy vấn, định nghĩa xếp hạng bảo thủ, mô tả M0/M1. Chương 3: bảng baseline Val (số lấy từ `e1_*_val.json`, chưa có).

**Rủi ro 2B:**
- Truy vấn known là crop của clip đã index và cửa sổ chỉ mục chồng lên đoạn đó, nên M1/M2 có thể đạt điểm rất cao trên truy vấn sạch; đây là định nghĩa bài toán chứ không phải rò rỉ, nhưng đừng diễn giải thành hiệu năng thực tế. Sức phân biệt thật nằm ở truy vấn biến đổi và unknown (Phase 4).
- Ngưỡng và FMR trong JSON được chọn và báo cáo trên cùng tập Val, nên FMR đạt mục tiêu theo cấu trúc; không dùng làm kết quả cuối.
- `sr`, `hop_length` trong `baseline.yaml` trùng với `features.yaml`; nếu bạn đã đổi `features.yaml` thì phải đổi theo (mình chưa thấy file đó).
- Chưa chạy trên cache thật: thời gian dựng chỉ mục M1 cho ~800 bài và fit scaler trên 1.500 bài Train chưa đo (ước vài phút). Dán `timing_sec` trong JSON.
- Còn lại: 2C (M2 fingerprint), 2D (bảng baseline, chốt `targets.yaml`).

**Kết quả chạy thật 2B (Val, nguồn `e1_m0_val.json`, `e1_m1_val.json`):** M0 R@1 = 0,0017 (CI 0,0004-0,0037, chứa mức ngẫu nhiên 0,0012: hợp lý). M1 R@1 = 0,9967, R@5 = 1,0, MRR = 0,9982, AUC = 0,9195, EER = 0,1674; tại FMR <= 1% (tau_high = 0,955) M1 chỉ nhận đúng 34,6% truy vấn known (65,4% bị từ chối, 0% nhận nhầm ID). Nhận xét: M1 truy hồi gần hoàn hảo trên truy vấn sạch nhưng từ chối bài lạ kém; đó là khoảng trống để M2/M3 cải thiện. Thời gian M1: dựng chỉ mục 3,1 s, chấm 4803 truy vấn 2,7 s.

## 7. Phần 2C: M2 fingerprint tự cài
| File | Vai trò | Yêu cầu GV |
|---|---|---|
| `src/features/peaks.py` | `extract_peaks` (cực đại cục bộ cao hơn nền cục bộ >= `prominence_db`, giới hạn đỉnh/giây), `hashes_from_peaks` (cặp neo-đích, hash 20 bit) | 2 |
| `src/models/m2_fingerprint.py` | `M2Fingerprint`: chỉ mục băm đã sắp; truy vấn bỏ phiếu theo (bài, độ lệch thời gian); điểm = phiếu cực đại / số băm truy vấn, trong [0,1] | 2 |
| `src/run_baselines.py`, `configs/baseline.yaml` | Thêm nhánh `m2`, khối `m2:`, ghi `index_hashes` vào JSON | 3 |
| `tests/test_m2_fingerprint.py` | 6 test: đóng gói hash, ràng buộc vùng đích, phổ phẳng không có đỉnh, bất biến độ lợi, nhận đúng/từ chối trên dữ liệu tổng hợp, lỗi khi chưa dựng chỉ mục | 4 |

**Thuật toán:** đỉnh trên log-mel -> mỗi điểm neo ghép với tối đa `fan_out` điểm đích sau 1..63 khung và lệch <= `df_max` dải -> hash `(f_neo<<13)|(f_dich<<6)|dt` -> chỉ mục đảo (mảng sắp + `searchsorted`) -> truy vấn: tra băm khớp, bỏ phiếu theo `(bài, t_chỉ_mục - t_truy_vấn)`, điểm bài = đỉnh histogram.

**Điểm nối:** cùng giao diện `build_index` / `score_matrix` và cùng bộ truy vấn sạch như M0/M1; điểm trong [0,1], cao = giống. Phase 4 chỉ cần tạo truy vấn biến đổi (audio -> log-mel) rồi gọi lại `score_matrix`.

**Cách chạy (PowerShell):**
```
pytest -q                                              # mong đợi: 62 passed
python -m src.run_baselines --split val --models m2
```
Kết quả mong đợi: một dòng `[m2] R@1 ... AUC ... EER ...` và `e1_m2_val.json` (có `index_hashes`, `timing_sec`). Mình KHÔNG dự đoán số trên FMA thật. Trong sandbox, trên phổ tổng hợp 100 bài x 1292 khung: dựng chỉ mục 0,5 s (~2.200 băm/bài), 300 truy vấn 0,3 s; số này không đại diện cho FMA thật.

**Quyết định lệch Phase 0:** M2 tính trên log-mel đã cache (độ phân giải thời gian 23 ms, tần số theo thang mel) thay vì STFT dựng lại từ audio, để dùng chung cache và tránh phụ thuộc đường dẫn audio. Hệ quả: dải cao kém phân giải hơn STFT; đã ghi vào hạn chế.

**Rủi ro 2C:**
- Tham số `m2:` (đặc biệt `prominence_db`, `max_peaks_per_sec`, `fan_out`) là đề xuất, chưa tinh chỉnh trên FMA thật. Nếu `index_hashes`/bài quá thấp (< vài trăm) hoặc quá cao (> vài chục nghìn), báo để chỉnh. Chỉ chỉnh trên Val.
- `delta_bin = 1` yêu cầu căn thời gian khớp từng khung; với tempo 0,9-1,1 (Phase 4) fingerprint dự kiến thất bại (điểm mù đã nêu ở Phase 0), không phải lỗi.
- Truy vấn sạch là crop của clip đã index nên M2 dự kiến rất cao ở đây; sức phân biệt thật nằm ở truy vấn biến đổi và unknown.
- Thông số `frames_per_sec` được truyền từ `sr / hop_length` của `baseline.yaml`.
- Còn lại: 2D (bảng baseline Val, chốt `targets.yaml`).

**Kết quả chạy thật 2C (Val, `e1_m2_val.json`):** M2 R@1 = R@5 = MRR = AUC = 1,0, EER = 0; chỉ mục 1.777.523 băm (~2.216 băm/bài, nằm trong khoảng hợp lý), chấm 4.803 truy vấn 25,9 s (5,4 ms/truy vấn). Điểm của M2 bão hòa trên truy vấn sạch vì truy vấn là crop của chính clip đã index; tau_high = 0,031 (tỷ lệ băm cùng độ lệch) vẫn tách hoàn toàn known/unknown. Kết luận: truy vấn sạch không còn phân biệt được M2 với M3 sau này, nên cần bộ truy vấn biến đổi (2D).

## 8. Phần 2D: truy vấn robustness trên Val và chốt `targets.yaml`
| File | Vai trò | Yêu cầu GV |
|---|---|---|
| `src/data/transforms.py` | Biến đổi audio: nhiễu theo SNR, lowpass/highpass, trộn theo SNR, pitch shift, time stretch, vòng MP3 qua ffmpeg | 1, 3 |
| `src/data/robust_queries.py` | 17 cấu hình / 7 nhóm (cắt 3/5/10 s; MP3 64/128k; SNR 5/10/20 dB; lowpass 3 kHz, highpass 500 Hz; pitch ±1, ±2; tempo 0,90/1,10; trộn 5 dB). Mỗi bài nguồn: giải mã một lần, cùng đoạn gốc s0 cho mọi cấu hình, peak-normalize riêng từng truy vấn, ghi `data/interim/robust/robust_val_<cfg>.npz` + manifest | 1, 3 |
| `src/evaluation/robustness.py` | `choose_tau_pooled` (tau chọn từ unknown GỘP mọi cấu hình, FMR gộp <= 1%), `evaluate_robust` (tau cho trước dùng lại ở Test), macro theo 7 nhóm | 3 |
| `src/evaluation/make_targets.py` | Sinh và khóa `configs/targets.yaml` theo quy tắc cố định; từ chối khi thiếu nhóm hoặc đã khóa | 3 |
| `src/run_robustness.py` | CLI `build` / `eval`; chỉ `--split val` | 3, 4 |
| `tests/test_robustness_2d.py` | 13 test: SNR đúng, bộ lọc, độ dài tempo, tái lập, chọn tau gộp, chặn rò rỉ unknown, chặn macro thiếu nhóm | 4 |

**Giao thức:** 200 bài known + 200 bài unknown của Val làm nguồn (có thể ít hơn nếu Val không đủ); chỉ mục là 802 bài known của Val như 2B. Với mỗi mô hình, MỘT tau duy nhất cho mọi cấu hình. Số chính: **macro Recall@1** và **macro tỷ lệ nhận đúng tại FMR gộp <= 1%**, tính theo 7 nhóm rồi lấy trung bình (mỗi nhóm trung bình các cấu hình của nó). Nhóm `cut` gồm `crop_5s` là truy vấn sạch đi qua đường audio.

**Quy tắc `targets.yaml` (chốt trước khi chạm Test):** `gain = max(0,02; 0,25 x (1 - giá trị M1 trên Val))` cho từng chỉ số macro; M3 chỉ "ĐẠT" khi vượt M1 tối thiểu `gain` VÀ khoảng tin cậy bootstrap cặp đôi (theo bài nguồn) của hiệu có cận dưới > 0. M2 chỉ để tham khảo. Vector từng truy vấn (`*_perquery.npz`) được lưu để Phase 4 tính bootstrap cặp đôi.

**Điểm nối cho Phase 3/4:** `evaluate_robust(scores_by_cfg, tau=None|float)`; M3 chỉ cần cài `build_index`/`score_matrix`, thêm nhánh trong `build_model`. Ở Phase 4, Test dùng `tau` đã lưu trong `e5_robust_val_<model>.json`.

**Cách chạy (PowerShell):**
```
pytest -q
ffmpeg -version
python -m src.run_robustness build --split val --limit-sources 20   # chạy thử
python -m src.run_robustness build --split val                      # chạy đủ (ước 3-10 phút)
python -m src.run_robustness eval --split val --models m0 m1 m2
python -m src.evaluation.make_targets
```
Kết quả mong đợi: `build` in số cấu hình xong (17/17) và thoát mã 0; thoát mã 3 nếu có cấu hình bị bỏ qua (vd thiếu ffmpeg); `eval` in mỗi mô hình một dòng macro và 7 điểm theo nhóm; `make_targets` in các mục tiêu và khóa file. Thiếu đầu vào thì thoát mã 2 "CHƯA CHẠY". Trong sandbox, `build` cho 20 bài nguồn mất 9 s trên audio tổng hợp (~0,45 s/bài nguồn; máy bạn có thể chậm hơn). Mình KHÔNG dự đoán số trên FMA thật.

**Rủi ro 2D:**
- `audio_dir` là giả định (`data/raw/fma_small`); loader thử `000/000002.mp3` rồi tìm đệ quy. Sai thư mục thì lệnh thoát mã 2.
- Thiếu ffmpeg trên Windows: nhóm `codec` bị bỏ; `make_targets` từ chối, tránh macro 6 nhóm bị so với macro 7 nhóm.
- Truy vấn robustness peak-normalize riêng từng truy vấn, còn chỉ mục dùng log-mel chuẩn hóa đỉnh theo cả clip: M1 có thể kém hơn ở `crop_5s` so với kết quả 2B. Đó là hiệu ứng thật của đường vào audio.
- Nhiễu là nhiễu trắng; bộ lọc Butterworth bậc 4; tempo chỉ 0,90/1,10; trộn dùng nguồn từ bài Train ở mức 5 dB. Các lựa chọn này là đề xuất; nói rõ trong báo cáo.
- 200 bài nguồn mỗi loại: số truy vấn mỗi cấu hình khá nhỏ, nên đừng đọc chênh lệch vài phần trăm giữa các cấu hình riêng lẻ; số macro và bootstrap ở Phase 4 mới là căn cứ.
- Test chưa bị chạm: mọi lệnh từ chối `--split test`.
- Sau 2D: đã chạy thật và khóa `targets.yaml` (mục 9-12); bước tiếp theo là Phase 3 (M3).

## 9. Kết quả chạy thật 2D (Val; nguồn `e5_robust_val_m{0,1,2}.json`, `configs/targets.yaml`)
- `build`: 17/17 cấu hình, 400 bài nguồn (200 known + 200 unknown), 6.800 truy vấn; ffmpeg có nên đủ 7 nhóm. Có một cảnh báo libmpg123 `dequantization failed` ở một bài nguồn (khung mp3 hỏng, không dừng chương trình).
- Số mình đã tự kiểm tra lại từ JSON: macro R@1 M1 = 0,6221, M2 = 0,7898; macro nhận đúng@FMR M1 = 0,1068, M2 = 0,7093; FMR gộp của M1 và M2 đều = 0,0100; gain = 0,0945 và 0,2233, các ngưỡng tham chiếu 0,7166 và 0,3301 khớp công thức.

| Mô hình | cut | codec | noise | eq | pitch | tempo | overlay | macro R@1 | macro nhận đúng@FMR |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,0000 | 0,0000 |
| M1 | 0,987 | 0,980 | 0,183 | 0,165 | 0,533 | 0,968 | 0,540 | 0,6221 | 0,1068 |
| M2 | 1,000 | 1,000 | 0,995 | 1,000 | 0,056 | 0,508 | 0,970 | 0,7898 | 0,7093 |

Nhận xét (số đo thật, chưa phải kết luận cuối vì mới là Val):
- M1 gần như hoàn hảo ở cut/codec/tempo, sụp ở nhiễu (R@1 0,05-0,40) và EQ (0,12-0,21); pitch 0,20-0,76 tùy hướng; overlay 0,54.
- `crop_5s` qua đường audio của M1 đạt 0,99 (so với 0,9967 truy vấn sạch trên cache): hiệu ứng chuẩn hóa đỉnh riêng từng truy vấn nhỏ, rủi ro đã nêu ở 2D không xảy ra đáng kể.
- M2 mạnh ở mọi nhóm trừ pitch (0,035-0,085) và tempo (0,47-0,55): đúng điểm mù đã dự báo.
- Khoảng trống cho M3: noise, eq, pitch, overlay (M1 yếu) và nhận đúng@FMR (M1 chỉ 0,107).

## 10. Chốt cách xét ĐẠT/KHÔNG ĐẠT cho Phase 4 (không sửa `targets.yaml`, chỉ làm rõ cách đọc)
`targets.yaml` có cả mức tăng (`gain`) lẫn mức tuyệt đối (`min_macro_*`). Quy ước, cố định từ đây:
1. **Chỉ số chính:** macro Recall@1 trên Test (7 nhóm, một tau lấy từ Val). **ĐẠT** khi (M3 - M1) đo trên chính Test >= `gain` (0,0945) VÀ cận dưới CI 95% bootstrap cặp đôi (theo bài nguồn, 1.000 lần) của hiệu > 0.
2. **Chỉ số phụ:** macro nhận đúng@FMR, cùng quy tắc với `gain` 0,2233; báo trạng thái riêng, không quyết định kết luận chung.
3. Các số `min_macro_*` là tham chiếu suy từ Val (M1_Val + gain), không dùng làm tiêu chí trên Test.
4. FMR trên Test (với tau lấy từ Val) chỉ báo cáo kèm CI, không gán ĐẠT/KHÔNG ĐẠT, vì không thể bảo đảm FMR đúng 1% trên dữ liệu mới.
5. M2 và M0 chỉ tham khảo. M3 không bắt buộc vượt M2 (M2 tuyệt đối ở các nhóm M3 khó vượt; điểm mù của M2 là pitch/tempo).
6. Kết quả KHÔNG ĐẠT vẫn được báo cáo trung thực kèm phân tích.

## 11. Bốn quyết định thiết kế M3 (mặc định; báo nếu muốn đổi)
1. **Đầu vào:** crop log-mel 5 s từ cache (không giải mã audio khi huấn luyện), chỉ bài `split==train` (4.794 bài).
2. **Augmentation trên miền log-mel (xấp xỉ):** cộng/trừ độ lợi, nhiễu nền (thêm công suất nhiễu), đường cong EQ kiểu lowpass/highpass, co giãn trục tần số (xấp xỉ pitch) và trục thời gian (xấp xỉ tempo), SpecAugment. Đánh giá vẫn dùng biến đổi THẬT trên miền audio (bộ truy vấn 2D), nên khoảng cách xấp xỉ-thật được đo trung thực.
3. **Họ giữ ra khỏi huấn luyện (ISSUE-05):** `overlay` (chồng âm) và `codec` (nén MP3 lại) không có trong augmentation; báo cáo tách nhóm "đã thấy" và "chưa thấy".
4. **Mô hình và loss:** ResNet nhỏ (4 khối, 16→128 kênh) -> embedding 128 chiều chuẩn hóa L2; học tương phản NT-Xent (hai crop khác nhau, augmentation khác nhau của cùng bài là cặp dương); điểm bài = max cosine trên cửa sổ 5 s của chỉ mục, như M1. Huấn luyện trên Colab, checkpoint mỗi epoch, 3 seed; ablation: bật/tắt từng loại augmentation, độ dài crop, số chiều embedding.

## 12. Danh sách ISSUE phát hiện khi chốt Phase 2
| # | Vấn đề | Mức | Quyết định |
|---|---|---|---|
| P2-01 | M0 cho ma trận ngẫu nhiên GIỐNG HỆT ở cả 17 cấu hình (cùng seed và cùng kích thước nên mọi cấu hình có cùng kết quả) | Thấp | Không sửa (đổi seed sẽ lệch so với số đã khóa trong `targets.yaml`); ghi chú: M0 chỉ là mốc, không đọc từng cấu hình |
| P2-02 | FMR gộp = 1% nhưng không đều theo cấu hình (M2: 11,5% ở crop 3 s; M1: 4,5% ở SNR 5 dB) vì điểm M2 chuẩn hóa theo số băm nên phụ thuộc độ dài truy vấn | Trung bình | Giữ tau gộp (một ngưỡng cho cả hệ thống); báo FMR từng cấu hình trong báo cáo; demo dùng truy vấn >= 5 s |
| P2-03 | `targets.yaml` có cả `gain` và `min_macro_*` nên cách xét ĐẠT mơ hồ | Trung bình | Chốt ở mục 10 |
| P2-04 | Chỉ 200 bài unknown nguồn, mỗi bài 17 truy vấn không độc lập: ước lượng FMR 1% nhiễu | Trung bình | Không đổi baseline; Phase 4 báo FMR kèm CI bootstrap theo bài nguồn |
| P2-05 | Cảnh báo libmpg123 `dequantization failed` ở một bài nguồn | Thấp | Không xử lý; khung mp3 hỏng, truy vấn vẫn sinh được |
| P2-06 | Tập truy vấn in-index nên M2 rất cao | Thấp | Đúng định nghĩa bài toán; nêu trong hạn chế báo cáo |

## 13. Đóng góp báo cáo (Phase 2)
- Chương 2: giao thức truy vấn (clean + 17 cấu hình/7 nhóm), xếp hạng bảo thủ khi hòa điểm, tau gộp, quy tắc chốt mục tiêu (mục 10), mô tả M0/M1/M2.
- Chương 3: hai bảng baseline Val ở mục 9 và `phase-tracker.md`; nêu rõ đây là số Val dùng để đặt mục tiêu, không phải kết quả cuối.
- Hạn chế: P2-01 đến P2-06.
