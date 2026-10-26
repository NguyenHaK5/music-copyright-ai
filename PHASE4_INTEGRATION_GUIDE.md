# PHASE 4 - INTEGRATION GUIDE (Evaluation & Robustness trên Test)
Cập nhật: 2026-10-29 · **Phase 4 XONG về nội dung** (chờ bạn chép bản vá luật giấy phép v2 và gắn tag `phase4-final`): Test ĐẠT, 4C đã chạy (mục 12), 4D đã chạy độ phủ và được vá (mục 13). Tiếp theo Phase 5 (demo).

## 1. Kế hoạch chia phần
| Phần | Nội dung | Trạng thái |
|---|---|---|
| 4A | Cổng bảo vệ Test (`test_gate.py`), vá `run_robustness.py` cho `--split test --final-test`, lưu điểm unknown để tính FMR | **XONG** (đã chạy Test) |
| 4B | `test_verdict.py`: quy tắc ĐẠT/KHÔNG ĐẠT, bootstrap cặp đôi theo bài nguồn, FMR kèm CI, bảng chính | **XONG**: ĐẠT |
| 4C | Phân tích lỗi (theo cấu hình, Val-Test, thể loại), độ trễ và dung lượng chỉ mục | **XONG** (đã chạy; mục 12) |
| 4D | Lớp luật giấy phép T2 (`src/rights/license_rules.py`, `configs/license_rules.yaml`) + test đơn vị | **XONG** (đã chạy độ phủ; vá v2; mục 13) |

## 2. Quy tắc đặt trước (CỐ ĐỊNH trước khi chạm Test; không đổi sau khi thấy kết quả)
- Mô hình đánh giá: M0, M1, M2 và **M3 = cấu hình cuối `noise_coherent_k2`, 3 seed (42, 43, 44), `best.pt`**. Cấu hình gốc của M3 và các ablation KHÔNG chạy trên Test (tránh nhìn Test thêm lần nào).
- tau của mỗi mô hình lấy NGUYÊN từ file Val (`e5_robust_val_<nhãn>.json`), không chọn lại. Mỗi seed của M3 dùng tau Val của chính nó.
- Bộ truy vấn Test: cùng 17 cấu hình / 7 nhóm như Val, nhiễu/biến đổi sinh từ audio Test, bài trộn lấy từ Train. Kế hoạch ban đầu: **400 bài known + 400 bài unknown** (Val dùng 200+200). **Thực tế đã chạy: 200+200 (xem mục 7.2, lệch giao thức).**
- Chỉ số chính: macro Recall@1. M3 = trung bình 3 seed. **ĐẠT** khi (M3 - M1) >= gain trong `targets.yaml` VÀ cận dưới CI 95% của hiệu > 0 (bootstrap cặp đôi theo bài nguồn, 1.000 lần, seed 0). Chỉ số phụ: macro nhận đúng@FMR, cùng quy tắc, báo riêng. Báo thêm từng seed có đạt gain không (không ảnh hưởng kết luận).
- FMR trên Test (tau cố định từ Val) chỉ báo cáo kèm CI, không gán ĐẠT/KHÔNG ĐẠT. M0, M2 chỉ để tham khảo; M3 không bắt buộc vượt M2.
- Test chạy đúng MỘT lần: mỗi nhãn chỉ ghi `e5_robust_test_<nhãn>.json` một lần (không ghi đè); bộ truy vấn Test không sinh lại. Nếu một lệnh thất bại giữa chừng (chưa ghi file) thì chạy lại được; không chạy lại để "thử số khác".
- Kết quả KHÔNG ĐẠT vẫn được báo cáo trung thực kèm phân tích.

## 3. Phần 4A: những gì đã có
| File | Vai trò |
|---|---|
| `src/evaluation/test_gate.py` | `check_test_allowed(...)` trả danh sách lý do TỪ CHỐI: cần `--final-test`; `freeze_phase3` đạt và `verify_freeze` không báo file nào đổi (kể cả `targets.yaml`, `m3.yaml`, `baseline.yaml`, ba checkpoint); mô hình thuộc {m0, m1, m2, m3}; m3 chỉ với `--tag abl_noise_coherent_k2_seed<N>`; không ghi đè kết quả Test; có tau Val; không sinh lại bộ truy vấn Test |
| `src/run_robustness.py` | Vá: cho phép `--split test --final-test` qua cổng trên; tau Test = tau Val của đúng nhãn; lưu điểm top-1 của truy vấn unknown (`__unk_top1`, `__unk_src`) vào file perquery để tính FMR kèm CI; ghi `tau_source`; ghi chú "KẾT QUẢ TEST (một lần)". Chép ĐÈ nguyên file (là bản vá trên file đã vá ở 3D-2) |
| `tests/test_phase4_ab.py` | 10 test (có cả các test của 4B) |
`run_baselines.py` giữ nguyên: vẫn từ chối `--split test` (truy vấn sạch không cần trên Test).

## 4. Phần 4B: kết luận trên Test
| File | Vai trò |
|---|---|
| `src/evaluation/test_verdict.py` | `per_source_macro`, `paired_bootstrap`, `compare`, `fmr_pooled`, `load_gains`, `build_verdict`, `to_markdown`; CLI ghi `experiments/results/e4_test_verdict.json` và `reports/tables/test_main_table.md` |
- Hiệu M3 - M1 được tính trên từng BÀI NGUỒN (mỗi bài có đủ 17 truy vấn ở 7 nhóm; nhóm là trung bình các cấu hình, macro là trung bình 7 nhóm), M3 là trung bình 3 seed theo từng bài; bootstrap lấy mẫu lại bài nguồn.
- `load_gains` tìm khóa chứa "gain" trong `targets.yaml` (khóa chứa "recall" cho R@1; "correct"/"accept"/"nhan" cho nhận đúng); mình CHƯA thấy cấu trúc file đó nên có bước kiểm tra bắt buộc bên dưới; nếu không đọc được, truyền `--gain-r1 0.0945 --gain-ca 0.2233` (hai giá trị ghi trong `phase-tracker.md`).
- Bảng chính gồm M0/M1/M2, M3 (mean ± std) và từng seed: macro, 7 nhóm, FMR gộp kèm CI; dưới bảng là bảng kết luận ĐẠT/KHÔNG ĐẠT.

## 5. Cách chạy
**Bước 0 (bắt buộc, chưa chạm Test):**
```
pytest -q                                                   # local mong đợi 145 passed (135 + 10); sandbox: 70
python -c "from src.evaluation.test_verdict import load_gains; print(load_gains('configs/targets.yaml'))"
python -m src.evaluation.freeze_phase3                      # vẫn phải ĐÓNG BĂNG ĐẠT sau khi chép gói này
```
Dòng thứ hai phải in `gain_recall_at_1` = 0,0945 và `gain_correct_accept` = 0,2233 (khớp tracker). Nếu in số khác hoặc báo lỗi: DỪNG và dán kết quả cho mình, không chạy Test. Chép gói này chỉ đổi code, không đổi file đã băm; nếu `freeze_phase3` báo khác, dừng.
Commit mã nguồn (`git commit`) trước bước 1.

**Bước 1-4 (chạy Test, đúng một lần; local CPU đủ, audio phải có sẵn như ở Val):**
```
python -m src.run_robustness build --split test --final-test --limit-sources 400
python -m src.run_robustness eval --split test --final-test --models m0 m1 m2
python -m src.run_robustness eval --split test --final-test --models m3 --m3-ckpt experiments/runs/m3_abl_noise_coherent_k2_seed42/best.pt --tag abl_noise_coherent_k2_seed42
python -m src.run_robustness eval --split test --final-test --models m3 --m3-ckpt experiments/runs/m3_abl_noise_coherent_k2_seed43/best.pt --tag abl_noise_coherent_k2_seed43
python -m src.run_robustness eval --split test --final-test --models m3 --m3-ckpt experiments/runs/m3_abl_noise_coherent_k2_seed44/best.pt --tag abl_noise_coherent_k2_seed44
python -m src.evaluation.test_verdict
```
Kết quả mong đợi: `build` in 17/17 cấu hình và 800 bài nguồn (có thể ít hơn nếu Test không đủ; thoát mã 3 nếu có cấu hình bị bỏ qua: DỪNG, báo mình trước khi chạy `eval`); mỗi `eval` in một dòng `tau`, macro và 7 nhóm; `test_verdict` in bảng chính và kết luận. Thời gian ước (chưa đo): build khoảng 6 phút; mỗi seed M3 khoảng 4-5 phút (Val: dựng chỉ mục 81 s, 11,6 ms/truy vấn trên CPU; Test có 13.600 truy vấn). Mình KHÔNG dự đoán kết quả.
Dán lại: log từng lệnh, `e4_test_verdict.json` và `reports/tables/test_main_table.md`.

## 6. Rủi ro 4A/4B
- Cổng Test chặn nhầm lẫn (ghi đè, sai mô hình, thay đổi cấu hình), không chặn một người cố tình xóa file kết quả để chạy lại; việc giữ kỷ luật "một lần" vẫn do bạn.
- Chưa chạy vá `run_robustness.py` với dữ liệu thật: mình test cổng và bộ kết luận bằng dữ liệu tổng hợp; nhánh `eval` ghi unknown top-1 và dùng tau Val chỉ kiểm bằng đọc code và cú pháp (các module `cache`, `robust_queries`, ... không có trong sandbox). Nếu lệnh Test đầu tiên báo lỗi KeyError/đường dẫn, dán nguyên văn trước khi chạy lại.
- `load_gains` đoán khóa trong `targets.yaml`; đã có bước 0 để bắt trường hợp đoán sai trước khi chạm Test.
- Test có thể không đủ 400 bài mỗi loại (800 known, 797 unknown sau chia nên đủ; nếu `select_sources` hành xử khác, `build` sẽ báo).
- Hiệu M3 - M1 mang tính một lần trên một bộ truy vấn Test; CI chỉ phản ánh lấy mẫu bài nguồn, không phản ánh biến thiên seed huấn luyện (xem ± std của 3 seed).
- Còn lại: 4C (phân tích lỗi, độ trễ, dung lượng chỉ mục), 4D (lớp luật T2), rồi Phase 5 (demo Streamlit) và Phase 6 (báo cáo, slide).

## 7. Kết quả Test (một lần; nguồn: log bạn dán, `e4_test_verdict.json`, `test_main_table.md`)
### 7.1. Kiểm tra trước khi chạy
- tau trong `targets.yaml` khớp tau trong các file Val: m0 0,999984; m1 0,944159; m2 0,034091 (cả hai nguồn trùng đến 6 chữ số).
- `load_gains` đọc đúng `gain_recall_at_1` = 0,0945 và `gain_correct_accept` = 0,2233 từ `targets.m3_vs_m1_min_gain.*`.
- `freeze_phase3`: 37/37 PASS trước khi chạy Test; `pytest` 145 passed.

### 7.2. Lệch giao thức đã ghi nhận
Giao thức đặt trước là 400 bài known + 400 bài unknown làm nguồn; thực tế Test được dựng với **200 + 200** (log: "400/400 bài nguồn"; `n_sources` = 200 bài known trong kết quả; 6.800 truy vấn mỗi mô hình = 17 x 400, nếu 400+400 sẽ là 13.600). Nguyên nhân khả dĩ: bộ truy vấn được dựng bằng `python -m src.data.robust_queries --split test` (theo ghi chú của bạn), không qua `run_robustness build ... --limit-sources 400` của guide (lệnh đó cũng là số bài MỖI vai trò, nên dùng giá trị mặc định 200). Hệ quả: (a) cổng `--final-test` không áp cho bước dựng bộ truy vấn (bước `eval` vẫn qua cổng); (b) khoảng tin cậy rộng hơn dự kiến nhưng vẫn rất hẹp so với biên đạt: cận dưới CI +0,349 so với gain 0,0945; (c) không ảnh hưởng chiều của kết luận. **Không dựng lại**: dựng lại sau khi đã thấy kết quả sẽ phá nguyên tắc một lần. Báo cáo phải ghi: Test dùng 200 bài known + 200 bài unknown làm nguồn (cùng cỡ với Val).

### 7.3. Bảng chính (Test; tau cố định từ Val)
| Mô hình | macro R@1 | macro nhận đúng@FMR | cut | codec | noise | eq | pitch | tempo | overlay | FMR gộp [CI95] |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 0,0000 | 0,0000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,0100 [0,0000; 0,0250] |
| M1 | 0,5939 | 0,0965 | 0,983 | 0,920 | 0,152 | 0,168 | 0,435 | 0,975 | 0,525 | 0,0176 [0,0097; 0,0279] |
| M2 | 0,7883 | 0,7137 | 1,000 | 1,000 | 0,997 | 1,000 | 0,044 | 0,497 | 0,980 | 0,0059 [0,0032; 0,0088] |
| **M3 (3 seed, mean ± std)** | **0,9635 ± 0,0037** | **0,8832 ± 0,0129** | 1,000 | 1,000 | 0,971 | 0,988 | 0,998 | 1,000 | 0,787 | 0,0099 |
| M3 seed 42 | 0,9597 | 0,8711 | 1,000 | 1,000 | 0,957 | 0,982 | 0,999 | 1,000 | 0,780 | 0,0050 [0,0009; 0,0112] |
| M3 seed 43 | 0,9670 | 0,8968 | 1,000 | 1,000 | 0,987 | 0,990 | 0,998 | 1,000 | 0,795 | 0,0138 [0,0047; 0,0271] |
| M3 seed 44 | 0,9638 | 0,8816 | 1,000 | 1,000 | 0,970 | 0,992 | 0,999 | 1,000 | 0,785 | 0,0109 [0,0032; 0,0212] |

### 7.4. Kết luận theo quy tắc đặt trước (M3 = trung bình 3 seed, so với M1)
| Chỉ số | M1 | M3 | Hiệu (CI95, bootstrap theo bài nguồn) | Gain | Kết luận |
|---|---|---|---|---|---|
| macro Recall@1 (chính) | 0,5939 | 0,9635 | +0,3696 [+0,3495; +0,3883] | 0,0945 | **ĐẠT** |
| macro nhận đúng@FMR (phụ) | 0,0965 | 0,8832 | +0,7867 [+0,7635; +0,8102] | 0,2233 | **ĐẠT** |
Từng seed riêng lẻ cũng đều đạt gain ở cả hai chỉ số (hiệu so với M1: +0,366 / +0,373 / +0,370 cho R@1).

### 7.5. Đọc kết quả (trung thực)
- Số Test khớp số Val, không có dấu hiệu quá khớp Val: M3 macro R@1 0,9635 (Test) so với 0,9615 (Val); nhóm noise 0,971 so với 0,968. Đáng chú ý nhất: k của nhiễu coherent được chọn bằng truy vấn nhiễu thật của Val, vậy kết quả noise trên Test (bài mới, nhiễu mới) là bằng chứng độc lập cho cải thiện này.
- M1 giảm nhẹ trên Test (0,5939 so với 0,6221); M2 ổn định (0,7883 so với 0,7898). Gain không bị "bơm" bởi việc M1 yếu đi bất thường: ngay cả với M1 ở mức Val, hiệu vẫn gần +0,34.
- M3 không thắng ở mọi nhóm: M2 vẫn tốt hơn ở noise (0,997 so với 0,971), eq (1,000 so với 0,988) và rõ nhất ở overlay (0,980 so với 0,787); M3 thắng ở pitch (0,998 so với 0,044) và tempo (1,000 so với 0,497). Về macro nhận đúng@FMR M3 hơn M2 (0,883 so với 0,714). Hai mô hình bổ trợ nhau (hướng M5, chưa làm).
- FMR gộp với tau lấy từ Val: M3 trung bình 0,99% (khoảng 0,5-1,4% theo seed), M2 0,59%, M1 1,76% (CI 0,97-2,79%, cao hơn mục tiêu 1%). tau Val không bảo đảm FMR đúng 1% trên dữ liệu mới, đúng như đã nêu; CI của seed 43 lên tới 2,7%. FMR của M0 vô nghĩa (tau gần 1; không có điểm nào đủ cao nên R@1 bằng 0).
- Giới hạn: một bộ truy vấn Test (200+200 bài nguồn); CI chỉ phản ánh lấy mẫu bài nguồn, không gồm biến thiên seed huấn luyện (std 0,0037 ghi riêng); số Test là kết quả của in-index crop và biến đổi nhân tạo, không phải bản cover/live thật (mục hạn chế báo cáo).

## 8. Phần 4C: phân tích lỗi và chi phí
| File | Vai trò |
|---|---|
| `src/evaluation/error_analysis.py` | Từ kết quả Test đã có (không chấm lại): bảng theo cấu hình (M1/M2/M3 mean ± std), chênh Val -> Test của M3 theo cấu hình (cờ khi |Δ| > 0,05), bảng theo thể loại (n < 15 gắn cờ), 5 cấu hình yếu nhất; ghi `e4_error_analysis.json`, `reports/tables/test_error_analysis.md` |
| `src/evaluation/measure_cost.py` | E9: dựng chỉ mục, độ trễ 1 truy vấn (trung vị, p95), ms/truy vấn khi chấm lô, dung lượng chỉ mục (byte mảng numpy), kèm cấu hình máy; đo trên VAL (không chạm Test); ghi `e9_cost.json`, `reports/tables/cost.md` |
| `tests/test_phase4c.py` | 4 test |
Giới hạn: file perquery không lưu ID dự đoán nên không có "ví dụ sai cụ thể" trên Test; nếu cần ví dụ sai cho báo cáo, làm trên Val.

## 9. Phần 4D: lớp luật giấy phép T2
| File | Vai trò |
|---|---|
| `configs/license_rules.yaml` | Bảng luật có version 1: câu chữ kết quả, điều kiện (BY/NC/ND/SA), disclaimer |
| `src/rights/license_rules.py` | `parse_license` (nhận dạng chuỗi văn bản lẫn URL CC; CC0, Public Domain Mark, Free Art License), `describe(raw, status, rules)`, `coverage`, CLI độ phủ |
| `tests/test_license_rules.py` | 26 test (tham số hóa): các tổ hợp CC, CC0/PDM/Free Art, các ca UNKNOWN, không xuất thông tin khi không phải THUỘC_DANH_MỤC, không có từ cấm trong mọi đầu ra |
- Chỉ chạy khi nhãn là `THUỘC_DANH_MỤC`; kết quả là "Giấy phép mở (kèm điều kiện)" kèm tên giấy phép, điều kiện, mô tả dùng thương mại/phái sinh, hoặc "UNKNOWN – cần xác minh" (thiếu, không nhận ra, thiếu Attribution, hay mâu thuẫn ND+SA). Không bao giờ xuất "không có bản quyền", "hợp lệ", "được phép dùng". Disclaimer: chỉ mô tả giấy phép của bản ghi theo metadata, không phải kết luận pháp lý, không gồm quyền đối với tác phẩm.
- Không sửa `src/data/license_norm.py` (giữ nguyên test cũ); module mới độc lập. Tracker ghi cột license của FMA là tên văn bản nên cả 8.000 bài từng thành `unknown` ở bộ chuẩn hóa cũ.
- Mình CHƯA thấy các giá trị `license_raw` thật (đã xin hai lần chưa nhận được); bộ nhận dạng dựa trên từ khóa, nên cần chạy CLI độ phủ để xem còn bao nhiêu chuỗi rơi vào UNKNOWN và là chuỗi nào.

## 10. Cách chạy 4C và 4D (không chạm Test thêm)
```
pytest -q                                                  # local mong đợi 175 passed (145 + 30); sandbox: 100
python -m src.evaluation.error_analysis                    # đọc kết quả Test đã có
python -m src.evaluation.measure_cost                      # đo trên Val; vài phút CPU (dựng chỉ mục M3 khoảng 80 s trên máy trước đó)
python -m src.rights.license_rules                         # độ phủ giấy phép trên tracks.parquet
```
Dán lại: `reports/tables/test_error_analysis.md`, `reports/tables/cost.md`, và `e8_license_coverage.json`.
`measure_cost` dùng `best.pt` seed 42 của cấu hình cuối (đổi bằng `--m3-ckpt`); nếu đường dẫn khác, chỉnh tham số.

## 11. Rủi ro 4C/4D và việc còn lại
- 4C và 4D chưa chạy trên dữ liệu thật; test dùng dữ liệu tổng hợp. `error_analysis` giả định cột `genre` có trong `tracks.parquet` (đúng theo schema ở Phase 1); `measure_cost` gọi `build_model` như `run_baselines` nên phụ thuộc cùng cấu hình `baseline.yaml`.
- Phân tích theo thể loại chỉ có khoảng 200 bài known trên 8 thể loại, mỗi thể loại vài chục bài: chỉ nêu xu hướng rõ rệt, không diễn giải chênh lệch nhỏ.
- Lớp luật T2 chưa nối vào demo (Phase 5). T3 đã cắt (cổng giấy phép FAIL từ Phase 1).
- Còn lại: chạy 4C, 4D; chốt Phase 4 (commit, tag `phase4-final`); Phase 5 (demo Streamlit); Phase 6 (báo cáo, slide).

## 12. Kết quả chạy thật 4C (175 passed local; nguồn `test_error_analysis.md`, `cost.md`)
**Theo cấu hình (Test; M3 = mean ± std qua 3 seed):** M3 đạt R@1 = 1,000 ở cut (3/5/10 s), codec (MP3 64k/128k), EQ lowpass, pitch ±1, tempo 0,90/1,10; yếu nhất:
| Cấu hình | M3 R@1 | M3 nhận đúng@tau | M3 FMR | M2 R@1 |
|---|---|---|---|---|
| overlay_5db | 0,787 ± 0,008 | 0,527 ± 0,003 | 0,000 | 0,980 |
| snr_5db | 0,930 ± 0,038 | 0,543 ± 0,116 | 0,015 | 0,990 |
| eq_highpass_500 | 0,977 ± 0,010 | 0,758 ± 0,068 | 0,007 | 1,000 |
| snr_10db | 0,983 ± 0,010 | 0,927 ± 0,053 | 0,012 | 1,000 |
| pitch_p2 | 0,995 ± 0,000 | 0,925 ± 0,033 | 0,003 | 0,025 |
- **Kiểu thất bại chủ yếu là TỪ CHỐI, không phải nhận nhầm:** ở snr_5db R@1 0,930 nhưng nhận đúng@tau chỉ 0,543; ở overlay_5db 0,787 và 0,527. Tức nhiều truy vấn có bài đúng ở hạng 1 nhưng điểm thấp hơn tau Val, nên rơi vào vùng không đủ chắc chắn (an toàn: chuyển người xem xét). Số nhận nhầm ID tối đa là 1 - R@1 (7% ở 5 dB; 21% ở overlay); mình không có số "nhận nhầm ID" trực tiếp trên Test vì file perquery không lưu ID dự đoán.
- **Chênh lệch Val -> Test của M3 theo cấu hình: tất cả <= 0,012, không cấu hình nào bị gắn cờ** (lớn nhất: overlay_5db +0,012, snr_5db +0,008): kết quả ổn định giữa Val và Test.
- **FMR theo cấu hình (tau Val):** M3 0,003-0,015, ổn định giữa các cấu hình; M2 lên 0,085 ở crop_3s (điểm yếu đã biết P2-02: điểm M2 phụ thuộc độ dài truy vấn); M1 lên 0,105 ở snr_5db.
- **Theo thể loại (macro R@1 theo bài nguồn, 19-30 bài mỗi thể loại, không thể loại nào bị gắn cờ n nhỏ):** M3 từ 0,916 (Folk) đến 0,996 (Pop); thấp nhất: Folk 0,916, Instrumental 0,937, Experimental 0,948; cao nhất: Pop 0,996, Rock 0,991, Hip-Hop 0,985. Với cỡ mẫu này chỉ nên nêu xu hướng nhẹ (Folk/Instrumental thấp hơn), chưa kiểm định chênh lệch; M1 và M2 cũng dao động theo thể loại (M2 0,767-0,812) nên không có thể loại nào yếu bất thường.

**Chi phí (Val, chỉ mục 802 bài; 8 luồng CPU Windows, torch 2.2.2+cpu; benchmark trên máy này):**
| Mô hình | dựng chỉ mục (s) | 1 truy vấn, trung vị / p95 (ms) | ms/truy vấn (lô 200) | chỉ mục (MB) |
|---|---|---|---|---|
| M1 | 12,1 | 1,2 / 1,3 | 0,4 | 10,78 |
| M2 | 6,6 | 2,6 / 3,3 | 2,7 | 21,33 |
| M3 | 68,7 | 12,8 / 14,2 | 11,7 | 4,11 |
- Chỉ mục M3 nhỏ nhất (8.020 cửa sổ x 128 chiều float32 = khoảng 4,1 MB), dù chậm nhất khi chấm (khoảng 13 ms/truy vấn trên CPU). Đủ nhanh cho demo.
- Lưu ý: thời gian chưa gồm giải mã audio và tính log-mel (truy vấn đã là log-mel); độ trễ đầu-cuối của demo (Phase 5) phải đo riêng. Dung lượng chỉ tính mảng numpy của chỉ mục, không tính trọng số mô hình (checkpoint M3 vài MB).

## 13. Kết quả chạy thật 4D (nguồn `e8_license_coverage.json`) và bản vá v2
Độ phủ trên 8.000 bài trước vá: 63 bài UNKNOWN (0,79%). Chuỗi không nhận ra: `Public Domain` 40, `Noncommercial Sampling Plus` 8, `Art Libre` 6, `None` 5, `Sampling Plus` 3, `Music Sharing` 1.
- **Vá v2 (`license_rules.yaml` version 2):** nhận `Public Domain` trơn (kind `pd_claim`, nhãn "Phạm vi công cộng (theo khai báo của metadata, cần xác minh)") và `Art Libre` (Free Art License). Sau vá, UNKNOWN còn 17 bài (0,21%; mình tính từ danh sách trên, sẽ xác nhận bằng lần chạy độ phủ lại).
- **Cố ý để UNKNOWN:** `Sampling Plus`/`Noncommercial Sampling Plus` (giấy phép CC đã ngừng, điều khoản khác CC thường nên không dám ánh xạ vào BY/NC/ND/SA), `Music Sharing` trơn (không rõ biến thể), `None` (thiếu). Nguyên tắc: bảng luật chỉ nhận cái nó mô tả chính xác, còn lại UNKNOWN + cần xác minh.
- **Thống kê từ chính độ phủ (mình tính tay từ bảng bạn dán):** khoảng 82% bài (6.553/8.000) có điều kiện NC (phi thương mại) và khoảng 35% có ND (không phái sinh); CC0 và Public Domain Mark chỉ 72 bài. Ý nghĩa cho demo và báo cáo: "giấy phép mở" ở đây phần lớn KHÔNG cho dùng thương mại, nên giao diện phải hiện điều kiện nổi bật, và đây là minh chứng cụ thể cho việc nhận diện đúng bài chưa nghĩa là được dùng tự do.
- Tests: 33 test (26 cũ + 7 mới) cho luật giấy phép; local mong đợi 182 passed; sandbox 107.

## 14. Chốt Phase 4
Chép bản vá (`license_rules.py`, `license_rules.yaml`, `test_license_rules.py`), `pytest -q` (mong đợi 182 passed), chạy lại `python -m src.rights.license_rules` (mong đợi `unknown_fraction` khoảng 0,0021 = 17/8000), rồi `git commit` và gắn tag `phase4-final`.
**Đã đạt ở Phase 4:** E4 (ngưỡng chọn trên Val), E5 (ma trận robustness Test, bootstrap, FMR kèm CI), E6 (ablation, ở Phase 3), E8 (T2 + test đơn vị), E9 (độ trễ/dung lượng). **Cắt có chủ đích:** T3 (cổng giấy phép FAIL từ Phase 1), M4, M5, FastAPI. **Chưa có:** ví dụ sai cụ thể trên Test (perquery không lưu ID dự đoán); nếu báo cáo cần, làm một công cụ nhỏ trên Val (xin xác nhận).
**Hạn chế ghi vào báo cáo (Phase 4):** Test dựng với 200+200 bài nguồn (lệch kế hoạch 400+400); crop và biến đổi nhân tạo, không có cover/live thật; FMR trên Test chỉ gần mục tiêu 1% (M1 1,76%); overlay M3 0,787 so với M2 0,980.
