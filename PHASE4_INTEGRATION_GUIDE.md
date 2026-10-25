# PHASE 4 - INTEGRATION GUIDE (Evaluation & Robustness trên Test)
Cập nhật: 2026-10-27 · **Phase 4 ĐANG LÀM**: 4A (cổng Test một lần + chạy Test) và 4B (kết luận, bootstrap, bảng chính) code + test xong, CHƯA chạm Test. 4C (phân tích lỗi, độ trễ) và 4D (lớp luật T2) chưa làm.

## 1. Kế hoạch chia phần
| Phần | Nội dung | Trạng thái |
|---|---|---|
| 4A | Cổng bảo vệ Test (`test_gate.py`), vá `run_robustness.py` cho `--split test --final-test`, lưu điểm unknown để tính FMR | Code + test xong |
| 4B | `test_verdict.py`: quy tắc ĐẠT/KHÔNG ĐẠT, bootstrap cặp đôi theo bài nguồn, FMR kèm CI, bảng chính | Code + test xong |
| 4C | Phân tích lỗi (theo nhóm/SNR/thể loại), độ trễ và dung lượng chỉ mục kèm cấu hình máy | Chưa làm |
| 4D | Lớp luật giấy phép T2 + test đơn vị (cần sửa `license_norm.py` cho tên giấy phép dạng văn bản) | Chưa làm |

## 2. Quy tắc đặt trước (CỐ ĐỊNH trước khi chạm Test; không đổi sau khi thấy kết quả)
- Mô hình đánh giá: M0, M1, M2 và **M3 = cấu hình cuối `noise_coherent_k2`, 3 seed (42, 43, 44), `best.pt`**. Cấu hình gốc của M3 và các ablation KHÔNG chạy trên Test (tránh nhìn Test thêm lần nào).
- tau của mỗi mô hình lấy NGUYÊN từ file Val (`e5_robust_val_<nhãn>.json`), không chọn lại. Mỗi seed của M3 dùng tau Val của chính nó.
- Bộ truy vấn Test: cùng 17 cấu hình / 7 nhóm như Val, **400 bài known + 400 bài unknown** làm nguồn (Val dùng 200+200; tăng để CI chặt hơn, chọn trước khi thấy kết quả), nhiễu/biến đổi sinh từ audio Test, bài trộn lấy từ Train.
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
