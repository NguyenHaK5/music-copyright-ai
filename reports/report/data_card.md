# Data Card - FMA small (danh mục tham chiếu mô phỏng có kiểm soát)

> Mọi con số lấy từ `experiments/results/e0_smoke_check.json` và `e0_eda.json`. "CHƯA CHẠY" nghĩa là chưa có số đo.
> Đây là phân loại kỹ thuật, không phải kết luận pháp lý.

| Mục | Nội dung |
|---|---|
| Nguồn | Free Music Archive (repo chính thức mdeff/fma), tập `small`, kiểm SHA1 |
| Số bài theo metadata | 8000 |
| Số bài đạt sau làm sạch + khử trùng | 7992 |
| Lý do loại | too_short: 3, decode_error: 3, silent: 2 |
| Số nghệ sĩ / album (bài đạt) | 2307 / 2462 |
| Số bài mỗi nghệ sĩ (min/median/max) | 1 / 1.0 / 208 |
| Thể loại | Pop: 1000, International: 1000, Folk: 999, Experimental: 999, Rock: 999, Electronic: 999, Instrumental: 999, Hip-Hop: 997 |
| Họ giấy phép | unknown: 7992 |
| Thời lượng giây (min/median/max) | 29.96 / 29.98 / 30.0 |
| Cặp nghi trùng gần (gắn cờ) | 77 (cùng nghệ sĩ: 60) |
| Số nhóm trùng có >1 bài | 47 |
| Cổng smoke check | artist_id_missing_rate=PASS; n_artists=PASS; min_artists_per_genre=PASS; license_families_ge_min_tracks=FAIL; free_disk_gb=PASS |
| Audio thật | Có |
| Nhãn T1 | `known`/`unknown` do người làm đồ án gán theo split |
| Nhãn T2 | Giấy phép Creative Commons thật từ metadata FMA |
| Giới hạn | Toàn bộ nhạc giấy phép CC (vẫn có bản quyền); clip 30 s; chủ yếu nhạc quốc tế; không có cover/live thật; không có nhạc thương mại |
