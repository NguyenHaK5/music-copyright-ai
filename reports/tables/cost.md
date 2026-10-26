| Mô hình | dựng chỉ mục (s) | độ trễ 1 truy vấn, trung vị / p95 (ms) | ms/truy vấn khi chấm cả lô | chỉ mục (MB) |
|---|---|---|---|---|
| m1 | 12.1 | 1.2 / 1.3 | 0.4 | 10.78 |
| m2 | 6.6 | 2.6 / 3.3 | 2.7 | 21.33 |
| m3 | 68.7 | 12.8 / 14.2 | 11.7 | 4.11 |

Máy: Windows-10-10.0.19045-SP0, 8 luồng CPU, torch 2.2.2+cpu (CUDA: False); chỉ mục 802 bài Val, 200 truy vấn crop 5 s.
