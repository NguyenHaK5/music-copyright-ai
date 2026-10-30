| Mô hình | macro R@1 | macro nhận đúng@FMR | cut | codec | noise | eq | pitch | tempo | overlay |
|---|---|---|---|---|---|---|---|---|---|
| m1 | 0.6221 | 0.1068 | 0.987 | 0.980 | 0.183 | 0.165 | 0.532 | 0.968 | 0.540 |
| m2 | 0.7898 | 0.7093 | 1.000 | 1.000 | 0.995 | 1.000 | 0.056 | 0.508 | 0.970 |
| M3 (n=1 seed, mean ± std) | 0.9095 | 0.8102 | 1.000 | 1.000 | 0.635 | 0.988 | 0.996 | 0.998 | 0.750 |

**Ablation / tuning (Recall@1 theo nhóm; Δ so với `m3_seed` cùng seed)**

| Biến thể | macro R@1 (Δ) | macro nhận đúng@FMR (Δ) | cut | codec | noise | eq | pitch | tempo | overlay |
|---|---|---|---|---|---|---|---|---|---|
