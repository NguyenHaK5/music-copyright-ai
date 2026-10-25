"""Sinh Data Card (markdown) từ e0_smoke_check.json và e0_eda.json. Thiếu số -> 'CHƯA CHẠY'.

Chạy:  python -m src.data.data_card
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.config import load_config
from src.common.paths import resolve

NA = "CHƯA CHẠY"


def _g(d: dict | None, *keys, default=NA):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt_counts(d) -> str:
    return NA if d == NA else (", ".join(f"{k}: {v}" for k, v in d.items()) or "không có")


def render_data_card(smoke: dict | None, eda: dict | None) -> str:
    return f"""# Data Card - FMA small (danh mục tham chiếu mô phỏng có kiểm soát)

> Mọi con số lấy từ `experiments/results/e0_smoke_check.json` và `e0_eda.json`. "{NA}" nghĩa là chưa có số đo.
> Đây là phân loại kỹ thuật, không phải kết luận pháp lý.

| Mục | Nội dung |
|---|---|
| Nguồn | Free Music Archive (repo chính thức mdeff/fma), tập `small`, kiểm SHA1 |
| Số bài theo metadata | {_g(eda, 'n_tracks_metadata')} |
| Số bài đạt sau làm sạch + khử trùng | {_g(eda, 'n_tracks_ok')} |
| Lý do loại | {_fmt_counts(_g(eda, 'drop_reason_counts'))} |
| Số nghệ sĩ / album (bài đạt) | {_g(eda, 'n_artists_ok')} / {_g(eda, 'n_albums_ok')} |
| Số bài mỗi nghệ sĩ (min/median/max) | {_g(eda, 'tracks_per_artist', 'min')} / {_g(eda, 'tracks_per_artist', 'median')} / {_g(eda, 'tracks_per_artist', 'max')} |
| Thể loại | {_fmt_counts(_g(eda, 'genre_counts'))} |
| Họ giấy phép | {_fmt_counts(_g(eda, 'license_family_counts'))} |
| Thời lượng giây (min/median/max) | {_g(eda, 'duration_sec', 'min')} / {_g(eda, 'duration_sec', 'median')} / {_g(eda, 'duration_sec', 'max')} |
| Cặp nghi trùng gần (gắn cờ) | {_g(eda, 'near_pairs_flagged')} (cùng nghệ sĩ: {_g(eda, 'near_pairs_same_artist')}) |
| Số nhóm trùng có >1 bài | {_g(eda, 'n_dup_groups_gt1')} |
| Cổng smoke check | {_fmt_gates(smoke)} |
| Audio thật | Có |
| Nhãn T1 | `known`/`unknown` do người làm đồ án gán theo split |
| Nhãn T2 | Giấy phép Creative Commons thật từ metadata FMA |
| Giới hạn | Toàn bộ nhạc giấy phép CC (vẫn có bản quyền); clip 30 s; chủ yếu nhạc quốc tế; không có cover/live thật; không có nhạc thương mại |
"""


def _fmt_gates(smoke: dict | None) -> str:
    if not smoke or "gates" not in smoke:
        return NA
    return "; ".join(f"{g['gate']}={'PASS' if g['passed'] else 'FAIL'}" for g in smoke["gates"])


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    dcfg = load_config("data")
    res = resolve(dcfg["paths"]["results_dir"])
    text = render_data_card(_load(res / "e0_smoke_check.json"), _load(res / "e0_eda.json"))
    out = resolve(dcfg["interim"]["data_card"]); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Đã ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
