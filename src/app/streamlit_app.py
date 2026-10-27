"""Demo Streamlit (Phase 5): Upload -> Kết quả -> Bằng chứng -> Disclaimer.  Chạy: streamlit run src/app/streamlit_app.py"""
from __future__ import annotations

import streamlit as st

from src.app.engine import build_engine
from src.app.pipeline import analyze_bytes
from src.app.render import BANNER, explain, label_caption, rights_lines, top_rows

st.set_page_config(page_title="Nhận diện nhạc theo danh mục tham chiếu", page_icon="🎵", layout="centered")


@st.cache_resource(show_spinner="Đang nạp mô hình và chỉ mục (lần đầu có thể mất vài chục giây)...")
def load_engine():
    return build_engine(log=lambda *_: None)


st.title("🎵 Nhận diện nhạc theo danh mục tham chiếu")
st.warning(BANNER)

try:
    engine = load_engine()
except Exception as e:                                           # thiếu checkpoint/hiệu chuẩn: báo rõ, không crash
    st.error(f"Không khởi động được hệ thống: {e}")
    st.stop()

cfg = engine.cfg
up = st.file_uploader(f"Tải lên tệp âm thanh ({', '.join(cfg.allowed_ext)}; tối đa {int(cfg.max_mb)} MB; tối thiểu {cfg.min_seconds:.0f} giây)",
                      type=[e.lstrip(".") for e in cfg.allowed_ext])
if up is not None:
    data = up.getvalue()
    st.audio(data)
    if st.button("Phân tích", type="primary"):
        with st.spinner("Đang phân tích..."):
            res = analyze_bytes(data, up.name, engine)
        if not res["ok"]:
            st.warning(f"Không phân tích được: {res['message']}")
        else:
            kind, caption = label_caption(res["label"])
            getattr(st, kind)(f"**{caption}**")
            st.write(explain(res["label"], res["top"][0]["score"], res["tau_low"], res["tau_high"]))
            if res["label"] == "THUỘC_DANH_MỤC":
                st.subheader("Giấy phép của bản ghi khớp")
                for line in rights_lines(res["rights"]):
                    st.markdown(line)
            st.subheader("Bằng chứng: các bản ghi giống nhất")
            st.dataframe(top_rows(res["top"]), hide_index=True)
            t = res["timing_ms"]
            st.caption(f"Phân tích {res['analyzed_seconds']:.1f} s âm thanh · giải mã {t['decode']:.0f} ms · log-mel {t['logmel']:.0f} ms · "
                       f"so khớp {t['match']:.0f} ms")
            st.info(res["disclaimer"])

with st.expander("Thông tin hệ thống"):
    i = engine.info
    st.write({"mô hình": i["ckpt"], "SHA256 (12 ký tự)": i["ckpt_sha256"], "số bài trong danh mục": i["n_index_tracks"],
              "số cửa sổ trong chỉ mục": i["index_windows"], "ngưỡng chắc chắn (tau_high)": round(engine.tau_high, 4),
              "ngưỡng loại (tau_low)": round(engine.tau_low, 4), "ngưỡng lấy từ": "Val (không dùng Test)"})
    st.caption("Không lưu âm thanh người dùng: tệp tạm bị xóa ngay sau khi phân tích. Chỉ phân tích tối đa "
               f"{cfg.query_seconds:.0f} giây (file dài hơn lấy đoạn giữa).")
st.caption(BANNER)
