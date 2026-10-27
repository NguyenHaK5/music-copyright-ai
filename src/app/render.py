"""Chuỗi hiển thị cho giao diện (thuần Python, kiểm thử được). Không dùng từ cấm: "hợp lệ", "không có bản quyền", "được phép dùng"."""
from __future__ import annotations

from src.app.pipeline import IN, OUT, UNSURE

STYLE = {IN: ("success", "Thuộc danh mục tham chiếu"),
         OUT: ("info", "Ngoài danh mục tham chiếu"),
         UNSURE: ("warning", "Không xác định – cần người xem xét")}
BANNER = ("Đây là phân loại kỹ thuật theo DANH MỤC THAM CHIẾU của đồ án (nhạc giấy phép Creative Commons từ FMA), "
          "không phải kết luận pháp lý.")


def label_caption(label: str) -> tuple[str, str]:
    return STYLE[label]


def explain(label: str, score: float, tau_low: float, tau_high: float) -> str:
    if label == IN:
        return f"Điểm giống {score:.3f} ≥ ngưỡng {tau_high:.3f}: đoạn âm thanh khớp một bản ghi trong danh mục."
    if label == OUT:
        return f"Điểm giống cao nhất chỉ {score:.3f} < {tau_low:.3f}: không có bản ghi nào trong danh mục đủ giống."
    return (f"Điểm giống {score:.3f} nằm giữa {tau_low:.3f} và {tau_high:.3f}: chưa đủ bằng chứng để kết luận, "
            "nên chuyển người xem xét (có thể do nhiễu hoặc đoạn quá ngắn).")


def rights_lines(rights: dict | None) -> list[str]:
    """Các dòng mô tả giấy phép (chỉ khi nhãn THUỘC_DANH_MỤC). Cảnh báo nổi bật khi có NC hoặc ND."""
    if not rights:
        return []
    lines = [f"**{rights['result']}**"]
    if rights.get("license"):
        lines.append(f"Giấy phép (theo metadata): {rights['license']}")
        lines += [f"- {c}" for c in rights["conditions"]]
        lines.append(f"Dùng thương mại: {rights['commercial_use']}")
        lines.append(f"Tác phẩm phái sinh: {rights['derivatives']}")
        if any("NonCommercial" in c for c in rights["conditions"]):
            lines.append("⚠ Giấy phép giới hạn phi thương mại: nhận diện đúng bài KHÔNG có nghĩa là được dùng tự do.")
    elif rights.get("raw") is not None:
        lines.append(f"Chuỗi giấy phép trong metadata: {rights['raw']}")
    return lines


def top_rows(top: list[dict]) -> list[dict]:
    return [{"Hạng": t["rank"], "Bài": t["title"] or f"track {t['track_id']}", "Nghệ sĩ": t["artist"] or "-",
             "Thể loại": t["genre"] or "-", "Điểm giống": round(t["score"], 3)} for t in top]
