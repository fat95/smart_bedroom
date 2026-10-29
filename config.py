"""Hằng số dùng chung cho cả nhóm.

!!! BẢN TẠM do A viết để chạy được asr.py / commands.py trước khi C chốt config.py chính thức.
!!! C: khi làm config.py thật, hãy giữ nguyên tên và cấu trúc DEVICES (hoặc báo A để sửa commands.py).

Quy ước giá trị `action` (dùng chung cho commands.py, decision.py và index.html):
    on / off          bật / tắt (không có giá trị)
    up / down         tăng / giảm một lượng `value` (so với trạng thái hiện tại)
    set               đặt thẳng về `value`
    channel           chuyển kênh `value` (chỉ tivi)
    mode_cool / mode_fan / mode_dry   đổi chế độ điều hòa (value = None)
Khoảng giá trị hợp lệ nằm trong DEVICES[thiết bị]["actions"][hành động]["range"] (None = không có giá trị).
"""
from __future__ import annotations

# ---------------------------------------------------------------- âm thanh
SAMPLE_RATE = 16000
RECORD_SECONDS = 4

# ---------------------------------------------------------------- mô hình ASR
# Mặc định hệ thống dùng bản small; base vẫn có thể chọn làm phương án nhẹ hơn trên CPU.
PHOWHISPER_DEMO = "vinai/PhoWhisper-small"
PHOWHISPER_EVAL = "vinai/PhoWhisper-small"
ASR_MODELS = {          # tên ngắn (dùng ở dòng lệnh và báo cáo) -> id trên Hugging Face
    "phowhisper-base": "vinai/PhoWhisper-base",
    "phowhisper-small": "vinai/PhoWhisper-small",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
}
ASR_DEFAULT = "phowhisper-small"

# ---------------------------------------------------------------- ngưỡng
COMMAND_THRESHOLD = 0.75   # điểm khớp mờ tối thiểu (0..1) để chấp nhận một lệnh
SPEAKER_THRESHOLD = 0.5    # KHÔNG DÙNG: ngưỡng người nói nằm trong models/speaker_model.joblib (train_speaker.py chọn).
                           # Muốn ép tay thì đặt biến môi trường SPEAKER_THRESHOLD khi chạy app.

# ---------------------------------------------------------------- thiết bị
# id thiết bị khớp với index.html: ac, light-main, bed, tv.
# keywords: từ khóa nhận ra thiết bị (viết KHÔNG DẤU, chữ thường).
# range: (min, max) cho `value`; None nghĩa là hành động không có giá trị.
DEVICES = {
    "ac": {
        "label": "Điều hòa",
        "keywords": ["dieu hoa", "may lanh", "dieu hoa nhiet do"],
        "actions": {
            "on":        {"range": None},
            "off":       {"range": None},
            "up":        {"range": (1, 5)},     # tăng 1-5 độ, kết quả phải nằm trong 16-30
            "down":      {"range": (1, 5)},
            "set":       {"range": (16, 30)},
            "mode_cool": {"range": None},
            "mode_fan":  {"range": None},
            "mode_dry":  {"range": None},
        },
        "limits": (16, 30),
    },
    "light-main": {
        "label": "Đèn trần",
        "keywords": ["den tran", "den"],
        "actions": {
            "on":   {"range": None},
            "off":  {"range": None},
            "up":   {"range": (1, 100)},
            "down": {"range": (1, 100)},
            "set":  {"range": (0, 100)},
        },
        "limits": (0, 100),
    },
    "bed": {
        "label": "Giường nâng đầu",
        "keywords": ["dau giuong", "giuong nang", "giuong"],
        "actions": {
            "up":   {"range": (1, 60)},
            "down": {"range": (1, 60)},
            "set":  {"range": (0, 60)},         # 0 = nằm phẳng; dashboard hiện cho 0-60 độ
        },
        "limits": (0, 60),
    },
    "tv": {
        "label": "Tivi",
        "keywords": ["tivi", "ti vi", "tv", "truyen hinh", "television"],
        "actions": {
            "on":      {"range": None},
            "off":     {"range": None},
            "channel": {"range": (1, 9)},
            "up":      {"range": (1, 100)},     # âm lượng
            "down":    {"range": (1, 100)},
            "set":     {"range": (0, 100)},
        },
        "limits": (0, 100),
    },
}


def value_in_range(device: str, action: str, value) -> bool:
    """True nếu `value` hợp lệ cho (thiết bị, hành động). Hành động không cần giá trị thì value phải là None."""
    rng = DEVICES[device]["actions"][action]["range"]
    if rng is None:
        return value is None
    return value is not None and rng[0] <= value <= rng[1]


# Danh sách người đăng ký / người lạ KHÔNG khai báo ở đây:
#   - người đăng ký = tên người trong dataset/ (speaker.py tự đọc từ tên file)
#   - người lạ      = tên la, la1, la2, ... hoặc tên ghi trong strangers.txt
