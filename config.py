# config.py
RECORD_SECONDS = 4

# Danh sách thiết bị và khoảng giá trị hợp lệ
DEVICES = {
    "điều hòa": {"min": 16, "max": 30},
    "đèn trần": {"min": 0, "max": 100},  # % độ sáng
    "giường": {"min": 0, "max": 100},    # độ nghiêng
    "tivi": {"min": 0, "max": 100}       # mức âm lượng hoặc kênh
}

ENROLLED_SPEAKERS = ["speaker_1", "speaker_2", "speaker_3"] 
STRANGERS = ["la_1", "la_2"]
SPEAKER_VERIFICATION_THRESHOLD = 0.75