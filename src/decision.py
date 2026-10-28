"""
Module: decision.py
Chức năng chính: Bộ kiểm tra logic và ra quyết định thực thi (Decision Engine).
Nhiệm vụ:
  1. Tiếp nhận thông tin lệnh từ Mô hình ASR (Người A).
  2. Tiếp nhận thông tin người nói từ Mô hình Speaker Verification (Người B).
  3. Kiểm tra tính hợp lệ của thiết bị và khoảng giá trị hoạt động an toàn.
  4. Đánh giá quyền truy cập và độ tin cậy của người phát ra câu lệnh.
  5. Trả về kết quả cho phép thực thi ("ok") hoặc lý do từ chối ("rejected").
"""

from config import DEVICES, ENROLLED_SPEAKERS, SPEAKER_VERIFICATION_THRESHOLD


def decide(command_data, speaker_data):
    """Hàm trung tâm đánh giá tính hợp lệ của lệnh điều khiển và danh tính người nói.

    Tham số:
        command_data (dict): Dữ liệu lệnh gồm 'device', 'action', 'value'.
        speaker_data (dict): Dữ liệu người nói gồm 'speaker_id', 'score'.

    Trả về:
        dict: Quyết định cuối cùng dạng {"status": "ok"|"rejected", ...}
    """

    # BƯỚC 1: Kiểm tra xem có nhận diện được lệnh giọng nói hợp lệ hay không
    if not command_data or not command_data.get("device"):
        return {
            "status": "rejected",
            "reason": "no_speech",
            "message": "Không nhận diện được nội dung câu lệnh.",
        }

    device = command_data.get("device")
    action = command_data.get("action")
    value = command_data.get("value")

    # BƯỚC 2: Kiểm tra thiết bị yêu cầu có nằm trong danh mục hỗ trợ của hệ thống hay không
    if device not in DEVICES:
        return {
            "status": "rejected",
            "reason": "unknown_command",
            "message": f"Thiết bị '{device}' không tồn tại trong cấu hình hệ thống.",
        }

    # BƯỚC 3: Kiểm tra thông số thiết lập (nhiệt độ, độ sáng...) có nằm trong ngưỡng an toàn không
    if value is not None:
        try:
            val = float(value)
            min_val = DEVICES[device]["min"]
            max_val = DEVICES[device]["max"]

            # Từ chối nếu giá trị vượt quá khoảng cho phép (ví dụ: điều hòa 50 độ hoặc -10 độ)
            if val < min_val or val > max_val:
                return {
                    "status": "rejected",
                    "reason": "invalid_value",
                    "message": f"Giá trị {val} vượt ngưỡng an toàn ({min_val} - {max_val}) của {device}.",
                }
        except ValueError:
            return {
                "status": "rejected",
                "reason": "invalid_value",
                "message": "Giá trị truyền vào không phải là định dạng số hợp lệ.",
            }

    # BƯỚC 4: Kiểm tra dữ liệu đầu ra từ mô hình xác thực giọng nói
    if not speaker_data:
        return {
            "status": "rejected",
            "reason": "unauthorized_speaker",
            "message": "Thiếu dữ liệu nhận diện người nói.",
        }

    score = speaker_data.get("score", 0)
    speaker_id = speaker_data.get("speaker_id")

    # BƯỚC 5: Kiểm tra danh tính người nói và ngưỡng độ tự tin (Threshold)
    # Nếu điểm tin cậy < 0.75 hoặc người nói không nằm trong danh sách thành viên -> Từ chối
    if (
        score < SPEAKER_VERIFICATION_THRESHOLD
        or speaker_id not in ENROLLED_SPEAKERS
    ):
        return {
            "status": "rejected",
            "reason": "unauthorized_speaker",
            "message": "Người nói không có quyền thực thi lệnh này.",
        }

    # BƯỚC 6: Khi tất cả các điều kiện đều hợp lệ -> Chấp nhận lệnh
    return {
        "status": "ok",
        "device": device,
        "action": action,
        "value": value,
        "speaker": speaker_id,
    }