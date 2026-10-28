"""
Module: decision.py
Chức năng chính: Bộ kiểm tra logic và ra quyết định thực thi (Decision Engine).
Nhiệm vụ:
  1. Tiếp nhận thông tin lệnh từ mô hình nhận dạng lệnh (Người A).
  2. Tiếp nhận thông tin người nói từ mô hình xác thực người nói (Người B).
  3. Kiểm tra người nói trước, rồi kiểm tra thiết bị, hành động và khoảng giá trị an toàn.
  4. Trả về "ok" (cho phép thực thi) hoặc "rejected" kèm lý do.

Dữ liệu vào:
  command_data = {"device": "điều hòa", "action": "tăng", "target": "nhiệt độ", "value": 2}
      - bật/tắt: chỉ cần device + action ("bật" / "tắt"), value bỏ qua được.
      - tăng/giảm/đặt: cần thêm target (thuộc tính nào) và value.
  speaker_data = {"speaker_id": "phat", "score": 0.93, "is_enrolled": True}
      - is_enrolled do module của B quyết định bằng ngưỡng đã chọn khi huấn luyện.
        Không đặt ngưỡng cố định thêm ở đây, vì thang điểm phụ thuộc phương pháp
        (xác suất hoặc điểm biên) và đổi mỗi lần train lại.

Lý do từ chối: unauthorized_speaker, ambiguous_device, unknown_command, invalid_value.
"""

# Thiết bị -> thuộc tính -> hành động -> khoảng hợp lệ.
#   (lo, hi)           : số trong khoảng
#   ("lạnh", "quạt")   : giá trị phải thuộc tập này
# "tăng"/"giảm" là mức thay đổi, "đặt" là giá trị tuyệt đối.
ACTION_RULES = {
    ("điều hòa", "nhiệt độ"): {"tăng": (1, 5), "giảm": (1, 5), "đặt": (16, 30)},
    ("điều hòa", "chế độ"): {"đặt": ("lạnh", "quạt", "hút ẩm")},
    ("đèn trần", "độ sáng"): {"tăng": (1, 100), "giảm": (1, 100), "đặt": (0, 100)},
    ("giường", "góc"): {"tăng": (1, 60), "giảm": (1, 60), "đặt": (0, 60)},
    ("tivi", "kênh"): {"đặt": (1, 9)},
    ("tivi", "âm lượng"): {"tăng": (1, 100), "giảm": (1, 100), "đặt": (0, 100)},
}
DEVICES = {device for device, _ in ACTION_RULES}
POWER_ACTIONS = {"bật", "tắt"}
POWER_DEVICES = {"điều hòa", "đèn trần", "tivi"}  # giường không có nút bật/tắt


def _reject(reason, message):
    return {"status": "rejected", "reason": reason, "message": message}


def decide(command_data, speaker_data):
    """Đánh giá người nói và lệnh. Trả về {"status": "ok"|"rejected", ...}."""

    # BƯỚC 1: Người nói. Kiểm tra trước để người lạ luôn nhận cùng một phản hồi,
    # không biết được lệnh có hợp lệ hay khoảng giá trị nào được chấp nhận.
    if not speaker_data or not speaker_data.get("is_enrolled", False):
        return _reject("unauthorized_speaker", "Giọng nói không được phép điều khiển hệ thống.")
    speaker_id = speaker_data.get("speaker_id")

    # BƯỚC 2: Có nhận ra lệnh và thiết bị không
    command_data = command_data or {}
    device = command_data.get("device")
    action = command_data.get("action")
    target = command_data.get("target")
    value = command_data.get("value")

    if not device:
        if action:
            return _reject("ambiguous_device", "Không rõ lệnh dành cho thiết bị nào.")
        return _reject("unknown_command", "Không nhận ra nội dung câu lệnh.")
    if device not in DEVICES:
        return _reject("unknown_command", f"Thiết bị '{device}' không có trong hệ thống.")

    # BƯỚC 3: Bật / tắt
    if action in POWER_ACTIONS:
        if device not in POWER_DEVICES:
            return _reject("unknown_command", f"{device} không có lệnh bật/tắt.")
        return {"status": "ok", "device": device, "action": action, "target": None,
                "value": value, "speaker": speaker_id}

    # BƯỚC 4: Tăng / giảm / đặt: phải có thuộc tính hợp lệ cho thiết bị
    rule = ACTION_RULES.get((device, target), {}).get(action)
    if rule is None:
        return _reject("unknown_command", f"Hệ thống không hỗ trợ lệnh '{action} {target}' cho {device}.")
    if value is None:
        return _reject("invalid_value", "Thiếu giá trị cho lệnh này.")

    # BƯỚC 5: Khoảng giá trị an toàn
    if isinstance(rule[0], str):                      # tập giá trị rời rạc (chế độ)
        if value not in rule:
            return _reject("invalid_value", f"Giá trị '{value}' không hợp lệ (chấp nhận: {', '.join(rule)}).")
    else:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return _reject("invalid_value", "Giá trị truyền vào không phải là số hợp lệ.")
        lo, hi = rule
        if num < lo or num > hi:
            return _reject("invalid_value", f"Giá trị {num:g} nằm ngoài khoảng cho phép ({lo} - {hi}) của {device}.")
        value = int(num) if num == int(num) else num

    # BƯỚC 6: Hợp lệ
    return {"status": "ok", "device": device, "action": action, "target": target,
            "value": value, "speaker": speaker_id}