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

# Bảng luật được SINH TỪ config.DEVICES (nguồn duy nhất cho khoảng giá trị), dịch sang tiếng Việt
# bằng các bảng trong adapter.py vì index.html và JSON trả về dùng tiếng Việt.
#   ACTION_RULES[(thiết bị, thuộc tính)][hành động] = (lo, hi) | tập giá trị rời rạc
# "tăng"/"giảm" là mức thay đổi, "đặt" là giá trị tuyệt đối.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from adapter import ACT_VI, DEV_VI, MODE, TARGET  # noqa: E402


def _build_rules():
    rules, power = {}, set()
    for dev, spec in config.DEVICES.items():
        vi, modes = DEV_VI[dev], []
        for act, a in spec["actions"].items():
            if act in ("up", "down", "set"):
                rules.setdefault((vi, TARGET[dev]), {})[ACT_VI[act]] = a["range"]
            elif act == "channel":
                rules.setdefault((vi, "kênh"), {})["đặt"] = a["range"]
            elif act in MODE:
                modes.append(MODE[act])
            elif act in ("on", "off"):
                power.add(vi)
        if modes:
            rules.setdefault((vi, "chế độ"), {})["đặt"] = tuple(modes)
    return rules, power


ACTION_RULES, POWER_DEVICES = _build_rules()
DEVICES = {device for device, _ in ACTION_RULES} | POWER_DEVICES
POWER_ACTIONS = {"bật", "tắt"}


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
                "value": None, "speaker": speaker_id}

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