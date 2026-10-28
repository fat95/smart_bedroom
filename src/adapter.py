"""Chuyển kết quả của commands.py (mã tiếng Anh) sang dạng decision.decide() và index.html dùng (tiếng Việt)."""
from __future__ import annotations

DEV_VI = {"ac": "điều hòa", "light-main": "đèn trần", "bed": "giường", "tv": "tivi"}
TARGET = {"ac": "nhiệt độ", "light-main": "độ sáng", "bed": "góc", "tv": "âm lượng"}
ACT_VI = {"on": "bật", "off": "tắt", "up": "tăng", "down": "giảm", "set": "đặt"}
MODE = {"mode_cool": "lạnh", "mode_fan": "quạt", "mode_dry": "hút ẩm"}


def to_decision_input(r: dict) -> dict:
    """Kết quả recognize_command()/parse_command() -> command_data cho decide()."""
    reason = r.get("reason")
    if reason == "ambiguous_device":
        return {"device": None, "action": "x"}            # decide -> ambiguous_device
    if reason in ("unknown_command", "empty_text"):
        return {"device": None, "action": None}           # decide -> unknown_command
    dev, act, val = r.get("device"), r.get("action"), r.get("value")
    if dev not in DEV_VI or act is None:
        return {"device": None, "action": None}
    if reason == "missing_value":                         # decide -> invalid_value (thiếu giá trị)
        act_vi = "đặt" if act == "channel" else ACT_VI.get(act, act)
        target = "kênh" if act == "channel" else TARGET[dev]
        return {"device": DEV_VI[dev], "action": act_vi, "target": target, "value": None}
    if act in MODE:
        return {"device": DEV_VI[dev], "action": "đặt", "target": "chế độ", "value": MODE[act]}
    if act == "channel":
        return {"device": DEV_VI[dev], "action": "đặt", "target": "kênh", "value": val}
    if act in ("on", "off"):
        return {"device": DEV_VI[dev], "action": ACT_VI[act], "target": None, "value": None}
    return {"device": DEV_VI[dev], "action": ACT_VI[act], "target": TARGET[dev], "value": val}