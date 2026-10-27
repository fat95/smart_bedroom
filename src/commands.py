#!/usr/bin/env python3
"""Tách lệnh từ văn bản: "giảm điều hòa hai độ" -> {device: ac, action: down, value: 2}.

Văn bản (từ PhoWhisper) đi qua các bước:
  1. chuẩn hóa: bỏ dấu, chữ thường, bỏ dấu câu          ("Đặt điều hòa 26°C." -> "dat dieu hoa 26 do c")
  2. tìm TỪ KHÓA THIẾT BỊ bằng khớp mờ                    (ASR nghe sai nhẹ vẫn nhận: "diu hoa" -> điều hòa)
  3. tìm HÀNH ĐỘNG bằng khớp mờ                           (bật/tắt/tăng/giảm/đặt/chuyển kênh/chế độ)
  4. đọc GIÁ TRỊ: chữ số "26" hoặc số đọc bằng chữ "hai mươi sáu"
  5. kiểm tra hành động có thuộc thiết bị đó không, có đủ giá trị không

Hàm chính (đúng hợp đồng nhóm):  parse_command(text) -> dict
    {"text": str, "device": str|None, "action": str|None, "value": int|None, "score": float,
     "reason": str|None}      # reason = None nếu hiểu được; thêm trường này so với hợp đồng gốc

Giá trị `reason` khi không hiểu được lệnh (decision.py của C dùng để chọn lý do từ chối):
    "ambiguous_device"  không có từ khóa thiết bị (hoặc có nhiều thiết bị) nhưng câu có vẻ là lệnh
    "unknown_command"   không nhận ra lệnh, hoặc hành động không thuộc thiết bị đó
    "missing_value"     hành động cần giá trị (tăng/giảm/đặt/kênh) mà câu không có số
    "empty_text"        văn bản rỗng

Việc kiểm tra giá trị nằm trong khoảng (invalid_value) là của decision.py: dùng config.value_in_range().

Thử nhanh:   python src/commands.py "đặt điều hòa hai mươi sáu độ"
Chạy test:   python -m unittest discover -s tests -v
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

try:  # rapidfuzz nhanh hơn, nhưng không bắt buộc: thiếu thì dùng difflib có sẵn trong Python
    from rapidfuzz import fuzz as _fuzz

    def _ratio(a: str, b: str) -> float:
        return float(_fuzz.ratio(a, b))
except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return 100.0 * SequenceMatcher(None, a, b).ratio()


# ------------------------------------------------------------------ chuẩn hóa văn bản
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def normalize(text: str) -> list[str]:
    """Văn bản -> danh sách token không dấu, chữ thường. "phần trăm" và "%" thành token đơn vị "pt"."""
    s = strip_accents(text).lower()
    s = s.replace("%", " pt ").replace("°", " do ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", s)   # "26c" -> "26 c"
    toks = s.split()
    out, i = [], 0
    while i < len(toks):                                          # "phan tram" -> bỏ, để "tram" không bị đọc là 100
        if toks[i] == "phan" and i + 1 < len(toks) and toks[i + 1] == "tram":
            out.append("pt")                                      # đơn vị phần trăm
            i += 2
            continue
        out.append(toks[i])
        i += 1
    return out


# ------------------------------------------------------------------ số đọc bằng chữ
ONES = {"khong": 0, "mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9}
AFTER_TENS = {"mot": 1, "tu": 4, "lam": 5, "nham": 5}   # "hai mươi mốt / tư / lăm": chỉ hợp lệ ngay sau "muoi"
CONNECT = {"linh", "le"}                                  # "một trăm linh năm"


def _is_num_token(tok: str, prev: str | None) -> bool:
    if tok.isdigit():
        return True
    if tok in ONES or tok in ("muoi", "tram") or tok in CONNECT:
        return True
    return tok in AFTER_TENS and prev == "muoi"


def _run_to_int(run: list[str]) -> int | None:
    if len(run) == 1 and run[0].isdigit():
        return int(run[0])
    if all(t.isdigit() for t in run):                      # "2 6" -> 26
        return int("".join(run)) if len(run) <= 3 else None
    ws = [t for t in run if not t.isdigit() and t not in CONNECT]
    if not ws:
        return None
    if "muoi" not in ws and "tram" not in ws:               # đọc từng chữ số: "hai sáu" -> 26
        digs = [ONES.get(w, AFTER_TENS.get(w)) for w in ws]
        return int("".join(map(str, digs))) if None not in digs and len(digs) <= 3 else None
    total, pend = 0, None
    for w in ws:
        if w == "tram":
            total += (pend if pend is not None else 1) * 100
            pend = None
        elif w == "muoi":
            total += (pend if pend is not None else 1) * 10
            pend = None
        else:
            pend = ONES.get(w, AFTER_TENS.get(w))
    return total + (pend or 0)


def find_number(tokens: list[str], skip: set[int] | None = None) -> tuple[int | None, tuple[int, int] | None]:
    """Tìm nhóm số đầu tiên trong tokens. Trả về (giá trị, (đầu, cuối)); không có thì (None, None)."""
    skip = skip or set()
    starts = set(ONES) | {"muoi", "tram"}
    i, n = 0, len(tokens)
    while i < n:
        if i not in skip and (tokens[i].isdigit() or tokens[i] in starts):
            j = i + 1
            while j < n and j not in skip and _is_num_token(tokens[j], tokens[j - 1]):
                j += 1
            v = _run_to_int(tokens[i:j])
            if v is not None:
                return v, (i, j)
            i = j
        else:
            i += 1
    return None, None


# ------------------------------------------------------------------ khớp mờ
def _threshold(phrase: str) -> float:
    """Từ càng ngắn càng phải khớp chặt (tránh "nam" bị nhầm với "nang"): <=3 ký tự khớp đúng, <=5 ký tự >=88, còn lại >=82."""
    n = len(phrase.replace(" ", ""))
    return 100.0 if n <= 3 else (88.0 if n <= 5 else 82.0)


def _match_phrase(tokens: list[str], phrase: str, skip: set[int]) -> tuple[float, int, int] | None:
    """Tìm cụm `phrase` (không dấu) trong tokens. Trả về (điểm 0-100, đầu, cuối) tốt nhất."""
    words = phrase.split()
    n = len(words)
    best = None
    for i in range(len(tokens) - n + 1):
        if any(k in skip for k in range(i, i + n)):
            continue
        cand = " ".join(tokens[i:i + n])
        sc = _ratio(cand, phrase)
        if sc >= _threshold(phrase) and (best is None or sc > best[0]):
            best = (sc, i, i + n)
    return best


def find_devices(tokens: list[str]) -> dict[str, tuple[float, int, int]]:
    """Mọi thiết bị có từ khóa xuất hiện: {id: (điểm, đầu, cuối)}; mỗi thiết bị lấy cụm dài/khớp tốt nhất."""
    found = {}
    for dev, spec in config.DEVICES.items():
        best = None
        for kw in spec["keywords"]:
            m = _match_phrase(tokens, kw, set())
            if m and (best is None or (m[0], m[2] - m[1]) > (best[0], best[2] - best[1])):
                best = m
        if best:
            found[dev] = best
    return found


# từ động từ (không dấu) -> ý định chung
# Lưu ý: "bat" (bật) và "tat" (tắt) chỉ lệch 1 chữ cái nên KHÔNG nới khớp mờ chung cho từ ngắn, vì sẽ bật nhầm khi người
# dùng nói tắt. Thay vào đó liệt kê đúng các kiểu ASR nghe sai đã gặp: PhoWhisper hay nghe "bật" thành "bậc" (t/c cuối từ).
VERBS = {
    "on": ["bat", "bac", "mo", "khoi dong"],
    "off": ["tat", "tac", "ngat", "ngung"],
    "up": ["tang", "nang", "them"],
    "down": ["giam", "ha", "bot"],
    "set": ["dat", "chinh", "cai", "de"],
    "channel": ["chuyen", "kenh"],
}
MODES = {"mode_cool": ["lam lanh", "lanh"], "mode_fan": ["quat"], "mode_dry": ["hut am", "hut"]}


def _verb_variants(toks: list[str]) -> list[list[str]]:
    """ASR hay nghe lẫn phụ âm cuối -c/-t ("bật" -> "bậc", "tắt" -> "tắc"): thử thêm bản đổi -c thành -t cho từ 3 ký tự trở lên.
    Chỉ dùng khi dò động từ, không áp cho từ khóa thiết bị hay số."""
    out = [toks]
    if len(toks) == 1 and len(toks[0]) >= 3 and toks[0].endswith("c"):
        out.append([toks[0][:-1] + "t"])
    return out


def find_intents(tokens: list[str], skip: set[int]) -> list[tuple[int, str, float]]:
    """Mọi động từ tìm thấy: [(vị trí, ý định, điểm)] theo thứ tự xuất hiện trong câu."""
    hits = []
    for intent, words in VERBS.items():
        for w in words:
            n = len(w.split())
            for i in range(len(tokens)):
                if i in skip:
                    continue
                best = None
                for cand in _verb_variants(tokens[i:i + n]):
                    m = _match_phrase(cand, w, set())
                    if m and (best is None or m[0] > best):
                        best = m[0]
                if best is not None:
                    hits.append((i, intent, best))
    hits.sort()
    return hits


def find_mode(tokens: list[str], skip: set[int]) -> tuple[str, float, set[int]] | None:
    best = None
    for act, words in MODES.items():
        for w in words:
            m = _match_phrase(tokens, w, skip)
            if m and (best is None or m[0] > best[1]):
                best = (act, m[0], set(range(m[1], m[2])))
    return best


# ------------------------------------------------------------------ hàm chính
def _result(text, device=None, action=None, value=None, score=0.0, reason=None):
    return {"text": text, "device": device, "action": action, "value": value,
            "score": round(float(score), 3), "reason": reason}


def parse_command(text: str) -> dict:
    """Văn bản -> {text, device, action, value, score, reason}. Xem docstring đầu file."""
    raw = text or ""
    tokens = normalize(raw)
    if not tokens:
        return _result(raw, reason="empty_text")

    # 1) thiết bị
    devs = find_devices(tokens)
    flat = "phang" in tokens                                   # "nằm phẳng"
    skip = {k for (_, a, b) in devs.values() for k in range(a, b)}
    intents = find_intents(tokens, skip)
    num, span = find_number(tokens, skip)
    # PhoWhisper-base hay nghe "bật" thành "bọn" (gặp thật ở 2/36 file test). "bon" cũng là số 4 ("bốn"), nên chỉ coi là "bật" khi:
    # nó là từ ĐẦU câu, đứng một mình (không phải "bốn mươi..."), không có động từ nào khác và sau nó không phải "độ".
    if (len(devs) == 1 and not intents and tokens[0] == "bon" and 0 not in skip
            and (len(tokens) < 2 or tokens[1] != "do") and find_number(tokens, skip)[1] == (0, 1)):
        intents = [(0, "on", 70.0)]
        num, span = None, None
    has_action_cue = bool(intents) or num is not None or flat

    if len(devs) != 1:
        if not devs:
            return _result(raw, reason="ambiguous_device" if has_action_cue else "unknown_command")
        return _result(raw, reason="ambiguous_device")
    dev = next(iter(devs))
    dev_score = devs[dev][0] / 100.0
    allowed = config.DEVICES[dev]["actions"]

    # 2) hành động
    act, act_score = None, 0.8                                 # hành động ngầm định (không có động từ) bị trừ điểm
    mode = find_mode(tokens, skip) if dev == "ac" else None
    if mode:                                                   # "điều hòa chế độ làm lạnh"
        act, act_score = mode[0], mode[1] / 100.0
        skip |= mode[2]
        num, span = find_number(tokens, skip)
    elif flat and dev == "bed":                                # "hạ giường nằm phẳng" -> đặt góc 0
        act, act_score, num = "set", 1.0, 0
    else:
        usable = [h for h in intents if h[1] != "channel" or "channel" in allowed]
        channel = [h for h in usable if h[1] == "channel"]
        if channel:                                            # tivi + "kênh"/"chuyển" luôn là chuyển kênh
            act, act_score = "channel", channel[0][2] / 100.0
        elif usable:
            act, act_score = usable[0][1], usable[0][2] / 100.0
        elif num is not None and ({"do", "pt"} & set(tokens)):  # không có động từ nhưng có số + đơn vị: "điều hòa 26 độ"
            act = "set"                                         # số trần không đơn vị (vd "bọn" nghe thành "bốn") thì từ chối, không đoán
        if act is None:
            return _result(raw, device=dev, reason="unknown_command", score=dev_score * 0.5)

    if act not in allowed:
        return _result(raw, device=dev, reason="unknown_command", score=dev_score * 0.5)

    # 3) giá trị
    needs_value = allowed[act]["range"] is not None
    value = num if needs_value else None
    score = (dev_score + act_score) / 2.0
    if needs_value and value is None:
        return _result(raw, device=dev, action=act, score=score, reason="missing_value")
    return _result(raw, device=dev, action=act, value=value, score=score)


# ------------------------------------------------------------------ chuẩn hóa cho WER/CER
def normalize_for_wer(text: str) -> list[str]:
    """Token cho WER/CER: chữ thường, giữ dấu tiếng Việt, bỏ dấu câu, số đọc bằng chữ -> chữ số.

    "Đặt điều hòa hai mươi sáu độ." và "đặt điều hòa 26 độ" cho cùng kết quả, để ASR xuất chữ số
    hay chữ đều không bị tính là sai.
    """
    s = unicodedata.normalize("NFC", (text or "").lower()).replace("%", " phần trăm ").replace("°", " độ ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", s)
    toks = s.split()
    merged, i = [], 0
    while i < len(toks):                                       # gộp "phần trăm" thành một token "%"
        if toks[i] == "phần" and i + 1 < len(toks) and toks[i + 1] == "trăm":
            merged.append("%")
            i += 2
        else:
            merged.append(toks[i])
            i += 1
    plain = [strip_accents(t) for t in merged]                 # strip theo từng token: số token không đổi
    out, i = [], 0
    while i < len(merged):
        if plain[i] != "%" and _is_num_token(plain[i], plain[i - 1] if i else None):
            j = i + 1
            while j < len(merged) and plain[j] != "%" and _is_num_token(plain[j], plain[j - 1]):
                j += 1
            v = _run_to_int(plain[i:j]) if not all(p in CONNECT for p in plain[i:j]) else None
            if v is not None:
                out.append(str(v))
                i = j
                continue
        out.append(merged[i])
        i += 1
    return out


if __name__ == "__main__":
    import json
    for t in (sys.argv[1:] or ["đặt điều hòa hai mươi sáu độ"]):
        print(json.dumps(parse_command(t), ensure_ascii=False))
