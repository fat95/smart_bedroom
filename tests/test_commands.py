"""Test parse_command bằng văn bản mẫu (không cần mô hình, micro hay GPU).

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import commands as C  # noqa: E402

# (văn bản ASR, thiết bị, hành động, giá trị, reason)  -- reason None = hiểu được lệnh
OK = None
CASES = [
    # --- bật/tắt
    ("bật điều hòa", "ac", "on", None, OK),
    ("Tắt điều hòa.", "ac", "off", None, OK),
    ("bật đèn trần", "light-main", "on", None, OK),
    ("tắt đèn trần", "light-main", "off", None, OK),
    ("bật tivi", "tv", "on", None, OK),
    ("tắt tivi", "tv", "off", None, OK),
    ("bật ti vi", "tv", "on", None, OK),
    ("tắt TV", "tv", "off", None, OK),
    ("mở máy lạnh", "ac", "on", None, OK),
    ("bật đèn", "light-main", "on", None, OK),
    # --- điều hòa có tham số
    ("đặt điều hòa hai mươi sáu độ", "ac", "set", 26, OK),
    ("đặt điều hòa 26 độ", "ac", "set", 26, OK),
    ("Đặt điều hòa 26°C.", "ac", "set", 26, OK),
    ("đặt điều hòa hai mươi lăm độ", "ac", "set", 25, OK),
    ("đặt điều hòa hai mốt độ", "ac", "set", 21, OK),
    ("đặt điều hòa mười sáu độ", "ac", "set", 16, OK),
    ("đặt điều hòa ba mươi độ", "ac", "set", 30, OK),
    ("đặt điều hòa hai sáu độ", "ac", "set", 26, OK),
    ("chỉnh điều hòa hai mươi bốn độ", "ac", "set", 24, OK),
    ("điều hòa hai mươi hai độ", "ac", "set", 22, OK),
    ("giảm điều hòa hai độ", "ac", "down", 2, OK),
    ("tăng điều hòa ba độ", "ac", "up", 3, OK),
    ("giảm điều hòa 1 độ", "ac", "down", 1, OK),
    ("điều hòa chế độ làm lạnh", "ac", "mode_cool", None, OK),
    ("điều hòa chế độ quạt", "ac", "mode_fan", None, OK),
    ("đổi điều hòa sang chế độ hút ẩm", "ac", "mode_dry", None, OK),
    ("bật chế độ làm lạnh máy lạnh", "ac", "mode_cool", None, OK),
    # --- đèn trần
    ("tăng đèn trần hai mươi phần trăm", "light-main", "up", 20, OK),
    ("giảm đèn trần mười phần trăm", "light-main", "down", 10, OK),
    ("đặt đèn trần năm mươi phần trăm", "light-main", "set", 50, OK),
    ("đặt đèn trần 50%", "light-main", "set", 50, OK),
    ("đặt đèn trần một trăm phần trăm", "light-main", "set", 100, OK),
    ("đặt đèn trần 100 phần trăm", "light-main", "set", 100, OK),
    ("đặt đèn trần bảy mươi lăm phần trăm", "light-main", "set", 75, OK),
    ("đặt đèn trần không phần trăm", "light-main", "set", 0, OK),
    # --- giường
    ("nâng đầu giường mười độ", "bed", "up", 10, OK),
    ("nâng đầu giường ba mươi độ", "bed", "up", 30, OK),
    ("hạ đầu giường mười lăm độ", "bed", "down", 15, OK),
    ("đặt giường bốn mươi lăm độ", "bed", "set", 45, OK),
    ("hạ giường nằm phẳng", "bed", "set", 0, OK),
    ("giường nằm phẳng", "bed", "set", 0, OK),
    # --- tivi kênh / âm lượng
    ("chuyển tivi kênh năm", "tv", "channel", 5, OK),
    ("chuyển tivi kênh 5", "tv", "channel", 5, OK),
    ("chuyển kênh tivi sáu", "tv", "channel", 6, OK),
    ("tivi kênh chín", "tv", "channel", 9, OK),
    ("tăng âm lượng tivi mười", "tv", "up", 10, OK),
    ("giảm âm lượng tivi hai mươi", "tv", "down", 20, OK),
    ("đặt âm lượng tivi năm mươi", "tv", "set", 50, OK),
    # --- ASR nghe sai nhẹ vẫn phải hiểu (khớp mờ)
    ("bật điều hoà", "ac", "on", None, OK),
    ("bật diều hòa", "ac", "on", None, OK),
    ("bật điêu hòa", "ac", "on", None, OK),
    ("tắt đèn chần", "light-main", "off", None, OK),
    ("đặt điều hòa hai mươi sáu độ c", "ac", "set", 26, OK),
    ("Bật Tivi!", "tv", "on", None, OK),
    ("bậc ti vi.", "tv", "on", None, OK),            # PhoWhisper-base nghe "bật" thành "bậc" (gặp thật khi chạy mô hình)
    ("tắc đèn trần", "light-main", "off", None, OK),
    ("bậc điều hòa", "ac", "on", None, OK),
    # lỗi thật từ PhoWhisper-base: "bật" nghe thành "bọn" (= "bốn" ở đầu câu). Không được hiểu thành "đặt ... 4"
    ("bọn đèn chân.", "light-main", "on", None, OK),
    ("bọn tivi.", "tv", "on", None, OK),
    ("bốn mươi tivi", "tv", None, None, "unknown_command"),     # số thật ở đầu câu thì không phải "bật"
    ("bọn tivi.", "tv", "on", None, OK),              # "bật" bị nghe thành "bọn" (gặp thật)
    ("bọn đèn chân.", "light-main", "on", None, OK),
    ("bốn độ điều hòa", "ac", "set", 4, OK),           # "bốn" thật ở đầu câu vẫn là số, không bị coi là "bật"
    ("bốn mươi độ điều hòa", "ac", "set", 40, OK),
    # --- lỗi ASR thật đã gặp: PhoWhisper nghe "bật" thành "bậc"
    ("bậc ti vi.", "tv", "on", None, OK),
    ("bậc điều hòa", "ac", "on", None, OK),
    # lỗi thật từ PhoWhisper-base: "bật" nghe thành "bọn" (= "bốn" ở đầu câu). Không được hiểu thành "đặt ... 4"
    ("bọn đèn chân.", "light-main", "on", None, OK),
    ("bọn tivi.", "tv", "on", None, OK),
    ("bốn mươi tivi", "tv", None, None, "unknown_command"),     # số thật ở đầu câu thì không phải "bật"
    ("bọn tivi.", "tv", "on", None, OK),              # "bật" bị nghe thành "bọn" (gặp thật)
    ("bọn đèn chân.", "light-main", "on", None, OK),
    ("bốn độ điều hòa", "ac", "set", 4, OK),           # "bốn" thật ở đầu câu vẫn là số, không bị coi là "bật"
    ("bốn mươi độ điều hòa", "ac", "set", 40, OK),
    ("bậc đèn trần", "light-main", "on", None, OK),
    ("tắc tivi", "tv", "off", None, OK),
    ("tắc đèn trần", "light-main", "off", None, OK),
    # --- từ chối
    ("bật", None, None, None, "ambiguous_device"),          # có động từ nhưng thiếu từ khóa thiết bị
    ("tăng lên mười", None, None, None, "ambiguous_device"),
    ("tăng mười độ", None, None, None, "ambiguous_device"),
    ("nâng mười độ", None, None, None, "ambiguous_device"),
    ("bật điều hòa và tivi", None, None, None, "ambiguous_device"),
    ("xin chào các bạn", None, None, None, "unknown_command"),
    ("hôm nay trời đẹp quá", None, None, None, "unknown_command"),
    ("", None, None, None, "empty_text"),
    ("   ", None, None, None, "empty_text"),
    ("điều hòa", "ac", None, None, "unknown_command"),
    ("tăng điều hòa", "ac", "up", None, "missing_value"),
    ("đặt đèn trần", "light-main", "set", None, "missing_value"),
    ("chuyển tivi kênh", "tv", "channel", None, "missing_value"),
    ("chuyển kênh đèn trần năm", "light-main", None, None, "unknown_command"),   # số trần không đơn vị: từ chối, không đoán
    ("bật giường", "bed", None, None, "unknown_command"),    # giường không có hành động bật/tắt
]


class TestParseCommand(unittest.TestCase):
    def test_cases(self):
        for text, dev, act, val, reason in CASES:
            with self.subTest(text=text):
                r = C.parse_command(text)
                self.assertEqual((r["device"], r["action"], r["value"], r["reason"]), (dev, act, val, reason), r)

    def test_score_range_and_keys(self):
        r = C.parse_command("bật tivi")
        self.assertEqual(set(r), {"text", "device", "action", "value", "score", "reason"})
        self.assertTrue(0.0 <= r["score"] <= 1.0)
        self.assertEqual(r["text"], "bật tivi")

    def test_values_in_config_range(self):
        import config
        for dev, act, val in [("ac", "set", 26), ("tv", "channel", 5), ("bed", "set", 0)]:
            self.assertTrue(config.value_in_range(dev, act, val))
        self.assertFalse(config.value_in_range("ac", "set", 35))
        self.assertFalse(config.value_in_range("tv", "channel", 12))
        self.assertTrue(config.value_in_range("tv", "on", None))
        self.assertFalse(config.value_in_range("tv", "on", 3))


class TestNumbers(unittest.TestCase):
    def test_words(self):
        cases = {"mười": 10, "mười một": 11, "mười lăm": 15, "hai mươi": 20, "hai mươi mốt": 21, "hai mươi tư": 24,
                 "ba mươi lăm": 35, "chín mươi chín": 99, "một trăm": 100, "một trăm linh năm": 105,
                 "năm": 5, "không": 0, "hai sáu": 26, "26": 26, "bảy": 7}
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(C.find_number(C.normalize(text))[0], want)

    def test_no_number(self):
        self.assertIsNone(C.find_number(C.normalize("bật điều hòa"))[0])
        # "làm lạnh", "nằm phẳng" không được đọc thành số
        self.assertIsNone(C.find_number(C.normalize("chế độ làm lạnh"))[0])


class TestNormalizeForWer(unittest.TestCase):
    def test_numbers_unified(self):
        a = C.normalize_for_wer("Đặt điều hòa hai mươi sáu độ.")
        b = C.normalize_for_wer("đặt điều hòa 26 độ")
        self.assertEqual(a, b)
        self.assertEqual(a, ["đặt", "điều", "hòa", "26", "độ"])

    def test_percent(self):
        self.assertEqual(C.normalize_for_wer("đặt đèn trần 50%"), C.normalize_for_wer("đặt đèn trần năm mươi phần trăm"))

    def test_accents_kept(self):
        self.assertNotEqual(C.normalize_for_wer("bật"), C.normalize_for_wer("bắt"))


if __name__ == "__main__":
    unittest.main()
