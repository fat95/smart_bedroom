import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import rescore_asr  # noqa: E402


class TestRescore(unittest.TestCase):
    def test_uses_command_parser_and_summary_logic(self):
        rows = [{"hyp_text": "bậc tivi", "device": "tv", "action": "on", "value": "" , "speaker": "x", "ref_text": "bật tivi"}]
        rescored, report = rescore_asr.rescore_rows(rows)
        self.assertEqual(report["acc_exact"], 1.0)
        self.assertEqual(report["wrong_exec"], 0.0)
        self.assertEqual((rescored[0]["pred_device"], rescored[0]["pred_action"]), ("tv", "on"))


if __name__ == "__main__":
    unittest.main()
