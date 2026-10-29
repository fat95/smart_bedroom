"""Test eval_asr.py và asr.load_audio bằng dữ liệu tổng hợp và ASR giả (không cần mô hình hay GPU).

    python -m unittest discover -s tests -v
"""
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import asr  # noqa: E402
import eval_asr as E  # noqa: E402

SENTENCES = """id,text,device,action,value
s01,đặt điều hòa hai mươi sáu độ,ac,set,26
s02,bật tivi,tv,on,
s03,chuyển tivi kênh năm,tv,channel,5
s04,tăng đèn trần hai mươi phần trăm,light-main,up,20
"""


def _write_wav(path: Path, sr=16000, seconds=0.5):
    wavfile.write(str(path), sr, (np.random.default_rng(0).standard_normal(int(sr * seconds)) * 3000).astype("int16"))


class TestMetrics(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(E.edit_distance(list("kitten"), list("sitting")), 3)
        self.assertEqual(E.edit_distance([], ["a"]), 1)

    def test_wer_cer_perfect_and_number_forms(self):
        w, c = E.wer_cer(["đặt điều hòa hai mươi sáu độ"], ["Đặt điều hòa 26 độ."])
        self.assertEqual((w, c), (0.0, 0.0))

    def test_wer_counts_errors(self):
        w, _ = E.wer_cer(["bật điều hòa", "tắt tivi"], ["bật điều hòa", "tắt ti"])     # 1 lỗi / 5 từ
        self.assertAlmostEqual(w, 1 / 5)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        t = Path(self.tmp.name)
        (t / "audio").mkdir()
        (t / "sentences.csv").write_text(SENTENCES, encoding="utf-8")
        for name in ["vinh_s01_1", "vinh_s02_1", "vinh_s03_1", "la1_s01_1", "la1_s04_1", "vinh_zz9_1"]:
            _write_wav(t / "audio" / f"{name}.wav")
        self.t = t

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_testset_skips_unknown_ids(self):
        items = E.load_testset(self.t / "sentences.csv", self.t / "audio")
        self.assertEqual(len(items), 5)                                 # vinh_zz9_1 bị bỏ
        it = next(i for i in items if i["sid"] == "s01" and i["speaker"] == "la1")
        self.assertEqual((it["device"], it["action"], it["value"]), ("ac", "set", 26))
        self.assertIsNone(next(i for i in items if i["sid"] == "s02")["value"])

    def test_evaluate_with_fake_asr(self):
        items = E.load_testset(self.t / "sentences.csv", self.t / "audio")
        heard = {"vinh_s01_1": "đặt điều hòa hai mươi sáu độ",       # đúng
                 "vinh_s02_1": "bật ti vi",                          # đúng (khớp mờ)
                 "vinh_s03_1": "chuyển tivi kênh sáu",               # sai giá trị
                 "la1_s01_1": "đặt điều hòa 26 độ",                  # đúng, xuất chữ số
                 "la1_s04_1": "tăng điều hòa hai mươi phần trăm"}    # sai thiết bị
        fake = lambda p: heard[Path(p).stem]
        rows, rep = E.evaluate_items(items, fake, warmup=0)
        self.assertEqual(rep["n"], 5)
        self.assertAlmostEqual(rep["acc_exact"], 3 / 5, places=3)
        self.assertAlmostEqual(rep["acc_device"], 4 / 5, places=3)
        self.assertAlmostEqual(rep["acc_value"], 3 / 4, places=3)               # 4 câu có giá trị: s01, s01, s03, s04
        self.assertIn("la1", rep["by_speaker"])
        self.assertIsNotNone(rep["wer"])

        out = self.t / "out"
        E.write_outputs("fake", rows, rep, out, simple=False)
        E.write_summary({"fake": rep}, out, simple=False, baseline=None)
        for f in ["results_fake.csv", "report_fake.json", "confusion_fake_device.png", "confusion_fake_action.csv", "summary.md"]:
            self.assertTrue((out / f).exists(), f)

    def test_load_simple_follows_renamed_files(self):
        d = self.t / "dataset" / "bat_tivi"
        d.mkdir(parents=True)
        _write_wav(d / "vinh_bat_tivi_003.wav")                  # tên mới của nhánh phan-B
        (self.t / "manifest.csv").write_text(
            "file,command,speaker,split\r\ndataset\\bat_tivi\\vinh_003.wav,bat_tivi,vinh,test\r\n", encoding="utf-8")
        old_root = E.ROOT
        E.ROOT = self.t
        try:
            items = E.load_simple(self.t / "manifest.csv", "test")
        finally:
            E.ROOT = old_root
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["path"].endswith("vinh_bat_tivi_003.wav"))

    def test_simple_mode_noise_rejection(self):
        items = [dict(path="a.wav", speaker="x", sid="bat_tivi", rep="", ref_text="bật tivi", device="tv", action="on", value=None),
                 dict(path="b.wav", speaker="x", sid="noise", rep="", ref_text="", device=None, action=None, value=None),
                 dict(path="c.wav", speaker="x", sid="noise", rep="", ref_text="", device=None, action=None, value=None)]
        heard = {"a.wav": "bật tivi", "b.wav": "", "c.wav": "tắt đèn trần"}      # c: nhiễu bị nhận nhầm
        rows, rep = E.evaluate_items(items, lambda p: heard[p], warmup=0)
        self.assertAlmostEqual(rep["noise_rejected"], 1 / 2, places=3)
        self.assertAlmostEqual(rep["acc_exact"], 2 / 3, places=3)
        self.assertAlmostEqual(rep["wrong_exec"], 1 / 3, places=3)       # c: nhiễu bị thực thi thành "tắt đèn trần"
        out = self.t / "o2"
        E.write_outputs("fake", rows, rep, out, simple=True)
        self.assertTrue((out / "confusion_fake_simple.png").exists())

    def test_write_summary_simple_baseline_without_wrong_exec(self):
        out = Path(self.tmp.name) / "summary_baseline"
        out.mkdir()
        E.write_summary({}, out, simple=True,
                        baseline={"n": 42, "acc": 0.857, "latency_ms_mean": 49.0})
        summary = (out / "summary.md").read_text(encoding="utf-8")
        self.assertIn("baseline MFCC", summary)
        self.assertIn("–", summary)


class TestLoadAudio(unittest.TestCase):
    def test_resample_stereo_bytes(self):
        sr = 44100
        t = np.arange(sr) / sr
        stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
        buf = io.BytesIO()
        wavfile.write(buf, sr, (stereo * 20000).astype("int16"))
        y = asr.load_audio(buf.getvalue())
        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(y.ndim, 1)
        self.assertAlmostEqual(len(y) / 16000, 1.0, delta=0.01)
        self.assertLessEqual(float(np.abs(y).max()), 1.0)

    def test_numpy_passthrough(self):
        y = asr.load_audio(np.zeros(16000, dtype="float64"))
        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(len(y), 16000)

    def test_resolve_model(self):
        self.assertEqual(asr.resolve_model("phowhisper-small"), "vinai/PhoWhisper-small")
        self.assertEqual(asr.resolve_model("vinai/PhoWhisper-base"), "vinai/PhoWhisper-base")


if __name__ == "__main__":
    unittest.main()
