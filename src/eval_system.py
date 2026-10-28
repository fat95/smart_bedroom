"""
Đánh giá toàn hệ thống Smart Bedroom qua HTTP (câu 2.5).
Chạy app.py ở terminal khác trước.

    python src/eval_system.py                       # sạch
    python src/eval_system.py --snr 20 10 5         # thêm nhiễu trắng (mô phỏng phòng ồn)
    python src/eval_system.py --gain-db -10 -20     # giảm biên độ (mô phỏng xa micro)

File: testset/audio/<người>_<mã câu>_<số lần>.wav, đáp án: testset/sentences.csv (id,text,device,action,value).
Người lạ = tên la, la1, la2, ... (hoặc trong strangers.txt). Đáp án so với JSON của app (tiếng Việt).
"""
import argparse
import csv
import io
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
from adapter import to_decision_input  # noqa: E402

AUDIO_DIR = BASE / "testset" / "audio"
CSV_FILE = BASE / "testset" / "sentences.csv"
URL = "http://127.0.0.1:5000/process-audio"
STRANGER_RE = re.compile(r"^la\d*$", re.I)


def stranger_names():
    p = BASE / "strangers.txt"
    if not p.exists():
        return set()
    return {l.split("#", 1)[0].strip().lower() for l in p.read_text(encoding="utf-8-sig").splitlines()
            if l.split("#", 1)[0].strip()}


def load_sentences():
    out = {}
    with open(CSV_FILE, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("value") or "").strip()
            exp = to_decision_input({"device": (row.get("device") or "").strip() or None,
                                     "action": (row.get("action") or "").strip() or None,
                                     "value": int(float(v)) if v else None, "reason": None})
            out[row["id"].strip()] = exp
    return out


def wav_bytes(path, snr_db=None, gain_db=None, rng=None):
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if gain_db is not None:
        y = y * 10 ** (gain_db / 20)
    if snr_db is not None:
        p = float(np.mean(y ** 2)) + 1e-12
        y = y + rng.standard_normal(len(y)).astype("float32") * np.sqrt(p / 10 ** (snr_db / 10))
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def run(snr=None, gain=None):
    sents, strangers = load_sentences(), stranger_names()
    rng = np.random.default_rng(0)
    enr, stg, lat, skipped = [], [], [], 0
    for p in sorted(AUDIO_DIR.glob("*.wav")):
        parts = p.stem.split("_")
        if len(parts) < 3 or "_".join(parts[1:-1]) not in sents:
            skipped += 1
            continue
        spk, exp = parts[0].lower(), sents["_".join(parts[1:-1])]
        is_str = bool(STRANGER_RE.match(spk)) or spk in strangers
        t0 = time.perf_counter()
        try:
            r = requests.post(URL, files={"audio_data": (p.name, wav_bytes(p, snr, gain, rng), "audio/wav")},
                              timeout=60).json()
        except requests.exceptions.ConnectionError:
            sys.exit(f"Không kết nối được {URL}. Đã chạy 'python app.py' chưa?")
        lat.append((time.perf_counter() - t0) * 1000)
        ok = r.get("status") == "ok"
        if is_str:
            stg.append(not ok)                               # đúng = bị chặn
        else:
            enr.append(ok and all(r.get(k) == exp.get(k) for k in ("device", "action", "target", "value")))
            if not enr[-1]:
                print(f"  sai: {p.name}: mong đợi {exp} | nhận {r.get('status')} {r.get('reason')} "
                      f"'{r.get('text')}' {r.get('device')}/{r.get('action')}/{r.get('target')}/{r.get('value')}")
    pct = lambda x: "-" if not x else f"{100 * sum(x) / len(x):.1f}% ({sum(x)}/{len(x)})"
    tag = "sạch" if snr is None and gain is None else (f"SNR {snr} dB" if snr is not None else f"gain {gain} dB")
    print(f"[{tag}] người đăng ký thực thi đúng: {pct(enr)} | người lạ bị chặn: {pct(stg)} | "
          f"trễ TB: {statistics.mean(lat):.0f} ms" if lat else f"[{tag}] không có mẫu nào")
    if skipped:
        print(f"  (bỏ qua {skipped} file không khớp mã câu)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="*")
    ap.add_argument("--gain-db", type=float, nargs="*")
    a = ap.parse_args()
    if not CSV_FILE.exists():
        sys.exit(f"Không thấy {CSV_FILE}")
    run()
    for s in a.snr or []:
        run(snr=s)
    for g in a.gain_db or []:
        run(gain=g)


if __name__ == "__main__":
    main()