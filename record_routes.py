# -*- coding: utf-8 -*-
"""
record_routes.py - Trang thu âm dữ liệu (blueprint record_bp).

Hai chế độ thu:
  cmd      lệnh đơn giản   -> dataset/<lệnh>/<người>_<lệnh>_<số>.wav     (huấn luyện)
  sentence câu có tham số  -> testset/audio/<người>_<mã câu>_<số>.wav    (chỉ để kiểm thử)

Quy ước tên người: chữ thường không dấu và số, KHÔNG chứa "_" (vinh, phat, la1).
Người lạ đặt tên la1, la2, ... để speaker.py nhận ra.

Gắn vào app.py (nếu chưa có):
    from record_routes import record_bp
    app.register_blueprint(record_bp)
"""
from __future__ import annotations

import csv
import io
import re
import threading
import wave
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, render_template, request

record_bp = Blueprint("record", __name__)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
TESTSET_DIR = ROOT / "testset" / "audio"
SENTENCES_CSV = ROOT / "testset" / "sentences.csv"

SR = 16000
SECONDS = 4                 # độ dài mỗi bản thu
NAME_WITH_LABEL = True      # True: người_lệnh_số.wav   False: người_số.wav (kiểu cũ, chỉ cho dataset)
MIN_RMS = 0.004             # nhỏ hơn thì coi là quá nhỏ (trừ lớp noise)
MAX_TAIL_ZERO_S = 0.3       # đuôi toàn số 0 dài hơn mức này -> bản thu bị hụt

NAME_RE = re.compile(r"^[a-z][a-z0-9]{1,15}$")
LABEL_RE = re.compile(r"^[a-z0-9_]+$")
FILE_RE = re.compile(r"^[a-z][a-z0-9]{1,15}_[a-z0-9_]*\d+\.wav$")
STRANGER_RE = re.compile(r"^la\d*$")
STRANGERS_FILE = ROOT / "strangers.txt"   # tên người lạ khác (mỗi dòng một tên), vd: vinh


def stranger_names() -> set[str]:
    if not STRANGERS_FILE.exists():
        return set()
    out = set()
    for line in STRANGERS_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            out.add(line)
    return out


def is_stranger(name: str) -> bool:
    return bool(STRANGER_RE.fullmatch(name)) or name in stranger_names()

# (nhãn thư mục, chữ hiển thị). Bỏ rèm cửa; thêm tắt tivi.
COMMANDS = [
    ("bat_dieu_hoa", "bật điều hòa"),
    ("tat_dieu_hoa", "tắt điều hòa"),
    ("bat_den_tran", "bật đèn trần"),
    ("tat_den_tran", "tắt đèn trần"),
    ("bat_tivi", "bật tivi"),
    ("tat_tivi", "tắt tivi"),
    ("noise", "(giữ im lặng hoặc tiếng ồn nền)"),
]

# Câu có tham số mặc định: id,text,device,action,value. Có thể sửa trực tiếp file CSV.
DEFAULT_SENTENCES = [
    ("c01", "tăng điều hòa hai độ", "ac", "temp_up", "2"),
    ("c02", "giảm điều hòa ba độ", "ac", "temp_down", "3"),
    ("c03", "đặt điều hòa hai mươi sáu độ", "ac", "temp_set", "26"),
    ("c04", "đặt điều hòa hai mươi hai độ", "ac", "temp_set", "22"),
    ("c05", "giảm điều hòa một độ", "ac", "temp_down", "1"),
    ("c06", "tăng điều hòa năm độ", "ac", "temp_up", "5"),
    ("c07", "đặt điều hòa mười sáu độ", "ac", "temp_set", "16"),
    ("c08", "đặt điều hòa mười chín độ", "ac", "temp_set", "19"),
    ("c09", "điều hòa chế độ làm lạnh", "ac", "mode_set", "cool"),
    ("c10", "tăng đèn trần hai mươi phần trăm", "light", "brightness_up", "20"),
    ("c11", "giảm đèn trần ba mươi phần trăm", "light", "brightness_down", "30"),
    ("c12", "đặt đèn trần năm mươi phần trăm", "light", "brightness_set", "50"),
    ("c13", "đặt đèn trần tám mươi phần trăm", "light", "brightness_set", "80"),
    ("c14", "đặt đèn trần một trăm phần trăm", "light", "brightness_set", "100"),
    ("c15", "nâng đầu giường mười độ", "bed", "bed_up", "10"),
    ("c16", "hạ đầu giường mười lăm độ", "bed", "bed_down", "15"),
    ("c17", "đặt đầu giường ba mươi độ", "bed", "bed_set", "30"),
    ("c18", "hạ giường nằm phẳng", "bed", "bed_flat", "0"),
    ("c19", "chuyển tivi kênh năm", "tv", "channel_set", "5"),
    ("c20", "chuyển tivi kênh chín", "tv", "channel_set", "9"),
    ("c21", "chuyển tivi kênh ba", "tv", "channel_set", "3"),
    ("c22", "tăng âm lượng tivi mười", "tv", "volume_up", "10"),
    ("c23", "giảm âm lượng tivi hai mươi", "tv", "volume_down", "20"),
    ("c24", "đặt âm lượng tivi bốn mươi", "tv", "volume_set", "40"),
    ("c25", "bật điều hòa", "ac", "power_on", ""),
    ("c26", "tắt đèn trần", "light", "power_off", ""),
    ("c27", "bật tivi", "tv", "power_on", ""),
    ("c28", "tắt tivi", "tv", "power_off", ""),
]

_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------
def load_sentences() -> list[dict]:
    if not SENTENCES_CSV.exists():
        SENTENCES_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(SENTENCES_CSV, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text", "device", "action", "value"])
            w.writerows(DEFAULT_SENTENCES)
    out = []
    with open(SENTENCES_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("id") or "").strip()
            if sid and LABEL_RE.fullmatch(sid) and "_" not in sid:
                out.append({"id": sid, "text": (row.get("text") or "").strip(),
                            "device": (row.get("device") or "").strip(),
                            "action": (row.get("action") or "").strip(),
                            "value": (row.get("value") or "").strip()})
    return out


def valid_labels(mode: str) -> set[str]:
    if mode == "cmd":
        return {c for c, _ in COMMANDS}
    return {s["id"] for s in load_sentences()}


def folder_for(mode: str, label: str) -> Path:
    return DATASET_DIR / label if mode == "cmd" else TESTSET_DIR


def index_of(stem: str, speaker: str, label: str):
    """Số thứ tự trong tên file; hiểu cả kiểu mới (người_nhãn_số) lẫn kiểu cũ (người_số)."""
    pre = speaker + "_"
    if not stem.startswith(pre):
        return None
    rest = stem[len(pre):]
    if rest.isdigit():
        return int(rest)
    if rest.startswith(label + "_") and rest[len(label) + 1:].isdigit():
        return int(rest[len(label) + 1:])
    return None


def indices(mode: str, speaker: str, label: str) -> list[int]:
    folder = folder_for(mode, label)
    if not folder.exists():
        return []
    got = (index_of(p.stem, speaker, label) for p in folder.glob(f"{speaker}_*.wav"))
    return [i for i in got if i is not None]


def counts_for(mode: str, speaker: str) -> dict:
    return {label: len(indices(mode, speaker, label)) for label in sorted(valid_labels(mode))}


def err(msg: str, code: int = 400):
    return jsonify(ok=False, error=msg), code


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@record_bp.route("/record")
def record_page():
    return render_template("record.html")


@record_bp.route("/record/config")
def record_config():
    return jsonify(seconds=SECONDS, sr=SR, minRms=MIN_RMS, maxTailZero=MAX_TAIL_ZERO_S,
                   strangers=sorted(stranger_names()),
                   commands=[{"id": c, "text": t} for c, t in COMMANDS],
                   sentences=load_sentences())


@record_bp.route("/sample-counts")
def sample_counts():
    mode = request.args.get("mode", "cmd")
    speaker = request.args.get("speaker", "").strip().lower()
    if mode not in ("cmd", "sentence") or not NAME_RE.fullmatch(speaker):
        return jsonify(counts={})
    return jsonify(counts=counts_for(mode, speaker))


@record_bp.route("/record/summary")
def record_summary():
    """Mỗi người đã thu bao nhiêu file (để nhóm kiểm tra tiến độ)."""
    tot: dict[str, dict] = {}

    def add(path: Path, key: str):
        sp = path.stem.split("_", 1)[0]
        t = tot.setdefault(sp, {"speaker": sp, "dataset": 0, "testset": 0,
                                "stranger": is_stranger(sp)})
        t[key] += 1

    if DATASET_DIR.exists():
        for d in DATASET_DIR.iterdir():
            if d.is_dir() and d.name != "noise":
                for p in d.glob("*.wav"):
                    add(p, "dataset")
    if TESTSET_DIR.exists():
        for p in TESTSET_DIR.glob("*.wav"):
            add(p, "testset")
    return jsonify(speakers=sorted(tot.values(), key=lambda x: x["speaker"]))


@record_bp.route("/save-sample", methods=["POST"])
def save_sample():
    mode = request.args.get("mode", "cmd")
    speaker = request.args.get("speaker", "").strip().lower()
    label = request.args.get("label", "")
    if mode not in ("cmd", "sentence"):
        return err("Chế độ không hợp lệ.")
    if not NAME_RE.fullmatch(speaker):
        return err("Tên người chỉ gồm chữ thường không dấu và số (2-16 ký tự), không có dấu gạch dưới.")
    if label not in valid_labels(mode):
        return err("Nhãn không hợp lệ.")

    try:
        with wave.open(io.BytesIO(request.get_data()), "rb") as w:
            ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError):
        return err("Dữ liệu gửi lên không phải file WAV hợp lệ.")
    if ch != 1 or width != 2 or rate != SR:
        return err(f"WAV phải là mono, 16-bit, {SR} Hz (nhận được {ch} kênh, {width * 8}-bit, {rate} Hz).")
    if abs(n / SR - SECONDS) > 0.1:
        return err(f"Bản thu dài {n / SR:.2f}s, cần {SECONDS}s.")

    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    nz = np.nonzero(np.abs(x) > 1e-4)[0]
    tail = (len(x) - 1 - int(nz[-1])) / SR if len(nz) else len(x) / SR
    is_noise = (mode == "cmd" and label == "noise")
    warnings = []
    if not is_noise:
        if rms < MIN_RMS:
            return err(f"Âm quá nhỏ (RMS {rms:.4f}). Hãy tăng âm lượng micro rồi thu lại.")
        if tail > MAX_TAIL_ZERO_S:
            return err(f"Bản thu bị hụt đuôi ({tail:.2f}s toàn số 0). Hãy thu lại.")
    elif rms < 1e-4:
        return err("Micro đang không có tín hiệu (toàn số 0). Kiểm tra lại micro.")
    if peak >= 0.99:
        warnings.append("Âm bị méo (chạm trần). Nên giảm âm lượng micro.")
    elif not is_noise and rms < 0.02:
        warnings.append("Âm hơi nhỏ, nên nói gần micro hơn.")

    folder = folder_for(mode, label)
    with _LOCK:
        folder.mkdir(parents=True, exist_ok=True)
        idx = indices(mode, speaker, label)
        n_next = (max(idx) if idx else 0) + 1
        if mode == "cmd" and not NAME_WITH_LABEL:
            name = f"{speaker}_{n_next:02d}.wav"
        else:
            name = f"{speaker}_{label}_{n_next:02d}.wav"
        tmp = folder / (name + ".part")
        tmp.write_bytes(request.get_data())
        tmp.replace(folder / name)
        count = len(idx) + 1
    return jsonify(ok=True, filename=name, count=count, rms=round(rms, 4),
                   peak=round(peak, 3), warnings=warnings)


@record_bp.route("/delete-sample", methods=["POST"])
def delete_sample():
    """Xóa một file vừa thu nhầm (chỉ trong dataset/ và testset/audio/)."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("filename", ""))
    if not FILE_RE.fullmatch(name):
        return err("Tên file không hợp lệ.")
    allowed = [TESTSET_DIR]
    if DATASET_DIR.exists():
        allowed += [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    with _LOCK:
        for d in allowed:
            p = d / name          # name đã qua FILE_RE nên không chứa dấu "/" hay ".."
            if p.is_file():
                p.unlink()
                return jsonify(ok=True, deleted=name)
    return err("Không tìm thấy file.", 404)