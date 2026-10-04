"""Routes thu âm dữ liệu. Thêm vào app.py:

    from record_routes import record_bp
    app.register_blueprint(record_bp)

Mở http://localhost:5000/record
"""
import re
from pathlib import Path
from flask import Blueprint, jsonify, render_template, request

record_bp = Blueprint("record", __name__)
DATASET = Path(__file__).resolve().parent / "dataset"
COMMANDS = ["bat_dieu_hoa", "tat_dieu_hoa", "bat_den_tran", "tat_den_tran", "mo_rem", "bat_tivi", "noise"]
SPEAKER_RE = re.compile(r"^[a-z0-9]{2,20}$")


def _indices(folder: Path, speaker: str):
    pat = re.compile(rf"{speaker}_(\d+)\.wav")
    return [int(m.group(1)) for p in folder.glob(f"{speaker}_*.wav") if (m := pat.fullmatch(p.name))]


@record_bp.route("/record")
def record_page():
    return render_template("record.html")


@record_bp.route("/sample-counts")
def sample_counts():
    speaker = request.args.get("speaker", "")
    if not SPEAKER_RE.match(speaker):
        return jsonify({})
    return jsonify({c: len(_indices(DATASET / c, speaker)) for c in COMMANDS})


@record_bp.route("/save-sample", methods=["POST"])
def save_sample():
    command = request.form.get("command", "")
    speaker = request.form.get("speaker", "")
    f = request.files.get("audio")
    if command not in COMMANDS or not SPEAKER_RE.match(speaker) or f is None:
        return jsonify({"error": "Dữ liệu gửi lên không hợp lệ"}), 400
    data = f.read(1_000_000)
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return jsonify({"error": "File không phải WAV"}), 400
    folder = DATASET / command
    folder.mkdir(parents=True, exist_ok=True)
    idx = max(_indices(folder, speaker), default=0) + 1
    (folder / f"{speaker}_{idx:03d}.wav").write_bytes(data)
    return jsonify({"count": len(_indices(folder, speaker)), "file": f"{speaker}_{idx:03d}.wav"})
