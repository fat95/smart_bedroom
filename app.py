"""Smart Bedroom – Flask backend.

Luồng: trình duyệt thu 2 s (WAV 16 kHz) -> POST /process-audio -> tiền xử lý + MFCC (src/preprocess.py)
       -> mô hình nhận dạng lệnh (models/command_model.joblib) -> JSON {command, action, confidence}
       -> index.html cập nhật thiết bị mô phỏng.

Chạy:  python app.py   rồi mở http://localhost:5000
Trang thu thập dữ liệu: http://localhost:5000/record
"""
import io
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
import preprocess as pp  # noqa: E402
from record_routes import record_bp  # noqa: E402

MODEL_PATH = BASE / "models" / "command_model.joblib"
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.5))  # dưới ngưỡng này -> "không hiểu lệnh"

# nhãn lệnh -> hành động trên giao diện (device khớp id trong index.html)
COMMANDS = {
    "bat_dieu_hoa": ("Bật điều hòa", {"device": "ac", "action": "on"}),
    "tat_dieu_hoa": ("Tắt điều hòa", {"device": "ac", "action": "off"}),
    "bat_den_tran": ("Bật đèn trần", {"device": "light-main", "action": "on", "value": 80}),
    "tat_den_tran": ("Tắt đèn trần", {"device": "light-main", "action": "off", "value": 0}),
    "mo_rem": ("Mở rèm", {"device": "curtain", "action": "open", "value": 100}),
    "bat_tivi": ("Bật tivi", {"device": "tv", "action": "on"}),
}

app = Flask(__name__)
app.register_blueprint(record_bp)
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
print("Mô hình:", bundle["name"] if bundle else f"CHƯA CÓ ({MODEL_PATH}) – hãy chạy src/train.py")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-audio", methods=["POST"])
def process_audio():
    if bundle is None:
        return jsonify({"error": "Chưa có mô hình. Chạy src/preprocess.py rồi src/train.py trước."}), 503
    f = request.files.get("audio_data")
    if f is None:
        return jsonify({"error": "Không nhận được file âm thanh"}), 400

    t0 = time.perf_counter()
    try:
        y, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=True)
    except Exception:
        return jsonify({"error": "Không đọc được âm thanh. Cần file WAV."}), 400
    y = y.mean(axis=1)
    if sr != pp.SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=pp.SR)

    r = pp.process_signal(y)
    proba = bundle["model"].predict_proba(r["stat"][None, :])[0]
    i = int(np.argmax(proba))
    label, conf = str(bundle["classes"][i]), float(proba[i])
    latency = round((time.perf_counter() - t0) * 1000, 1)
    print(f"[voice] {label} conf={conf:.2f} speech={r['vad']['found']} {latency} ms")

    base = {"command": label, "confidence": round(conf, 3), "latency_ms": latency}
    if not r["vad"]["found"]:
        return jsonify({**base, "status": "rejected", "reason": "no_speech"})
    if label not in COMMANDS:  # lớp noise hoặc nhãn lạ
        return jsonify({**base, "status": "rejected", "reason": "noise"})
    if conf < CONF_THRESHOLD:
        return jsonify({**base, "status": "rejected", "reason": "low_confidence"})
    text, action = COMMANDS[label]
    return jsonify({**base, "status": "ok", "text": text, "action": action})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)