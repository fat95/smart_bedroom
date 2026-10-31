"""
File: app.py
Smart Bedroom - Flask backend.

Luồng: trình duyệt thu 4 s (WAV 16 kHz mono) -> POST /process-audio
  1. đọc WAV, VAD: không có tiếng nói -> no_speech
  2. xác thực người nói (B): người lạ -> unauthorized_speaker NGAY, không chạy ASR,
     không trả transcript/lệnh (người lạ không biết lệnh có hợp lệ hay không)
  3. PhoWhisper + commands.py (A) -> {device, action, value}
  4. adapter -> decide() (C) -> JSON cho index.html

Chạy:  python app.py   rồi mở http://localhost:5000
Biến môi trường: ASR_MODEL (mặc định config.ASR_DEFAULT), FLASK_DEBUG=1
"""
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE))

import config  # noqa: E402
import preprocess as pp  # noqa: E402
from adapter import to_decision_input  # noqa: E402
from decision import decide  # noqa: E402
from record_routes import record_bp  # noqa: E402
from speaker import load_bundle as load_speaker_bundle  # noqa: E402
from speaker import verify_speaker  # noqa: E402

ASR_MODEL = os.environ.get("ASR_MODEL", config.ASR_DEFAULT)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 4 s WAV chỉ khoảng 128 KB
app.register_blueprint(record_bp)

# ---- nạp mô hình MỘT lần khi khởi động ----
spk_bundle = None
try:
    bundle = load_speaker_bundle()
    print("Mô hình người nói:", ", ".join(bundle["speakers"]),
          f"| điểm: {bundle['method']} | ngưỡng: {bundle['thresholds'][bundle['method']]:.3f}")
    verify_speaker(np.random.default_rng(0).standard_normal(pp.SR).astype("float32") * 0.1, sr=pp.SR)
    spk_bundle = bundle
except FileNotFoundError:
    print("Mô hình người nói: CHƯA CÓ - hãy chạy src/train_speaker.py")
except Exception as e:
    spk_bundle = None
    print("Cảnh báo khi khởi động mô hình người nói:", e)

asr_ready = False
try:
    import asr
    asr.warmup(ASR_MODEL)
    asr_ready = True
except Exception as e:  # thiếu torch/transformers, không tải được mô hình...
    print("ASR chưa sẵn sàng:", e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-audio", methods=["POST"])
def process_audio():
    if spk_bundle is None:
        return jsonify({"error": "Chưa có mô hình người nói. Chạy src/train_speaker.py trước."}), 503
    if not asr_ready:
        return jsonify({"error": "Mô hình nhận dạng lệnh (PhoWhisper) chưa sẵn sàng. Xem log khi khởi động."}), 503

    f = request.files.get("audio_data")
    if f is None:
        return jsonify({"error": "Không nhận được file âm thanh"}), 400

    t0 = time.perf_counter()
    elapsed = lambda: round((time.perf_counter() - t0) * 1000, 1)

    # --- 1. đọc và chuẩn hóa âm thanh ---
    try:
        y, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=True)
    except Exception:
        return jsonify({"error": "Không đọc được âm thanh. Cần file WAV."}), 400
    y = y.mean(axis=1)
    if sr != pp.SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=pp.SR)

    if not pp.detect_speech(y)["found"]:
        return jsonify({"status": "rejected", "reason": "no_speech", "latency_ms": elapsed()})

    # --- 2. người nói TRƯỚC ---
    spk = verify_speaker(y, sr=pp.SR)
    speaker_data = {"speaker_id": spk["speaker"], "score": spk["score"], "is_enrolled": spk["is_enrolled"]}
    print(f"[speaker] {spk['best_guess']} score={spk['score']:.3f} enrolled={spk['is_enrolled']}")
    base = {"speaker_score": round(spk["score"], 3)}

    if not spk["is_enrolled"]:
        final = decide(None, speaker_data)           # luôn là unauthorized_speaker
        return jsonify({**final, **base, "latency_ms": elapsed()})

    # --- 3. nhận dạng lệnh ---
    cmd = asr.recognize_command(y, ASR_MODEL)
    print(f"[asr] '{cmd['text']}' -> {cmd['device']}/{cmd['action']}/{cmd['value']} "
          f"score={cmd['score']} reason={cmd['reason']}")

    # --- 4. quyết định ---
    final = decide(to_decision_input(cmd), speaker_data)
    return jsonify({**final, **base, "text": cmd["text"], "command_score": cmd["score"],
                    "latency_ms": elapsed()})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)
