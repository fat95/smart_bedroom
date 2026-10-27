"""Smart Bedroom – Flask backend.

Luồng: trình duyệt thu 2 s (WAV 16 kHz) -> POST /process-audio -> tiền xử lý + MFCC (src/preprocess.py)
       -> mô hình nhận dạng lệnh (models/command_model.joblib) -> JSON {command, action, confidence}
       -> index.html cập nhật thiết bị mô phỏng.

Chạy:  python app.py   rồi mở http://localhost:5000
Trang thu thập dữ liệu: http://localhost:5000/record
"""
"""
File: app.py
Chức năng chính: Web Server Flask - Trung tâm tiếp nhận HTTP Request và điều phối hệ thống.
Nhiệm vụ:
  1. Lắng nghe yêu cầu POST gửi dữ liệu file âm thanh WAV từ giao diện web (Trình duyệt).
  2. Chuyển giao dữ liệu âm thanh cho module preprocess.py để trích xuất đặc trưng (MFCC).
  3. Sử dụng mô hình Machine Learning (Người A) để nhận dạng câu lệnh.
  4. Lấy kết quả xác thực người nói (Người B) để kiểm tra danh tính.
  5. Đẩy toàn bộ thông tin qua hàm decide() (Người C) để ra kết quả cuối cùng phản hồi dạng JSON.
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

# Thêm đường dẫn thư mục src vào hệ thống để Python có thể import các file nội bộ
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

import preprocess as pp  # Module tiền xử lý tín hiệu âm thanh
from decision import decide  # Hàm kiểm tra logic & ra quyết định thực thi
from record_routes import record_bp  # Router phục vụ giao diện thu âm (/record)

# Đường dẫn đến file mô hình đã huấn luyện của Người A
MODEL_PATH = BASE / "models" / "command_model.joblib"
# Ngưỡng tin cậy tối thiểu để chấp nhận câu lệnh (mặc định 0.5)
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.5))

# Bảng ánh xạ từ mã nhãn của mô hình ASR sang đối tượng chi tiết tương ứng trong hệ thống
COMMANDS = {
    "bat_dieu_hoa": {
        "text": "Bật điều hòa",
        "device": "điều hòa",
        "action": "bật",
        "value": None,
    },
    "tat_dieu_hoa": {
        "text": "Tắt điều hòa",
        "device": "điều hòa",
        "action": "tắt",
        "value": None,
    },
    "bat_den_tran": {
        "text": "Bật đèn trần",
        "device": "đèn trần",
        "action": "bật",
        "value": 100,
    },
    "tat_den_tran": {
        "text": "Tắt đèn trần",
        "device": "đèn trần",
        "action": "tắt",
        "value": 0,
    },
    "bat_tivi": {
        "text": "Bật tivi",
        "device": "tivi",
        "action": "bật",
        "value": None,
    },
    "tat_tivi": {
        "text": "Tắt tivi",
        "device": "tivi",
        "action": "tắt",
        "value": None,
    },
}

app = Flask(__name__)
app.register_blueprint(record_bp)

# Nạp mô hình ASR vào bộ nhớ khi khởi động Server
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
print(
    "Mô hình ASR:",
    (
        bundle["name"]
        if bundle
        else f"CHƯA CÓ ({MODEL_PATH}) – hãy chạy src/train.py"
    ),
)


@app.route("/")
def index():
    """Hàm trả về giao diện điều khiển chính (Dashboard)."""
    return render_template("index.html")


@app.route("/process-audio", methods=["POST"])
def process_audio():
    """API chính nhận file WAV từ Frontend, chạy toàn bộ luồng ASR -> Speaker Verification -> Logic."""
    if bundle is None:
        return (
            jsonify(
                {
                    "error": (
                        "Chưa có mô hình. Chạy src/preprocess.py rồi"
                        " src/train.py trước."
                    )
                }
            ),
            503,
        )

    # Lấy file âm thanh từ request được tải lên từ trình duyệt
    f = request.files.get("audio_data")
    if f is None:
        return jsonify({"error": "Không nhận được file âm thanh"}), 400

    t0 = time.perf_counter()

    # --- PHẦN 1: ĐỌC VÀ CHUẨN HÓA TÍN HIỆU ÂM THANH ---
    try:
        y, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=True)
    except Exception:
        return jsonify({"error": "Không đọc được âm thanh. Cần file WAV."}), 400

    # Chuyển kênh âm thanh về dạng Mono (1 kênh)
    y = y.mean(axis=1)

    # Chuyển đổi tần số lấy mẫu (Resample) về đúng chuẩn 16000 Hz nếu khác tần số
    if sr != pp.SR:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=pp.SR)

    # --- PHẦN 2: TRÍCH XUẤT ĐẶC TRƯNG VÀ DỰ ĐOÁN LỆNH (NGƯỜI A) ---
    # Phân tích tín hiệu âm thanh và kiểm tra VAD (phát hiện tiếng nói)
    r = pp.process_signal(y)

    # Đưa đặc trưng âm thanh vào mô hình dự đoán xác suất từng câu lệnh
    proba = bundle["model"].predict_proba(r["stat"][None, :])[0]
    i = int(np.argmax(proba))
    label, conf = str(bundle["classes"][i]), float(proba[i])
    latency = round((time.perf_counter() - t0) * 1000, 1)

    print(
        f"[voice] {label} conf={conf:.2f} speech={r['vad']['found']}"
        f" {latency} ms"
    )
    base_info = {
        "command": label,
        "confidence": round(conf, 3),
        "latency_ms": latency,
    }

    # Bắt lỗi 1: Không tìm thấy đoạn tiếng nói trong file ghi âm (Chỉ có tiếng ồn hoặc im lặng)
    if not r["vad"]["found"]:
        return jsonify(
            {**base_info, "status": "rejected", "reason": "no_speech"}
        )

    # Bắt lỗi 2: Độ tự tin của mô hình ASR quá thấp (không rõ từ nói gì)
    if conf < CONF_THRESHOLD:
        return jsonify(
            {**base_info, "status": "rejected", "reason": "unknown_command"}
        )

    # Bắt lỗi 3: Nhãn dự đoán nằm ngoài từ điển hỗ trợ
    if label not in COMMANDS:
        return jsonify(
            {**base_info, "status": "rejected", "reason": "unknown_command"}
        )

    command_data = COMMANDS[label]

    # --- PHẦN 3: XÁC THỰC GIỌNG NÓI NGƯỜI DÙNG (NGƯỜI B) ---
    # Dữ liệu giả lập mô hình Speaker Verification (Người B sẽ thay thế bằng hàm dự đoán thực tế)
    speaker_data = {
        "speaker_id": "speaker_1",  # ID thành viên hợp lệ trong hệ thống
        "score": 0.85,  # Điểm tương đồng giọng nói (85%)
    }

    # --- PHẦN 4: ĐỐI CHIẾU DỮ LIỆU VÀ TỔNG HỢP KẾT QUẢ (NGƯỜI C) ---
    # Gọi hàm decide() để kiểm tra tính hợp lệ toàn diện
    final_decision = decide(command_data, speaker_data)

    # Ghép dữ liệu đo lường hệ thống với kết quả quyết định để phản hồi về Client
    response = {**base_info, **final_decision}
    response["text"] = command_data.get("text", "")

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)