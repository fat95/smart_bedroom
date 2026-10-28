"""
Script đánh giá toàn diện hệ thống Smart Bedroom (Người C).
Yêu cầu: Phải chạy file app.py (Flask server) ở một terminal khác trước khi chạy script này.
"""

import csv
import time
import requests
from pathlib import Path

# Cấu hình đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent
TESTSET_DIR = BASE_DIR / "testset"
AUDIO_DIR = TESTSET_DIR / "audio"
CSV_FILE = TESTSET_DIR / "sentences.csv"
SERVER_URL = "http://127.0.0.1:5000/process-audio"

def run_evaluation():
    if not CSV_FILE.exists():
        print(f"❌ Không tìm thấy file {CSV_FILE}. Hãy đảm bảo bạn đã tạo dữ liệu test.")
        return

    total_samples = 0
    correct_recognition = 0
    total_latency = 0
    
    print(f"{'ID':<5} | {'THIẾT BỊ KỲ VỌNG':<18} | {'DỰ ĐOÁN':<15} | {'TRẠNG THÁI':<12} | {'ĐỘ TRỄ (ms)'}")
    print("-" * 75)

    with open(CSV_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_id = row['id']
            expected_device = row['device']
            
            # Tìm file audio (hỗ trợ cả người quen và người lạ, ví dụ: 001.wav, la_1_001.wav)
            # Tạm định dạng file audio theo ID trong file csv
            audio_path = AUDIO_DIR / f"{audio_id}.wav"
            
            if not audio_path.exists():
                continue
            
            total_samples += 1
            
            with open(audio_path, 'rb') as audio_file:
                files = {'audio_data': (f"{audio_id}.wav", audio_file, 'audio/wav')}
                try:
                    response = requests.post(SERVER_URL, files=files)
                    result = response.json()
                    
                    pred_device = result.get('device', 'N/A')
                    status = result.get('status', 'error')
                    latency = result.get('latency_ms', 0)
                    reason = result.get('reason', '')
                    
                    total_latency += latency
                    
                    # Kiểm tra tính đúng đắn: Lệnh hợp lệ và thiết bị nhận diện đúng
                    is_correct = (status == 'ok' and pred_device == expected_device)
                    
                    # Nếu kịch bản là test từ chối (vd: file của người lạ), kỳ vọng status phải là 'rejected'
                    if "la_" in audio_id and status == 'rejected':
                        is_correct = True
                        
                    if is_correct:
                        correct_recognition += 1
                        
                    display_status = status if status == 'ok' else f"từ chối ({reason})"
                    print(f"{audio_id:<5} | {expected_device:<18} | {pred_device:<15} | {display_status:<12} | {latency}")
                    
                except requests.exceptions.ConnectionError:
                    print(f"❌ Lỗi: Không thể kết nối đến {SERVER_URL}. Bạn đã chạy 'python app.py' chưa?")
                    return
                except Exception as e:
                    print(f"{audio_id:<5} | Lỗi xử lý: {e}")

    # In báo cáo tổng kết
    if total_samples > 0:
        accuracy = (correct_recognition / total_samples) * 100
        avg_latency = total_latency / total_samples
        print("-" * 75)
        print("TỔNG KẾT ĐÁNH GIÁ HỆ THỐNG:")
        print(f"- Tổng số mẫu test đã chạy : {total_samples}")
        print(f"- Tỷ lệ chính xác (Accuracy): {accuracy:.2f}%")
        print(f"- Độ trễ trung bình         : {avg_latency:.2f} ms")
    else:
        print("⚠️ Không có mẫu test nào được thực thi. Hãy kiểm tra lại thư mục testset/audio/")

if __name__ == "__main__":
    run_evaluation()