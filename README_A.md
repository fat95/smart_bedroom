# Phần A – Nhận dạng lệnh bằng PhoWhisper

## File của A
| File | Việc |
|---|---|
| `config.py` | **Bản tạm** do A viết để chạy được. C thay bằng bản chính, giữ nguyên cấu trúc `DEVICES` |
| `src/commands.py` | `parse_command(text)` -> `{device, action, value, score, reason}` (chuẩn hóa, số đọc bằng chữ, khớp mờ) |
| `src/asr.py` | `recognize_command(wav)`: nạp PhoWhisper một lần, nhận dạng, gọi `parse_command` |
| `src/eval_asr.py` | WER, CER, độ chính xác thiết bị/hành động/giá trị, ma trận nhầm lẫn, độ trễ; so với Whisper và baseline MFCC |
| `notebooks/asr_colab.ipynb` | Chạy `eval_asr.py` trên Colab GPU, có hướng dẫn từng bước |
| `tests/` | Test cho `commands.py` (khoảng 90 câu) và `eval_asr.py`, chạy không cần mô hình |
| `testset/sentences.csv` | **Bản nháp** 30 câu mẫu để C thu âm (C sửa được, miễn đúng cột) |

## Giá trị `action` (cả nhóm dùng chung)
`on`, `off`, `up`, `down`, `set`, `channel`, `mode_cool`, `mode_fan`, `mode_dry`. Âm lượng tivi và góc giường dùng `up/down/set`; "nằm phẳng" là `set` với `value = 0`.

## Hai điểm thêm so với hợp đồng gốc (cần báo B và C)
1. `recognize_command` trả thêm `reason`, `latency_ms`, `model`. **`decision.py` phải kiểm tra `reason` trước**; `reason` có thể là
   `ambiguous_device`, `unknown_command`, `missing_value`, `empty_text` (`missing_value` nên map sang `invalid_value`).
2. `invalid_value` (giá trị ngoài khoảng) do `decision.py` kiểm tra bằng `config.value_in_range(device, action, value)`.

## Chạy thử trên máy
```
pip install -r requirements-asr.txt
python -m unittest discover -s tests -v          # không cần mô hình
python src/commands.py "đặt điều hòa hai mươi sáu độ"
python src/asr.py dataset/bat_tivi/vinh_bat_tivi_001.wav   # lần đầu tải mô hình
```
Trong `app.py`: gọi `asr.warmup()` một lần khi khởi động rồi gọi `asr.recognize_command(wav_bytes)` mỗi request.

## Đo số liệu (Colab)
Mở `notebooks/asr_colab.ipynb`, làm theo từng bước. Cần có `testset/audio/*.wav` (do C thu) cho câu 2.3; câu 2.4 dùng `dataset/` có sẵn.
Đặt tên file âm thanh: `<người>_<mã câu>_<số lần>.wav`, ví dụ `vinh_s09_1.wav`, `la1_s09_2.wav` (mã câu lấy từ cột `id` của `sentences.csv`).
