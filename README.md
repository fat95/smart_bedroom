# 🛏️ Smart Bedroom – Hệ Thống Điều Khiển Phòng Ngủ Thông Minh Bằng Giọng Nói

Dự án môn học **Xử lý tiếng nói (XLTN)** xây dựng hệ thống điều khiển phòng ngủ thông minh bằng tiếng Việt, tích hợp hai lớp bảo mật và xử lý thông minh:
1. **Xác thực người nói (Speaker Verification):** Phân biệt thành viên gia đình (đã đăng ký) và người lạ bằng đặc trưng sinh trắc học âm thanh (MFCC + F0 + SVM). Kẻ lạ nói sẽ bị chặn ngay lập tức.
2. **Nhận dạng lệnh thoại (Speech-to-Intent):** Nhận dạng tiếng nói tiếng Việt qua **PhoWhisper** và phân tích cú pháp lệnh (thiết bị, hành động, tham số, số đọc bằng chữ, khớp mờ).
3. **Điều khiển & Quản lý trạng thái:** Mô phỏng thiết bị thực tế (Điều hòa, Đèn trần, Tivi, Giường thông minh) qua giao diện Web trực quan.

---

## 📁 Cấu Trúc Dự Án

```plaintext
smart-bedroom/
├── dataset/                     # Dữ liệu âm thanh các lệnh ngắn (đăng ký giọng nói & train baseline)
│   ├── bat_den_tran/
│   ├── tat_den_tran/
│   ├── bat_dieu_hoa/
│   ├── tat_dieu_hoa/
│   ├── bat_tivi/
│   ├── tat_tivi/
│   └── noise/                  # Nhiễu môi trường (tiếng quạt, tạp âm nền)
├── testset/                    # Dữ liệu kiểm thử độc lập văn bản (Text-independent)
│   ├── audio/                  # Các file wav câu lệnh có tham số
│   └── sentences.csv           # Đáp án nhãn (id, text, device, action, value)
├── features/                   # Đặc trưng âm thanh sau khi trích xuất
│   ├── features.npz            # MFCC sequence & statistics
│   ├── manifest.csv            # Bảng danh mục toàn bộ file dữ liệu
│   └── speaker_cache.joblib    # Cache đặc trưng người nói (tăng tốc độ train)
├── models/                     # Mô hình sau khi huấn luyện
│   ├── command_model.joblib    # Mô hình phân loại lệnh baseline (SVM/kNN/RF)
│   └── speaker_model.joblib    # Mô hình SVM xác thực người nói & ngưỡng từ chối
├── notebooks/                  # Jupyter Notebook chạy trên Google Colab GPU
│   └── asr_colab.ipynb         # Đo kiểm ASR (WER, CER, Latency)
├── src/                        # Mã nguồn cốt lõi
│   ├── preprocess.py           # VAD, chuẩn hóa âm lượng, trích xuất MFCC
│   ├── train.py                # Huấn luyện mô hình nhận dạng lệnh baseline
│   ├── evaluate.py             # Đánh giá mô hình nhận dạng lệnh baseline
│   ├── speaker.py              # Trích xuất đặc trưng giọng nói & suy luận người nói
│   ├── train_speaker.py        # Huấn luyện SVM người nói & chọn ngưỡng LOSO
│   ├── eval_speaker.py         # Đánh giá EER, FAR, FRR xác thực người nói
│   ├── asr.py                  # Wrapper nạp và suy luận PhoWhisper
│   ├── commands.py             # Parser phân tích văn bản ra {device, action, value}
│   ├── eval_asr.py             # Đánh giá PhoWhisper (WER, CER, độ chính xác lệnh)
│   ├── decision.py             # Logic ra quyết định & cập nhật trạng thái thiết bị
│   ├── adapter.py              # Bộ chuyển đổi dữ liệu giữa ASR và Decision engine
│   ├── eval_system.py          # Đánh giá toàn bộ hệ thống End-to-End qua HTTP
│   └── eval_ablation.py        # Nghiên cứu Ablation (hệ thống có vs. không có xác thực)
├── templates/                  # Giao diện Web (Flask UI)
│   ├── index.html              # Bảng điều khiển phòng ngủ thông minh
│   └── record.html             # Công cụ thu âm dữ liệu qua trình duyệt
├── tests/                      # Unit tests cho parser lệnh và ASR eval
├── app.py                      # Flask Backend chính của hệ thống
├── config.py                   # Cấu hình ngưỡng, thông số âm thanh và thiết bị
├── record_routes.py            # API & giao diện thu âm dữ liệu
├── strangers.txt               # Danh sách người lạ (ngoài quy tắc đặt tên la*)
└── requirements.txt            # Danh sách thư viện phụ thuộc
```

---

## ⚙️ Cài Đặt Môi Trường

### 1. Yêu cầu hệ thống
- Python `>= 3.9`
- Khuyên dùng môi trường ảo `venv` hoặc `conda`.

### 2. Cài đặt các gói phụ thuộc
```bash
# Tạo và kích hoạt môi trường ảo (ví dụ trên Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## 🎙️ Quy Ước Thu Âm & Đặt Tên Dữ Liệu

Hệ thống cung cấp sẵn trang thu âm chuẩn 16 kHz mono tại `http://localhost:5000/record`.

### Quy tắc đặt tên file:
* **Tập huấn luyện (`dataset/`):** `<người>_<tên_lệnh>_<stt>.wav`
  - Ví dụ: `vinh_bat_tivi_001.wav`, `phat_tat_den_tran_002.wav`
  - Tên người viết thường, không dấu, không chứa dấu gạch dưới (`_`).
* **Người lạ (Strangers):** 
  - Đặt tên bắt đầu bằng `la` (ví dụ: `la1_bat_tivi_01.wav`, `la2_s09_01.wav`) HOẶC thêm tên vào file [strangers.txt](file:///c:/Users/admin/Desktop/24110299/XLTN/smart-bedroom/strangers.txt).
  - *Người lạ tuyệt đối không được đưa vào huấn luyện mô hình người nói*.
* **Tập câu có tham số (`testset/audio/`):** `<người>_<mã_câu>_<lần_thu>.wav`
  - Ví dụ: `vinh_s09_1.wav`, `la1_s15_2.wav` (mã câu khớp với cột `id` trong [testset/sentences.csv](file:///c:/Users/admin/Desktop/24110299/XLTN/smart-bedroom/testset/sentences.csv)).

---

## 🚀 Hướng Dẫn Vận Hành Dự Án (Pipeline)

### Bước 1: Tiền xử lý & Trích xuất đặc trưng
Thực hiện VAD cắt khoảng lặng, chuẩn hóa âm lượng và trích xuất MFCC:
```bash
python src/preprocess.py --data dataset --out features
```

### Bước 2: Huấn luyện mô hình nhận dạng lệnh Baseline
Huấn luyện các mô hình phân loại lệnh (SVM, k-NN, Random Forest) trên đặc trưng thống kê MFCC:
```bash
python src/train.py --features features --out models
```
Đánh giá độ chính xác baseline và ma trận nhầm lẫn:
```bash
python src/evaluate.py
```

### Bước 3: Huấn luyện xác thực người nói (Speaker Verification)
Trích xuất đặc trưng sinh trắc học (MFCC + Pitch F0), huấn luyện SVM và tự động tìm ngưỡng tối ưu qua Leave-One-Speaker-Out (LOSO):
```bash
python src/train_speaker.py --feature-set plus
```
Đánh giá tỷ lệ lỗi EER, FAR (người lạ lọt qua), FRR (người nhà bị chặn):
```bash
python src/eval_speaker.py
```

### Mô hình ASR mặc định

Hệ thống dùng `phowhisper-small` làm ASR mặc định. Có thể chọn mô hình khác,
ví dụ bản nhẹ hơn trên CPU, bằng biến môi trường `ASR_MODEL`:

```powershell
$env:ASR_MODEL = "phowhisper-base"
python app.py
```

Các tên hợp lệ gồm `phowhisper-small`, `phowhisper-base`, `whisper-small`
và `whisper-base`. Mặc định giải mã là greedy; initial prompt và n-best không
được bật mặc định.

### Bước 4: Chạy ứng dụng Web (Smart Bedroom)
Khởi động máy chủ Flask:
```bash
python app.py
```
- Mở trình duyệt và truy cập: **`http://localhost:5000`**
- Giữ nút micro trên giao diện để ra lệnh (hệ thống thu âm 4 giây, xử lý VAD $\rightarrow$ Xác thực người nói $\rightarrow$ PhoWhisper nhận dạng $\rightarrow$ Cập nhật giao diện thiết bị).
- Truy cập **`http://localhost:5000/record`** nếu muốn thu âm bổ sung thêm dữ liệu.

---

## 📊 Đánh Giá & Thử Nghiệm Nâng Cao

| Kịch bản đánh giá | Lệnh thực hiện | Mục đích |
|---|---|---|
| **Đánh giá PhoWhisper ASR** | `python src/eval_asr.py` | Đo WER, CER, độ chính xác thực thi lệnh trên tập testset |
| **Ablation Study (A vs A+B)** | `python src/eval_ablation.py` | So sánh độ an toàn và khả năng sử dụng khi Có vs. Không có xác thực người nói |
| **Đánh giá toàn hệ thống (End-to-End)** | `python src/eval_system.py` | Giả lập gửi request HTTP có thêm nhiễu trắng (`--snr 20 10 5`) và suy giảm âm lượng (`--gain-db -10`) |
| **Chạy kiểm thử Unit Test** | `python -m unittest discover -s tests -v` | Kiểm tra độ phủ phân tích ngữ pháp lệnh `commands.py` (~90 test cases) |

---

## 🎛️ Thiết Bị Hỗ Trợ Trong Phòng Ngủ

| Thiết bị (`device`) | Từ khóa nhận diện | Hành động (`action`) | Khoảng giá trị (`value`) |
|---|---|---|---|
| **Điều hòa** (`ac`) | `dieu hoa`, `may lanh` | `on`, `off`, `up`, `down`, `set`, `mode_cool`, `mode_fan`, `mode_dry` | `16` - `30` (°C) |
| **Đèn trần** (`light-main`) | `den tran`, `den` | `on`, `off`, `up`, `down`, `set` | `0` - `100` (%) |
| **Tivi** (`tv`) | `tivi`, `ti vi` | `on`, `off`, `up`, `down`, `set` (âm lượng), `channel` (kênh) | Kênh: `1` - `99`, Âm lượng: `0` - `100` |
| **Giường thông minh** (`bed`) | `giuong`, `dau giuong` | `up`, `down`, `set` (gồm cả lệnh *"nằm phẳng"* $\rightarrow$ value = 0) | `0` - `45` (độ nghiêng) |

---

## 👥 Phân Công Công Việc Nhóm
- **Phần A (Speech Recognition & NLU):** Tích hợp PhoWhisper, xử lý văn bản tiếng Việt, khớp lệnh mờ `commands.py`, đánh giá ASR.
- **Phần B (Speaker Verification):** Trích xuất MFCC + F0 pitch, huấn luyện SVM, chọn ngưỡng LOSO, phát hiện & từ chối người lạ.
- **Phần C (System Integration & Web UI):** Xây dựng Flask backend, bộ điều khiển logic thiết bị `decision.py`, công cụ thu âm và giao diện điều khiển phòng ngủ.
