# 🎭 BTLKPDL - Face Recognition Project

Hệ thống nhận diện khuôn mặt và phân tích cảm xúc sử dụng Deep Learning, Flask, OpenCV và DeepFace.

## 📋 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Cấu trúc project](#cấu-trúc-project)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Ghi chú](#ghi-chú)

## 🎯 Giới thiệu

BTLKPDL (Bài Tập Lớn Khai Phá Dữ Liệu) là ứng dụng web cho phép người dùng:
- **Nhận diện danh tính**: Xác định người trong ảnh từ dataset được huấn luyện trước
- **Phân tích cảm xúc**: Nhận dạng các cảm xúc (vui, buồn, giận, sợ, ghê tởm, ngạc nhiên, bình thường)
- **Dự đoán tuổi tác**: Ước tính tuổi của người trong ảnh
- **Xác định giới tính**: Phân loại nam/nữ
- **Nhận diện hướng khuôn mặt**: Xác định người đang nhìn hướng nào (trái, phải, thẳng)

## ✨ Tính năng

### 📸 Tính năng chính
1. **Upload ảnh tĩnh**
   - Tải ảnh từ máy tính
   - Nhận diện toàn bộ thông tin trong một ảnh
   - Hiển thị kết quả chi tiết trên giao diện

2. **Nhận diện qua Webcam**
   - Truyền phát video real-time từ webcam
   - Nhận diện danh tính trong thời gian thực
   - Tối ưu hóa tốc độ xử lý

3. **Xử lý Video**
   - Tải video lên máy chủ
   - Nhận diện khuôn mặt trong suốt video
   - Tải xuống video đã xử lý

### 🔧 Các tính năng xử lý ảnh
- **Tiền xử lý ảnh**: Điều chỉnh độ sáng, tương phản bằng histogram equalization
- **Nhận diện khuôn mặt**: Sử dụng model Deep Learning đã được huấn luyện
- **Phân tích đặc trưng**: Cảm xúc, tuổi, giới tính bằng DeepFace
- **Nhận diện hướng**: Sử dụng facial landmarks từ dlib

## 🖥️ Yêu cầu hệ thống

- **Python**: 3.8 hoặc cao hơn
- **OS**: Windows, macOS hoặc Linux
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Webcam**: (tùy chọn) Cho tính năng nhận diện qua webcam
- **GPU**: (tùy chọn) NVIDIA GPU giúp tăng tốc độ xử lý

## 📦 Cài đặt

### 1. Clone hoặc tải project
```bash
git clone <repository-url>
cd btlkpdl
```

### 2. Tạo môi trường ảo (Virtual Environment)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

Nếu không có file `requirements.txt`, cài đặt các package sau:
```bash
pip install flask tensorflow opencv-python deepface dlib numpy scikit-learn
```

### 4. Chuẩn bị dữ liệu
- Đặt model đã huấn luyện: `face_recognition_model.h5` vào thư mục gốc
- Đặt label encoder: `label_encoder.pkl` vào thư mục gốc
- Đặt file dlib: `shape_predictor_68_face_landmarks.dat` vào thư mục gốc

### 5. Chạy ứng dụng
```bash
# Đảm bảo bạn đang ở thư mục gốc và có kích hoạt virtual environment
python btlkpdl/app.py
```

Ứng dụng sẽ chạy tại: `http://localhost:5000`

## 🚀 Cách sử dụng

### Bước 1: Truy cập ứng dụng
Mở trình duyệt và đi đến: `http://localhost:5000`

### Bước 2: Chọn chế độ nhận diện

#### Chế độ 1: Upload ảnh tĩnh
1. Nhấn "Chọn file" để chọn ảnh từ máy tính
2. Nhấn "Nhận diện"
3. Xem kết quả bao gồm:
   - Tên người được nhận diện
   - Độ chính xác (%)
   - Cảm xúc
   - Giới tính
   - Tuổi
   - Hướng khuôn mặt

#### Chế độ 2: Nhận diện qua Webcam
1. Nhấn vào link "Mở Webcam"
2. Cho phép trình duyệt truy cập webcam
3. Xem kết quả nhận diện real-time

#### Chế độ 3: Xử lý Video
1. Chọn file video từ máy tính
2. Nhấn "Tải lên & Nhận diện"
3. Chờ xử lý hoàn tất
4. Tải xuống hoặc xem video đã xử lý

## 📁 Cấu trúc Project

```
btlkpdl/
├── btlkpdl/
│   ├── app.py                           # File ứng dụng Flask chính
│   ├── templates/
│   │   └── index.html                   # Giao diện web
│   └── .ipynb_checkpoints/
├── static/
│   ├── style.css                        # CSS styling
│   └── uploads/                         # Thư mục lưu ảnh/video tải lên
├── dataset/                             # Dataset cho huấn luyện
│   ├── angelina jolie/
│   ├── cong thanh/
│   ├── ducphuc/
│   ├── elonmusk/
│   ├── leonardo dicaprio/
│   ├── quanglong/
│   └── Taylor Swift/
├── j.ipynb                              # Jupyter notebook (thử nghiệm)
├── nhandienkhuonmat.ipynb              # Notebook nhận diện khuôn mặt
├── face_recognition_model.h5            # Model nhận diện đã huấn luyện
├── label_encoder.pkl                    # Label encoder cho các lớp
├── shape_predictor_68_face_landmarks.dat # Dlib facial landmarks model
└── README.md                            # File này
```

## 🛠️ Công nghệ sử dụng

| Thư viện | Phiên bản | Mục đích |
|---------|----------|---------|
| Flask | 2.x+ | Framework web |
| TensorFlow | 2.x+ | Deep Learning framework |
| Keras | - | API cao cấp cho mô hình neural networks |
| OpenCV | 4.x+ | Xử lý ảnh và video |
| DeepFace | - | Nhận diện cảm xúc, tuổi, giới tính |
| dlib | - | Facial landmarks detection |
| NumPy | - | Tính toán số học |
| scikit-learn | - | Machine Learning utilities |

## 🧠 Mô hình máy học

### Model Nhận diện khuôn mặt
- **Type**: Convolutional Neural Network (CNN)
- **Input**: Ảnh kích thước 100x100 pixels
- **Output**: Dự đoán nhân dạng (7 người trong dataset)
- **Threshold**: 0.3 (độ tin cậy tối thiểu)

### DeepFace
- **Cảm xúc**: 7 loại (happy, sad, angry, surprised, fearful, disgusted, neutral)
- **Giới tính**: 2 loại (Nam, Nữ)
- **Tuổi**: Dự đoán giá trị liên tục

## ⚙️ Cài đặt quan trọng

### Tiền xử lý ảnh
```python
- Histogram Equalization (V channel của HSV)
- Scale factor: alpha = 1.3, beta = 10
```

### Ngưỡng nhận diện
- **Confidence threshold**: 0.3
- **Phân biệt giữa 2 lớp gần nhất**: < 0.01

### Webcam Stream
- **Độ phân giải**: 320x240
- **FPS**: 15
- **Cơ chế cache**: Lưu kết quả cuối cùng để tối ưu hóa tốc độ

## 🔐 Ghi chú bảo mật

- ⚠️ Ứng dụng hiện tại chỉ phù hợp cho phát triển/testing
- 🔒 Không lưu trữ dữ liệu người dùng theo mặc định
- 📸 Ảnh tải lên được lưu trong thư mục `static/uploads/`
- 🗑️ Nên xóa thư mục uploads định kỳ nếu triển khai production

## 🐛 Troubleshooting

### Lỗi: "face_recognition_model.h5 not found"
- Đảm bảo model file nằm trong thư mục gốc
- Kiểm tra tên file có đúng không

### Lỗi: "shape_predictor_68_face_landmarks.dat not found"
- Tải file từ: http://dlib.net/files/
- Đặt vào thư mục gốc

### Webcam không hoạt động
- Kiểm tra quyền truy cập webcam trên trình duyệt
- Cần HTTPS hoặc localhost để truy cập webcam

### Tốc độ xử lý chậm
- Giảm độ phân giải input
- Cân nhắc sử dụng GPU
- Giảm số frame được xử lý (nâng FPS)

## 📝 License

Dự án này được tạo cho mục đích học tập.

## 👥 Tác giả

Sinh viên - Đại học (Bài Tập Lớn - Khai Phá Dữ Liệu)

## 📧 Liên hệ

Để báo cáo bug hoặc đề xuất tính năng, vui lòng tạo issue hoặc liên hệ qua email.

---

**Cập nhật lần cuối**: Tháng 12, 2025
