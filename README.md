# VN Plate OCR - Nhận Diện Biển Số Xe Việt Nam
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
Đây là mô hình Nhận dạng Ký tự Quang học (OCR) gọn nhẹ và mạnh mẽ, được thiết kế đặc biệt để đọc Biển số xe Việt Nam. Dự án sử dụng kiến trúc CRNN (Convolutional Recurrent Neural Network) kết hợp với hàm suy hao CTC Loss, giúp đọc chính xác các ký tự từ ảnh cắt biển số (cả biển ngang của ô tô lẫn biển vuông 2 dòng của xe máy).
## Cơ Chế Hoạt Động & Thuật Toán (CRNN + CTC)
Quá trình biến một bức ảnh biển số thành chuỗi ký tự được xử lý qua 3 giai đoạn kiến trúc cốt lõi:
### 1. Trích xuất đặc trưng (CNN - Mạng Tích Chập)
Mạng CNN quét qua bức ảnh đầu vào $I$ để bóc tách các đặc trưng không gian. Đầu ra của CNN là một chuỗi các dải đặc trưng (feature maps) $X = (x_1, x_2, ..., x_T)$, trong đó $T$ là chiều dài của chuỗi (số cột đặc trưng).
### 2. Mô hình hóa chuỗi (Bi-LSTM - RNN Hai Chiều)
Chuỗi $X$ được đưa vào mạng Bi-LSTM. Khác với CNN chỉ nhìn cục bộ, Bi-LSTM quét chuỗi theo 2 chiều (từ trái qua phải và ngược lại) để nắm bắt ngữ cảnh. Tại mỗi bước $t$, trạng thái ẩn $h_t$ được tính bằng cách ghép trạng thái của cả 2 chiều:
\[ h_t = [\overrightarrow{h_t}, \overleftarrow{h_t}] \]
Đầu ra của phân lớp này là một ma trận xác suất $Y = (y_1, y_2, ..., y_T)$, trong đó $y_t$ chứa phân bố xác suất của toàn bộ tập ký tự (cộng thêm ký tự `Blank`) tại cột $t$.
### 3. Căn chỉnh và Giải mã (CTC Loss - Connectionist Temporal Classification)
Vấn đề lớn nhất của OCR là độ rộng của mỗi chữ cái không giống nhau. Hàm CTC giải quyết bài toán này bằng cách giới thiệu ký tự khoảng trắng (Blank), ký hiệu là `-`.
Hàm ánh xạ $\mathcal{B}$ sẽ tự động gộp các ký tự giống nhau đứng liền kề và xóa bỏ `-`. 
Ví dụ: $\mathcal{B}(\text{--22-99-C-11-}) = \text{29C1}$.
**Công thức CTC Loss:**
Mục tiêu của mô hình là cực đại hóa xác suất sinh ra chuỗi nhãn $S$ đúng từ đầu vào $X$. Xác suất của $S$ là tổng xác suất của mọi đường dẫn (paths) $\pi$ có thể ánh xạ ra $S$:
\[ p(S | X) = \sum_{\pi \in \mathcal{B}^{-1}(S)} p(\pi | X) \]
Trong đó $p(\pi | X) = \prod_{t=1}^T y_{\pi_t}^t$ là xác suất của một đường dẫn cụ thể. Hàm mất mát (Loss) cần tối thiểu hóa được định nghĩa là âm logarit:
\[ \mathcal{L}_{CTC} = -\log p(S | X) \]
## Tính Năng Nổi Bật
- Kiến trúc CRNN: Kết hợp CNN và Bi-LSTM chuẩn mực.
- Data Augmentation Mạnh Mẽ: Mô phỏng lại các điều kiện khắc nghiệt ngoài thực tế của camera hành trình (ảnh nhòe, xoay nghiêng, lóa sáng...) giúp tăng độ ổn định cho mô hình.
- Bộ Lọc Regex Thông Minh: Tích hợp sẵn bộ luật Regex xử lý được hầu hết các định dạng biển số tại VN (Xe dân sự, Xe máy điện, Xe ngoại giao, Xe liên doanh, Quân đội).
- Đánh Giá Tự Động: Script tự động chấm điểm Accuracy (Độ chính xác nguyên biển) và CER (Tỉ lệ lỗi ký tự) qua các epoch, tự động vẽ biểu đồ và trích xuất ra model tốt nhất.
## Cấu Trúc Thư Mục
```text
vn-plate-ocr/
├── model.py            # Kiến trúc CRNN (CNN + Bi-LSTM)
├── dataset.py          # Xử lý data, Dataloader & Data Augmentation
├── train.py            # Code huấn luyện mô hình (Training pipeline)
├── eval.py             # Code đánh giá (Tính Acc, CER, vẽ biểu đồ csv)
├── best_crnn.pth       # File trọng số tốt nhất (Sinh ra sau khi chạy eval.py)
└── yolo_plate_ocr_dataset/ # Dataset chứa ảnh ký tự và file txt chuẩn YOLO
```
## Dữ Liệu (Dataset)
Mô hình được huấn luyện trên bộ dataset ký tự biển số xe tự custom.
- Định dạng: Chuẩn YOLO annotation (`<class> <x_center> <y_center> <width> <height>`).
- Classes: 36 classes (`0-9`, `A-Z`).
- Tải về: [Nhấn vào đây để tải Dataset](https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/blob/main/doc/dataset.md) *(Vui lòng giải nén vào thư mục `yolo_plate_ocr_dataset/` trước khi train).*
## Cài Đặt
1. Clone repository về máy:
```bash
git clone https://github.com/your-username/vn-plate-ocr.git
cd vn-plate-ocr
```
2. Cài đặt thư viện cần thiết:
Đảm bảo bạn đã cài đặt PyTorch phù hợp với phiên bản CUDA của máy.
```bash
pip install torch torchvision tqdm matplotlib
```
## Hướng Dẫn Sử Dụng
### 1. Huấn Luyện Mô Hình (Training)
Để bắt đầu train CRNN từ đầu, chạy lệnh:
```bash
python train.py
```
*Các file trọng số sau mỗi epoch sẽ được lưu dưới dạng `crnn_epoch_X.pth`.*
### 2. Đánh Giá & Lấy Model Tốt Nhất
Chạy file này để tự động chấm điểm toàn bộ các file `.pth` đã lưu, xuất ra báo cáo CSV, vẽ biểu đồ học tập và tự động copy file xịn nhất thành `best_crnn.pth`:
```bash
python eval.py
```
## Hướng Dẫn Dùng Regex Cho Biển Số VN
Để lọc và chuẩn hóa chuỗi OCR đầu ra khi áp dụng vào thực tế (VD: Hệ thống Bãi đỗ xe thông minh), bạn có thể dùng bộ Regex bao quát dưới đây:
```python
# Bao trọn gói: Dân sự, Xe máy điện (MD), Ngoại giao/Nước ngoài (NG/NN), Liên Doanh, Quân đội.
PLATE_REGEX = r'^(\d{2}[A-Z][A-Z0-9]?\d{4,5}|\d{2}MD\d{5,6}|\d{2}(LD|NN|NG|KT|CD|DA)\d{4,5}|\d{5}NG\d{2}|[A-Z]{2}\d{4})$'
```
## Hiệu Năng (Performance)
Sau khi áp dụng Data Augmentation, mô hình đạt được kết quả cực kỳ ấn tượng trên tập dữ liệu camera hành trình thực tế nhiều nhiễu:
- Độ chính xác tuyệt đối (Plate Accuracy): `~95.83%` (95.83% biển số được đọc đúng 100% không trật ký tự nào)
- Tỉ lệ lỗi ký tự (CER): `~1.14%`
