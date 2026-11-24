# 🩺 EnsVTP: Diabetes Prediction System - AI Lab

Chào mừng bạn đến với **Diabetes Prediction System**, một giải pháp trí tuệ nhân tạo toàn diện được thiết kế để hỗ trợ sàng lọc sớm bệnh lý đái tháo đường. Dự án kết hợp các mô hình học máy tiên tiến với giao diện người dùng hiện đại, cung cấp cái nhìn minh bạch từ dữ liệu thô đến kết luận chẩn đoán.

---

## 🚀 Điểm nổi bật của Dự án
- **Hội đồng chuyên gia :** Kết hợp sức mạnh từ Logistic Regression, Decision Tree và SVM Ultimate thông qua cơ chế **Weighted Voting** và **Bagging**.
- **Xử lý dữ liệu mất cân bằng:** Áp dụng kỹ thuật **SMOTE** để đảm bảo mô hình không bị thiên kiến đối với nhóm người khỏe mạnh.
- **Explainable AI (XAI):** Minh họa logic xử lý nội bộ 4 bước, giúp chuyên gia y tế hiểu rõ "tại sao" mô hình đưa ra kết luận.
- **Dashboard chuyên nghiệp:** Giao diện tối giản, trực quan hóa rủi ro qua biểu đồ Gauge và phân tích thành phần (Ablation Analysis).

---

## 🎨 Trải nghiệm giao diện Dashboard

Hệ thống được tổ chức thành 4 phân vùng chính:

### 🏠 1. Tổng quan Dự án
Cung cấp cái nhìn toàn cảnh về mục tiêu, thách thức và quy trình thực hiện dự án. Hiển thị thông tin đội ngũ thực hiện và thông số tổng quát về tập dữ liệu 96.000 mẫu.

### 🔍 2. Khám phá Dữ liệu (EDA)
Không gian tương tác để phân tích sâu các chỉ số lâm sàng:
- Phân bổ độ tuổi, BMI, và nồng độ HbA1c qua các biểu đồ **Histogram** và **Box Plot**.
- Quan sát mối tương quan đa biến thông qua biểu đồ **Scatter Chart**.
- Theo dõi tỷ lệ mất cân bằng dữ liệu thực tế.

### 🩺 3. Chẩn đoán Thông minh
Hệ thống chẩn đoán thời gian thực với các tính năng:
- **Ngưỡng chẩn đoán linh hoạt:** Cho phép bác sĩ điều chỉnh điểm cắt xác suất (Threshold) để tối ưu hóa độ nhạy hoặc độ chính xác.
- **Trực quan hóa rủi ro:** Sử dụng biểu đồ **Gauge** để định vị mức độ nguy hiểm của bệnh nhân.
- **Chi tiết Voting:** Hiển thị ý kiến độc lập từ từng mô hình con trong hội đồng.
- **Demo Quy trình:** Mục mở rộng giải thích 4 giai đoạn biến đổi dữ liệu (Raw -> Feature Engineering -> Models -> Weighting).

### 📊 4. Phòng thí nghiệm (Ablation Analysis)
Nơi kiểm chứng tính đúng đắn của kiến trúc hệ thống:
- So sánh hiệu năng (F1-Score) giữa mô hình đầy đủ và các bản rút gọn.
- **Stress Test:** Chạy kiểm định áp lực trên 15% dữ liệu (15.000 mẫu) với ngưỡng động để đánh giá độ ổn định thực tế.

---

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.x
- **Frontend:** Streamlit (UI Framework)
- **Trực quan hóa:** Plotly, Plotly Express
- **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE), Joblib
- **Mô hình:** Ensemble (Bagging + Weighted Voting)

---

## ⚙️ Hướng dẫn Cài đặt & Chạy

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt các thư viện cần thiết:
```bash
pip install streamlit pandas numpy scikit-learn plotly imbalanced-learn joblib
```

### 2. Khởi chạy Dashboard
Mở terminal tại thư mục gốc của dự án và thực hiện lệnh:
```bash
streamlit run streamlit/app.py
```

---

## 👥 Đội ngũ thực hiện - Nhóm 7
- **Phan Hồng Phúc** - 23110141
- **Đặng Minh Tài** - 23110148
- **Nguyễn Thành Vinh** - 23110172
- **GVHD:** TS. Phan Thị Huyền Trang
- **Đơn vị:** HCM-UTE (04/2026)

---
*Dự án được xây dựng phục vụ mục đích học tập và nghiên cứu trong lĩnh vực Học máy tại HCM-UTE.*

