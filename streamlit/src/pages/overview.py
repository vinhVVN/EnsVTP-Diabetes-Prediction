import streamlit as st

def show_overview():
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🏥 Tổng quan Dự án</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 2px solid #3b82f6; margin-bottom: 30px;'>", unsafe_allow_html=True)
    
    col_main, col_side = st.columns([2.2, 1], gap="large")
    
    with col_main:
        st.markdown("### Giới thiệu")
        st.write("Hệ thống xây dựng trên nền tảng trí tuệ nhân tạo nhằm hỗ trợ sàng lọc sớm bệnh lý đái tháo đường từ thông tin lâm sàng và chỉ số sinh hóa. Giải pháp này kết hợp sức mạnh từ nhiều thuật toán khác nhau để đảm bảo kết quả chính xác và tin cậy cho đội ngũ y khoa.")
        
        st.markdown("### Mục tiêu")
        st.markdown("""
        - Nâng cao khả năng nhận diện bệnh lý thông qua các đặc trưng sinh hóa quan trọng.
        - So sánh và đánh giá hiệu quả thực tế giữa các phương pháp học tập khác nhau.
        - Xây dựng mô hình hỗn hợp có tính ổn định cao và khả năng tổng quát hóa tốt.
        """)
        
        st.markdown("### Thách thức")
        st.markdown("""
        - Sự chênh lệch lớn về số mẫu giữa các nhóm đối tượng trong tập số liệu.
        - Mối liên hệ đa chiều giữa các chỉ số như đường huyết và nồng độ huyết sắc tố đòi hỏi kỹ thuật xử lý chuyên biệt.
        """)
        
        st.markdown("### Quy trình thực hiện")
        st.markdown(f'''
            <div class="step-container">
                <div class="step-item">
                    <div class="step-number">1</div>
                    <div><b>Phân tích số liệu:</b> Làm sạch và khám phá các mối liên quan giữa các biến số y sinh.</div>
                </div>
                <div class="step-item">
                    <div class="step-number">2</div>
                    <div><b>Tiền xử lý:</b> Cân bằng các nhóm đối tượng và trích xuất thêm đặc trưng tương tác.</div>
                </div>
                <div class="step-item">
                    <div class="step-number">3</div>
                    <div><b>Huấn luyện tổng hợp:</b> Áp dụng kỹ thuật lấy mẫu lặp lại để xây dựng đa mô hình song song.</div>
                </div>
                <div class="step-item">
                    <div class="step-number">4</div>
                    <div><b>Kết hợp kết quả:</b> Sử dụng cơ chế bỏ phiếu có trọng số dựa trên hiệu năng thực tế.</div>
                </div>
                <div class="step-item">
                    <div class="step-number">5</div>
                    <div><b>Tối ưu ngưỡng:</b> Hiệu chỉnh điểm cắt xác suất để đạt điểm số đánh giá tốt nhất trên nhóm bệnh nhân.</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

    with col_side:
        st.markdown(f'''
            <div class="overview-card">
                <h3 style="margin-top:0; color: #1e40af; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;">Dữ liệu Dataset</h3>
                <p style="margin: 10px 0;"><b>Số mẫu sạch:</b> 96.146</p>
                <p style="margin: 10px 0;"><b>Đặc trưng gốc:</b> 9</p>
                <p style="margin: 10px 0;"><b>Đặc trưng tinh chỉnh:</b> 12</p>
                <p style="margin: 10px 0;"><b>Số lớp phân loại:</b> 2</p>
                <div style="background: #eff6ff; padding: 15px; border-radius: 8px; margin-top: 20px;">
                    <h4 style="margin-top:0; color: #1d4ed8;">Đối tượng chẩn đoán</h4>
                    <ul style="padding-left:20px; font-size: 0.9em; color: #1e293b; margin-bottom: 0;">
                        <li>Lớp 0: Người bình thường</li>
                        <li>Lớp 1: Bệnh nhân tiểu đường</li>
                    </ul>
                </div>
            </div>
        ''', unsafe_allow_html=True)
