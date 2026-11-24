import streamlit as st
import pandas as pd
import plotly.express as px
import os

@st.cache_data
def load_raw_data():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datas", "diabetes_prediction_dataset.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).drop_duplicates()
    return None

def show_dataset():
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🔍 Khám phá Tập số liệu</h1>", unsafe_allow_html=True)
    
    df = load_raw_data()
    if df is None:
        st.error("Không tìm thấy tệp dữ liệu huấn luyện!")
        return

    # 1. Thống kê tổng quan
    st.markdown("### Thống kê cơ bản")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Tổng số mẫu sạch", f"{len(df):,}")
    with c2: st.metric("Số lượng đặc trưng", f"{df.shape[1] - 1}")
    with c3: st.metric("Số ca dương tính", f"{len(df[df['diabetes']==1]):,}")
    with c4: st.metric("Tỷ lệ bệnh", f"{len(df[df['diabetes']==1])/len(df):.1%}")

    st.markdown("---")
    
    # 2. Phân phối lớp và Preview
    col_plot, col_preview = st.columns([1, 1], gap="large")
    
    with col_plot:
        st.markdown("#### Phân bổ các nhóm đối tượng")
        counts = df['diabetes'].value_counts().reset_index()
        counts.columns = ['Trạng thái', 'Số lượng']
        counts['Trạng thái'] = counts['Trạng thái'].map({0: 'Bình thường', 1: 'Tiểu đường'})
        
        fig_pie = px.pie(counts, values='Số lượng', names='Trạng thái', 
                         color_discrete_sequence=['#93c5fd', '#1e40af'],
                         hole=0.4)
        fig_pie.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_preview:
        st.markdown("#### Xem trước dữ liệu")
        num_rows = st.slider("Số dòng hiển thị:", 5, 50, 10)
        st.dataframe(df.head(num_rows), use_container_width=True)
        st.info(f"Hiển thị {num_rows} dòng đầu tiên từ tập số liệu gốc.")

    st.markdown("---")
    
    # 3. Phân tích chi tiết các chỉ số
    st.markdown("### Phân tích phân phối đặc trưng")
    
    tab1, tab2, tab3 = st.tabs(["Biểu đồ sinh hiệu", "Tương quan rủi ro", "Phân tích sinh hóa"])
    
    with tab1:
        c_alt1, c_alt2 = st.columns(2)
        with c_alt1:
            fig_age = px.histogram(df, x='age', color='diabetes', 
                                   title='Phân bố độ tuổi theo tình trạng bệnh',
                                   labels={'age': 'Tuổi', 'diabetes': 'Trạng thái'},
                                   color_discrete_map={0: '#94a3b8', 1: '#ef4444'},
                                   barmode='overlay')
            st.plotly_chart(fig_age, use_container_width=True)
        with c_alt2:
            fig_bmi = px.histogram(df, x='bmi', color='diabetes', 
                                   title='Chỉ số khối cơ thể (BMI)',
                                   labels={'bmi': 'Chỉ số BMI', 'diabetes': 'Trạng thái'},
                                   color_discrete_map={0: '#94a3b8', 1: '#ef4444'},
                                   barmode='overlay')
            st.plotly_chart(fig_bmi, use_container_width=True)

    with tab2:
        st.markdown("#### Mối liên hệ giữa BMI và Đường huyết")
        # Lấy mẫu 2000 điểm để scatter plot chạy mượt
        df_sample = df.sample(min(2000, len(df)))
        fig_scatter = px.scatter(df_sample, x='blood_glucose_level', y='bmi', 
                                 color='diabetes',
                                 title='Tương quan giữa Đường huyết và BMI (Mẫu 2000 trường hợp)',
                                 labels={'blood_glucose_level': 'Mức đường huyết', 'bmi': 'Chỉ số BMI', 'diabetes': 'Trạng thái'},
                                 color_discrete_map={0: '#94a3b8', 1: '#ef4444'},
                                 opacity=0.6)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.markdown("#### Các chỉ số xét nghiệm huyết thanh")
        fig_hba1c = px.box(df, x='diabetes', y='HbA1c_level', 
                           color='diabetes',
                           title='Chỉ số HbA1c trung bình',
                           labels={'HbA1c_level': 'Nồng độ HbA1c', 'diabetes': 'Trạng thái'},
                           color_discrete_map={0: '#93c5fd', 1: '#1e40af'})
        st.plotly_chart(fig_hba1c, use_container_width=True)
