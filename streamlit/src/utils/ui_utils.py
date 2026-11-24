import streamlit as st
import os

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
        .main-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .metric-card { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        </style>
        """, unsafe_allow_html=True)

def inject_sidebar_info(model, model_no_svm, model_no_bagging):
    with st.sidebar:
        # Layout thông tin nhóm và ảnh ở đầu sidebar
        col_info, col_img = st.columns([1.6, 1])
        with col_info:
            st.markdown("""
                <div style="font-size: 12px; line-height: 1.4; color: #cbd5e0; margin-top: 5px;">
                    <b style="color: #63b3ed; font-size: 13px;">Nhóm 7 - Học Máy</b><br>
                    Phan Hồng Phúc - 23110141<br>
                    Đặng Minh Tài - 23110148<br>
                    Nguyễn Thành Vinh - 23110172<br>
                    <hr style="margin: 5px 0; border: 0; border-top: 1px solid #4a5568;">
                    <b>GVHD:</b> TS. Phan Thị Huyền Trang<br>
                    <span style="font-style: italic; font-size: 11px;">HCM-UTE - 04/2026</span>
                </div>
            """, unsafe_allow_html=True)
        with col_img:
            st.image("https://img.icons8.com/clouds/150/medical-doctor.png", use_container_width=True)
            
        st.markdown("<h2 style='margin-top: 0; padding-top: 0;'>Medical AI Portal</h2>", unsafe_allow_html=True)
        if model:
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3; margin-bottom: 20px;">
                <p style="margin:0; color: #0d47a1; font-weight: bold;">Mô hình: Ensemble V3</p>
                <p style="margin:0; color: #1565c0; font-size: 0.9em;">Ngưỡng tối ưu: {model.best_threshold:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🛠️ ĐIỀU HƯỚNG")
