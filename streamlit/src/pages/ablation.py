import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from sklearn.metrics import f1_score, classification_report

def show_ablation(model, model_no_svm, model_no_bagging, perform_bulk_feature_engineering):
    st.title("📊 Phòng thí nghiệm (Ablation Analysis)")
    st.markdown("Trang này giúp đánh giá vai trò của các thành phần trong hệ thống (SVM, Bagging) bằng cách loại bỏ chúng.")

    # 1. So sánh từ JSON
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "results.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            ablation_results = json.load(f)
            
        st.subheader("1. So sánh F1-Score trên tập Test (Pre-calculated)")
        df_results = pd.DataFrame(ablation_results)
        
        fig = px.bar(df_results, x='experiment', y='f1', title='Đánh giá độ giảm hiệu năng khi loại bỏ các thành phần', color='f1', text_auto='.4f', color_continuous_scale='Blues')
        fig.update_layout(xaxis_title="", yaxis_title="F1-Score")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Kiểm thử Mô hình Động (Dynamic Stress Test - 15% Dữ liệu)")
    
    c_left, c_right = st.columns([1, 2])
    with c_left:
        config_choice = st.radio("Chọn cấu hình muốn chạy thực tế:",
                                ["Full Ensemble (Baseline V3)", 
                                "Ablation: No SVM", 
                                "Ablation: No SVM & No Bagging"])
        
        # Thêm slider threshold
        # Xác định model tạm thời để lấy threshold mặc định
        temp_model = model
        if "No SVM & No Bagging" in config_choice: temp_model = model_no_bagging
        elif "No SVM" in config_choice: temp_model = model_no_svm
        
        default_t = float(temp_model.best_threshold) if (temp_model and hasattr(temp_model, 'best_threshold')) else 0.5
        selected_t = st.slider("Ngưỡng chẩn đoán động:", 0.1, 0.9, default_t, step=0.01, key="ablation_threshold")
        
        run_ablation = st.button("🚀 CHẠY KIỂM THỬ MÔ HÌNH", type="primary")

    with c_right:
        if run_ablation:
            target_model = temp_model # Đã gán ở trên
                
            if target_model is None:
                st.error("Không tìm thấy tệp trọng số (pkl) cho mô hình này!")
            else:
                csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datas", "diabetes_prediction_dataset.csv")
                if os.path.exists(csv_path):
                    with st.spinner(f"Hệ thống đang chạy dự đoán với Ngưỡng {selected_t}..."):
                        t_df = pd.read_csv(csv_path).drop_duplicates().sample(frac=0.15, random_state=42)
                        X_b, y_b = perform_bulk_feature_engineering(t_df)
                        
                        # Dự đoán với ngưỡng tùy chỉnh
                        probas = target_model._get_weighted_probas(X_b)
                        y_p = (probas >= selected_t).astype(int)
                        
                        st.success(f"Đã xử lý xong bằng cấu hình: **{config_choice}** (T={selected_t})")
                        
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("F1-Score Tạm tính", f"{f1_score(y_b, y_p):.4f}")
                        mc2.metric("Precision", f"{pd.DataFrame(classification_report(y_b, y_p, output_dict=True)).T.loc['1', 'precision']:.4f}")
                        mc3.metric("Recall (Nhạy)", f"{pd.DataFrame(classification_report(y_b, y_p, output_dict=True)).T.loc['1', 'recall']:.4f}")
                        
                        st.markdown("**Classification Report (Nhãn 1 = Có bệnh):**")
                        st.dataframe(pd.DataFrame(classification_report(y_b, y_p, output_dict=True)).T)
