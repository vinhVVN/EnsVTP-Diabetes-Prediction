import streamlit as st
import pandas as pd
import os
from sklearn.metrics import f1_score, classification_report

def show_testing(model, perform_bulk_feature_engineering):
    st.title("🧪 Stress Test: Kiểm thử tập dữ liệu lớn")
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datas", "diabetes_prediction_dataset.csv")
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("⚙️ Cấu hình Kiểm thử")
        # Lấy ngưỡng mặc định từ mô hình nếu có, không thì 0.5
        default_t = float(model.best_threshold) if hasattr(model, 'best_threshold') else 0.5
        selected_t = st.slider("Điều chỉnh Ngưỡng (Threshold):", 0.1, 0.9, default_t, step=0.01, help="Kéo thấp xuống để tăng Recall, kéo cao lên để tăng Precision.")
        run_test = st.button("🚀 BẮT ĐẦU CHẠM TRÁN THỰC TẾ", type="primary")

    with c2:
        if run_test:
            if model and os.path.exists(csv_path):
                with st.spinner("Đang phân tích 15% dữ liệu ngẫu nhiên..."):
                    t_df = pd.read_csv(csv_path).drop_duplicates().sample(frac=0.15, random_state=42)
                    X_b, y_b = perform_bulk_feature_engineering(t_df)
                    
                    # Dự đoán với ngưỡng tùy chỉnh
                    probas = model._get_weighted_probas(X_b)
                    y_p = (probas >= selected_t).astype(int)
                    
                    st.success(f"Hoàn thành kiểm thử trên {len(y_b):,} mẫu với Ngưỡng {selected_t}!")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("F1-Score", f"{f1_score(y_b, y_p):.4f}")
                    col_m2.metric("Precision", f"{pd.DataFrame(classification_report(y_b, y_p, output_dict=True)).T.loc['1', 'precision']:.4f}")
                    col_m3.metric("Recall (Nhạy)", f"{pd.DataFrame(classification_report(y_b, y_p, output_dict=True)).T.loc['1', 'recall']:.4f}")
                    st.dataframe(pd.DataFrame(classification_report(y_b, y_p, output_dict=True)).T)
