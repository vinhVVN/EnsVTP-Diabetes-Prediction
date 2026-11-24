import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def show_diagnosis(model, model_no_svm, model_no_bagging, FEATURE_NAMES):
    st.title("🩺 Chẩn đoán thông minh")
    
    diagnostic_model_choice = st.selectbox("Lựa chọn Mô hình Chẩn đoán:", 
                                         ["Full Ensemble (Baseline V3)", 
                                          "Ablation: No SVM", 
                                          "Ablation: No SVM & No Bagging"])
    
    active_model = model
    if "No SVM & No Bagging" in diagnostic_model_choice:
        active_model = model_no_bagging
    elif "No SVM" in diagnostic_model_choice:
        active_model = model_no_svm
    
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("👤 Thông tin cơ bản")
        age = st.slider("Tuổi bệnh nhân", 1, 100, 45)
        gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])
        bmi = st.number_input("Chỉ số BMI cơ thể", 10.0, 60.0, 24.5)
        smoke = st.selectbox("Thói quen hút thuốc", ["never", "current", "ever", "former", "not current", "No Info"])
    with c2:
        st.subheader("🧪 Chỉ số xét nghiệm")
        hba1c = st.slider("Chỉ số HbA1c (%)", 3.0, 15.0, 5.5, step=0.1)
        glucose = st.slider("Đường huyết (mg/dL)", 50, 400, 110)
        col_t1, col_t2 = st.columns(2)
        with col_t1: hyper = st.toggle("Cao huyết áp", False)
        with col_t2: heart = st.toggle("Bệnh tim mạch", False)
        
        st.markdown("---")
        # Lấy ngưỡng mặc định từ mô hình đang chọn
        default_t = float(active_model.best_threshold) if (active_model and hasattr(active_model, 'best_threshold')) else 0.5
        diag_threshold = st.slider("Ngưỡng chẩn đoán (Diagnostic Threshold):", 0.1, 0.9, default_t, step=0.01, help="Chỉnh thấp để tăng độ nhạy, chỉnh cao để tăng độ chính xác.")
    
    btn = st.button("🎬 BẮT ĐẦU PHÂN TÍCH HỘI CHẨN", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if btn and active_model:
        with st.spinner("Đang tổng hợp ý kiến từ Hội đồng..."):
            t_gender = "Other" if gender == "Khác" else ("Male" if gender == "Nam" else "Female")
            f = {k: 0.0 for k in FEATURE_NAMES}
            f.update({
                'age': float(age), 'hypertension': 1. if hyper else 0., 
                'heart_disease': 1. if heart else 0., 'bmi': float(bmi), 
                'HbA1c_level': float(hba1c), 'blood_glucose_level': float(glucose), 
                'glucose_hba1c_prod': float(glucose * hba1c), 'risk_flag': 1. if (hyper or heart) else 0., 
                'bmi_log': np.log1p(bmi)
            })
            if t_gender == 'Male': f['gender_Male'] = 1.
            elif t_gender == 'Other': f['gender_Other'] = 1.
            if smoke in ['current', 'ever', 'former', 'never', 'not current']: f[f'smoking_history_{smoke}'] = 1.
            if bmi >= 30: f['bmi_category_Obese'] = 1.
            elif 25 <= bmi < 30: f['bmi_category_Overweight'] = 1.
            elif bmi < 18.5: f['bmi_category_Underweight'] = 1.
            x_in = pd.DataFrame([f])[FEATURE_NAMES]
            
            prob_final = active_model._get_weighted_probas(x_in)[0]; pred_final = 1 if prob_final >= diag_threshold else 0
            indiv_probs = [m.predict_proba(x_in)[0, 1] for m in active_model.trained_models]
            
            if "No SVM" in diagnostic_model_choice:
                names = ["Logistic Regression", "Decision Tree"]
            else:
                names = ["Logistic Regression", "Decision Tree", "SVM Ultimate"]

            st.markdown("---")
            res_col, chart_col = st.columns([4, 5])
            with res_col:
                st.subheader("🏆 Kết luận")
                color = "#d32f2f" if pred_final == 1 else "#2e7d32"
                status = "DƯƠNG TÍNH" if pred_final == 1 else "ÂM TÍNH"
                st.markdown(f"""
                    <div style="background-color: {color}15; padding: 25px; border-radius: 15px; border: 2px solid {color}; text-align: center;">
                        <h1 style="color: {color}; margin:0; font-size: 2.5em;">{status}</h1>
                        <p style="color: #333; margin-top: 10px; font-size: 1.2em;">Khả năng bệnh: <b>{prob_final:.2%}</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 🗳️ Chi tiết Voting")
                vote_data = pd.DataFrame({
                    "Thành viên": names,
                    "Xác suất": [f"{p:.1%}" for p in indiv_probs],
                    "Đánh giá": ["🔴 Bệnh" if p >= diag_threshold else "🟢 An toàn" for p in indiv_probs]
                })
                st.table(vote_data)

            with chart_col:
                st.subheader("📊 Biểu đồ Rủi ro")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob_final * 100,
                    title = {'text': "Ngưỡng chẩn đoán", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [{'range': [0, diag_threshold*100], 'color': "#e8f5e9"}, 
                                  {'range': [diag_threshold*100, 100], 'color': "#ffebee"}],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'value': diag_threshold * 100}
                    }
                ))
                fig.update_layout(height=400, margin=dict(l=30, r=30, t=30, b=30))
                st.plotly_chart(fig, use_container_width=True)

            # --- PHẦN MINH HỌA QUY TRÌNH (DEMO) ---
            st.markdown("---")
            with st.expander("🔍 Xem chi tiết quy trình xử lý nội bộ (Minh họa logic hệ thống)", expanded=False):
                st.info("Dưới đây là mô phỏng 5 giai đoạn biến đổi dữ liệu từ lúc bạn nhập vào cho đến khi hệ thống đưa ra kết luận.")
                
                # Bước 1: Dữ liệu thô
                st.markdown("#### Bước 1: Tiếp nhận Dữ liệu thô")
                raw_input_data = {
                    "Chỉ số": ["Tuổi", "Giới tính", "BMI", "Hút thuốc", "HbA1c", "Đường huyết", "Cao huyết áp", "Bệnh tim"],
                    "Giá trị nhập": [age, gender, bmi, smoke, hba1c, glucose, "Có" if hyper else "Không", "Có" if heart else "Không"]
                }
                st.table(pd.DataFrame(raw_input_data))

                # Bước 2: Feature Engineering
                st.markdown("#### Bước 2: Tinh chỉnh Đặc trưng (Feature Engineering)")
                st.write("Hệ thống tự động tạo ra các biến số mới để tăng khả năng nhận diện bệnh lý.")
                fe_data = {
                    "Đặc trưng mới": ["Tương tác G-H (Glucose * HbA1c)", "Chỉ số Rủi ro (Risk Flag)", "BMI Logarit"],
                    "Công thức": ["Đường huyết × HbA1c", "1 nếu có Cao huyết áp/Tim mạch", "log(BMI + 1)"],
                    "Giá trị tính toán": [f['glucose_hba1c_prod'], f['risk_flag'], f['bmi_log']]
                }
                st.table(pd.DataFrame(fe_data))

                # Bước 3: Phân tích từ Hội đồng chuyên gia
                st.markdown("#### Bước 3: Phân tích từ Hội đồng chuyên gia")
                st.write("Dữ liệu sau khi tinh chỉnh được đưa vào bộ các mô hình máy học. Mỗi mô hình con đưa ra một xác suất mắc bệnh độc lập.")
                
                indiv_probs_pct = [p * 100 for p in indiv_probs]
                fig_bar = go.Figure(go.Bar(
                    x=names, y=indiv_probs_pct,
                    text=[f"{p:.2f}%" for p in indiv_probs_pct],
                    textposition='auto',
                    marker_color=['#1e40af', '#3b82f6', '#93c5fd'][:len(names)]
                ))
                fig_bar.update_layout(title="Xác suất mắc bệnh của từng mô hình", yaxis_title="Xác suất (%)", height=350)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Bước 4: Tổng hợp Kết quả (Weighted Voting)
                st.markdown("#### Bước 4: Tổng hợp Kết quả (Weighted Voting)")
                st.write("Kết quả cuối cùng được tính bằng cách lấy trung bình có trọng số từ các chuyên gia, dựa trên độ tin cậy của từng mô hình đã được kiểm chứng.")
                
                weights_pct = active_model.weights * 100
                fig_pie = go.Figure(go.Pie(
                    labels=names, values=weights_pct,
                    hole=0.4,
                    marker_colors=['#1e3a8a', '#2563eb', '#60a5fa']
                ))
                fig_pie.update_layout(title="Trọng số đóng góp của các mô hình", height=350)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.success(f"**Công thức kết hợp:** Xác suất cuối = Σ (Trọng số_i × Xác suất_i) = **{prob_final:.2%}**")
                if prob_final >= diag_threshold:
                    st.error(f"Vì {prob_final:.2%} ≥ Ngưỡng {diag_threshold:.0%}, hệ thống đưa ra kết luận: DƯƠNG TÍNH")
                else:
                    st.success(f"Vì {prob_final:.2%} < Ngưỡng {diag_threshold:.0%}, hệ thống đưa ra kết luận: ÂM TÍNH")
