import streamlit as st
import os
import sys
import joblib

# Thêm đường dẫn thư mục gốc và src vào sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

src_dir = os.path.join(os.path.dirname(__file__), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

training_dir = os.path.join(root_dir, "Raw_File", "Training")
if os.path.exists(training_dir) and training_dir not in sys.path:
    sys.path.insert(0, training_dir)

# Import các thành phần đã tách module
from models.ensemble import (
    MedicalEnsembleV3Ultimate, MedicalEnsembleNoSVM, MedicalEnsembleNoSVMNoBagging, MedicalEnsemble,
    LogisticRegressionManual, DecisionTree_Scratch, SVMLD
)
import svm.svm_model as svm_model
from utils.data_utils import FEATURE_NAMES, perform_bulk_feature_engineering
from utils.ui_utils import load_css, inject_sidebar_info
from pages import show_overview, show_dataset, show_diagnosis, show_ablation

# Fix for Pickle load "Can't get attribute on __main__" & alias custom modules
import __main__
__main__.MedicalEnsembleV3Ultimate = MedicalEnsembleV3Ultimate
__main__.MedicalEnsembleNoSVM = MedicalEnsembleNoSVM
__main__.MedicalEnsembleNoSVMNoBagging = MedicalEnsembleNoSVMNoBagging
__main__.MedicalEnsemble = MedicalEnsemble
__main__.LogisticRegressionManual = LogisticRegressionManual
__main__.DecisionTree_Scratch = DecisionTree_Scratch
__main__.SVMLD = SVMLD
__main__.SVMScratch = svm_model.SVMScratch
sys.modules['SVMHL'] = svm_model

# --- Streamlit UI Config ---
st.set_page_config(page_title="Diabetes AI Lab V3", page_icon="💊", layout="wide")
load_css()

@st.cache_resource
def load_all_models():
    base = os.path.join(os.path.dirname(__file__), "models", "final")
    m_full = joblib.load(os.path.join(base, "final_ensemble_model_v3.pkl")) if os.path.exists(os.path.join(base, "final_ensemble_model_v3.pkl")) else None
    m_no_svm = joblib.load(os.path.join(base, "ensemble_no_svm.pkl")) if os.path.exists(os.path.join(base, "ensemble_no_svm.pkl")) else None
    m_no_bagging = joblib.load(os.path.join(base, "ensemble_no_svm_no_bagging.pkl")) if os.path.exists(os.path.join(base, "ensemble_no_svm_no_bagging.pkl")) else None
    return m_full, m_no_svm, m_no_bagging

model, model_no_svm, model_no_bagging = load_all_models()

# --- Sidebar ---
inject_sidebar_info(model, model_no_svm, model_no_bagging)
with st.sidebar:
    page = st.selectbox("Chọn tác vụ dự án:", 
                        ["🏠 Tổng quan dự án", "🔍 Khám phá Dữ liệu", "🩺 Chẩn đoán Thực tế", "📊 Kết quả Lab (Ablation)"])

# --- Main Page Routing ---
if page == "🏠 Tổng quan dự án":
    show_overview()

elif page == "🔍 Khám phá Dữ liệu":
    show_dataset()

elif page == "🩺 Chẩn đoán Thực tế":
    show_diagnosis(model, model_no_svm, model_no_bagging, FEATURE_NAMES)

elif page == "📊 Kết quả Lab (Ablation)":
    show_ablation(model, model_no_svm, model_no_bagging, perform_bulk_feature_engineering)
