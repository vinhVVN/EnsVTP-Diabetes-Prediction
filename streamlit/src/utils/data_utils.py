import pandas as pd
import numpy as np

FEATURE_NAMES = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
    'glucose_hba1c_prod', 'risk_flag', 'bmi_log', 'gender_Male', 'gender_Other', 
    'smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 
    'smoking_history_never', 'smoking_history_not current', 
    'bmi_category_Obese', 'bmi_category_Overweight', 'bmi_category_Underweight'
]

def perform_bulk_feature_engineering(df):
    df_fe = df.copy()
    df_fe['glucose_hba1c_prod'] = df_fe['blood_glucose_level'] * df_fe['HbA1c_level']
    df_fe['risk_flag'] = ((df_fe['hypertension'] == 1) | (df_fe['heart_disease'] == 1)).astype(int)
    df_fe['bmi_log'] = np.log1p(df_fe['bmi'])
    bmi_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df_fe['bmi_category'] = pd.cut(df_fe['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=bmi_order)
    df_fe['gender_Male'] = (df_fe['gender'] == 'Male').astype(float)
    df_fe['gender_Other'] = (df_fe['gender'] == 'Other').astype(float)
    for s in ['current', 'ever', 'former', 'never', 'not current']:
        df_fe[f'smoking_history_{s}'] = (df_fe['smoking_history'] == s).astype(float)
    for b in ['Obese', 'Overweight', 'Underweight']:
        df_fe[f'bmi_category_{b}'] = (df_fe['bmi_category'] == b).astype(float)
    return df_fe[FEATURE_NAMES], (df_fe['diabetes'] if 'diabetes' in df_fe.columns else None)
