import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Cấu hình giao diện
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def eda_process(file_path, target_col='diabetes'):
    # Đọc dữ liệu
    df1 = pd.read_csv(file_path)
    
    print("1. THỐNG KÊ MÔ TẢ (DATASET 1)")
    print(f"\nKích thước mẫu: {df1.shape[0]} dòng, {df1.shape[1]} cột")

    print("\nThông tin bộ nhớ và kiểu dữ liệu:")
    df1.info()
    # Kiểm tra trùng lặp
    duplicate_count = df1.duplicated().sum()
    print(f"- Số lượng dòng trùng lặp: {duplicate_count}")

    # Kiểm tra giá trị thiếu chi tiết
    missing_data = df1.isnull().sum().to_frame(name='Số lượng thiếu')
    missing_data['Tỷ lệ (%)'] = (missing_data['Số lượng thiếu'] / len(df1)) * 100
    print("\n- Bảng kiểm tra giá trị thiếu:")
    display(missing_data)

    print("\nCác chỉ số thống kê chi tiết (Biến số):")
    num_df = df1.select_dtypes(include=[np.number])
    stats = num_df.describe().T
    stats['median'] = num_df.median()
    stats['skew'] = num_df.skew()
    display(stats)

    print("\nThống kê các biến định danh (Categorical):")
    display(df1.describe(include=['object']).T)
    
    # TRỰC QUAN HÓA DỮ LIỆU 
    print("\n2. TRỰC QUAN HÓA DỮ LIỆU")
    num_cols = df1.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    df1[num_cols].hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
    plt.suptitle("Histogram: Phân bố giá trị gốc", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df1[num_cols], palette="Set3")
    plt.title("Boxplot: Phát hiện Outliers trong dữ liệu gốc", fontsize=16)
    plt.show()

    # Stripplot (Mối quan hệ với Target)
    important_features = ['bmi', 'blood_glucose_level', 'HbA1c_level']
    for col in important_features:
        plt.figure(figsize=(8, 5))
        sns.stripplot(data=df1, x=target_col, y=col, jitter=True, alpha=0.3, palette="viridis", hue=target_col, legend=False)
        plt.title(f"Mối quan hệ giữa {col} và {target_col}", fontsize=14)
        plt.show()

    # PHÂN TÍCH TƯƠNG QUAN 
    print("\n3. PHÂN TÍCH TƯƠNG QUAN (DATASET 1)")
    corr1 = df1.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr1, annot=True, cmap='RdBu', fmt='.2f', center=0)
    plt.title("Ma trận tương quan Dataset 1")
    plt.show()

    print("\nKỸ THUẬT ĐẶC TRƯNG (Tạo Dataset 2)")
    df2 = df1.copy()

    # Interaction Feature
    df2['glucose_hba1c_prod'] = df2['blood_glucose_level'] * df2['HbA1c_level']
    
    # Comorbidity Flag
    df2['risk_flag'] = ((df2['hypertension'] == 1) | (df2['heart_disease'] == 1)).astype(int)
    
    # Binning BMI và định nghĩa thứ tự (Ordinal Encoding)
    # Định nghĩa danh sách các nhóm theo thứ tự tăng dần về mức độ rủi ro
    bmi_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    
    # Tạo cột bmi_category với bins như cũ
    df2['bmi_category'] = pd.cut(df2['bmi'], 
                                 bins=[0, 18.5, 24.9, 29.9, 100], 
                                 labels=bmi_order)

    # Ép kiểu sang Category với thứ tự đã định sẵn
    df2['bmi_category'] = pd.Categorical(df2['bmi_category'], 
                                         categories=bmi_order, 
                                         ordered=True)

    # Transformation (Log Transform dựa trên Skewness)
    df2['bmi_log'] = np.log1p(df2['bmi'])
    
    print("- Đã tạo: glucose_hba1c_prod, risk_flag, bmi_category, bmi_log")
    display(df2[['glucose_hba1c_prod', 'risk_flag', 'bmi_category', 'bmi_log']].head())


    print("\nPHÂN TÍCH TƯƠNG QUAN (DATASET 2)")
    temp_df = df2.copy()
    if 'bmi_category' in temp_df.columns:
        temp_df['bmi_category_code'] = temp_df['bmi_category'].cat.codes
    
    corr2 = temp_df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr2, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title("Ma trận tương quan Dataset 2 (Đã thêm đặc trưng)")
    plt.show()

    target_corr = corr2[target_col].sort_values(ascending=False)
    print("\nMức độ tương quan của các đặc trưng trong Dataset 2 với Target:")
    print(target_corr)

    return df1, df2

