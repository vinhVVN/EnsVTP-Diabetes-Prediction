import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTENC


def preprocess(df1, df2, target_column='diabetes', random_state=42):
    results = {}

    for dataset_name, df_raw in [("Dataset_1", df1), ("Dataset_2", df2)]:
        print(f"\n--- Đang thực thi Tiền xử lý cho: {dataset_name} ---")

        df_cleaned = df_raw.drop_duplicates().reset_index(drop=True)

        X = df_cleaned.drop(columns=[target_column])
        y = df_cleaned[target_column]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=random_state, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
        )

        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        categorical_columns = [
            col for col in X_train.columns
            if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category'
        ]
        numeric_columns = [col for col in X_train.columns if col not in categorical_columns]

        # Xử lý missing dựa trên train
        for col in numeric_columns:
            fill_value = X_train[col].median()
            X_train[col] = X_train[col].fillna(fill_value)
            X_val[col] = X_val[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)

        for col in categorical_columns:
            mode_series = X_train[col].mode()
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            X_train[col] = X_train[col].fillna(fill_value)
            X_val[col] = X_val[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)

            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")
            X_test[col] = X_test[col].astype("category")

        # SMOTENC
        if len(categorical_columns) > 0:
            categorical_indices = [X_train.columns.get_loc(col) for col in categorical_columns]
            smote_nc_model = SMOTENC(
                categorical_features=categorical_indices,
                random_state=random_state
            )
            X_train_resampled, y_train_resampled = smote_nc_model.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train.copy(), y_train.copy()

        # One-Hot Encoding
        if len(categorical_columns) > 0:
            try:
                encoder = OneHotEncoder(
                    drop='first',
                    sparse_output=False,
                    handle_unknown='ignore'
                )
            except TypeError:
                encoder = OneHotEncoder(
                    drop='first',
                    sparse=False,
                    handle_unknown='ignore'
                )

            encoder.fit(X_train_resampled[categorical_columns])
            new_categorical_names = encoder.get_feature_names_out(categorical_columns)

            def transform_and_merge(X_input):
                encoded_data = encoder.transform(X_input[categorical_columns])
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=new_categorical_names,
                    index=X_input.index
                )
                return pd.concat(
                    [X_input[numeric_columns].reset_index(drop=True),
                     encoded_df.reset_index(drop=True)],
                    axis=1
                )

            X_train_final = transform_and_merge(X_train_resampled)
            X_val_final = transform_and_merge(X_val)
            X_test_final = transform_and_merge(X_test)

        else:
            # Không có cột categorical thì giữ nguyên numeric
            X_train_final = X_train_resampled[numeric_columns].reset_index(drop=True)
            X_val_final = X_val[numeric_columns].reset_index(drop=True)
            X_test_final = X_test[numeric_columns].reset_index(drop=True)

        # Scaling chỉ cho biến liên tục
        continuous_features = [
            'age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
            'glucose_hba1c_prod', 'bmi_log'
        ]
        features_to_scale = [col for col in continuous_features if col in X_train_final.columns]

        if len(features_to_scale) > 0:
            scaler_model = StandardScaler()
            X_train_final[features_to_scale] = scaler_model.fit_transform(X_train_final[features_to_scale])
            X_val_final[features_to_scale] = scaler_model.transform(X_val_final[features_to_scale])
            X_test_final[features_to_scale] = scaler_model.transform(X_test_final[features_to_scale])
        else:
            scaler_model = None

        results[dataset_name] = {
            'X_train': X_train_final,
            'y_train': y_train_resampled.reset_index(drop=True),
            'X_val': X_val_final,
            'y_val': y_val.reset_index(drop=True),
            'X_test': X_test_final,
            'y_test': y_test.reset_index(drop=True),
            'categorical_columns': categorical_columns,
            'numeric_columns': numeric_columns,
            'scaled_columns': features_to_scale,
            'scaler': scaler_model
        }

        print(f"     Train: {X_train_final.shape}, Val: {X_val_final.shape}, Test: {X_test_final.shape}")

    return results  


def clean_and_encode(df1, df2, target_column='diabetes', random_state=42):
    results = {}
    for dataset_name, df_raw in [("Dataset_1", df1), ("Dataset_2", df2)]:
        df_cleaned = df_raw.drop_duplicates().reset_index(drop=True)
        X = df_cleaned.drop(columns=[target_column])
        y = df_cleaned[target_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp)
        X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

        categorical_cols = [col for col in X_train.columns if X_train[col].dtype in ['object', 'category']]
        numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

        for col in numeric_cols:
            fill_value = X_train[col].median()
            X_train[col], X_val[col], X_test[col] = X_train[col].fillna(fill_value), X_val[col].fillna(fill_value), X_test[col].fillna(fill_value)

        for col in categorical_cols:
            mode_series = X_train[col].mode()
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            X_train[col], X_val[col], X_test[col] = X_train[col].fillna(fill_value), X_val[col].fillna(fill_value), X_test[col].fillna(fill_value)

        # One-Hot Encoding
        if len(categorical_cols) > 0:
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            encoder.fit(X_train[categorical_cols])
            new_cat_names = encoder.get_feature_names_out(categorical_cols)

            def transform_merge(X_input):
                encoded_df = pd.DataFrame(encoder.transform(X_input[categorical_cols]), columns=new_cat_names, index=X_input.index)
                return pd.concat([X_input[numeric_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

            X_train_final = transform_merge(X_train)
            X_val_final = transform_merge(X_val)
            X_test_final = transform_merge(X_test)
        else:
            X_train_final, X_val_final, X_test_final = X_train[numeric_cols], X_val[numeric_cols], X_test[numeric_cols]

        results[dataset_name] = {
            'X_train': X_train_final, 'y_train': y_train.reset_index(drop=True),
            'X_val': X_val_final, 'y_val': y_val.reset_index(drop=True),
            'X_test': X_test_final, 'y_test': y_test.reset_index(drop=True)
        }
    return results