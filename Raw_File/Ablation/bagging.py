import numpy as np
import pandas as pd

def bagging_sampler(X, y, n_subsets=3, sample_size_ratio=1.0, random_state=42):

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X phải là pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y phải là pandas Series.")
    if n_subsets <= 0:
        raise ValueError("n_subsets phải > 0.")
    if not (0 < sample_size_ratio <= 1.0):
        raise ValueError("sample_size_ratio phải nằm trong khoảng (0, 1].")

    rng = np.random.default_rng(random_state)
    bags = []

    n_samples = X.shape[0]
    n_size = int(n_samples * sample_size_ratio)
    all_indices = np.arange(n_samples) 

    print(f"BAGGING: Băm dữ liệu thành {n_subsets} tập Bootstrap")

    for i in range(n_subsets):
        # Bốc ngẫu nhiên có hoàn lại
        train_indices = rng.choice(n_samples, size=n_size, replace=True)
        
        # Tìm các index không được bốc trúng (Out-of-Bag)
        oob_indices = np.setdiff1d(all_indices, train_indices)

        # Cắt dữ liệu và reset index
        X_train_bag = X.iloc[train_indices].reset_index(drop=True)
        y_train_bag = y.iloc[train_indices].reset_index(drop=True)
        
        X_oob_bag = X.iloc[oob_indices].reset_index(drop=True)
        y_oob_bag = y.iloc[oob_indices].reset_index(drop=True)

        # Đóng gói vào Dictionary
        bags.append({
            'train_X': X_train_bag,
            'train_y': y_train_bag,
            'oob_X': X_oob_bag,
            'oob_y': y_oob_bag
        })
        
        oob_ratio = (len(oob_indices) / n_samples) * 100
        print(f"  Túi {i+1}: Dữ liệu Train = {len(train_indices)} mẫu | Dữ liệu OOB = {len(oob_indices)} mẫu (~{oob_ratio:.1f}%)")

    return bags