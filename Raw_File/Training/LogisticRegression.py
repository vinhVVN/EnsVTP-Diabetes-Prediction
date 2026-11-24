import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class LogisticRegressionManual:
    def __init__(self, learning_rate=0.1, epochs=2000, lambda_param=0.1, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.threshold = threshold
        self.weights = None 
        self.features = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _compute_loss(self, y, y_hat, m):
        core_loss = -(1/m) * np.sum(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))
        l2_loss = (self.lambda_param / (2 * m)) * np.sum(np.square(self.weights[1:]))
        return core_loss + l2_loss

    def _prepare_data(self, X, ds_name):
        X_temp = X.copy()
        if ds_name == "Dataset_2":
            cols_to_drop = [col for col in X_temp.columns if col == 'bmi' or 'bmi_category' in col]
            # Chỉ drop nếu cột đó thực sự tồn tại
            cols_to_drop = [c for c in cols_to_drop if c in X_temp.columns]
            X_temp = X_temp.drop(columns=cols_to_drop)
        return X_temp

    def fit(self, X, y, ds_name="Dataset_1"):
        X_processed = self._prepare_data(X, ds_name)
        self.features = X_processed.columns.tolist()
        
        X_np = X_processed.values
        y_np = y.values
        m, n = X_np.shape
        
        X_bias = np.c_[np.ones(m), X_np]
        self.weights = np.zeros(n + 1)
        self.loss_history = []

        for i in range(self.epochs):
            z = np.dot(X_bias, self.weights)
            y_hat = self._sigmoid(z)
            gradient = np.dot(X_bias.T, (y_hat - y_np)) / m
            l2_penalty = (self.lambda_param / m) * self.weights
            l2_penalty[0] = 0 
            self.weights -= self.learning_rate * (gradient + l2_penalty)
            
            if i % 100 == 0:
                self.loss_history.append(self._compute_loss(y_np, y_hat, m))
        
        # Sửa lỗi in Loss nếu list rỗng
        last_loss = self.loss_history[-1] if self.loss_history else 0
        print(f"  [LR-Manual] {ds_name} Done! Final Loss: {last_loss:.4f}")
        return self

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Lỗi: Bạn phải gọi hàm .fit() trước khi dự đoán!")
        # Đảm bảo lấy đúng cột features
        X_processed = X[self.features].values
        m = X_processed.shape[0]
        X_bias = np.c_[np.ones(m), X_processed]
        return self._sigmoid(np.dot(X_bias, self.weights))

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

class LogisticRegressionLibrary:
    def __init__(self, lambda_param=0.1, max_iter=2000):
        self.lambda_param = lambda_param
        self.C = 1.0 / lambda_param
        self.model = LogisticRegression(C=self.C, max_iter=max_iter, solver='lbfgs')
        self.features = None

    def _prepare_data(self, X, ds_name):
        X_temp = X.copy()
        if ds_name == "Dataset_2":
            cols_to_drop = [col for col in X_temp.columns if col == 'bmi' or 'bmi_category' in col]
            cols_to_drop = [c for c in cols_to_drop if c in X_temp.columns]
            X_temp = X_temp.drop(columns=cols_to_drop)
        return X_temp

    def fit(self, X, y, ds_name="Dataset_1"):
        X_processed = self._prepare_data(X, ds_name)
        self.features = X_processed.columns.tolist()
        self.model.fit(X_processed, y)
        print(f"  [LR-Library] {ds_name} Done! (C={self.C})")
        return self

    def predict_proba(self, X):
        if self.features is None:
            raise ValueError("Lỗi: Bạn phải gọi hàm .fit() trước khi dự đoán!")
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict(self, X):
        if self.features is None:
            raise ValueError("Lỗi: Bạn phải gọi hàm .fit() trước khi dự đoán!")
        return self.model.predict(X[self.features])