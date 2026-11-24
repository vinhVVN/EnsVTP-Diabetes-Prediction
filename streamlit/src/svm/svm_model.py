import numpy as np
from sklearn.metrics import hinge_loss

class SVMScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000, threshold=0.5):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.threshold = threshold
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y, ds_name=None):
        X = np.asarray(X, dtype=np.float64)
        y_original = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        y_ = np.where(y_original <= 0.5, -1, 1) 
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []

        for epoch in range(self.epochs):
            distances = y_ * (np.dot(X, self.w) - self.b)
            misclassified_indices = np.where(distances < 1)[0]
            dw = 2 * self.lambda_param * self.w
            if len(misclassified_indices) > 0:
                dw -= np.dot(y_[misclassified_indices], X[misclassified_indices])
                db = np.sum(y_[misclassified_indices])
            else:
                db = 0
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 100 == 0:
                margin = y_ * (np.dot(X, self.w) - self.b)
                loss = np.mean(np.maximum(0, 1 - margin)) + self.lambda_param * np.dot(self.w, self.w)
                self.loss_history.append(loss)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        approx = np.dot(X, self.w) - self.b
        predictions = np.where(approx >= 0, 1, 0)
        return predictions

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = np.dot(X, self.w) - self.b
        scores = np.clip(scores, -500, 500)
        proba = 1 / (1 + np.exp(-scores))
        return proba
