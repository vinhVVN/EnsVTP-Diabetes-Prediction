import numpy as np

class MedicalEnsembleV3Ultimate:
    def __init__(self, random_state=42):
        self.trained_models = []
        self.weights = []
        self.best_threshold = 0.5
    
    def _get_weighted_probas(self, X):
        probas = np.array([m.predict_proba(X)[:, 1] for m in self.trained_models])
        return np.average(probas, axis=0, weights=self.weights)

    def predict(self, X):
        weighted_probas = self._get_weighted_probas(X)
        return (weighted_probas >= self.best_threshold).astype(int)

class MedicalEnsembleNoSVM:
    def __init__(self, random_state=42):
        self.trained_models = []
        self.weights = []
        self.best_threshold = 0.5
    
    def _get_weighted_probas(self, X):
        probas = np.array([m.predict_proba(X)[:, 1] for m in self.trained_models])
        return np.average(probas, axis=0, weights=self.weights)

    def predict(self, X):
        weighted_probas = self._get_weighted_probas(X)
        return (weighted_probas >= self.best_threshold).astype(int)

class MedicalEnsembleNoSVMNoBagging:
    def __init__(self, random_state=42):
        self.trained_models = []
        self.weights = []
        self.best_threshold = 0.5
    
    def _get_weighted_probas(self, X):
        probas = np.array([m.predict_proba(X)[:, 1] for m in self.trained_models])
        return np.average(probas, axis=0, weights=self.weights)

    def predict(self, X):
        weighted_probas = self._get_weighted_probas(X)
        return (weighted_probas >= self.best_threshold).astype(int)

class MedicalEnsemble:
    def __init__(self, random_state=42):
        self.trained_models = []
        self.weights = []
        self.best_threshold = 0.5
    
    def _get_weighted_probas(self, X):
        probas = np.array([m.predict_proba(X)[:, 1] for m in self.trained_models])
        return np.average(probas, axis=0, weights=self.weights)

    def predict(self, X):
        weighted_probas = self._get_weighted_probas(X)
        return (weighted_probas >= self.best_threshold).astype(int)

class LogisticRegressionManual:
    def __init__(self, learning_rate=0.1, epochs=2000, lambda_param=0.1, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)
        m, n = X_np.shape
        self.weights = np.zeros(n)
        self.bias = 0
        return self

    def predict_proba(self, X):
        X_np = np.asarray(X, dtype=np.float64)
        z = np.dot(X_np, self.weights) + self.bias
        proba_positive = self._sigmoid(z)
        return np.array([[1 - p, p] for p in proba_positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

class DecisionTree_Scratch:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None
    
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.zeros((len(X), 2))

class SVMLD:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma=None, coef0=1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
    
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.zeros((len(X), 2))
