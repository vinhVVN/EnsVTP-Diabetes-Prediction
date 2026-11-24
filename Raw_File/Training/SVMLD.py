import numpy as np
from cvxopt import matrix, solvers

class SVMLD:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma=None, coef0=1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)

        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree

        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)

    def _compute_gram_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])

        return K

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.astype(float)

        if self.gamma is None:
            self.gamma = 1 / n_features

        # Gram matrix
        K = self._compute_gram_matrix(X)

        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))

        G = matrix(np.vstack((
            -np.eye(n_samples),
            np.eye(n_samples)
        )))
        h = matrix(np.hstack((
            np.zeros(n_samples),
            np.ones(n_samples) * self.C
        )))

        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        alphas = np.ravel(solution['x'])

        # Support vectors
        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]

        # Bias
        self.b = 0
        for i in range(len(self.alphas)):
            s = 0
            for j in range(len(self.alphas)):
                s += self.alphas[j] * self.sv_y[j] * \
                     self._kernel_function(self.sv_X[j], self.sv_X[i])
            self.b += self.sv_y[i] - s
        self.b /= len(self.alphas)

    def predict(self, X):
        y_pred = []

        for x in X:
            s = 0
            for i in range(len(self.alphas)):
                s += self.alphas[i] * self.sv_y[i] * \
                     self._kernel_function(self.sv_X[i], x)
            y_pred.append(s + self.b)

        return np.sign(y_pred)