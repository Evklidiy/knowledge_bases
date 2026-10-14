import numpy as np


class MinDistanceClassifier:
    """Решающая функция на основе минимального расстояния"""

    def __init__(self):
        self.prototypes = {}

    def fit(self, X, y):
        # Нахождение объектов-прототипов (центров масс) для каждого класса
        for c in np.unique(y):
            self.prototypes[c] = np.mean(X[y == c], axis=0)

    def predict(self, x):
        best_class = None
        max_d = -float('inf')
        details = {}

        for c, p in self.prototypes.items():
            # Формула: Di = 2 * sum(X * Pi) - sum(Pi^2)
            d = 2 * np.dot(x, p) - np.dot(p, p)
            details[c] = d
            if d > max_d:
                max_d = d
                best_class = c

        return best_class, details


class PerceptronClassifier:
    """Разделяющая решающая функция на основе алгоритма восприятия"""

    def __init__(self):
        self.w = None
        self.w0 = 0
        self.iterations = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.w0 = 0

        E = 0  # Счетчик правильно распознанных объектов
        max_iter = 1000
        self.iterations = 0

        while E < n_samples and self.iterations < max_iter:
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]

                # D12 = W1*X1 + ... + Wn*Xn + W0
                d12 = np.dot(self.w, xi) + self.w0

                if yi == 1 and d12 <= 0:
                    # Коррекция в сторону увеличения
                    self.w += xi
                    self.w0 += 1
                    E = 0
                elif yi == 2 and d12 >= 0:
                    # Коррекция в сторону уменьшения
                    self.w -= xi
                    self.w0 -= 1
                    E = 0
                else:
                    E += 1

                if E == n_samples:
                    break
            self.iterations += 1

    def predict(self, x):
        d12 = np.dot(self.w, x) + self.w0
        predicted_class = 1 if d12 > 0 else 2
        return predicted_class, d12