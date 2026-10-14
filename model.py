import numpy as np


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


class KMeansClustering:
    """Метод К-средних (Разделение на заданное число кластеров)"""

    def __init__(self, n_clusters=3, max_iter=100):
        self.K = n_clusters
        self.max_iter = max_iter
        self.centroids = []
        self.labels = []
        self.iterations = 0

    def fit(self, X):
        # Шаг 2: Первые K объектов становятся начальными центроидами
        self.centroids = X[:self.K].copy()

        for iteration in range(self.max_iter):
            self.iterations += 1

            # Шаг 4: Распределение объектов по ближайшим кластерам
            distances = np.array([[euclidean_distance(x, c) for c in self.centroids] for x in X])
            self.labels = np.argmin(distances, axis=1)

            # Шаг 5: Пересчет центроидов как среднее арифметическое
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.K):
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = self.centroids[k]

            # Шаг 6: Проверка условия окончания
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return self.labels, self.centroids


class MaximinClustering:
    """Метод максимина (Автоматическое определение числа кластеров)"""

    def __init__(self):
        self.centroids = []
        self.labels = []
        self.iterations = 0

    def fit(self, X):
        # Шаг 1: Первый прототип - первый объект
        self.centroids = [X[0]]

        # Шаг 2: Второй прототип - наиболее удаленный от первого
        dists_to_first = [euclidean_distance(x, self.centroids[0]) for x in X]
        self.centroids.append(X[np.argmax(dists_to_first)])

        # Шаг 3: Первоначальное пороговое расстояние T
        T = euclidean_distance(self.centroids[0], self.centroids[1]) / 2.0

        while True:
            self.iterations += 1

            # Шаг 4: Распределение объектов
            distances = np.array([[euclidean_distance(x, c) for c in self.centroids] for x in X])
            self.labels = np.argmin(distances, axis=1)

            # Шаг 5: Поиск наиболее удаленного объекта в каждом кластере (Y_k)
            new_candidates = []
            for k in range(len(self.centroids)):
                indices = np.where(self.labels == k)[0]
                if len(indices) == 0: continue

                dists_in_cluster = distances[indices, k]
                max_idx = indices[np.argmax(dists_in_cluster)]
                max_dist = np.max(dists_in_cluster)

                # Шаг 6: Проверка условия Y_k < T
                if max_dist > T:
                    new_candidates.append(X[max_idx])

            if not new_candidates:
                break  # Алгоритм завершен, новых кластеров нет

            # Добавляем новые прототипы кластеров
            self.centroids.extend(new_candidates)

            # Шаг 7: Расчет нового порога T = Сумма парных расстояний / (K * (K - 1))
            K_count = len(self.centroids)
            sum_dist = 0
            for i in range(K_count - 1):
                for j in range(i + 1, K_count):
                    sum_dist += euclidean_distance(self.centroids[i], self.centroids[j])
            T = sum_dist / (K_count * (K_count - 1))

        return self.labels, np.array(self.centroids)