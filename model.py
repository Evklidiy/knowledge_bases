import numpy as np


class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        """Обучение сети по правилу Хебба"""
        for p in patterns:
            # Формирование весов: w_ij = a_i * a_j
            self.weights += np.outer(p, p)
        # Обнуление главной диагонали (отсутствие обратной связи на себя)
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, max_iter=100):
        """Асинхронное распознавание образа"""
        state = np.copy(pattern)
        for _ in range(max_iter):
            prev_state = np.copy(state)

            # Асинхронное обновление: случайный порядок обхода нейронов
            indices = np.arange(self.n_neurons)
            np.random.shuffle(indices)

            for i in indices:
                net_input = np.dot(self.weights[i], state)
                # Бинарная/биполярная пороговая функция активации
                if net_input > 0:
                    state[i] = 1
                elif net_input < 0:
                    state[i] = -1

            # Проверка на стабилизацию сети (энергетический минимум)
            if np.array_equal(state, prev_state):
                break

        return state