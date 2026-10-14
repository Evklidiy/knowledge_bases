import numpy as np


def build_normalized_matrix(raw_data, goals):
    """Нормирование критериев путем приведения к масштабу [0; 1]"""
    norm_mat = np.zeros(raw_data.shape, dtype=float)
    for col_idx in range(raw_data.shape[1]):
        column = raw_data[:, col_idx]
        if goals[col_idx] == 1:
            norm_mat[:, col_idx] = column / np.max(column)
        else:
            norm_mat[:, col_idx] = np.min(column) / column
    return norm_mat


def get_pareto_front(matrix):
    """Выделение множества Парето-оптимальных альтернатив"""
    rows = matrix.shape[0]
    pareto_front = np.ones(rows, dtype=bool)

    for i in range(rows):
        for j in range(rows):
            if i != j:
                # Если j-я альтернатива не хуже i-й по всем критериям и строго лучше хотя бы по одному
                if np.all(matrix[j] >= matrix[i]) and np.any(matrix[j] > matrix[i]):
                    pareto_front[i] = False
                    break
    return pareto_front


def compute_criteria_weights(expert_matrix):
    """Вычисление весов критериев методом ранга"""
    rank_sums = np.sum(expert_matrix, axis=0)
    return rank_sums / np.sum(rank_sums)


def execute_electre(norm_matrix, weights, threshold_c, threshold_d):
    """Основной алгоритм метода ЭЛЕКТРА"""
    n_alts = norm_matrix.shape[0]
    c_indices = np.zeros((n_alts, n_alts))
    d_indices = np.zeros((n_alts, n_alts))

    for i in range(n_alts):
        for j in range(n_alts):
            if i == j:
                c_indices[i, j] = 1.0
                d_indices[i, j] = 0.0
                continue

            # Подмножество критериев, где i-я альтернатива не хуже j-й
            better_or_eq = norm_matrix[i] >= norm_matrix[j]
            c_indices[i, j] = np.sum(weights[better_or_eq])

            # Подмножество критериев, где i-я альтернатива хуже j-й
            worse = norm_matrix[i] < norm_matrix[j]
            if np.any(worse):
                d_indices[i, j] = np.max(norm_matrix[j][worse] - norm_matrix[i][worse])
            else:
                d_indices[i, j] = 0.0

    # Предельные значения индексов согласия и несогласия для каждой альтернативы
    c_limits = np.array([np.min(np.delete(c_indices[row], row)) for row in range(n_alts)])
    d_limits = np.array([np.max(np.delete(d_indices[row], row)) for row in range(n_alts)])

    # Формирование ядра (лучших альтернатив)
    core_set = (c_limits >= threshold_c) & (d_limits <= threshold_d)

    return c_limits, d_limits, core_set