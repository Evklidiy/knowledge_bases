import numpy as np

# Размерность сетки: 7 строк, 5 столбцов = 35 нейронов
GRID_ROWS = 7
GRID_COLS = 5
N_NEURONS = GRID_ROWS * GRID_COLS

# Биполярные образы (-1 и 1) для букв
PATTERN_P = np.array([
    1, 1, 1, 1, 1,
    1,-1,-1,-1, 1,
    1,-1,-1,-1, 1,
    1,-1,-1,-1, 1,
    1,-1,-1,-1, 1,
    1,-1,-1,-1, 1,
    1,-1,-1,-1, 1
])

PATTERN_L = np.array([
   -1,-1, 1,-1,-1,
   -1, 1,-1, 1,-1,
    1,-1,-1,-1, 1,
    1,-1,-1,-1, 1,
    1,-1,-1,-1, 1,
    1,-1,-1,-1, 1,
    1,-1,-1,-1, 1
])

PATTERN_E = np.array([
    1, 1, 1, 1, 1,
    1,-1,-1,-1,-1,
    1,-1,-1,-1,-1,
    1, 1, 1, 1,-1,
    1,-1,-1,-1,-1,
    1,-1,-1,-1,-1,
    1, 1, 1, 1, 1
])

PATTERNS = {
    "Буква 'П'": PATTERN_P,
    "Буква 'Л'": PATTERN_L,
    "Буква 'Е'": PATTERN_E
}

def get_noisy_pattern(pattern, noise_percent):
    """Вносит заданный процент зашумления путем инверсии случайных битов"""
    noisy = np.copy(pattern)
    num_flips = int(N_NEURONS * (noise_percent / 100.0))
    if num_flips > 0:
        flip_indices = np.random.choice(N_NEURONS, num_flips, replace=False)
        for idx in flip_indices:
            noisy[idx] = -noisy[idx]
    return noisy