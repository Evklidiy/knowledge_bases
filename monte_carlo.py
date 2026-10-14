from lfsr import LFSR

def run_simulation(a, b, c, d, n=10000):
    rng = LFSR(seed=987654321)
    results = []
    defect_prob = a / 100.0

    # Перебор процента входного контроля от 0% до 100% с шагом 5%
    for proc in range(0, 101, 5):
        p = proc / 100.0
        total_cost = 0.0

        for _ in range(n):
            r_control = rng.random()
            r_defect = rng.random()

            is_controlled = r_control < p
            is_defective = r_defect < defect_prob

            if is_controlled:
                total_cost += b
                if is_defective:
                    total_cost += c  # Подгонка при выявлении брака
            else:
                if is_defective:
                    total_cost += d  # Замена в готовом изделии

        results.append((proc, total_cost))

    return results