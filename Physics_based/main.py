import numpy as np
from sa import simulated_annealing
from benchmark import sphere, rastrigin, ackley
from visualize_swarm import plot_convergence

# =====================
# CẤU HÌNH THÍ NGHIỆM
# =====================
DIM = 5
LOWER_BOUND = -5
UPPER_BOUND = 5

T0 = 100
T_MIN = 1e-3
ALPHA = 0.95
MAX_ITER = 100

RUNS = 30


def benchmark(f, name):
    best_values = []
    histories = []

    for _ in range(RUNS):
        _, best_f, history = simulated_annealing(
            f,
            DIM,
            LOWER_BOUND,
            UPPER_BOUND,
            T0,
            T_MIN,
            ALPHA,
            MAX_ITER
        )
        best_values.append(best_f)
        histories.append(history)

    best_values = np.array(best_values)

    print(f"===== {name} =====")
    print("Best:", np.min(best_values))
    print("Mean:", np.mean(best_values))
    print("Std :", np.std(best_values))
    print()

    return histories


def main():
    hist_sphere = benchmark(sphere, "Sphere")
    hist_rastrigin = benchmark(rastrigin, "Rastrigin")
    hist_ackley = benchmark(ackley, "Ackley")

    # Vẽ hội tụ (lấy 1 run đại diện)
    plot_convergence(
        [hist_sphere[0], hist_rastrigin[0], hist_ackley[0]],
        ["Sphere", "Rastrigin", "Ackley"],
        "SA Convergence on Benchmark Functions"
    )


if __name__ == "__main__":
    main()
