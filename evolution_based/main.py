# main.py
import numpy as np
from benchmark_lib import BenchmarkFunc
from GA import GA
from DE import DE
from visualize_swarm import plot_convergence
DIM = 30                    # Số chiều bài toán
BOUNDS = [(-5, 5)] * DIM    # Miền tìm kiếm
RUNS = 5                    # Số lần chạy độc lập

bench = BenchmarkFunc()
func = bench.rastrigin      # đổi sang sphere / ackley nếu cần
def run_algo(algo_class):
    best_vals = []
    histories = []

    for _ in range(RUNS):
        algo = algo_class(func, BOUNDS)
        best, history = algo.optimize()

        best_vals.append(func(best))
        histories.append(history)

    return {
        "best": np.min(best_vals),
        "mean": np.mean(best_vals),
        "std": np.std(best_vals),
        # Lấy trung bình lịch sử hội tụ
        "history": np.mean(histories, axis=0)
    }

if __name__ == "__main__":
    ga_res = run_algo(GA)
    de_res = run_algo(DE)
    # In kết quả số liệu
    print("GA  | Best:", ga_res["best"],
          "Mean:", ga_res["mean"],
          "Std:", ga_res["std"])

    print("DE  | Best:", de_res["best"],
          "Mean:", de_res["mean"],
          "Std:", de_res["std"])
    # HIỂN THỊ ĐỒ THỊ HỘI TỤ
    plot_convergence(
        histories=[ga_res["history"], de_res["history"]],
        labels=["GA", "DE"],
        title="So sánh hội tụ GA và DE trên hàm Rastrigin"
    )
