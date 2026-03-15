"""
per_problem_benchmark.py
========================
Tạo biểu đồ so sánh chi tiết cho TỪNG bài toán (Problem).
Mỗi bài toán xuất ra 1 ảnh PNG chứa 2 biểu đồ:
1. Fitness (Thanh - Median Best Fitness)
2. Time (Đường/Thanh - Execution Time)

Chạy:
    python scripts/per_problem_benchmark.py
"""

import sys, time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / 'results' / 'visualize' / 'per_problem'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports Problems ──────────────────────────────────────────────────────────
from problems.continuous.Sphere     import Sphere
from problems.continuous.Rastrigin  import Rastrigin
from problems.continuous.Ackley     import Ackley
from problems.continuous.Rosenbrock import Rosenbrock
from problems.continuous.Griewank   import Griewank

from problems.discrete.TSP           import TSPProblem
from problems.discrete.Knapsack      import KnapsackProblem
from problems.discrete.GraphColoring import GraphColoringProblem
from problems.discrete.ShortestPath  import ShortestPathProblem

# ── Imports Algorithms ────────────────────────────────────────────────────────
from algorithms.biology.CS          import CuckooSearch
from algorithms.biology.ABC         import ArtificialBeeColony
from algorithms.biology.FA          import FireflyAlgorithm
from algorithms.biology.PSO         import ParticleSwarmOptimization
from algorithms.biology.ACO         import AntColonyOptimization
from algorithms.evolution.GA        import GeneticAlgorithm
from algorithms.physics.SA          import SimulatedAnnealing
from algorithms.human.TLBO          import TLBO
from algorithms.classical.Hill_climbing import HillClimbing
from algorithms.classical.A_star    import AStar
from algorithms.classical.BFS       import BreadthFirstSearch
from algorithms.classical.DFS       import DepthFirstSearch

# ── Config ───────────────────────────────────────────────────────────────────
N_RUNS = 10
ITERATIONS = 100

# Danh sách tất cả các bài toán và thuật toán hỗ trợ tương ứng
PROBLEMS_MAP = {
    'Sphere': (Sphere, {'dim': 10}, ['CS', 'ABC', 'FA', 'PSO', 'GA', 'SA', 'HC', 'TLBO']),
    'Rastrigin': (Rastrigin, {'dim': 10}, ['CS', 'ABC', 'FA', 'PSO', 'GA', 'SA', 'HC', 'TLBO']),
    'Ackley': (Ackley, {'dim': 10}, ['CS', 'ABC', 'FA', 'PSO', 'GA', 'SA', 'HC', 'TLBO']),
    'Rosenbrock': (Rosenbrock, {'dim': 10}, ['CS', 'ABC', 'FA', 'PSO', 'GA', 'SA', 'HC', 'TLBO']),
    'Griewank': (Griewank, {'dim': 10}, ['CS', 'ABC', 'FA', 'PSO', 'GA', 'SA', 'HC', 'TLBO']),
    'TSP': (TSPProblem, {}, ['ACO', 'GA', 'SA', 'HC', 'TLBO']),
    'Knapsack': (KnapsackProblem, {}, ['GA', 'SA', 'HC', 'ABC', 'TLBO']),
    'GraphColoring': (GraphColoringProblem, {}, ['GA', 'SA', 'HC', 'TLBO']),
    'ShortestPath': (ShortestPathProblem, {}, ['A_star', 'BFS', 'DFS']),
}

ALGO_CLASSES = {
    'CS': (CuckooSearch, {'n': 40, 'pa': 0.25}),
    'ABC': (ArtificialBeeColony, {'pop_size': 40}),
    'FA': (FireflyAlgorithm, {'pop_size': 30, 'alpha': 0.5}),
    'PSO': (ParticleSwarmOptimization, {'pop_size': 40, 'w': 0.7, 'c1': 1.5, 'c2': 1.5}),
    'GA': (GeneticAlgorithm, {'pop_size': 50, 'cx_rate': 0.8, 'mut_rate': 0.2}),
    'SA': (SimulatedAnnealing, {'T0': 1.0, 'alpha': 0.99}),
    'HC': (HillClimbing, {'restarts': 3, 'step_scale': 0.1}),
    'TLBO': (TLBO, {'pop_size': 30}),
    'ACO': (AntColonyOptimization, {'n_ants': 20, 'alpha': 1.0, 'beta': 2.0}),
    'A_star': (AStar, {}),
    'BFS': (BreadthFirstSearch, {}),
    'DFS': (DepthFirstSearch, {}),
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_fitness_value(res):
    if isinstance(res, dict):
        for key in ['total_cost', 'total_penalty', 'penalty', 'total_distance']:
            if key in res: return float(res[key])
        return float(res.get('score', 0))
    return float(res)

def apply_mapping(prob, pname):
    if pname == 'TSP':
        prob.dim, prob.bounds = prob.n_cities, (0, 1)
    elif pname == 'Knapsack':
        prob.dim, prob.bounds = len(prob.items), (0, 1)
    elif pname == 'GraphColoring':
        prob.dim, prob.bounds = prob.n_nodes, (0, prob.n_colors - 1)
    
    orig_eval = prob.evaluate
    def wrapped_eval(x):
        discrete_x = x
        if pname == 'TSP': discrete_x = list(np.argsort(x))
        elif pname == 'Knapsack': discrete_x = [1 if v > 0.5 else 0 for v in x]
        elif pname == 'GraphColoring': discrete_x = {i: int(np.clip(np.round(v), 0, prob.n_colors-1)) for i, v in enumerate(x)}
        return get_fitness_value(orig_eval(discrete_x))
    prob.evaluate = wrapped_eval

# ── Main Loop ────────────────────────────────────────────────────────────────
def run_benchmark():
    for pname, (pcls, pkwargs, supported_algos) in PROBLEMS_MAP.items():
        print(f"\nBenchmarking Problem: {pname}")
        data_fitness = {}
        data_time = {}

        for aname in supported_algos:
            if aname not in ALGO_CLASSES: continue
            acls, aparams = ALGO_CLASSES[aname]
            
            fits, times = [], []
            print(f"  Running {aname}...", end=' ', flush=True)
            
            for _ in range(N_RUNS):
                # Khởi tạo bài toán
                if pname in ['Sphere', 'Rastrigin', 'Ackley', 'Rosenbrock', 'Griewank']:
                    prob = pcls(**pkwargs)
                else:
                    prob = pcls.medium()
                    # Apply mapping if Metaheuristic
                    if aname in ['GA', 'SA', 'HC', 'ABC', 'PSO', 'CS', 'FA']:
                        apply_mapping(prob, pname)
                
                alg = acls(prob, params=dict(aparams))
                t0 = time.perf_counter()
                try:
                    alg.solve(iterations=ITERATIONS)
                    fits.append(float(alg.best_fitness or 0))
                    times.append(time.perf_counter() - t0)
                except Exception: pass
            
            if fits:
                data_fitness[aname] = fits
                data_time[aname] = times
                print(f"Done.")
            else:
                print(f"Failed.")

        if data_fitness:
            plot_per_problem(pname, data_fitness, data_time)

def plot_per_problem(pname, data_fitness, data_time):
    # Chuẩn bị dữ liệu vẽ
    algos = list(data_fitness.keys())
    
    # Kết hợp fitness: dùng Mean và Std cho sát thực tế ảnh mẫu
    mean_fits = [np.mean(data_fitness[a]) for a in algos]
    std_fits = [np.std(data_fitness[a]) for a in algos]
    
    # Kết hợp time: dùng Mean
    avg_times = [np.mean(data_time[a]) for a in algos]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Thiết lập giao diện sáng (White) như ảnh mẫu mới
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax1.set_title(f"Performance (Fitness & Time) - {pname}", fontsize=14, fontweight='bold', pad=20)

    # Trục Y bên trái cho Fitness (Bar Chart)
    x = np.arange(len(algos))
    bar_width = 0.6
    
    # Lớp cột Fitness: màu xanh nhạt với viền xanh đậm
    # Sử dụng yerr để hiển thị sai số (Std) như trên ảnh mẫu
    bars = ax1.bar(x, mean_fits, bar_width, yerr=std_fits, 
                   color='#add8e6', edgecolor='#4169e1', alpha=0.8,
                   label='Mean Best Fitness ± Std', capsize=5)
    
    ax1.set_ylabel("Best Fitness", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='-', alpha=0.3)

    # Trục Y bên phải cho Execution Time (Line Chart)
    ax2 = ax1.twinx()
    line = ax2.plot(x, avg_times, color='red', marker='o', markersize=8, 
                    linestyle='--', linewidth=2, label='Mean Execution Time')
    
    ax2.set_ylabel("Execution Time (s)", color='red', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')

    # Hợp nhất Legend từ cả hai trục
    # Ở ảnh mẫu, legend nằm ở top left
    ax1.legend(loc='upper left', shadow=True, fancybox=True)
    ax2.legend(loc='upper right', shadow=True, fancybox=True)

    # Chú thích phía dưới ảnh
    plt.figtext(0.5, -0.05, f"(b) Performance trên {pname}", 
                ha='center', fontsize=24, fontfamily='serif')

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"benchmark_{pname.lower()}_combined.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: benchmark_{pname.lower()}_combined.png")

if __name__ == "__main__":
    print("Starting Per-Problem Detailed Benchmarks...")
    run_benchmark()
    print(f"\nAll DONE. Results saved in {OUT_DIR}")
