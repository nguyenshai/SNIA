"""
compare_all_problems.py
=======================
Tổng hợp benchmark cho cả Continuous và Discrete Problems.
Kẻ bảng cho từng thuật toán với những bài toán phù hợp (tương thích).
Mỗi thuật toán xuất ra 1 ảnh PNG riêng biệt.

Chạy:
    python scripts/compare_all_problems.py
"""

import sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / 'results' / 'tables' / 'per_algo'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports Problems ──────────────────────────────────────────────────────────
# Continuous
from problems.continuous.Sphere     import Sphere
from problems.continuous.Rastrigin  import Rastrigin
from problems.continuous.Ackley     import Ackley
from problems.continuous.Rosenbrock import Rosenbrock
from problems.continuous.Griewank   import Griewank

# Discrete
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
from algorithms.evolution.DE        import DifferentialEvolution
from algorithms.physics.SA          import SimulatedAnnealing
from algorithms.human.TLBO          import TLBO
from algorithms.classical.Hill_climbing import HillClimbing
from algorithms.classical.A_star    import AStar
from algorithms.classical.BFS       import BreadthFirstSearch
from algorithms.classical.DFS       import DepthFirstSearch

# ── Config ───────────────────────────────────────────────────────────────────
N_RUNS = 10
ITERATIONS = 50

# Định nghĩa các bài toán
CONTINUOUS_PROBLEMS = {
    'Sphere':     (Sphere, 10),
    'Rastrigin':  (Rastrigin, 10),
    'Ackley':     (Ackley, 10),
    'Rosenbrock': (Rosenbrock, 10),
    'Griewank':   (Griewank, 10),
}

DISCRETE_PROBLEMS = {
    'TSP':            TSPProblem,
    'Knapsack':       KnapsackProblem,
    'GraphColoring':  GraphColoringProblem,
    'ShortestPath':   ShortestPathProblem,
}

# Định nghĩa thuật toán và các bài toán mà chúng có thể giải
# Format: { AlgoName: (AlgoClass, Params, [List of Supported Problem Keys]) }
ALL_CONTINUOUS = list(CONTINUOUS_PROBLEMS.keys())
ALL_DISCRETE = list(DISCRETE_PROBLEMS.keys())
ALL_PROBLEMS = ALL_CONTINUOUS + ALL_DISCRETE

ALGO_CONFIG = {
    # Metaheuristics (Giải được cả Continuous và Discrete qua mapping)
    'CS':   (CuckooSearch,             {'n': 40, 'pa': 0.25}, ALL_PROBLEMS),
    'ABC':  (ArtificialBeeColony,      {'pop_size': 40},      ALL_PROBLEMS),
    'FA':   (FireflyAlgorithm,         {'pop_size': 30, 'alpha': 0.5}, ALL_PROBLEMS),
    'PSO':  (ParticleSwarmOptimization, {'pop_size': 40, 'w': 0.7, 'c1': 1.5, 'c2': 1.5}, ALL_PROBLEMS),
    'HC':   (HillClimbing,             {'restarts': 3, 'step_scale': 0.1}, ALL_PROBLEMS),
    'GA':   (GeneticAlgorithm,         {'pop_size': 40, 'cx_rate': 0.8, 'mut_rate': 0.1}, ALL_PROBLEMS),
    'DE':   (DifferentialEvolution,    {'pop_size': 40, 'F': 0.8, 'CR': 0.9}, ALL_PROBLEMS),
    'SA':   (SimulatedAnnealing,       {'T0': 1.0, 'alpha': 0.99}, ALL_PROBLEMS),
    'TLBO': (TLBO,                     {'pop_size': 30}, ALL_PROBLEMS),
    
    # Thuật toán chuyên dụng (Vẫn cho phép thử trên các bài toán khác nếu logic hỗ trợ)
    'ACO':  (AntColonyOptimization,    {'n_ants': 20, 'alpha': 1.0, 'beta': 2.0}, ALL_PROBLEMS),
    'A_star': (AStar, {}, ALL_PROBLEMS),
    'BFS':    (BreadthFirstSearch, {}, ALL_PROBLEMS),
    'DFS':    (DepthFirstSearch, {}, ALL_PROBLEMS),
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_fitness_value(res):
    if isinstance(res, dict):
        for key in ['total_cost', 'total_penalty', 'penalty', 'total_distance']:
            if key in res: return float(res[key])
        return float(res.get('score', 0))
    return float(res)

def apply_mapping(prob, pname):
    """Mapping cho Metaheuristic chạy trên Discrete."""
    if pname == 'TSP':
        prob.dim, prob.bounds = prob.n_cities, (0, 1)
    elif pname == 'Knapsack':
        prob.dim, prob.bounds = len(prob.items), (0, 1)
    elif pname == 'GraphColoring':
        prob.dim, prob.bounds = prob.n_nodes, (0, prob.n_colors - 1)
    elif pname == 'ShortestPath':
        prob.dim, prob.bounds = prob.grid_size * 2, (0, prob.grid_size - 1)
    
    orig_eval = prob.evaluate
    def wrapped_eval(x):
        discrete_x = x
        if pname == 'TSP': discrete_x = list(np.argsort(x))
        elif pname == 'Knapsack': discrete_x = [1 if v > 0.5 else 0 for v in x]
        elif pname == 'GraphColoring': discrete_x = {i: int(np.clip(np.round(v), 0, prob.n_colors-1)) for i, v in enumerate(x)}
        elif pname == 'ShortestPath':
            half = len(x) // 2
            path = [(int(np.clip(np.round(x[i]), 0, prob.grid_size-1)), 
                     int(np.clip(np.round(x[i+half]), 0, prob.grid_size-1))) 
                    for i in range(half)]
            path = [prob.start] + path + [prob.goal]
            discrete_x = path
        return get_fitness_value(orig_eval(discrete_x))
    prob.evaluate = wrapped_eval

# ── Benchmark ────────────────────────────────────────────────────────────────
def run_benchmark():
    results = {} # {algo_name: {prob_name: metrics}}

    for aname, (acls, params, supported_probs) in ALGO_CONFIG.items():
        print(f"\nProcessing Algorithm: {aname}")
        results[aname] = {}
        
        for pname in supported_probs:
            fits, times = [], []
            print(f"  Testing on {pname}...", end=' ', flush=True)
            
            for _ in range(N_RUNS):
                # 1. Khởi tạo bài toán
                if pname in CONTINUOUS_PROBLEMS:
                    pcls, dim = CONTINUOUS_PROBLEMS[pname]
                    prob = pcls(dim=dim)
                else:
                    pcls = DISCRETE_PROBLEMS[pname]
                    prob = pcls.medium()
                    # Apply mapping nếu là Metaheuristic chạy Discrete
                    if aname not in ['ACO', 'A_star', 'BFS', 'DFS'] and pname in ['TSP', 'Knapsack', 'GraphColoring', 'ShortestPath']:
                        apply_mapping(prob, pname)
                
                # 2. Chạy thuật toán
                if aname in ('A_star', 'BFS', 'DFS'):
                    # Search algorithms are deterministic on their intended problems
                    algIdx = acls(prob, params=dict(params))
                    t0 = time.perf_counter()
                    try:
                        algIdx.solve(iterations=None)
                        fits.append(float(algIdx.best_fitness or 0))
                        times.append(time.perf_counter() - t0)
                    except Exception: pass
                    break # Only one run needed
                else:
                    algIdx = acls(prob, params=dict(params))
                    t0 = time.perf_counter()
                    try:
                        algIdx.solve(iterations=ITERATIONS)
                        fits.append(float(algIdx.best_fitness or 0))
                        times.append(time.perf_counter() - t0)
                    except Exception: pass
            
            if fits:
                results[aname][pname] = {
                    'mbf': np.mean(fits), 'sbf': np.std(fits), 
                    'met': np.mean(times), 'set': np.std(times)
                }
                print(f"Done.")
            else:
                print(f"Failed.")
    
    return results

def save_tables(results):
    for aname, prob_data in results.items():
        if not prob_data:
            print(f"Skipping {aname}: No data.")
            continue
        
        data = []
        for pname, res in prob_data.items():
            if res is None: continue
            data.append([
                pname, 
                f"{res['mbf']:.4f}" if abs(res['mbf']) > 1e-4 else f"{res['mbf']:.2e}",
                f"{res['sbf']:.4f}" if abs(res['sbf']) > 1e-4 else f"{res['sbf']:.2e}",
                f"{res['met']:.4f}", 
                f"{res['set']:.4f}"
            ])
        
        if not data:
            print(f"Skipping {aname}: Data list is empty.")
            continue

        try:
            # Điều chỉnh chiều cao dựa trên số lượng hàng (tối thiểu 4, cộng thêm 0.8 cho mỗi hàng)
            fig_height = max(4, len(data) * 0.8 + 1)
            fig, ax = plt.subplots(figsize=(12, fig_height))
            ax.axis('off')
            ax.set_title(f"Benchmark Results: {aname}", fontsize=14, pad=20)
            
            col_labels = ['Problem', 'Mean Best Fit', 'Std Best Fit', 'Mean Time (s)', 'Std Time (s)']
            table = ax.table(cellText=data, colLabels=col_labels, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2.0)

            clean_aname = aname.replace('*', '_star').lower()
            save_path = OUT_DIR / f"table_{clean_aname}_combined.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"Table saved for {aname} at {save_path}")
        except Exception as e:
            print(f"Error saving table for {aname}: {e}")
            plt.close()

if __name__ == "__main__":
    print("Starting Combined Benchmark (Continuous & Discrete)...")
    res = run_benchmark()
    save_tables(res)
    print("\nAll done. Tables are in results/tables/per_algo/")
