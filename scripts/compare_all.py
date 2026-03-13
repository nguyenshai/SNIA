"""
Algorithm Performance Comparison for SNIA.

Compares algos on CONTINUOUS and DISCRETE benchmark functions
using two main evaluation criteria:

  1. CONVERGENCE SPEED
  2. ROBUSTNESS
  3. PERFORMANCE (Solution Quality)

Continuous Output: results/compare/continuous

Discrete comparison - ALL algorithms × ALL discrete problems:
  Criteria:  1) Speed (execution time)   2) Convergence (curve + T90)

Discrete Output:   results/compare/discrete

Usage:
    python scripts/compare_all.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Project root --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR_CONT = PROJECT_ROOT / 'results' / 'compare' / 'continuous'
OUT_DIR_CONT.mkdir(parents=True, exist_ok=True)

OUT_DIR_DISC = PROJECT_ROOT / 'results' / 'compare' / 'discrete'
OUT_DIR_DISC.mkdir(parents=True, exist_ok=True)

# =====================================================================
#  Imports
# =====================================================================

# -- Continuous --
from problems.continuous.Sphere import Sphere
from problems.continuous.Rastrigin import Rastrigin
from problems.continuous.Rosenbrock import Rosenbrock
from problems.continuous.Ackley import Ackley
from problems.continuous.Griewank import Griewank

# -- Discrete --
from problems.discrete.TSP import TSPProblem
from problems.discrete.Knapsack import KnapsackProblem
from problems.discrete.GraphColoring import GraphColoringProblem
from problems.discrete.ShortestPath import ShortestPathProblem

# -- Algorithms --
from algorithms.biology.CS import CuckooSearch
from algorithms.biology.ABC import ArtificialBeeColony
from algorithms.biology.FA import FireflyAlgorithm
from algorithms.biology.PSO import ParticleSwarmOptimization
from algorithms.biology.ACO import AntColonyOptimization
from algorithms.evolution.GA import GeneticAlgorithm
from algorithms.evolution.DE import DifferentialEvolution
from algorithms.classical.Hill_climbing import HillClimbing
from algorithms.physics.SA import SimulatedAnnealing
from algorithms.classical.BFS import BreadthFirstSearch
from algorithms.classical.DFS import DepthFirstSearch
from algorithms.classical.A_star import AStar
from algorithms.human.TLBO import TLBO


# =====================================================================
#  Style constants
# =====================================================================
BG      = '#0d1117'
PANEL   = '#161b22'
TXT     = '#e6edf3'
GRID_C  = '#30363d'
BORDER  = '#30363d'

ALGO_COLORS = {
    'CS':   '#a371f7',
    'ABC':  '#00d4aa',
    'FA':   '#f0883e',
    'PSO':  '#58a6ff',
    'GA':   '#ff6b6b',
    'DE':   '#ffd93d',
    'HC':   '#54a0ff',
    'SA':   '#ff9ff3',
    'TLBO': '#79c0ff',
    'ACO':  '#d2a8ff',
    'BFS':  '#ff6b6b',
    'DFS':  '#54a0ff',
    'A*':   '#a371f7',
}

ALGO_MARKERS = {
    'CS': 'v', 'ABC': 'D', 'FA': 'P', 'PSO': '^',
    'GA': 'o', 'DE': 's', 'HC': '*', 'SA': 'X',
    'TLBO': 'h', 'ACO': 'd', 'BFS': 'p', 'DFS': 'H', 'A*': '8',
}

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.15, color='#484f58')
    if title:
        ax.set_title(title, color=TXT, fontsize=11, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)


def compute_t90(curve):
    """Return iteration index where 90% of total improvement is achieved."""
    if not curve or len(curve) < 2:
        return len(curve) - 1 if curve else 0
    first, last = curve[0], curve[-1]
    improvement = first - last
    if improvement <= 1e-12:
        return len(curve) - 1
    target = first - 0.9 * improvement
    for i, v in enumerate(curve):
        if v <= target:
            return i
    return len(curve) - 1


# =====================================================================
#  CONTINUOUS Setup
# =====================================================================

CONT_ALGO_NAMES = ['CS', 'ABC', 'FA', 'PSO', 'GA', 'HC', 'SA']

CONT_ALGO_REGISTRY = {
    'CS':  (CuckooSearch,             {'n': 40, 'pa': 0.25}),
    'ABC': (ArtificialBeeColony,      {'pop_size': 40}),
    'FA':  (FireflyAlgorithm,         {'pop_size': 30, 'alpha': 0.5, 'beta0': 1.0, 'gamma': 1.0}),
    'PSO': (ParticleSwarmOptimization, {'pop_size': 40, 'w': 0.7, 'c1': 1.5, 'c2': 1.5}),
    'GA':  (GeneticAlgorithm,          {'pop_size': 40, 'cx_rate': 0.8, 'mut_rate': 0.1}),
    'HC':  (HillClimbing,            {'restarts': 5, 'step_scale': 0.1}),
    'SA':  (SimulatedAnnealing,       {'T0': 1.0, 'alpha': 0.99}),
}

CONT_PROBLEM_REGISTRY = {
    'Sphere':     (Sphere,     10),
    'Rastrigin':  (Rastrigin,  10),
    'Ackley':     (Ackley,     10),
    'Rosenbrock': (Rosenbrock, 10),
    'Griewank':   (Griewank,   10),
}

CONT_ITERATIONS = 200
CONT_N_RUNS = 30

FITNESS_THRESHOLDS = {
    'Sphere':     1.0,
    'Rastrigin':  50.0,
    'Ackley':     5.0,
    'Rosenbrock': 100.0,
    'Griewank':   1.0,
}

def run_continuous_benchmark():
    data = {}
    for prob_name, (prob_cls, dim) in CONT_PROBLEM_REGISTRY.items():
        data[prob_name] = {}
        print(f'\n  {prob_name}:', end='', flush=True)

        for aname in CONT_ALGO_NAMES:
            algo_cls, params = CONT_ALGO_REGISTRY[aname]
            runs = []
            for _ in range(CONT_N_RUNS):
                prob = prob_cls(dim=dim)
                alg = algo_cls(prob, params=dict(params))
                t0 = time.perf_counter()
                try:
                    alg.solve(iterations=CONT_ITERATIONS)
                except Exception as e:
                    print(f' [{aname} FAIL: {e}]', end='')
                    break
                conv = [h.get('global_best_fit', None) for h in alg.history]
                runs.append({
                    'convergence': conv,
                    'best_fit': alg.best_fitness,
                    'time_s': time.perf_counter() - t0,
                })
            data[prob_name][aname] = runs
            print(f' {aname}✓', end='', flush=True)

    print()
    return data

# ---- Continuous Charts ----
def chart_cont_convergence_curves(data):
    print('\n--- [Cont] Convergence Curves ---')
    prob_names = list(CONT_PROBLEM_REGISTRY.keys())
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()
    for i in range(len(prob_names), len(axes)): axes[i].set_visible(False)

    for idx, prob_name in enumerate(prob_names):
        ax = axes[idx]
        style_ax(ax, title=prob_name, xlabel='Iteration', ylabel='Best Fitness')

        for aname in CONT_ALGO_NAMES:
            runs = data[prob_name].get(aname, [])
            if not runs: continue
            all_c = [r['convergence'] for r in runs]
            ml = min(len(c) for c in all_c)
            if ml == 0: continue
            arr = np.array([c[:ml] for c in all_c])
            med = np.median(arr, axis=0)
            lo, hi = np.min(arr, axis=0), np.max(arr, axis=0)
            x = np.arange(1, ml + 1)
            c = ALGO_COLORS[aname]
            ax.plot(x, med, label=aname, color=c, linewidth=2,
                    marker=ALGO_MARKERS[aname], markevery=max(1, ml // 8), markersize=5)
            ax.fill_between(x, lo, hi, alpha=0.08, color=c)

        ax.set_yscale('log')
        ax.legend(facecolor='#21262d', edgecolor=BORDER, labelcolor='#c9d1d9', fontsize=7, ncol=2)

    for j in range(len(prob_names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Convergence Speed - All Algorithms × All Problems', color=TXT, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_CONT / 'convergence_curves.png'), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

def chart_cont_t90(data):
    print('--- [Cont] T90 Convergence Speed ---')
    prob_names = list(CONT_PROBLEM_REGISTRY.keys())
    cols = min(3, max(1, len(prob_names)))
    rows = (len(prob_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 6.0 * rows))
    if rows * cols == 1: axes_flat = [axes]
    else: axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for _i in range(len(prob_names), len(axes_flat)): axes_flat[_i].set_visible(False)
    fig.patch.set_facecolor(BG)

    for ax, prob_name in zip(axes_flat, prob_names):
        style_ax(ax, title=prob_name, ylabel='T90 (iteration)')
        vals = []
        for aname in CONT_ALGO_NAMES:
            runs = data[prob_name].get(aname, [])
            t90_vals = [compute_t90(r['convergence']) for r in runs if r['convergence']]
            vals.append(np.median(t90_vals) if t90_vals else CONT_ITERATIONS)
            
        cols = [ALGO_COLORS[a] for a in CONT_ALGO_NAMES]
        x_pos = np.arange(len(CONT_ALGO_NAMES))
        bars = ax.bar(x_pos, vals, color=cols, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONT_ALGO_NAMES, color='#c9d1d9', fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.0f}', ha='center', va='bottom', color='#c9d1d9', fontsize=8, fontweight='bold')

    fig.suptitle('Convergence Speed - T90 (lower = faster)', color=TXT, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_CONT / 'convergence_t90.png'), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

def chart_cont_robustness_boxplots(data):
    print('--- [Cont] Robustness: Box Plots ---')
    prob_names = list(CONT_PROBLEM_REGISTRY.keys())
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()
    for i in range(len(prob_names), len(axes)): axes[i].set_visible(False)

    for idx, prob_name in enumerate(prob_names):
        ax = axes[idx]
        style_ax(ax, title=prob_name, ylabel='Final Fitness (log scale)')

        box_data, labels, cols = [], [], []
        for aname in CONT_ALGO_NAMES:
            runs = data[prob_name].get(aname, [])
            fits = [r['best_fit'] for r in runs if r.get('best_fit') is not None]
            if fits:
                box_data.append(fits)
                labels.append(aname)
                cols.append(ALGO_COLORS[aname])

        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                            medianprops={'color': 'white', 'linewidth': 2},
                            flierprops={'markerfacecolor': '#ff6b6b', 'markersize': 4})
            for patch, c in zip(bp['boxes'], cols):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_xticklabels(labels, color='#c9d1d9', fontsize=9)
            ax.set_yscale('log')

    for j in range(len(prob_names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Robustness - Final Fitness Distribution', color=TXT, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_CONT / 'robustness_boxplots.png'), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

def chart_cont_robustness_cv(data):
    print('--- [Cont] Robustness: CV Analysis ---')
    prob_names = list(CONT_PROBLEM_REGISTRY.keys())
    cv_matrix = np.zeros((len(CONT_ALGO_NAMES), len(prob_names)))
    
    for j, prob_name in enumerate(prob_names):
        for i, aname in enumerate(CONT_ALGO_NAMES):
            runs = data[prob_name].get(aname, [])
            fits = [r['best_fit'] for r in runs if r.get('best_fit') is not None]
            cv_matrix[i, j] = np.std(fits) / np.mean(fits) if fits and np.mean(fits) > 1e-12 else 0.0

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(PANEL)
    im = ax1.imshow(cv_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1.5)
    cb = fig.colorbar(im, ax=ax1)
    cb.set_label('CV (σ/μ)', color='#8b949e')
    cb.ax.tick_params(colors='#8b949e')

    ax1.set_xticks(range(len(prob_names)))
    ax1.set_xticklabels(prob_names, color='#c9d1d9', fontsize=10, rotation=20, ha='right')
    ax1.set_yticks(range(len(CONT_ALGO_NAMES)))
    ax1.set_yticklabels(CONT_ALGO_NAMES, color='#c9d1d9', fontsize=10)

    for i in range(len(CONT_ALGO_NAMES)):
        for j in range(len(prob_names)):
            ax1.text(j, i, f'{cv_matrix[i, j]:.3f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    plt.title('CV Heatmap (lower = more robust)', color=TXT)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_CONT / 'robustness_cv.png'), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

def chart_cont_execution_time(data):
    print('--- [Cont] Performance: Execution Time ---')
    prob_names = list(CONT_PROBLEM_REGISTRY.keys())
    cols = min(3, max(1, len(prob_names)))
    rows = (len(prob_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 5.0 * rows))
    if rows * cols == 1: axes_flat = [axes]
    else: axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for _i in range(len(prob_names), len(axes_flat)): axes_flat[_i].set_visible(False)
    fig.patch.set_facecolor(BG)

    for ax, prob_name in zip(axes_flat, prob_names):
        style_ax(ax, title=prob_name, ylabel='Median Time (s)')
        vals = []
        for aname in CONT_ALGO_NAMES:
            runs = data[prob_name].get(aname, [])
            times = [r.get('time_s', 0.0) for r in runs if 'time_s' in r]
            vals.append(np.median(times) if times else 0.0)
            
        color_list = [ALGO_COLORS[a] for a in CONT_ALGO_NAMES]
        x_pos = np.arange(len(CONT_ALGO_NAMES))
        bars = ax.bar(x_pos, vals, color=color_list, alpha=0.9, edgecolor='white', linewidth=1, width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONT_ALGO_NAMES, color='#c9d1d9', fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{val:.3f}s', ha='center', va='bottom', color='#c9d1d9', fontsize=8, rotation=90)
            
    fig.suptitle('Speed - Median Execution Time (Continuous)', color=TXT, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_CONT / 'execution_time_bars.png'), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

def chart_cont_performance_bars(data):
    print('--- [Cont] Performance: Best Fitness ---')
    prob_names = list(CONT_PROBLEM_REGISTRY.keys())
    cols = min(3, max(1, len(prob_names)))
    rows = (len(prob_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 6.0 * rows))
    if rows * cols == 1: axes_flat = [axes]
    else: axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for _i in range(len(prob_names), len(axes_flat)): axes_flat[_i].set_visible(False)
    fig.patch.set_facecolor(BG)

    for ax, prob_name in zip(axes_flat, prob_names):
        style_ax(ax, title=prob_name, ylabel='Median Best Fitness (lower=better)')
        vals = []
        for aname in CONT_ALGO_NAMES:
            runs = data[prob_name].get(aname, [])
            fits = [r['best_fit'] for r in runs if r.get('best_fit') is not None]
            vals.append(np.median(fits) if fits else 1e9)
            
        cols = [ALGO_COLORS[a] for a in CONT_ALGO_NAMES]
        x_pos = np.arange(len(CONT_ALGO_NAMES))
        bars = ax.bar(x_pos, vals, color=cols, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONT_ALGO_NAMES, color='#c9d1d9', fontsize=9)
        ax.set_yscale('log')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05, f'{val:.1e}', ha='center', va='bottom', color='#c9d1d9', fontsize=7, fontweight='bold', rotation=90)

    fig.suptitle('Performance - Median Best Fitness across Runs', color=TXT, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_CONT / 'performance_bars.png'), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


# =====================================================================
#  DISCRETE Setup - ALL algorithms × ALL problems
#  Criteria:  1) Speed (execution time)   2) Convergence (curve + T90)
# =====================================================================

class DiscreteWrapper:
    """Wraps a discrete problem so continuous-based optimisers can solve it.

    Maps a real-valued vector  →  discrete solution  →  fitness.
    """
    def __init__(self, prob_type, problem):
        self.problem = problem
        self.prob_type = prob_type
        self.name = getattr(problem, 'name', prob_type)

        if prob_type == 'TSP':
            self.dim = problem.n_cities
            self.bounds = np.array([[0.0, 1.0]] * self.dim)
        elif prob_type == 'Knapsack':
            self.dim = problem.n_items
            self.bounds = np.array([[0.0, 1.0]] * self.dim)
        elif prob_type == 'GraphColoring':
            self.dim = problem.n_nodes
            self.bounds = np.array([[0.0, float(problem.n_colors - 1e-6)]] * self.dim)
        elif prob_type == 'ShortestPath':
            n = problem.grid_size
            path_len = 2 * n
            self.dim = path_len * 2          # (row, col) pairs
            self.bounds = np.array([[0.0, float(n - 1)]] * self.dim)
        else:
            self.dim = 2
            self.bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

    def evaluate(self, sol_vector):
        if self.prob_type == 'TSP':
            route = np.argsort(sol_vector).tolist()
            res = self.problem.evaluate(route)
            return res.get('total_cost', float('inf'))

        elif self.prob_type == 'Knapsack':
            sel = (np.array(sol_vector) > 0.5).astype(int).tolist()
            res = self.problem.evaluate(sel)
            return -res.get('total_value', 0) if res.get('feasible', False) else 1e6

        elif self.prob_type == 'GraphColoring':
            coloring = {i: int(np.clip(np.floor(v), 0, self.problem.n_colors - 1))
                        for i, v in enumerate(sol_vector)}
            res = self.problem.evaluate(coloring)
            return res.get('total_penalty', 0.0) + res.get('n_violations', 0) * 1000.0

        elif self.prob_type == 'ShortestPath':
            prob = self.problem
            n = prob.grid_size
            half = self.dim // 2
            rows = np.clip(np.round(sol_vector[:half]).astype(int), 0, n - 1)
            cols = np.clip(np.round(sol_vector[half:]).astype(int), 0, n - 1)
            path = [(int(rows[k]), int(cols[k])) for k in range(half)]
            # Force start / goal
            path = [prob.start] + path + [prob.goal]
            # Remove consecutive duplicates
            clean = [path[0]]
            for p in path[1:]:
                if p != clean[-1]:
                    clean.append(p)
            res = prob.evaluate(clean)
            cost = res.get('total_cost', float('inf'))
            cost += len(res.get('invalid_moves', [])) * 500.0
            cost -= res.get('waypoint_reward', 0)
            return cost

        return float('inf')


# ── Problem instances ────────────────────────────────────────────────
DISC_PROBLEM_REGISTRY = {
    'TSP-15':           ('TSP',           lambda seed: TSPProblem.generate(15, 'medium', seed=seed)),
    'Knapsack-30':      ('Knapsack',      lambda seed: KnapsackProblem.generate(30, 'medium', seed=seed)),
    'GraphColoring-30': ('GraphColoring', lambda seed: GraphColoringProblem.generate(30, 5, 'clustered', 'medium', seed=seed)),
    'ShortestPath-30':  ('ShortestPath',  lambda seed: ShortestPathProblem.generate(30, 'medium', seed=seed)),
}

# ── Algorithm registry (all 13) ─────────────────────────────────────
DISC_ALGO_REGISTRY = {
    # Population-based metaheuristics (via DiscreteWrapper)
    'PSO':  (ParticleSwarmOptimization, {'pop_size': 30, 'w': 0.7, 'c1': 1.5, 'c2': 1.5}),
    'GA':   (GeneticAlgorithm,          {'pop_size': 30, 'cx_rate': 0.8, 'mut_rate': 0.1}),
    'DE':   (DifferentialEvolution,     {'pop_size': 30, 'F': 0.8, 'CR': 0.9}),
    'ABC':  (ArtificialBeeColony,       {'pop_size': 30}),
    'CS':   (CuckooSearch,              {'n': 30, 'pa': 0.25}),
    'FA':   (FireflyAlgorithm,          {'pop_size': 30, 'alpha': 0.5, 'beta0': 1.0, 'gamma': 1.0}),
    'TLBO': (TLBO,                      {'pop_size': 30}),
    # Single-solution (via DiscreteWrapper)
    'HC':   (HillClimbing,              {'restarts': 5, 'step_scale': 0.1}),
    'SA':   (SimulatedAnnealing,        {'T0': 1.0, 'alpha': 0.99}),
    # Discrete-native
    'ACO':  (AntColonyOptimization,     {'n_ants': 30, 'alpha': 1.0, 'beta': 3.0, 'rho': 0.5}),
    'BFS':  (BreadthFirstSearch,        {}),
    'DFS':  (DepthFirstSearch,          {}),
    'A*':   (AStar,                     {}),
}

# Continuous-based algorithms that need DiscreteWrapper
_WRAPPED_ALGOS = {'PSO', 'GA', 'DE', 'ABC', 'CS', 'FA', 'TLBO', 'HC', 'SA'}

DISC_ITERATIONS = 80
DISC_N_RUNS = 30


def get_disc_applicable_algos(prob_type):
    """Return list of algorithm names applicable to the given problem type."""
    wrapped = ['PSO', 'GA', 'DE', 'ABC', 'CS', 'FA', 'TLBO', 'HC', 'SA']
    if prob_type == 'TSP':
        return wrapped + ['ACO']
    elif prob_type in ('Knapsack', 'GraphColoring'):
        return wrapped
    elif prob_type == 'ShortestPath':
        return ['BFS', 'DFS', 'A*'] + wrapped
    return wrapped


# ── Benchmark runner ─────────────────────────────────────────────────
def run_discrete_benchmark():
    """Run all applicable algorithms on every discrete problem.

    Returns {prob_name: {algo_name: [run_dict, ...]}}
    Each run_dict has: convergence, best_fit, time_s, steps
    """
    data = {pn: {} for pn in DISC_PROBLEM_REGISTRY}

    for pname, (ptype, pgen) in DISC_PROBLEM_REGISTRY.items():
        print(f'\n  {pname}:', end='', flush=True)
        algos = get_disc_applicable_algos(ptype)

        for aname in algos:
            algo_cls, params = DISC_ALGO_REGISTRY[aname]
            runs = []
            for run_idx in range(DISC_N_RUNS):
                # Use a fixed seed so we test the algorithm's robustness 
                # (stochasticity) on the exact same problem instance 30 times
                prob = pgen(42)

                # Choose interface
                if aname in _WRAPPED_ALGOS:
                    prob_input = DiscreteWrapper(ptype, prob)
                else:
                    prob_input = prob

                alg = algo_cls(prob_input, params=dict(params))
                t0 = time.perf_counter()
                try:
                    if aname in ('BFS', 'DFS', 'A*'):
                        alg.solve(iterations=None)
                        elapsed = time.perf_counter() - t0
                        cost = float('inf')
                        if alg.best_solution:
                            res = prob.evaluate(alg.best_solution)
                            cost = res.get('total_cost', float('inf'))
                        # Build a pseudo-convergence: nodes explored → cost
                        n_steps = len(alg.history)
                        conv = [float('inf')] * max(n_steps - 1, 0) + [cost]
                        runs.append({
                            'convergence': conv,
                            'best_fit': cost,
                            'time_s': elapsed,
                            'steps': n_steps,
                        })
                    elif aname == 'ACO':
                        alg.solve(iterations=DISC_ITERATIONS)
                        elapsed = time.perf_counter() - t0
                        conv = [h.get('global_best_fit', float('inf'))
                                for h in alg.history]
                        runs.append({
                            'convergence': conv,
                            'best_fit': alg.best_fitness,
                            'time_s': elapsed,
                            'steps': DISC_ITERATIONS,
                        })
                    else:
                        # Wrapped continuous algo
                        alg.solve(iterations=DISC_ITERATIONS)
                        elapsed = time.perf_counter() - t0
                        conv = [h.get('global_best_fit', float('inf'))
                                for h in alg.history]
                        runs.append({
                            'convergence': conv,
                            'best_fit': alg.best_fitness,
                            'time_s': elapsed,
                            'steps': DISC_ITERATIONS,
                        })
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    print(f' [{aname} FAIL: {e}]', end='')
                    runs.append({
                        'convergence': [],
                        'best_fit': float('inf'),
                        'time_s': elapsed,
                        'steps': 0,
                    })

            data[pname][aname] = runs
            print(f' {aname}✓', end='', flush=True)

    print()
    return data


# =====================================================================
#  Algorithm categories for grouped comparison
# =====================================================================

# Classical (traditional) algorithms
CLASSICAL_ALGOS = {'BFS', 'DFS', 'A*', 'HC', 'SA'}
# Modern metaheuristic algorithms
META_ALGOS = {'PSO', 'GA', 'DE', 'ABC', 'CS', 'FA', 'TLBO', 'ACO'}

ALGO_CATEGORY_COLORS = {
    'classical': '#ff6b6b',
    'meta':      '#58a6ff',
}

def algo_category(name):
    return 'classical' if name in CLASSICAL_ALGOS else 'meta'


# =====================================================================
#  DISCRETE Charts - Criterion 1: SPEED  (execution time)
# =====================================================================

def chart_disc_robustness_boxplots(data):
    print('--- [Disc] Robustness: Box Plots ---')
    prob_names = list(DISC_PROBLEM_REGISTRY.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()
    for i in range(len(prob_names), len(axes)): axes[i].set_visible(False)

    for idx, prob_name in enumerate(prob_names):
        ax = axes[idx]
        style_ax(ax, title=prob_name, ylabel='Final Fitness')

        box_data, labels, cols = [], [], []
        # Specifically use META_ALGOS + maybe exclude classical completely
        applicable = get_disc_applicable_algos(DISC_PROBLEM_REGISTRY[prob_name][0])
        # Only test Swarm/Metaheuristics (ignoring BFS, DFS, A*, HC, SA if we treat SA/HC as classical-like)
        # Actually user said "các thuật toán Swarm" - we'll just check META_ALGOS array defined above.
        for aname in applicable:
            if aname not in META_ALGOS:
                continue
            runs = data[prob_name].get(aname, [])
            fits = [r['best_fit'] for r in runs if r.get('best_fit') is not None and r['best_fit'] != float('inf')]
            if fits:
                box_data.append(fits)
                labels.append(aname)
                cols.append(ALGO_COLORS.get(aname, '#ffffff'))

        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                            medianprops={'color': 'white', 'linewidth': 2},
                            flierprops={'markerfacecolor': '#ff6b6b', 'markersize': 4})
            for patch, c in zip(bp['boxes'], cols):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_xticklabels(labels, color='#c9d1d9', fontsize=9)
            ax.set_yscale('log') if prob_name.startswith('TSP') or prob_name.startswith('Knapsack') else None

    fig.suptitle('Robustness - Final Fitness Distribution (Metaheuristics)', color=TXT, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_DISC / 'robustness_boxplots.png'), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

def chart_disc_speed_bars(data):
    """Bar chart: median execution time per algorithm per problem, grouped Classical vs Meta."""
    print('\n--- [Disc] Speed - Execution Time ---')
    prob_names = list(DISC_PROBLEM_REGISTRY.keys())
    cols = min(3, max(1, len(prob_names)))
    rows = (len(prob_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 7.0 * rows))
    if rows * cols == 1: axes_flat = [axes]
    else: axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for _i in range(len(prob_names), len(axes_flat)): axes_flat[_i].set_visible(False)
    fig.patch.set_facecolor(BG)

    for ax, pname in zip(axes_flat, prob_names):
        ptype = DISC_PROBLEM_REGISTRY[pname][0]
        algos = get_disc_applicable_algos(ptype)
        # Sort: classical first, then meta
        algos_sorted = sorted(algos, key=lambda a: (0 if algo_category(a) == 'classical' else 1, a))
        style_ax(ax, title=pname, ylabel='Median Time (s)')

        vals = []
        for a in algos_sorted:
            times = [r['time_s'] for r in data[pname].get(a, []) if r['time_s'] > 0]
            vals.append(np.median(times) if times else 0.0)

        x_pos = np.arange(len(algos_sorted))
        # Color by category
        cols = [ALGO_COLORS.get(a, '#8b949e') for a in algos_sorted]
        edge_cols = [('#ff6b6b' if algo_category(a) == 'classical' else '#58a6ff') for a in algos_sorted]
        bars = ax.bar(x_pos, vals, color=cols, alpha=0.85,
                      edgecolor=edge_cols, linewidth=2.0, width=0.65)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algos_sorted, color='#c9d1d9', fontsize=8, rotation=45, ha='right')

        for bar, val in zip(bars, vals):
            label = f'{val:.3f}' if val < 1 else f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha='center', va='bottom', color='#c9d1d9',
                    fontsize=7, fontweight='bold')

        # Draw separator line between classical and meta groups
        n_classical = sum(1 for a in algos_sorted if algo_category(a) == 'classical')
        if 0 < n_classical < len(algos_sorted):
            ax.axvline(x=n_classical - 0.5, color='#484f58', linewidth=1.5, linestyle='--', alpha=0.8)
            ax.text(n_classical - 0.5, ax.get_ylim()[1] * 0.98,
                    ' Classical | Meta ', color='#8b949e', fontsize=7,
                    ha='center', va='top', style='italic')

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='#8b949e', edgecolor='#ff6b6b', linewidth=2, label='Classical'),
                    Patch(facecolor='#8b949e', edgecolor='#58a6ff', linewidth=2, label='Metaheuristic')]
    fig.legend(handles=legend_elems, loc='upper right', facecolor='#21262d',
               edgecolor=BORDER, labelcolor='#c9d1d9', fontsize=9, framealpha=0.9)

    fig.suptitle('Speed - Median Execution Time: Classical vs Metaheuristic (lower = faster)',
                 color=TXT, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_DISC / 'disc_speed_bars.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def chart_disc_speed_heatmap(data):
    """Heatmap: execution time across all problems - rows sorted Classical first, then Meta."""
    print('--- [Disc] Speed - Heatmap ---')
    prob_names = list(DISC_PROBLEM_REGISTRY.keys())
    # Sort rows: classical first, then meta
    all_algo_names = sorted(DISC_ALGO_REGISTRY.keys(),
                            key=lambda a: (0 if algo_category(a) == 'classical' else 1, a))

    time_matrix = np.full((len(all_algo_names), len(prob_names)), np.nan)
    for j, pname in enumerate(prob_names):
        ptype = DISC_PROBLEM_REGISTRY[pname][0]
        applicable = get_disc_applicable_algos(ptype)
        for i, aname in enumerate(all_algo_names):
            if aname in applicable:
                times = [r['time_s'] for r in data[pname].get(aname, [])
                         if r['time_s'] > 0]
                time_matrix[i, j] = np.median(times) if times else np.nan

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color=PANEL)
    masked = np.ma.masked_invalid(time_matrix)

    im = ax.imshow(masked, cmap=cmap, aspect='auto')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Time (s)', color='#8b949e')
    cb.ax.tick_params(colors='#8b949e')

    ax.set_xticks(range(len(prob_names)))
    ax.set_xticklabels(prob_names, color='#c9d1d9', fontsize=10, rotation=20, ha='right')
    ax.set_yticks(range(len(all_algo_names)))

    # Color y-tick labels by category
    ytick_labels = []
    for aname in all_algo_names:
        cat_color = '#ff6b6b' if algo_category(aname) == 'classical' else '#58a6ff'
        ytick_labels.append(aname)
    ax.set_yticklabels(ytick_labels, fontsize=10)
    for tick, aname in zip(ax.get_yticklabels(), all_algo_names):
        tick.set_color('#ff6b6b' if algo_category(aname) == 'classical' else '#58a6ff')

    # Draw separator between classical and meta rows
    n_classical = sum(1 for a in all_algo_names if algo_category(a) == 'classical')
    if 0 < n_classical < len(all_algo_names):
        ax.axhline(y=n_classical - 0.5, color='#8b949e', linewidth=1.5, linestyle='--', alpha=0.7)

    for i in range(len(all_algo_names)):
        for j in range(len(prob_names)):
            v = time_matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, '-', ha='center', va='center', color='#484f58', fontsize=9)
            else:
                label = f'{v:.3f}' if v < 1 else f'{v:.1f}'
                ax.text(j, i, label, ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='#161b22', edgecolor='#ff6b6b', linewidth=2, label='Classical'),
                    Patch(facecolor='#161b22', edgecolor='#58a6ff', linewidth=2, label='Metaheuristic')]
    ax.legend(handles=legend_elems, loc='upper right', facecolor='#21262d',
              edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)

    plt.title('Execution Time Heatmap (s) - Classical (red) vs Meta (blue) - lower = faster',
              color=TXT, fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_DISC / 'disc_speed_heatmap.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


# =====================================================================
#  DISCRETE Charts - Criterion 2: CONVERGENCE
# =====================================================================

def chart_disc_convergence_curves(data):
    """Convergence curves per problem - Classical (solid thick) vs Meta (dashed thin)."""
    print('--- [Disc] Convergence - Curves ---')
    prob_names = list(DISC_PROBLEM_REGISTRY.keys())
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()
    for i in range(len(prob_names), len(axes)): axes[i].set_visible(False)

    for idx, pname in enumerate(prob_names):
        ax = axes[idx]
        ptype = DISC_PROBLEM_REGISTRY[pname][0]
        algos = get_disc_applicable_algos(ptype)
        # Sort: classical first
        algos_sorted = sorted(algos, key=lambda a: (0 if algo_category(a) == 'classical' else 1, a))

        ylabel = 'Best Cost (lower = better)'
        if ptype == 'Knapsack':
            ylabel = 'Negative Value (lower = better)'

        style_ax(ax, title=pname, xlabel='Iteration', ylabel=ylabel)

        for aname in algos_sorted:
            runs = data[pname].get(aname, [])
            valid_c = [r['convergence'] for r in runs if r['convergence'] and len(r['convergence']) > 1]
            if not valid_c:
                continue

            ml = min(len(c) for c in valid_c)
            if ml < 2:
                continue
            arr = np.array([c[:ml] for c in valid_c])
            med = np.median(arr, axis=0)
            lo, hi = np.min(arr, axis=0), np.max(arr, axis=0)
            x = np.arange(1, ml + 1)

            c = ALGO_COLORS.get(aname, '#8b949e')
            is_classical = algo_category(aname) == 'classical'
            lw = 2.5 if is_classical else 1.5
            ls = '-' if is_classical else '--'
            label = f'[C] {aname}' if is_classical else f'[M] {aname}'
            ax.plot(x, med, label=label, color=c, linewidth=lw, linestyle=ls,
                    marker=ALGO_MARKERS.get(aname, '.'),
                    markevery=max(1, ml // 8), markersize=5 if is_classical else 4)
            ax.fill_between(x, lo, hi, alpha=0.05 if not is_classical else 0.10, color=c)

        ax.legend(facecolor='#21262d', edgecolor=BORDER, labelcolor='#c9d1d9',
                  fontsize=7, ncol=2, loc='upper right',
                  title='[C]=Classical [M]=Meta', title_fontsize=6)

    fig.suptitle('Convergence - Classical (solid) vs Metaheuristic (dashed) × All Discrete Problems',
                 color=TXT, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_DISC / 'disc_convergence_curves.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def chart_disc_t90(data):
    """T90 convergence speed bar chart per problem - Classical vs Meta grouped."""
    print('--- [Disc] Convergence - T90 ---')
    prob_names = list(DISC_PROBLEM_REGISTRY.keys())
    cols = min(3, max(1, len(prob_names)))
    rows = (len(prob_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 7.0 * rows))
    if rows * cols == 1: axes_flat = [axes]
    else: axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for _i in range(len(prob_names), len(axes_flat)): axes_flat[_i].set_visible(False)
    fig.patch.set_facecolor(BG)

    for ax, pname in zip(axes_flat, prob_names):
        ptype = DISC_PROBLEM_REGISTRY[pname][0]
        algos = get_disc_applicable_algos(ptype)
        # Sort: classical first
        algos_sorted = sorted(algos, key=lambda a: (0 if algo_category(a) == 'classical' else 1, a))
        style_ax(ax, title=pname, ylabel='T90 (iteration - lower = faster)')

        vals, valid_algos = [], []
        for a in algos_sorted:
            runs = data[pname].get(a, [])
            t90_list = [compute_t90(r['convergence'])
                        for r in runs if r['convergence'] and len(r['convergence']) > 1]
            if t90_list:
                vals.append(np.median(t90_list))
                valid_algos.append(a)

        if not valid_algos:
            ax.text(0.5, 0.5, 'No convergence data', transform=ax.transAxes,
                    ha='center', va='center', color='#8b949e', fontsize=10)
            continue

        x_pos = np.arange(len(valid_algos))
        cols = [ALGO_COLORS.get(a, '#8b949e') for a in valid_algos]
        edge_cols = [('#ff6b6b' if algo_category(a) == 'classical' else '#58a6ff') for a in valid_algos]
        bars = ax.bar(x_pos, vals, color=cols, alpha=0.85,
                      edgecolor=edge_cols, linewidth=2.0, width=0.65)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_algos, color='#c9d1d9', fontsize=8, rotation=45, ha='right')

        # Separator between classical and meta
        n_classical = sum(1 for a in valid_algos if algo_category(a) == 'classical')
        if 0 < n_classical < len(valid_algos):
            ax.axvline(x=n_classical - 0.5, color='#484f58', linewidth=1.5, linestyle='--', alpha=0.8)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', color='#c9d1d9',
                    fontsize=8, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='#8b949e', edgecolor='#ff6b6b', linewidth=2, label='Classical'),
                    Patch(facecolor='#8b949e', edgecolor='#58a6ff', linewidth=2, label='Metaheuristic')]
    fig.legend(handles=legend_elems, loc='upper right', facecolor='#21262d',
               edgecolor=BORDER, labelcolor='#c9d1d9', fontsize=9)

    fig.suptitle('Convergence Speed - T90: Classical vs Metaheuristic (lower = faster)',
                 color=TXT, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_DISC / 'disc_convergence_t90.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def chart_disc_convergence_heatmap(data):
    """Heatmap: T90 convergence speed across all problems × algorithms - Classical rows first."""
    print('--- [Disc] Convergence - Heatmap ---')
    prob_names = list(DISC_PROBLEM_REGISTRY.keys())
    all_algo_names = sorted(DISC_ALGO_REGISTRY.keys(),
                            key=lambda a: (0 if algo_category(a) == 'classical' else 1, a))

    t90_matrix = np.full((len(all_algo_names), len(prob_names)), np.nan)
    for j, pname in enumerate(prob_names):
        ptype = DISC_PROBLEM_REGISTRY[pname][0]
        applicable = get_disc_applicable_algos(ptype)
        for i, aname in enumerate(all_algo_names):
            if aname in applicable:
                runs = data[pname].get(aname, [])
                t90_list = [compute_t90(r['convergence'])
                            for r in runs if r['convergence'] and len(r['convergence']) > 1]
                if t90_list:
                    t90_matrix[i, j] = np.median(t90_list)

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color=PANEL)
    masked = np.ma.masked_invalid(t90_matrix)

    im = ax.imshow(masked, cmap=cmap, aspect='auto')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('T90 (iteration)', color='#8b949e')
    cb.ax.tick_params(colors='#8b949e')

    ax.set_xticks(range(len(prob_names)))
    ax.set_xticklabels(prob_names, color='#c9d1d9', fontsize=10, rotation=20, ha='right')
    ax.set_yticks(range(len(all_algo_names)))
    ax.set_yticklabels(all_algo_names, fontsize=10)
    for tick, aname in zip(ax.get_yticklabels(), all_algo_names):
        tick.set_color('#ff6b6b' if algo_category(aname) == 'classical' else '#58a6ff')

    # Draw separator between classical and meta rows
    n_classical = sum(1 for a in all_algo_names if algo_category(a) == 'classical')
    if 0 < n_classical < len(all_algo_names):
        ax.axhline(y=n_classical - 0.5, color='#8b949e', linewidth=1.5, linestyle='--', alpha=0.7)

    for i in range(len(all_algo_names)):
        for j in range(len(prob_names)):
            v = t90_matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, '-', ha='center', va='center', color='#484f58', fontsize=9)
            else:
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='#161b22', edgecolor='#ff6b6b', linewidth=2, label='Classical'),
                    Patch(facecolor='#161b22', edgecolor='#58a6ff', linewidth=2, label='Metaheuristic')]
    ax.legend(handles=legend_elems, loc='upper right', facecolor='#21262d',
              edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)

    plt.title('T90 Convergence Heatmap - Classical (red) vs Meta (blue) - lower = faster',
              color=TXT, fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_DISC / 'disc_convergence_heatmap.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def chart_disc_solution_quality(data):
    """Bar chart: median best fitness (solution quality) per algo per problem."""
    print('--- [Disc] Solution Quality - Best Fitness ---')
    prob_names = list(DISC_PROBLEM_REGISTRY.keys())
    cols = min(3, max(1, len(prob_names)))
    rows = (len(prob_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 7.0 * rows))
    if rows * cols == 1: axes_flat = [axes]
    else: axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for _i in range(len(prob_names), len(axes_flat)): axes_flat[_i].set_visible(False)
    fig.patch.set_facecolor(BG)

    for ax, pname in zip(axes_flat, prob_names):
        ptype = DISC_PROBLEM_REGISTRY[pname][0]
        algos = get_disc_applicable_algos(ptype)
        algos_sorted = sorted(algos, key=lambda a: (0 if algo_category(a) == 'classical' else 1, a))

        ylabel = 'Median Best Fitness (lower = better)'
        if ptype == 'Knapsack':
            ylabel = 'Median Best Fitness (higher = better value)'
        style_ax(ax, title=pname, ylabel=ylabel)

        vals, valid_algos = [], []
        for a in algos_sorted:
            fits = [r['best_fit'] for r in data[pname].get(a, [])
                    if r.get('best_fit') is not None and r['best_fit'] != float('inf')]
            if fits:
                vals.append(np.median(fits))
                valid_algos.append(a)

        if not valid_algos:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='#8b949e', fontsize=10)
            continue

        x_pos = np.arange(len(valid_algos))
        cols = [ALGO_COLORS.get(a, '#8b949e') for a in valid_algos]
        edge_cols = [('#ff6b6b' if algo_category(a) == 'classical' else '#58a6ff') for a in valid_algos]
        bars = ax.bar(x_pos, vals, color=cols, alpha=0.85,
                      edgecolor=edge_cols, linewidth=2.0, width=0.65)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_algos, color='#c9d1d9', fontsize=8, rotation=45, ha='right')

        # Separator between classical and meta
        n_classical = sum(1 for a in valid_algos if algo_category(a) == 'classical')
        if 0 < n_classical < len(valid_algos):
            ax.axvline(x=n_classical - 0.5, color='#484f58', linewidth=1.5, linestyle='--', alpha=0.8)

        for bar, val in zip(bars, vals):
            label = f'{val:.1e}' if abs(val) >= 1e4 or (abs(val) < 0.01 and val != 0) else f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha='center', va='bottom', color='#c9d1d9',
                    fontsize=7, fontweight='bold', rotation=45)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='#8b949e', edgecolor='#ff6b6b', linewidth=2, label='Classical'),
                    Patch(facecolor='#8b949e', edgecolor='#58a6ff', linewidth=2, label='Metaheuristic')]
    fig.legend(handles=legend_elems, loc='upper right', facecolor='#21262d',
               edgecolor=BORDER, labelcolor='#c9d1d9', fontsize=9)

    fig.suptitle('Solution Quality - Classical vs Metaheuristic (Median Best Fitness)',
                 color=TXT, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_DISC / 'disc_solution_quality.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def chart_disc_classical_vs_meta_summary(data):
    """Summary radar/bar chart: for each problem, compare best-of-classical vs best-of-meta
    on 3 criteria: Speed, Convergence (T90), Solution Quality."""
    print('--- [Disc] Classical vs Meta Summary ---')
    prob_names = list(DISC_PROBLEM_REGISTRY.keys())

    summary = {}
    for pname in prob_names:
        ptype = DISC_PROBLEM_REGISTRY[pname][0]
        algos = get_disc_applicable_algos(ptype)
        classical_algos = [a for a in algos if algo_category(a) == 'classical']
        meta_algos = [a for a in algos if algo_category(a) == 'meta']

        def best_speed(algo_list):
            times = []
            for a in algo_list:
                t = [r['time_s'] for r in data[pname].get(a, []) if r['time_s'] > 0]
                if t:
                    times.append(np.median(t))
            return min(times) if times else np.nan

        def best_t90(algo_list):
            t90s = []
            for a in algo_list:
                t_list = [compute_t90(r['convergence'])
                          for r in data[pname].get(a, [])
                          if r['convergence'] and len(r['convergence']) > 1]
                if t_list:
                    t90s.append(np.median(t_list))
            return min(t90s) if t90s else np.nan

        def best_fitness(algo_list):
            fits = []
            for a in algo_list:
                f = [r['best_fit'] for r in data[pname].get(a, [])
                     if r.get('best_fit') is not None and r['best_fit'] != float('inf')]
                if f:
                    fits.append(np.median(f))
            return min(fits) if fits else np.nan

        summary[pname] = {
            'classical': {
                'speed':   best_speed(classical_algos),
                't90':     best_t90(classical_algos),
                'fitness': best_fitness(classical_algos),
            },
            'meta': {
                'speed':   best_speed(meta_algos),
                't90':     best_t90(meta_algos),
                'fitness': best_fitness(meta_algos),
            },
        }

    # Draw grouped bar chart with 3 criteria per problem
    criteria = ['Speed (s)', 'T90 (iter)', 'Best Fitness']
    n_criteria = len(criteria)
    n_problems = len(prob_names)

    fig, axes = plt.subplots(1, n_criteria, figsize=(6 * n_criteria, 7))
    fig.patch.set_facecolor(BG)

    for ci, (crit_label, crit_key) in enumerate(zip(criteria, ['speed', 't90', 'fitness'])):
        ax = axes[ci]
        style_ax(ax, title=crit_label,
                 ylabel=crit_label + ' (lower = better' + (')' if crit_key != 'fitness' else ', except Knapsack)'))

        x_pos = np.arange(n_problems)
        width = 0.35

        classical_vals = [summary[p]['classical'][crit_key] for p in prob_names]
        meta_vals = [summary[p]['meta'][crit_key] for p in prob_names]

        bars_c = ax.bar(x_pos - width / 2, classical_vals, width, label='Best Classical',
                        color='#ff6b6b', alpha=0.85, edgecolor='white', linewidth=0.8)
        bars_m = ax.bar(x_pos + width / 2, meta_vals, width, label='Best Meta',
                        color='#58a6ff', alpha=0.85, edgecolor='white', linewidth=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.split('-')[0] for p in prob_names],
                           color='#c9d1d9', fontsize=9, rotation=30, ha='right')

        for bar, val in zip(list(bars_c) + list(bars_m),
                            classical_vals + meta_vals):
            if np.isnan(val):
                continue
            label = f'{val:.2f}' if val < 100 else f'{val:.0f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha='center', va='bottom', color='#c9d1d9',
                    fontsize=7, fontweight='bold')

        ax.legend(facecolor='#21262d', edgecolor=BORDER, labelcolor='#c9d1d9', fontsize=8)

    fig.suptitle('Classical vs Metaheuristic - Best Result per Criterion per Problem',
                 color=TXT, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR_DISC / 'disc_classical_vs_meta_summary.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


# =====================================================================
#  Main
# =====================================================================

def main():
    print('=' * 62)
    print('  SNIA - Algorithm Performance Comparison (Cont & Disc)')
    print('=' * 62)

    t0 = time.time()

    # --- Continuous (unchanged) ---
    print('\n' + '═' * 62 + '\n  [PART 1] CONTINUOUS BENCHMARKS \n' + '═' * 62)
    data_cont = run_continuous_benchmark()
    chart_cont_convergence_curves(data_cont)
    chart_cont_t90(data_cont)
    chart_cont_robustness_boxplots(data_cont)
    chart_cont_robustness_cv(data_cont)
    chart_cont_performance_bars(data_cont)
    chart_cont_execution_time(data_cont)

    # --- Discrete (rewritten) ---
    print('\n' + '═' * 62 + '\n  [PART 2] DISCRETE BENCHMARKS \n' + '═' * 62)
    data_disc = run_discrete_benchmark()

    # Robustness Discrete
    chart_disc_robustness_boxplots(data_disc)

    # Criterion 1: Speed
    chart_disc_speed_bars(data_disc)
    chart_disc_speed_heatmap(data_disc)

    # Criterion 2: Convergence
    chart_disc_convergence_curves(data_disc)
    chart_disc_t90(data_disc)
    chart_disc_convergence_heatmap(data_disc)

    # Criterion 3: Solution Quality
    chart_disc_solution_quality(data_disc)

    # Summary: Classical vs Metaheuristic
    chart_disc_classical_vs_meta_summary(data_disc)

    print(f'\n{"=" * 62}')
    print(f'  Done in {time.time() - t0:.0f}s!')
    print(f'  CONT → {OUT_DIR_CONT}')
    print(f'  DISC → {OUT_DIR_DISC}')
    print(f'{"=" * 62}')


if __name__ == '__main__':
    main()
