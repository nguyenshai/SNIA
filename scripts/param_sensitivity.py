"""
Parameter Sensitivity Analysis
================================
Thử nghiệm ảnh hưởng của tham số lên hiệu suất thuật toán:

  TSP          + ACO : ρ (rho / pheromone evaporation) ∈ [0.1, 0.5, 0.9]
  Knapsack     + CS  : pa (nest abandonment rate)       ∈ [0.1, 0.25, 0.8]
  GraphColoring+ GA  : mut_rate (mutation probability)  ∈ [0.01, 0.1, 0.5]
  ShortestPath + SA  : alpha (cooling rate)             ∈ [0.8, 0.95, 0.99]

Output: results/param_sensitivity/
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Ensure repo root is importable ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUT_DIR = os.path.join(BASE_DIR, 'results', 'param_sensitivity')
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
ITERATIONS = 100
N_RUNS = 3   # independent runs per config for statistical robustness

# ══════════════════════════════════════════════════════════════════════
#  Palette / Style
# ══════════════════════════════════════════════════════════════════════
COLORS = ['#58a6ff', '#ffd93d', '#ff6b6b', '#6bcb77', '#a371f7', '#f0883e', '#00d4aa']
BG_DARK = '#0d1117'
AX_DARK = '#161b22'
TEXT_CLR = '#e6edf3'
GRID_CLR = '#30363d'

def _style_ax(ax):
    ax.set_facecolor(AX_DARK)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.title.set_color(TEXT_CLR)
    for sp in ax.spines.values():
        sp.set_color(GRID_CLR)
    ax.grid(color=GRID_CLR, linestyle='--', linewidth=0.5, alpha=0.6)


def _savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
#  Adapters (discrete → continuous-interface)
# ══════════════════════════════════════════════════════════════════════

class KnapsackAdapter:
    """Wraps KnapsackProblem for continuous-coded metaheuristics (CS, GA …)."""
    def __init__(self, prob):
        self._prob = prob
        self.dim    = prob.n_items
        self.bounds = [0.0, 1.0]
        self.opt_type = 'min'   # we minimise –value

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        selection = (x > 0.5).astype(int).tolist()
        result = self._prob.evaluate(selection)
        return -result['total_value']          # negate → minimise


class GraphColoringAdapter:
    """Wraps GraphColoringProblem for continuous-coded metaheuristics."""
    def __init__(self, prob):
        self._prob = prob
        self.dim    = prob.n_nodes
        self.bounds = [0.0, float(prob.n_colors)]
        self.opt_type = 'min'

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        coloring = {i: min(int(xi), self._prob.n_colors - 1)
                    for i, xi in enumerate(x)}
        result = self._prob.evaluate(coloring)
        return result['total_penalty']


class ShortestPathSAAdapter:
    """
    Wraps ShortestPathProblem for Simulated Annealing.

    SA operates in continuous space (dim = grid_size²).
    Each dimension encodes the preference for a cell.
    A greedy decoding builds the path from strongest preference neighbours.
    """
    def __init__(self, prob):
        self._prob = prob
        n = prob.grid_size
        self.dim    = n * n
        self.bounds = [0.0, 1.0]
        self.opt_type = 'min'

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        prob = self._prob
        n    = prob.grid_size

        # Reshape to grid
        weight_grid = x.reshape(n, n)

        # Greedy path: always pick the admissible neighbour with highest weight
        path  = [prob.start]
        visited = {prob.start}
        r, c = prob.start
        MAX_STEPS = n * n * 2

        for _ in range(MAX_STEPS):
            if (r, c) == prob.goal:
                break
            nbrs = prob.neighbors(r, c)
            if not nbrs:
                break
            # Prefer unvisited; among those pick highest weight
            unvisited = [nb for nb in nbrs if nb not in visited]
            if unvisited:
                nr, nc = max(unvisited, key=lambda nb: weight_grid[nb])
            else:
                # allow revisit to avoid stuck
                nr, nc = max(nbrs, key=lambda nb: weight_grid[nb])

            path.append((nr, nc))
            visited.add((nr, nc))
            r, c = nr, nc

        result = prob.evaluate(path)
        return result['total_cost']


# ══════════════════════════════════════════════════════════════════════
#  Experiment runner
# ══════════════════════════════════════════════════════════════════════

def run_aco_tsp(prob, rho, n_runs=N_RUNS):
    """Run ACO on TSP with given rho, return convergence curves (list of arrays)."""
    from algorithms.biology.ACO import AntColonyOptimization
    curves = []
    for run in range(n_runs):
        params = {'n_ants': 30, 'alpha': 1.0, 'beta': 5.0, 'rho': rho, 'q': 1.0}
        alg = AntColonyOptimization(prob, params=params)
        alg.solve(iterations=ITERATIONS)
        best_per_iter = [h['global_best_fit'] for h in alg.history]
        curves.append(best_per_iter)
    return curves


def run_cs_knapsack(adapter, pa, n_runs=N_RUNS):
    """Run Cuckoo Search on Knapsack adapter with given pa."""
    from algorithms.biology.CS import CuckooSearch
    curves = []
    for run in range(n_runs):
        params = {'n': 40, 'pa': pa}
        alg = CuckooSearch(adapter, params=params)
        alg.solve(iterations=ITERATIONS)
        best_per_iter = [h['global_best_fit'] for h in alg.history]
        curves.append(best_per_iter)
    # CS minimises –value, convert back
    return [[-v for v in c] for c in curves]


def run_ga_graphcoloring(adapter, mut_rate, n_runs=N_RUNS):
    """Run GA on GraphColoring adapter with given mut_rate."""
    from algorithms.evolution.GA import GeneticAlgorithm
    curves = []
    for run in range(n_runs):
        params = {'pop_size': 50, 'cx_rate': 0.7, 'mut_rate': mut_rate}
        alg = GeneticAlgorithm(adapter, params=params)
        alg.solve(iterations=ITERATIONS)
        best_per_iter = [h['global_best_fit'] for h in alg.history]
        curves.append(best_per_iter)
    return curves


def run_sa_shortestpath(adapter, alpha, n_runs=N_RUNS):
    """Run SA on ShortestPath adapter with given cooling rate alpha."""
    from algorithms.physics.SA import SimulatedAnnealing
    curves = []
    for run in range(n_runs):
        params = {'T0': 1.0, 'alpha': alpha}
        alg = SimulatedAnnealing(adapter, params=params)
        alg.solve(iterations=ITERATIONS)
        best_per_iter = [h['global_best_fit'] for h in alg.history]
        curves.append(best_per_iter)
    return curves


# ══════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ══════════════════════════════════════════════════════════════════════

def plot_sensitivity(experiments, title, ylabel, param_name, param_values,
                     filename, higher_is_better=False):
    """
    experiments : list of lists-of-arrays  # [param_idx][run_idx][iter]
    """
    n_params = len(param_values)
    iters    = np.arange(1, ITERATIONS + 1)

    fig = plt.figure(figsize=(18, 12), facecolor=BG_DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top row: individual convergence curves per param value ──────
    ax_conv = [fig.add_subplot(gs[0, i]) for i in range(n_params)]

    for pi, (pv, curves) in enumerate(zip(param_values, experiments)):
        ax = ax_conv[pi]
        _style_ax(ax)
        arr = np.array(curves)   # (n_runs, ITERATIONS)

        mean_ = arr.mean(axis=0)
        std_  = arr.std(axis=0)
        best_ = arr.min(axis=0) if not higher_is_better else arr.max(axis=0)

        ax.fill_between(iters, mean_ - std_, mean_ + std_,
                         alpha=0.2, color=COLORS[pi], zorder=1)
        for run_i, curve in enumerate(curves):
            ax.plot(iters, curve, color=COLORS[pi], alpha=0.35, linewidth=0.9, zorder=2)
        ax.plot(iters, mean_, color=COLORS[pi], linewidth=2.0, zorder=3,
                label=f'Mean ± Std')

        ax.set_title(f'{param_name} = {pv}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, facecolor=AX_DARK, labelcolor=TEXT_CLR)

    # ── Bottom-left: overlay all param values ───────────────────────
    ax_overlay = fig.add_subplot(gs[1, 0:2])
    _style_ax(ax_overlay)
    legend_handles = []
    for pi, (pv, curves) in enumerate(zip(param_values, experiments)):
        arr   = np.array(curves)
        mean_ = arr.mean(axis=0)
        std_  = arr.std(axis=0)
        ax_overlay.fill_between(iters, mean_ - std_, mean_ + std_,
                                  alpha=0.12, color=COLORS[pi])
        line, = ax_overlay.plot(iters, mean_, color=COLORS[pi], linewidth=2.0,
                                 label=f'{param_name} = {pv}')
        legend_handles.append(line)

    ax_overlay.set_title(f'Convergence Comparison — {param_name}', fontsize=12, fontweight='bold')
    ax_overlay.set_xlabel('Iteration')
    ax_overlay.set_ylabel(ylabel)
    ax_overlay.legend(handles=legend_handles, fontsize=9,
                      facecolor=AX_DARK, labelcolor=TEXT_CLR)

    # ── Bottom-right: final result box-whisker + table ───────────────
    ax_box = fig.add_subplot(gs[1, 2])
    _style_ax(ax_box)

    finals = [np.array(curves)[:, -1] for curves in experiments]
    bp = ax_box.boxplot(finals,
                        patch_artist=True,
                        medianprops={'color': '#ffffff', 'linewidth': 2},
                        whiskerprops={'color': '#8b949e'},
                        capprops={'color': '#8b949e'},
                        flierprops={'markerfacecolor': '#8b949e',
                                    'marker': 'o', 'markersize': 4})
    for patch, c in zip(bp['boxes'], COLORS[:n_params]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax_box.set_xticks(range(1, n_params + 1))
    ax_box.set_xticklabels([str(v) for v in param_values], fontsize=9)
    ax_box.set_xlabel(param_name)
    ax_box.set_ylabel(f'Final {ylabel}')
    ax_box.set_title('Final Value Distribution', fontsize=11, fontweight='bold')

    # ── Super-title ──────────────────────────────────────────────────
    fig.suptitle(title, fontsize=16, fontweight='bold', color=TEXT_CLR, y=1.01)

    return _savefig(fig, filename)


def plot_summary_table(results_summary, filename='summary_table.png'):
    """
    results_summary : list of dicts with keys:
        problem, algorithm, param_name, param_values,
        means_final, stds_final, best_finals
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor=BG_DARK)
    fig.suptitle('Parameter Sensitivity — Summary Dashboard', fontsize=18,
                 fontweight='bold', color=TEXT_CLR, y=1.01)

    plot_configs = [
        ('TSP + ACO',           'ρ (Evaporation)',     results_summary[0], False),
        ('Knapsack + CS',       'pa (Abandonment)',    results_summary[1], True),
        ('Graph Coloring + GA', 'mut_rate (Mutation)', results_summary[2], False),
        ('Shortest Path + SA',  'α (Cooling rate)',    results_summary[3], False),
    ]

    for idx, (prob_name, param_label, res, higher) in enumerate(plot_configs):
        ax = axes[idx // 2][idx % 2]
        _style_ax(ax)

        pvs   = res['param_values']
        means = res['means_final']
        stds  = res['stds_final']
        bests = res['best_finals']
        x     = np.arange(len(pvs))

        bars = ax.bar(x, means, color=COLORS[:len(pvs)], alpha=0.75,
                      edgecolor='white', linewidth=0.6, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt='none',
                    color='#ffffff', capsize=5, linewidth=1.5, zorder=4)

        # Mark best final value
        for xi, (mean, best, std) in enumerate(zip(means, bests, stds)):
            marker = '▲' if higher else '▼'
            ax.annotate(f'{marker} {best:.2g}\nμ={mean:.2g}',
                        xy=(xi, mean + std + (max(means) - min(means)) * 0.02),
                        ha='center', va='bottom', fontsize=8, color=TEXT_CLR)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{param_label}={v}' for v in pvs], fontsize=8)
        ax.set_ylabel('Final Best Value', fontsize=9)
        ax.set_title(prob_name, fontsize=13, fontweight='bold')

    plt.tight_layout()
    return _savefig(fig, filename)


# ══════════════════════════════════════════════════════════════════════
#  Analysis: effect commentary
# ══════════════════════════════════════════════════════════════════════

def print_analysis(label, param_name, param_values, experiments, higher_better=False):
    print(f'\n{"─"*60}')
    print(f'  {label}  |  {param_name}')
    print(f'{"─"*60}')
    finals = []
    for pv, curves in zip(param_values, experiments):
        arr = np.array(curves)[:, -1]
        mu  = arr.mean()
        sd  = arr.std()
        be  = arr.min() if not higher_better else arr.max()
        finals.append(mu)
        dir_sym = '↑' if higher_better else '↓'
        print(f'  {param_name}={pv:5}  mean={mu:10.3f}  std={sd:8.3f}  best={be:10.3f}  [{dir_sym}]')

    best_idx = np.argmax(finals) if higher_better else np.argmin(finals)
    print(f'\n  → Best setting: {param_name} = {param_values[best_idx]}'
          f'  (mean final = {finals[best_idx]:.3f})')


# ══════════════════════════════════════════════════════════════════════
#  Main driver
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print('=' * 60)
    print('  Parameter Sensitivity Analysis')
    print('=' * 60)

    results_summary = []   # collected at end for dashboard

    # ──────────────────────────────────────────────────────────────
    # 1.  TSP  +  ACO  :  ρ  ∈ {0.1, 0.5, 0.9}
    # ──────────────────────────────────────────────────────────────
    print('\n[1/4] TSP + ACO — varying ρ (pheromone evaporation rate)')
    from problems.discrete.TSP import TSPProblem
    tsp_prob = TSPProblem.easy(seed=SEED)

    rho_values = [0.1, 0.5, 0.9]
    aco_experiments = []
    for rho in rho_values:
        print(f'       ρ = {rho} …', end=' ', flush=True)
        t1 = time.time()
        curves = run_aco_tsp(tsp_prob, rho)
        aco_experiments.append(curves)
        print(f'done ({time.time()-t1:.1f}s)')

    plot_sensitivity(
        aco_experiments,
        title='TSP (Storm Chaser) + ACO — Pheromone Evaporation Rate ρ',
        ylabel='Best Tour Cost',
        param_name='ρ',
        param_values=rho_values,
        filename='tsp_aco_rho.png',
        higher_is_better=False,
    )
    print_analysis('TSP + ACO', 'ρ', rho_values, aco_experiments, higher_better=False)

    finals_aco = [np.array(c)[:, -1] for c in aco_experiments]
    results_summary.append({
        'param_values': rho_values,
        'means_final':  [f.mean() for f in finals_aco],
        'stds_final':   [f.std()  for f in finals_aco],
        'best_finals':  [f.min()  for f in finals_aco],
    })

    # ──────────────────────────────────────────────────────────────
    # 2.  Knapsack  +  CS  :  pa  ∈ {0.1, 0.25, 0.8}
    # ──────────────────────────────────────────────────────────────
    print('\n[2/4] Knapsack + CS — varying pa (nest abandonment rate)')
    from problems.discrete.Knapsack import KnapsackProblem
    ks_prob    = KnapsackProblem.easy(seed=SEED)
    ks_adapter = KnapsackAdapter(ks_prob)

    pa_values = [0.1, 0.25, 0.8]
    cs_experiments = []
    for pa in pa_values:
        print(f'       pa = {pa} …', end=' ', flush=True)
        t1 = time.time()
        curves = run_cs_knapsack(ks_adapter, pa)
        cs_experiments.append(curves)
        print(f'done ({time.time()-t1:.1f}s)')

    plot_sensitivity(
        cs_experiments,
        title='Knapsack (Space Cargo) + Cuckoo Search — Abandonment Rate pa',
        ylabel='Total Value',
        param_name='pa',
        param_values=pa_values,
        filename='knapsack_cs_pa.png',
        higher_is_better=True,     # maximise value
    )
    print_analysis('Knapsack + CS', 'pa', pa_values, cs_experiments, higher_better=True)

    finals_cs = [np.array(c)[:, -1] for c in cs_experiments]
    results_summary.append({
        'param_values': pa_values,
        'means_final':  [f.mean() for f in finals_cs],
        'stds_final':   [f.std()  for f in finals_cs],
        'best_finals':  [f.max()  for f in finals_cs],
    })

    # ──────────────────────────────────────────────────────────────
    # 3.  GraphColoring  +  GA  :  mut_rate  ∈ {0.01, 0.1, 0.5}
    # ──────────────────────────────────────────────────────────────
    print('\n[3/4] GraphColoring + GA — varying mut_rate (mutation probability)')
    from problems.discrete.GraphColoring import GraphColoringProblem
    gc_prob    = GraphColoringProblem.easy(seed=SEED)
    gc_adapter = GraphColoringAdapter(gc_prob)

    mut_values = [0.01, 0.1, 0.5]
    ga_experiments = []
    for mr in mut_values:
        print(f'       mut_rate = {mr} …', end=' ', flush=True)
        t1 = time.time()
        curves = run_ga_graphcoloring(gc_adapter, mr)
        ga_experiments.append(curves)
        print(f'done ({time.time()-t1:.1f}s)')

    plot_sensitivity(
        ga_experiments,
        title='Graph Coloring (Festival Scheduling) + GA — Mutation Rate',
        ylabel='Total Penalty',
        param_name='mut_rate',
        param_values=mut_values,
        filename='graphcoloring_ga_mutrate.png',
        higher_is_better=False,    # minimise penalty
    )
    print_analysis('GraphColoring + GA', 'mut_rate', mut_values, ga_experiments, higher_better=False)

    finals_ga = [np.array(c)[:, -1] for c in ga_experiments]
    results_summary.append({
        'param_values': mut_values,
        'means_final':  [f.mean() for f in finals_ga],
        'stds_final':   [f.std()  for f in finals_ga],
        'best_finals':  [f.min()  for f in finals_ga],
    })

    # ──────────────────────────────────────────────────────────────
    # 4.  ShortestPath  +  SA  :  alpha  ∈ {0.8, 0.95, 0.99}
    # ──────────────────────────────────────────────────────────────
    print('\n[4/4] ShortestPath + SA — varying α (cooling rate)')
    from problems.discrete.ShortestPath import ShortestPathProblem
    sp_prob    = ShortestPathProblem.easy(seed=SEED)
    sp_adapter = ShortestPathSAAdapter(sp_prob)

    alpha_values = [0.8, 0.95, 0.99]
    sa_experiments = []
    for alpha in alpha_values:
        print(f'       α = {alpha} …', end=' ', flush=True)
        t1 = time.time()
        curves = run_sa_shortestpath(sp_adapter, alpha)
        sa_experiments.append(curves)
        print(f'done ({time.time()-t1:.1f}s)')

    plot_sensitivity(
        sa_experiments,
        title='Shortest Path (Mars Rover) + SA — Cooling Rate α',
        ylabel='Path Cost',
        param_name='α',
        param_values=alpha_values,
        filename='shortestpath_sa_alpha.png',
        higher_is_better=False,    # minimise cost
    )
    print_analysis('ShortestPath + SA', 'α', alpha_values, sa_experiments, higher_better=False)

    finals_sa = [np.array(c)[:, -1] for c in sa_experiments]
    results_summary.append({
        'param_values': alpha_values,
        'means_final':  [f.mean() for f in finals_sa],
        'stds_final':   [f.std()  for f in finals_sa],
        'best_finals':  [f.min()  for f in finals_sa],
    })

    # ──────────────────────────────────────────────────────────────
    # 5.  Combined Summary Dashboard
    # ──────────────────────────────────────────────────────────────
    print('\n[5/5] Generating combined summary dashboard …')
    plot_summary_table(results_summary)

    elapsed = time.time() - t0
    print(f'\n{"═"*60}')
    print(f'  All done in {elapsed:.1f}s')
    print(f'  Output → {OUT_DIR}')
    print(f'{"═"*60}')


if __name__ == '__main__':
    main()
