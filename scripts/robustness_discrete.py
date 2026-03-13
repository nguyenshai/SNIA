"""
Robustness Analysis — Discrete Problems (30 independent runs)
==============================================================
Mục đích: Chứng minh thuật toán Swarm không bị quá may rủi
          bằng cách chạy 30 lần độc lập và vẽ Boxplot phân phối kết quả.

  TSP           → ACO  (30 runs, 3 mức ρ để xem thêm ảnh hưởng tham số)
  Knapsack      → GA, ABC, CS      (30 runs mỗi thuật toán)
  GraphColoring → GA, SA           (30 runs mỗi thuật toán)
  ShortestPath  → SA               (30 runs — dùng grid-weight adapter)

Output: results/robustness_discrete/
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUT_DIR = os.path.join(BASE_DIR, 'results', 'robustness_discrete')
os.makedirs(OUT_DIR, exist_ok=True)

SEED       = 42
N_RUNS     = 30
ITERATIONS = 50    # đủ để hội tụ, vừa đủ nhanh cho 30 lần

# ══════════════════════════════════════════════════════════════════════
#  Visual style
# ══════════════════════════════════════════════════════════════════════
BG_DARK  = '#0d1117'
AX_DARK  = '#161b22'
TEXT_CLR = '#e6edf3'
GRID_CLR = '#30363d'

PALETTES = {
    'TSP':          ['#58a6ff', '#ffd93d', '#ff6b6b'],   # 3 configs ACO
    'Knapsack':     ['#58a6ff', '#ffd93d', '#ff6b6b'],   # GA / ABC / CS
    'GraphColoring':['#58a6ff', '#ffd93d'],               # GA / SA
    'ShortestPath': ['#58a6ff'],                          # SA only
}

def _style_ax(ax):
    ax.set_facecolor(AX_DARK)
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.title.set_color(TEXT_CLR)
    for sp in ax.spines.values():
        sp.set_color(GRID_CLR)
    ax.grid(axis='y', color=GRID_CLR, linestyle='--', linewidth=0.5, alpha=0.6)

def _savefig(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {p}')
    return p


# ══════════════════════════════════════════════════════════════════════
#  Discrete adapters (same as param_sensitivity.py)
# ══════════════════════════════════════════════════════════════════════

class KnapsackAdapter:
    def __init__(self, prob):
        self._prob = prob
        self.dim   = prob.n_items
        self.bounds = [0.0, 1.0]
        self.opt_type = 'min'

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        selection = (x > 0.5).astype(int).tolist()
        return -self._prob.evaluate(selection)['total_value']


class GraphColoringAdapter:
    def __init__(self, prob):
        self._prob = prob
        self.dim   = prob.n_nodes
        self.bounds = [0.0, float(prob.n_colors)]
        self.opt_type = 'min'

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        coloring = {i: min(int(xi), self._prob.n_colors - 1)
                    for i, xi in enumerate(x)}
        return self._prob.evaluate(coloring)['total_penalty']


class ShortestPathSAAdapter:
    def __init__(self, prob):
        self._prob = prob
        n = prob.grid_size
        self.dim   = n * n
        self.bounds = [0.0, 1.0]
        self.opt_type = 'min'

    def evaluate(self, x):
        x    = np.asarray(x, dtype=float)
        prob = self._prob
        n    = prob.grid_size
        wg   = x.reshape(n, n)
        path = [prob.start]
        visited = {prob.start}
        r, c = prob.start
        for _ in range(n * n * 2):
            if (r, c) == prob.goal:
                break
            nbrs = prob.neighbors(r, c)
            if not nbrs:
                break
            unvis = [nb for nb in nbrs if nb not in visited]
            choose = max(unvis, key=lambda nb: wg[nb]) \
                     if unvis else max(nbrs, key=lambda nb: wg[nb])
            path.append(choose)
            visited.add(choose)
            r, c = choose
        return prob.evaluate(path)['total_cost']


# ══════════════════════════════════════════════════════════════════════
#  Single-run helpers
# ══════════════════════════════════════════════════════════════════════

def _run_aco(prob, rho=0.5):
    from algorithms.biology.ACO import AntColonyOptimization
    alg = AntColonyOptimization(prob,
          params={'n_ants': 20, 'alpha': 1.0, 'beta': 5.0, 'rho': rho, 'q': 1.0})
    alg.solve(iterations=ITERATIONS)
    return alg.best_fitness

def _run_ga(adapter, pop=40):
    from algorithms.evolution.GA import GeneticAlgorithm
    alg = GeneticAlgorithm(adapter,
          params={'pop_size': pop, 'cx_rate': 0.7, 'mut_rate': 0.1})
    alg.solve(iterations=ITERATIONS)
    return alg.best_fitness

def _run_abc(adapter, pop=40):
    from algorithms.biology.ABC import ArtificialBeeColony
    alg = ArtificialBeeColony(adapter,
          params={'pop_size': pop, 'limit': 20})
    alg.solve(iterations=ITERATIONS)
    return alg.best_fitness

def _run_cs(adapter, pop=40):
    from algorithms.biology.CS import CuckooSearch
    alg = CuckooSearch(adapter,
          params={'n': pop, 'pa': 0.25})
    alg.solve(iterations=ITERATIONS)
    return alg.best_fitness

def _run_sa(adapter):
    from algorithms.physics.SA import SimulatedAnnealing
    alg = SimulatedAnnealing(adapter,
          params={'T0': 1.0, 'alpha': 0.95})
    alg.solve(iterations=ITERATIONS)
    return alg.best_fitness


# ══════════════════════════════════════════════════════════════════════
#  Boxplot builder
# ══════════════════════════════════════════════════════════════════════

def draw_boxplot(ax, data_dict, colors, ylabel, title,
                 higher_is_better=False, note=''):
    """
    data_dict : OrderedDict  {label: np.array(shape=N_RUNS)}
    """
    labels  = list(data_dict.keys())
    arrays  = [data_dict[l] for l in labels]
    x_pos   = np.arange(1, len(labels) + 1)

    _style_ax(ax)

    bp = ax.boxplot(
        arrays,
        positions=x_pos,
        patch_artist=True,
        widths=0.45,
        medianprops={'color': '#ffffff', 'linewidth': 2.5},
        whiskerprops={'color': '#8b949e', 'linewidth': 1.5},
        capprops={'color': '#8b949e', 'linewidth': 1.5},
        flierprops={'markerfacecolor': '#8b949e', 'marker': 'D',
                    'markersize': 4, 'alpha': 0.6},
    )

    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_linewidth(1.5)
        patch.set_edgecolor('#ffffff')

    # Individual run jitter (strip plot overlay)
    rng_jitter = np.random.default_rng(0)
    for xi, (arr, c) in enumerate(zip(arrays, colors), start=1):
        jitter = rng_jitter.uniform(-0.15, 0.15, size=len(arr))
        ax.scatter(xi + jitter, arr, color=c, alpha=0.35, s=18, zorder=5)

    # Annotations: median + std
    for xi, arr in enumerate(arrays, start=1):
        med = np.median(arr)
        std = arr.std()
        best = arr.min() if not higher_is_better else arr.max()
        sym = '▲' if higher_is_better else '▼'
        ax.text(xi, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                f'μ={np.mean(arr):.3g}\nσ={std:.2g}\n{sym}{best:.3g}',
                ha='center', va='bottom', fontsize=7.5,
                color=TEXT_CLR, fontfamily='monospace')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    if note:
        ax.text(0.5, -0.13, note, transform=ax.transAxes,
                ha='center', fontsize=7.5, color='#8b949e', style='italic')

    # Recalculate ylim after scatter
    all_vals = np.concatenate(arrays)
    span = all_vals.max() - all_vals.min() if all_vals.max() != all_vals.min() else 1
    pad = span * 0.25
    ax.set_ylim(all_vals.min() - span * 0.05, all_vals.max() + pad)


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print('=' * 60)
    print(f'  Robustness Analysis — Discrete Problems ({N_RUNS} runs)')
    print('=' * 60)

    # Load problems once
    from problems.discrete.TSP          import TSPProblem
    from problems.discrete.Knapsack     import KnapsackProblem
    from problems.discrete.GraphColoring import GraphColoringProblem
    from problems.discrete.ShortestPath  import ShortestPathProblem

    tsp_prob = TSPProblem.easy(seed=SEED)
    ks_prob  = KnapsackProblem.easy(seed=SEED)
    gc_prob  = GraphColoringProblem.easy(seed=SEED)
    sp_prob  = ShortestPathProblem.easy(seed=SEED)

    ks_ad = KnapsackAdapter(ks_prob)
    gc_ad = GraphColoringAdapter(gc_prob)
    sp_ad = ShortestPathSAAdapter(sp_prob)

    # ────────── 1. TSP — ACO with 3 evaporation rates ────────────────
    print(f'\n[1/4] TSP + ACO  ({N_RUNS} runs × 3 configs)')
    tsp_data = {}
    for rho in [0.1, 0.5, 0.9]:
        key = f'ACO\nρ={rho}'
        vals = []
        for run in range(N_RUNS):
            v = _run_aco(tsp_prob, rho=rho)
            vals.append(v)
        tsp_data[key] = np.array(vals)
        mu = np.mean(vals); sd = np.std(vals)
        print(f'   ρ={rho}  μ={mu:.3f}  σ={sd:.3f}  best={min(vals):.3f}')

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG_DARK)
    draw_boxplot(
        ax, tsp_data,
        colors=PALETTES['TSP'],
        ylabel='Best Tour Cost (minimize ↓)',
        title=f'TSP "Storm Chaser" + ACO — Robustness ({N_RUNS} independent runs)',
        higher_is_better=False,
        note=f'Easy difficulty | {tsp_prob.n_cities} cities | {ITERATIONS} iters/run'
    )
    _savefig(fig, 'tsp_aco_robustness.png')

    # ────────── 2. Knapsack — GA vs ABC vs CS ────────────────────────
    print(f'\n[2/4] Knapsack + GA / ABC / CS  ({N_RUNS} runs each)')
    ks_data = {}

    print('   GA  …', end=' ', flush=True); t1 = time.time()
    ks_data['GA'] = np.array([-_run_ga(ks_ad) for _ in range(N_RUNS)])
    print(f'done ({time.time()-t1:.1f}s)  '
          f'μ={ks_data["GA"].mean():.2f}  σ={ks_data["GA"].std():.2f}')

    print('   ABC …', end=' ', flush=True); t1 = time.time()
    ks_data['ABC'] = np.array([-_run_abc(ks_ad) for _ in range(N_RUNS)])
    print(f'done ({time.time()-t1:.1f}s)  '
          f'μ={ks_data["ABC"].mean():.2f}  σ={ks_data["ABC"].std():.2f}')

    print('   CS  …', end=' ', flush=True); t1 = time.time()
    ks_data['CS'] = np.array([-_run_cs(ks_ad) for _ in range(N_RUNS)])
    print(f'done ({time.time()-t1:.1f}s)  '
          f'μ={ks_data["CS"].mean():.2f}  σ={ks_data["CS"].std():.2f}')

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG_DARK)
    draw_boxplot(
        ax, ks_data,
        colors=PALETTES['Knapsack'],
        ylabel='Total Value (maximize ↑)',
        title=f'Knapsack "Space Cargo" — Robustness ({N_RUNS} independent runs)',
        higher_is_better=True,
        note=(f'Easy difficulty | {ks_prob.n_items} items | '
              f'GA / ABC / CS via continuous encoding | {ITERATIONS} iters/run')
    )
    _savefig(fig, 'knapsack_robustness.png')

    # ────────── 3. GraphColoring — GA vs SA ──────────────────────────
    print(f'\n[3/4] GraphColoring + GA / SA  ({N_RUNS} runs each)')
    gc_data = {}

    print('   GA  …', end=' ', flush=True); t1 = time.time()
    gc_data['GA'] = np.array([_run_ga(gc_ad) for _ in range(N_RUNS)])
    print(f'done ({time.time()-t1:.1f}s)  '
          f'μ={gc_data["GA"].mean():.3f}  σ={gc_data["GA"].std():.3f}')

    print('   SA  …', end=' ', flush=True); t1 = time.time()
    gc_data['SA'] = np.array([_run_sa(gc_ad) for _ in range(N_RUNS)])
    print(f'done ({time.time()-t1:.1f}s)  '
          f'μ={gc_data["SA"].mean():.3f}  σ={gc_data["SA"].std():.3f}')

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_DARK)
    draw_boxplot(
        ax, gc_data,
        colors=PALETTES['GraphColoring'],
        ylabel='Total Penalty (minimize ↓)',
        title=f'Graph Coloring "Festival" — Robustness ({N_RUNS} independent runs)',
        higher_is_better=False,
        note=(f'Easy difficulty | {gc_prob.n_nodes} nodes, '
              f'{gc_prob.n_colors} colors | {ITERATIONS} iters/run')
    )
    _savefig(fig, 'graphcoloring_robustness.png')

    # ────────── 4. ShortestPath — SA ─────────────────────────────────
    print(f'\n[4/4] ShortestPath + SA  ({N_RUNS} runs)')
    sp_vals = []
    t1 = time.time()
    for run in range(N_RUNS):
        if run % 5 == 0:
            print(f'   run {run+1}/{N_RUNS}…', end=' ', flush=True)
        v = _run_sa(sp_ad)
        sp_vals.append(v)
    sp_data = {'SA\n(grid-weight)': np.array(sp_vals)}
    mu = np.mean(sp_vals); sd = np.std(sp_vals)
    print(f'\n   done ({time.time()-t1:.1f}s)  μ={mu:.2f}  σ={sd:.2f}  best={min(sp_vals):.2f}')

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG_DARK)
    draw_boxplot(
        ax, sp_data,
        colors=PALETTES['ShortestPath'],
        ylabel='Path Cost (minimize ↓)',
        title=f'Shortest Path "Mars Rover" — SA Robustness ({N_RUNS} runs)',
        higher_is_better=False,
        note=(f'Easy difficulty | {sp_prob.grid_size}×{sp_prob.grid_size} grid | '
              f'SA with grid-weight encoding | {ITERATIONS} iters/run')
    )
    _savefig(fig, 'shortestpath_sa_robustness.png')

    # ────────── 5. Combined 2×2 dashboard ────────────────────────────
    print('\n[5/5] Combined dashboard …')
    fig = plt.figure(figsize=(22, 16), facecolor=BG_DARK)
    fig.suptitle(
        f'Robustness Analysis — Discrete Problems  ({N_RUNS} Independent Runs)',
        fontsize=18, fontweight='bold', color=TEXT_CLR, y=1.01
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.35)

    configs = [
        (fig.add_subplot(gs[0, 0]), tsp_data,
         PALETTES['TSP'], 'Best Tour Cost (minimize ↓)',
         f'TSP + ACO  (3 ρ configs)', False,
         f'{tsp_prob.n_cities} cities, Easy'),

        (fig.add_subplot(gs[0, 1]), ks_data,
         PALETTES['Knapsack'], 'Total Value (maximize ↑)',
         f'Knapsack + GA / ABC / CS', True,
         f'{ks_prob.n_items} items, Easy'),

        (fig.add_subplot(gs[1, 0]), gc_data,
         PALETTES['GraphColoring'], 'Total Penalty (minimize ↓)',
         f'Graph Coloring + GA / SA', False,
         f'{gc_prob.n_nodes} nodes, {gc_prob.n_colors} colors, Easy'),

        (fig.add_subplot(gs[1, 1]), sp_data,
         PALETTES['ShortestPath'], 'Path Cost (minimize ↓)',
         f'Shortest Path + SA', False,
         f'{sp_prob.grid_size}×{sp_prob.grid_size} grid, Easy'),
    ]

    for ax, data, pal, ylabel, title, hib, note in configs:
        draw_boxplot(ax, data, pal, ylabel, title,
                     higher_is_better=hib, note=note)

    plt.tight_layout()
    _savefig(fig, 'dashboard_robustness.png')

    elapsed = time.time() - t0
    print(f'\n{"═"*60}')
    print(f'  Done in {elapsed:.1f}s  |  Output → {OUT_DIR}')
    print(f'{"═"*60}')


if __name__ == '__main__':
    main()
