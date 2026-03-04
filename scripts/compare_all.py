"""
Algorithm Comparison Benchmark for SNIA.

Compares algorithms on problems based on the recommended pairings
described in each problem file:

  Continuous:
    Sphere      -> HC vs PSO
    Rastrigin   -> GA vs SA
    Ackley      -> CS vs SA
    Rosenbrock  -> DE vs HC
    Griewank    -> FA vs SA

  Plus an overall comparison of all population-based algorithms:
    GA, DE, PSO, ABC, CS, FA, SA, HC, TLBO

Results saved to results/compare/

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

OUT_DIR = PROJECT_ROOT / 'results' / 'compare'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Imports --
from problems.continuous.Sphere import Sphere
from problems.continuous.Rastrigin import Rastrigin
from problems.continuous.Rosenbrock import Rosenbrock
from problems.continuous.Ackley import Ackley
from problems.continuous.Griewank import Griewank

from algorithms.evolution.GA import GeneticAlgorithm
from algorithms.evolution.DE import DifferentialEvolution
from algorithms.biology.PSO import ParticleSwarmOptimization
from algorithms.biology.ABC import ArtificialBeeColony
from algorithms.biology.CS import CuckooSearch
from algorithms.biology.FA import FireflyAlgorithm
from algorithms.physics.SA import SimulatedAnnealing
from algorithms.classical.Hill_climbing import HillClimbing
from algorithms.human.TLBO import TLBO

# =====================================================================
#  Style constants
# =====================================================================
BG      = '#0d1117'
PANEL   = '#161b22'
TXT     = '#e6edf3'
GRID_C  = '#30363d'
BORDER  = '#30363d'

COLORS = {
    'GA':   '#ff6b6b',
    'DE':   '#ffd93d',
    'PSO':  '#58a6ff',
    'ABC':  '#00d4aa',
    'CS':   '#a371f7',
    'FA':   '#f0883e',
    'SA':   '#ff9ff3',
    'HC':   '#54a0ff',
    'TLBO': '#01a3a4',
}

MARKERS = {
    'GA': 'o', 'DE': 's', 'PSO': '^', 'ABC': 'D',
    'CS': 'v', 'FA': 'P', 'SA': 'X', 'HC': '*', 'TLBO': 'p',
}


def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.15, color='#484f58')
    if title:
        ax.set_title(title, color=TXT, fontsize=12, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)


# =====================================================================
#  Algorithm & Problem registry
# =====================================================================

ALL_ALGOS = {
    'GA':   (GeneticAlgorithm,          {'pop_size': 40, 'cx_rate': 0.8, 'mut_rate': 0.1}),
    'DE':   (DifferentialEvolution,     {'pop_size': 40, 'F': 0.8, 'CR': 0.9}),
    'PSO':  (ParticleSwarmOptimization, {'pop_size': 40, 'w': 0.7, 'c1': 1.5, 'c2': 1.5}),
    'ABC':  (ArtificialBeeColony,      {'pop_size': 40}),
    'CS':   (CuckooSearch,             {'n': 40, 'pa': 0.25}),
    'FA':   (FireflyAlgorithm,         {'pop_size': 30, 'alpha': 0.5, 'beta0': 1.0, 'gamma': 1.0}),
    'SA':   (SimulatedAnnealing,       {'T0': 1.0, 'alpha': 0.99}),
    'HC':   (HillClimbing,            {'restarts': 5, 'step_scale': 0.1}),
    'TLBO': (TLBO,                     {'pop_size': 40}),
}

ALL_PROBLEMS = {
    'Sphere':     (Sphere,     10),
    'Rastrigin':  (Rastrigin,  10),
    'Rosenbrock': (Rosenbrock, 10),
    'Ackley':     (Ackley,     10),
    'Griewank':   (Griewank,   10),
}

# Recommended pairings from problem docstrings
RECOMMENDED_PAIRS = {
    'Sphere':     ['HC', 'PSO'],
    'Rastrigin':  ['GA', 'SA'],
    'Ackley':     ['CS', 'SA'],
    'Rosenbrock': ['DE', 'HC'],
    'Griewank':   ['FA', 'SA'],
}

ITERATIONS = 100
N_RUNS = 3


# =====================================================================
#  Benchmark runner
# =====================================================================

def run_algos(prob_cls, dim, algo_names, iters=ITERATIONS, n_runs=N_RUNS):
    """Run selected algorithms on a given problem. Returns dict[algo_name] -> list[run_dict]."""
    results = {}
    for aname in algo_names:
        algo_cls, params = ALL_ALGOS[aname]
        runs = []
        for _ in range(n_runs):
            prob = prob_cls(dim=dim)
            alg = algo_cls(prob, params=dict(params))
            try:
                alg.solve(iterations=iters)
            except Exception as e:
                print(f'    [{aname} FAIL: {e}]')
                break
            conv = [h.get('global_best_fit', None) for h in alg.history]
            runs.append({
                'convergence': conv,
                'best_fit': alg.best_fitness,
            })
        results[aname] = runs
    return results


# =====================================================================
#  1) Recommended pairing comparison (per problem)
# =====================================================================

def chart_recommended_pairs():
    """Generate a head-to-head comparison chart for each problem's recommended algorithms."""
    print('\n--- Recommended Algorithm Pairings ---')

    for prob_name, algo_names in RECOMMENDED_PAIRS.items():
        prob_cls, dim = ALL_PROBLEMS[prob_name]
        print(f'  {prob_name}: {" vs ".join(algo_names)}', end='', flush=True)

        data = run_algos(prob_cls, dim, algo_names)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                        gridspec_kw={'width_ratios': [2, 1]})
        fig.patch.set_facecolor(BG)

        # -- Left: Convergence --
        style_ax(ax1, title=f'{prob_name} - Convergence',
                 xlabel='Iteration', ylabel='Best Fitness')

        for aname, runs in data.items():
            if not runs:
                continue
            all_c = [r['convergence'] for r in runs]
            ml = min(len(c) for c in all_c)
            arr = np.array([c[:ml] for c in all_c])
            med = np.median(arr, axis=0)
            lo, hi = np.min(arr, axis=0), np.max(arr, axis=0)
            x = np.arange(1, ml + 1)
            c = COLORS[aname]
            m = MARKERS[aname]
            ax1.plot(x, med, label=aname, color=c, linewidth=2.5,
                     marker=m, markevery=max(1, ml // 8), markersize=6)
            if N_RUNS > 1:
                ax1.fill_between(x, lo, hi, alpha=0.1, color=c)

        ax1.set_yscale('log')
        ax1.legend(facecolor='#21262d', edgecolor=BORDER,
                   labelcolor='#c9d1d9', fontsize=10)

        # -- Right: Bar chart final fitness --
        style_ax(ax2, title=f'{prob_name} - Final Fitness',
                 ylabel='Best Fitness')

        names_bar = []
        vals_bar = []
        cols_bar = []
        for aname, runs in data.items():
            if runs:
                names_bar.append(aname)
                vals_bar.append(np.mean([r['best_fit'] for r in runs]))
                cols_bar.append(COLORS[aname])

        x_pos = np.arange(len(names_bar))
        bars = ax2.bar(x_pos, vals_bar, color=cols_bar, alpha=0.85,
                       edgecolor='white', linewidth=0.5, width=0.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names_bar, color='#c9d1d9', fontsize=11)
        for bar, val in zip(bars, vals_bar):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.2e}', ha='center', va='bottom',
                     color='#c9d1d9', fontsize=9, fontweight='bold')

        # Winner annotation
        if len(vals_bar) >= 2:
            winner = names_bar[np.argmin(vals_bar)]
            fig.text(0.5, 0.01, f'Winner: {winner}',
                     ha='center', color=COLORS[winner],
                     fontsize=12, fontweight='bold')

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        path = OUT_DIR / f'pair_{prob_name.lower()}.png'
        fig.savefig(str(path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {path.name}')


# =====================================================================
#  2) All algorithms convergence (per problem)
# =====================================================================

def chart_all_convergence():
    """Overlay all 9 algorithms on each problem."""
    print('\n--- All Algorithms Convergence ---')
    all_algo_names = list(ALL_ALGOS.keys())

    for prob_name, (prob_cls, dim) in ALL_PROBLEMS.items():
        print(f'  {prob_name}:', end='', flush=True)
        data = run_algos(prob_cls, dim, all_algo_names)

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor(BG)
        style_ax(ax, title=f'All Algorithms on {prob_name} (dim={dim})',
                 xlabel='Iteration', ylabel='Best Fitness')

        for aname, runs in data.items():
            if not runs:
                continue
            all_c = [r['convergence'] for r in runs]
            ml = min(len(c) for c in all_c)
            arr = np.array([c[:ml] for c in all_c])
            med = np.median(arr, axis=0)
            x = np.arange(1, ml + 1)
            ax.plot(x, med, label=aname, color=COLORS[aname], linewidth=1.8,
                    marker=MARKERS[aname], markevery=max(1, ml // 10),
                    markersize=5, alpha=0.9)

        ax.set_yscale('log')
        ax.legend(facecolor='#21262d', edgecolor=BORDER,
                  labelcolor='#c9d1d9', fontsize=8, ncol=3, loc='upper right')

        plt.tight_layout()
        path = OUT_DIR / f'all_conv_{prob_name.lower()}.png'
        fig.savefig(str(path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {path.name}')


# =====================================================================
#  3) Heatmap (algorithm x problem)
# =====================================================================

def chart_heatmap():
    """Heatmap of average best fitness for all algorithm-problem combos."""
    print('\n--- Heatmap ---')
    all_algo_names = list(ALL_ALGOS.keys())
    prob_names = list(ALL_PROBLEMS.keys())

    # Collect results (reuse cache if we want, but keep simple)
    matrix = np.full((len(all_algo_names), len(prob_names)), np.nan)

    for j, pname in enumerate(prob_names):
        prob_cls, dim = ALL_PROBLEMS[pname]
        print(f'  {pname}:', end='', flush=True)
        data = run_algos(prob_cls, dim, all_algo_names, n_runs=1)  # single run for speed
        for i, aname in enumerate(all_algo_names):
            runs = data.get(aname, [])
            if runs:
                matrix[i, j] = runs[0]['best_fit']
        print(' done')

    matrix_log = np.log10(np.clip(matrix, 1e-20, None))

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    im = ax.imshow(matrix_log, cmap='plasma', aspect='auto')
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('log10(Fitness)', color='#8b949e', fontsize=9)
    cb.ax.tick_params(colors='#8b949e', labelsize=8)

    ax.set_xticks(range(len(prob_names)))
    ax.set_xticklabels(prob_names, color='#c9d1d9', fontsize=10, rotation=25, ha='right')
    ax.set_yticks(range(len(all_algo_names)))
    ax.set_yticklabels(all_algo_names, color='#c9d1d9', fontsize=10)

    for i in range(len(all_algo_names)):
        for j in range(len(prob_names)):
            v = matrix[i, j]
            if not np.isnan(v):
                mid = np.nanmedian(matrix_log)
                tc = 'white' if matrix_log[i, j] < mid else '#222'
                ax.text(j, i, f'{v:.1e}', ha='center', va='center',
                        fontsize=7, color=tc, fontweight='bold')

    ax.set_title('Algorithm x Problem -- Best Fitness',
                 color=TXT, fontsize=14, fontweight='bold', pad=15)
    for sp in ax.spines.values():
        sp.set_color(BORDER)

    plt.tight_layout()
    path = OUT_DIR / 'heatmap_fitness.png'
    fig.savefig(str(path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path.name}')


# =====================================================================
#  4) Radar chart (normalized performance)
# =====================================================================

def chart_radar():
    """Radar chart: each algorithm's normalized score across problems."""
    print('\n--- Radar Chart ---')
    all_algo_names = list(ALL_ALGOS.keys())
    prob_names = list(ALL_PROBLEMS.keys())

    matrix = np.full((len(all_algo_names), len(prob_names)), np.nan)
    for j, pname in enumerate(prob_names):
        prob_cls, dim = ALL_PROBLEMS[pname]
        data = run_algos(prob_cls, dim, all_algo_names, n_runs=1)
        for i, aname in enumerate(all_algo_names):
            runs = data.get(aname, [])
            if runs:
                matrix[i, j] = runs[0]['best_fit']

    # Normalize per column (0=best, 1=worst), then invert
    norm = np.zeros_like(matrix)
    for j in range(len(prob_names)):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            cmin, cmax = valid.min(), valid.max()
            rng = cmax - cmin if cmax - cmin > 1e-12 else 1.0
            norm[:, j] = (col - cmin) / rng
    norm = 1.0 - norm  # higher = better

    n_vars = len(prob_names)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    for i, aname in enumerate(all_algo_names):
        vals = norm[i].tolist() + [norm[i, 0]]
        c = COLORS.get(aname, '#ccc')
        ax.plot(angles, vals, color=c, linewidth=2, label=aname, alpha=0.85)
        ax.fill(angles, vals, color=c, alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(prob_names, color='#c9d1d9', fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='#8b949e', fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.spines['polar'].set_color(BORDER)
    ax.tick_params(colors='#8b949e')
    ax.grid(color='#484f58', alpha=0.3)

    ax.set_title('Algorithm Performance Radar\n(1.0 = best, 0.0 = worst)',
                 color=TXT, fontsize=13, fontweight='bold', pad=25)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.1),
              facecolor='#21262d', edgecolor=BORDER,
              labelcolor='#c9d1d9', fontsize=8, ncol=3)

    plt.tight_layout()
    path = OUT_DIR / 'radar_performance.png'
    fig.savefig(str(path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path.name}')


# =====================================================================
#  5) Ranking per problem
# =====================================================================

def chart_rankings():
    """Horizontal bar ranking for each problem."""
    print('\n--- Rankings ---')
    all_algo_names = list(ALL_ALGOS.keys())

    for pname, (prob_cls, dim) in ALL_PROBLEMS.items():
        print(f'  {pname}:', end='', flush=True)
        try:
            data = run_algos(prob_cls, dim, all_algo_names, n_runs=1)

            entries = []
            for aname, runs in data.items():
                if runs:
                    entries.append((aname, runs[0]['best_fit']))
            entries.sort(key=lambda x: x[1])

            names = [e[0] for e in entries]
            vals  = [e[1] for e in entries]
            cols  = [COLORS.get(n, '#ccc') for n in names]

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(BG)
            style_ax(ax, title=f'Algorithm Ranking -- {pname}', xlabel='Best Fitness (lower is better)')

            y_pos = np.arange(len(names))
            bars = ax.barh(y_pos, vals, height=0.55, color=cols, alpha=0.85,
                           edgecolor='white', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, color='#c9d1d9', fontsize=10)
            ax.invert_yaxis()

            for bar, val in zip(bars, vals):
                ax.text(max(bar.get_width(), max(vals)*0.01),
                        bar.get_y() + bar.get_height() / 2,
                        f'  {val:.2e}', va='center', color='#c9d1d9', fontsize=8)

            # top 3 labels
            labels = ['#1', '#2', '#3']
            for i in range(min(3, len(names))):
                ax.annotate(labels[i], xy=(0, y_pos[i]),
                            xytext=(-10, 0), textcoords='offset points',
                            va='center', ha='right', fontsize=9,
                            color=cols[i], fontweight='bold')

            if max(vals) / (min(vals) + 1e-20) > 100:
                ax.set_xscale('log')
            plt.tight_layout()
            path = OUT_DIR / f'ranking_{pname.lower()}.png'
            fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
            print(f'  -> {path.name}')
        except Exception as e:
            print(f'  [ERROR: {e}]')
            plt.close('all')


# =====================================================================
#  Main
# =====================================================================

def main():
    print('=' * 58)
    print('  SNIA -- Algorithm Comparison Benchmark')
    print('=' * 58)
    print(f'  Problems:   {len(ALL_PROBLEMS)}')
    print(f'  Algorithms: {len(ALL_ALGOS)}')
    print(f'  Iterations: {ITERATIONS}')
    print(f'  Runs:       {N_RUNS}')
    print(f'  Output:     {OUT_DIR}')

    t0 = time.time()

    # 1. Recommended head-to-head pairs (from problem docs)
    chart_recommended_pairs()

    # 2. All algorithms convergence overlay
    chart_all_convergence()

    # 3. Heatmap
    chart_heatmap()

    # 4. Radar
    chart_radar()

    # 5. Rankings
    chart_rankings()

    elapsed = time.time() - t0
    n_files = len(list(OUT_DIR.glob('*.png')))
    print(f'\n{"=" * 58}')
    print(f'  Done! {n_files} charts in {elapsed:.0f}s')
    print(f'  {OUT_DIR}')
    print(f'{"=" * 58}')


if __name__ == '__main__':
    main()
