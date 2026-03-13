"""
Execution Time Analysis — Continuous Problems
==============================================
Mục đích: Đo và so sánh thời gian chạy (Median Execution Time, giây)
          của tất cả thuật toán Continuous trên 5 hàm benchmark.

Thuật toán: GA, PSO, DE, SA, HC, ABC, CS, FA, TLBO
Bài toán  : Sphere, Rastrigin, Ackley, Rosenbrock, Griewank

Mỗi cấu hình chạy N_TIMING_RUNS lần → lấy median thời gian.

Biểu đồ xuất ra:
  • 1 bar chart riêng cho mỗi bài toán  (5 files)
  • 1 heatmap tổng hợp  problems × algorithms
  • 1 stacked comparison bar chart

Output: results/exec_time_continuous/
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUT_DIR = os.path.join(BASE_DIR, 'results', 'exec_time_continuous')
os.makedirs(OUT_DIR, exist_ok=True)

DIM            = 10     # dimension for all problems
ITERATIONS     = 100    # same as benchmark experiments
N_TIMING_RUNS  = 7      # runs per config to get stable median

# ══════════════════════════════════════════════════════════════════════
#  Visual style
# ══════════════════════════════════════════════════════════════════════
BG_DARK  = '#0d1117'
AX_DARK  = '#161b22'
TEXT_CLR = '#e6edf3'
GRID_CLR = '#30363d'
MID_CLR  = '#8b949e'

# Per-algorithm colors (consistent across all charts)
ALGO_COLORS = {
    'GA'  : '#58a6ff',
    'PSO' : '#ffd93d',
    'DE'  : '#ff6b6b',
    'SA'  : '#6bcb77',
    'HC'  : '#a371f7',
    'ABC' : '#f0883e',
    'CS'  : '#00d4aa',
    'FA'  : '#f78166',
    'TLBO': '#d2a8ff',
}

# Per-problem accent colors
PROB_COLORS = {
    'Sphere'    : '#58a6ff',
    'Rastrigin' : '#ffd93d',
    'Ackley'    : '#ff6b6b',
    'Rosenbrock': '#6bcb77',
    'Griewank'  : '#a371f7',
}

ALGO_LABELS = list(ALGO_COLORS.keys())   # ordered list
PROB_LABELS = list(PROB_COLORS.keys())

def _style_ax(ax):
    ax.set_facecolor(AX_DARK)
    ax.tick_params(colors=MID_CLR, labelsize=9)
    ax.xaxis.label.set_color(MID_CLR)
    ax.yaxis.label.set_color(MID_CLR)
    ax.title.set_color(TEXT_CLR)
    for sp in ax.spines.values():
        sp.set_color(GRID_CLR)
    ax.grid(axis='y', color=GRID_CLR, linestyle='--', linewidth=0.5, alpha=0.7)

def _savefig(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {p}')
    return p


# ══════════════════════════════════════════════════════════════════════
#  Algorithm factory
# ══════════════════════════════════════════════════════════════════════

def make_runner(algo_key, problem):
    """Return a zero-arg callable that instantiates + solves the algorithm."""
    if algo_key == 'GA':
        from algorithms.evolution.GA import GeneticAlgorithm
        def run():
            GeneticAlgorithm(problem, params={'pop_size': 40, 'cx_rate': 0.7, 'mut_rate': 0.1}).solve(iterations=ITERATIONS)
    elif algo_key == 'PSO':
        from algorithms.biology.PSO import ParticleSwarmOptimization
        def run():
            ParticleSwarmOptimization(problem, params={'pop_size': 40, 'w': 0.7, 'c1': 1.5, 'c2': 1.5}).solve(iterations=ITERATIONS)
    elif algo_key == 'DE':
        from algorithms.evolution.DE import DifferentialEvolution
        def run():
            DifferentialEvolution(problem, params={'pop_size': 40, 'F': 0.5, 'CR': 0.9}).solve(iterations=ITERATIONS)
    elif algo_key == 'SA':
        from algorithms.physics.SA import SimulatedAnnealing
        def run():
            SimulatedAnnealing(problem, params={'T0': 1.0, 'alpha': 0.95}).solve(iterations=ITERATIONS)
    elif algo_key == 'HC':
        from algorithms.classical.Hill_climbing import HillClimbing
        def run():
            HillClimbing(problem, params={'restarts': 5, 'step_scale': 0.1}).solve(iterations=ITERATIONS)
    elif algo_key == 'ABC':
        from algorithms.biology.ABC import ArtificialBeeColony
        def run():
            ArtificialBeeColony(problem, params={'pop_size': 40, 'limit': 20}).solve(iterations=ITERATIONS)
    elif algo_key == 'CS':
        from algorithms.biology.CS import CuckooSearch
        def run():
            CuckooSearch(problem, params={'n': 40, 'pa': 0.25}).solve(iterations=ITERATIONS)
    elif algo_key == 'FA':
        from algorithms.biology.FA import FireflyAlgorithm
        def run():
            FireflyAlgorithm(problem, params={'pop_size': 20, 'alpha': 0.5, 'beta0': 1.0, 'gamma': 0.1}).solve(iterations=ITERATIONS)
    elif algo_key == 'TLBO':
        from algorithms.human.TLBO import TLBO
        def run():
            TLBO(problem, params={'pop_size': 40}).solve(iterations=ITERATIONS)
    else:
        raise ValueError(f'Unknown algo: {algo_key}')

    return run


# ══════════════════════════════════════════════════════════════════════
#  Timing engine
# ══════════════════════════════════════════════════════════════════════

def measure_times(runner, n_runs=N_TIMING_RUNS):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        runner()
        times.append(time.perf_counter() - t0)
    return np.array(times)


# ══════════════════════════════════════════════════════════════════════
#  Individual problem bar chart
# ══════════════════════════════════════════════════════════════════════

def plot_problem_bar(prob_name, medians, stds, filename):
    algos    = ALGO_LABELS
    med_vals = [float(medians[a]) for a in algos]
    std_vals = [float(stds[a])    for a in algos]
    colors   = [ALGO_COLORS[a] for a in algos]
    x        = np.arange(len(algos))

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG_DARK)
    _style_ax(ax)

    bars = ax.bar(x, med_vals, color=colors, alpha=0.80,
                  edgecolor='white', linewidth=0.6, zorder=3, width=0.6)
    ax.errorbar(x, med_vals, yerr=std_vals,
                fmt='none', color=TEXT_CLR, capsize=5, linewidth=1.5, zorder=4)

    # Value labels on top of bars
    for xi, (med, std) in enumerate(zip(med_vals, std_vals)):
        ax.text(xi, med + std + max(med_vals) * 0.01,
                f'{med:.3f}s', ha='center', va='bottom',
                fontsize=8.5, color=TEXT_CLR, fontfamily='monospace')

    # Fastest highlight
    best_idx = int(np.argmin(med_vals))
    bars[best_idx].set_edgecolor('#ffd93d')
    bars[best_idx].set_linewidth(2.5)
    ax.text(best_idx, med_vals[best_idx] / 2,
            '  fastest', ha='center', va='center',
            fontsize=8, color='#ffd93d', fontweight='bold', rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=11, fontweight='bold')
    ax.set_ylabel('Median Execution Time (seconds)', fontsize=10)
    ax.set_title(
        f'{prob_name} — Execution Time per Algorithm\n'
        f'(dim={DIM}, {ITERATIONS} iterations, median of {N_TIMING_RUNS} runs)',
        fontsize=13, fontweight='bold', pad=14
    )

    ax.text(0.99, 0.97,
            '* FA uses pop=20 due to O(N ) complexity',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5, color=MID_CLR, style='italic')

    fig.suptitle('', fontsize=1)   # spacer
    plt.tight_layout()
    return _savefig(fig, filename)


# ══════════════════════════════════════════════════════════════════════
#  Heatmap — problems × algorithms
# ══════════════════════════════════════════════════════════════════════

def plot_heatmap(time_matrix, filename):
    cmap = LinearSegmentedColormap.from_list(
        'dark_heat',
        ['#0d1117', '#1a3a6b', '#2563eb', '#fbbf24', '#ef4444'],
        N=256
    )

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)

    im = ax.imshow(time_matrix, cmap=cmap, aspect='auto')

    for pi, prob in enumerate(PROB_LABELS):
        for ai, algo in enumerate(ALGO_LABELS):
            val = time_matrix[pi, ai]
            text_color = '#ffffff' if val > time_matrix.max() * 0.5 else TEXT_CLR
            ax.text(ai, pi, f'{val:.3f}s',
                    ha='center', va='center',
                    fontsize=9, color=text_color,
                    fontfamily='monospace', fontweight='bold')

    ax.set_xticks(range(len(ALGO_LABELS)))
    ax.set_xticklabels(ALGO_LABELS, fontsize=11, fontweight='bold', color=TEXT_CLR)
    ax.set_yticks(range(len(PROB_LABELS)))
    ax.set_yticklabels(PROB_LABELS, fontsize=11, fontweight='bold')
    for tick, label in zip(ax.get_yticklabels(), PROB_LABELS):
        tick.set_color(PROB_COLORS[label])
    ax.tick_params(colors=MID_CLR)
    for sp in ax.spines.values():
        sp.set_color(GRID_CLR)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=MID_CLR, labelsize=9)
    cbar.set_label('Median Time (s)', color=MID_CLR, fontsize=10)

    ax.set_title(
        f'Execution Time Heatmap — Continuous Problems x Algorithms\n'
        f'(dim={DIM}, {ITERATIONS} iters, median of {N_TIMING_RUNS} runs)',
        fontsize=14, fontweight='bold', color=TEXT_CLR, pad=14
    )

    plt.tight_layout()
    return _savefig(fig, filename)


# ══════════════════════════════════════════════════════════════════════
#  Grouped bar chart — all problems side-by-side
# ══════════════════════════════════════════════════════════════════════

def plot_grouped_bars(time_matrix, filename):
    n_prob  = len(PROB_LABELS)
    n_algo  = len(ALGO_LABELS)
    x       = np.arange(n_algo)
    width   = 0.14
    offsets = np.linspace(-(n_prob - 1) / 2, (n_prob - 1) / 2, n_prob) * width

    fig, ax = plt.subplots(figsize=(20, 8), facecolor=BG_DARK)
    _style_ax(ax)
    ax.set_axisbelow(True)

    for pi, (prob, offset) in enumerate(zip(PROB_LABELS, offsets)):
        vals = time_matrix[pi]
        color = PROB_COLORS[prob]
        bars = ax.bar(x + offset, vals, width=width * 0.85,
                       label=prob, color=color, alpha=0.82,
                       edgecolor='white', linewidth=0.4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(ALGO_LABELS, fontsize=12, fontweight='bold')
    ax.set_ylabel('Median Execution Time (seconds)', fontsize=11)
    ax.set_title(
        'Execution Time — All Continuous Problems & Algorithms\n'
        f'(dim={DIM}, {ITERATIONS} iters, median of {N_TIMING_RUNS} runs)',
        fontsize=14, fontweight='bold', pad=14
    )

    legend = ax.legend(
        fontsize=10, facecolor=AX_DARK, labelcolor=TEXT_CLR,
        edgecolor=GRID_CLR, title='Problem', title_fontsize=10,
        loc='upper left'
    )
    legend.get_title().set_color(TEXT_CLR)

    ax.text(0.99, 0.97, '* FA uses pop=20',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color=MID_CLR, style='italic')

    plt.tight_layout()
    return _savefig(fig, filename)


# ══════════════════════════════════════════════════════════════════════
#  Per-problem summary subplot grid
# ══════════════════════════════════════════════════════════════════════

def plot_per_problem_grid(all_medians, all_stds, filename):
    fig = plt.figure(figsize=(22, 12), facecolor=BG_DARK)
    fig.suptitle(
        f'Execution Time per Problem — All Algorithms\n'
        f'(dim={DIM}, {ITERATIONS} iters, median of {N_TIMING_RUNS} runs)',
        fontsize=16, fontweight='bold', color=TEXT_CLR, y=1.01
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

    algos  = ALGO_LABELS
    colors = [ALGO_COLORS[a] for a in algos]
    x      = np.arange(len(algos))

    for idx, prob in enumerate(PROB_LABELS):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        _style_ax(ax)

        med_vals = [float(all_medians[prob][a]) for a in algos]
        std_vals = [float(all_stds[prob][a])    for a in algos]

        bars = ax.bar(x, med_vals, color=colors, alpha=0.80,
                      edgecolor='white', linewidth=0.5, zorder=3, width=0.6)
        ax.errorbar(x, med_vals, yerr=std_vals,
                    fmt='none', color=TEXT_CLR, capsize=3, linewidth=1.2, zorder=4)

        for xi, (med, std) in enumerate(zip(med_vals, std_vals)):
            ax.text(xi, med + std + max(med_vals) * 0.03,
                    f'{med:.2f}', ha='center', va='bottom',
                    fontsize=7.5, color=TEXT_CLR, fontfamily='monospace')

        best_idx = int(np.argmin(med_vals))
        bars[best_idx].set_edgecolor('#ffd93d')
        bars[best_idx].set_linewidth(2.0)

        acc_color = PROB_COLORS[prob]
        ax.set_title(prob, fontsize=12, fontweight='bold', color=acc_color, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=8, rotation=25, ha='right')
        ax.set_ylabel('Time (s)', fontsize=8)

    # 6th slot: legend
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.set_facecolor(AX_DARK)
    for sp in ax_leg.spines.values():
        sp.set_color(GRID_CLR)
    ax_leg.set_xticks([]); ax_leg.set_yticks([])

    legend_items = [
        plt.Rectangle((0, 0), 1, 1, color=ALGO_COLORS[a], alpha=0.8)
        for a in algos
    ]
    ax_leg.legend(legend_items, algos,
                  loc='center', fontsize=11,
                  facecolor=AX_DARK, labelcolor=TEXT_CLR,
                  edgecolor=GRID_CLR,
                  title='Algorithm', title_fontsize=12,
                  ncol=2)
    ax_leg.get_legend().get_title().set_color(TEXT_CLR)
    ax_leg.text(0.5, 0.08,
                '* Gold border = fastest per problem\n'
                '* FA pop=20 (O(N ) complexity)',
                ha='center', va='center', transform=ax_leg.transAxes,
                fontsize=9, color=MID_CLR, style='italic')

    plt.tight_layout()
    return _savefig(fig, filename)


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    t0_total = time.time()
    print('=' * 62)
    print(f'  Execution Time Analysis — Continuous Problems')
    print(f'  ({N_TIMING_RUNS} timing runs x {ITERATIONS} iters each)')
    print('=' * 62)

    # Load problems
    from problems.continuous.Sphere     import Sphere
    from problems.continuous.Rastrigin  import Rastrigin
    from problems.continuous.Ackley     import Ackley
    from problems.continuous.Rosenbrock import Rosenbrock
    from problems.continuous.Griewank   import Griewank

    problems = {
        'Sphere'    : Sphere(dim=DIM),
        'Rastrigin' : Rastrigin(dim=DIM),
        'Ackley'    : Ackley(dim=DIM),
        'Rosenbrock': Rosenbrock(dim=DIM),
        'Griewank'  : Griewank(dim=DIM),
    }

    time_matrix  = np.zeros((len(PROB_LABELS), len(ALGO_LABELS)))
    all_medians  = {p: {} for p in PROB_LABELS}
    all_stds     = {p: {} for p in PROB_LABELS}

    header = f'  {"Problem":<14}' + ''.join(f'{a:>8}' for a in ALGO_LABELS)
    print(f'\n{header}')
    print('  ' + '-' * (14 + 8 * len(ALGO_LABELS)))

    for pi, prob_name in enumerate(PROB_LABELS):
        prob = problems[prob_name]
        row_str = f'  {prob_name:<14}'

        for ai, algo in enumerate(ALGO_LABELS):
            runner = make_runner(algo, prob)
            times  = measure_times(runner)
            med    = float(np.median(times))
            std    = float(np.std(times))

            time_matrix[pi, ai]            = med
            all_medians[prob_name][algo]   = med
            all_stds[prob_name][algo]      = std
            row_str += f'{med:>7.3f}s'

        print(row_str)

    print('\n-- Generating individual bar charts ...')
    for prob_name in PROB_LABELS:
        plot_problem_bar(
            prob_name,
            all_medians[prob_name],
            all_stds[prob_name],
            filename=f'exec_time_{prob_name.lower()}.png',
        )

    print('\n-- Generating heatmap ...')
    try:
        plot_heatmap(time_matrix, 'exec_time_heatmap.png')
    except Exception as e:
        print(f"Heatmap failed: {e}")

    print('\n-- Generating grouped bar chart ...')
    try:
        plot_grouped_bars(time_matrix, 'exec_time_grouped.png')
    except Exception as e:
        print(f"Grouped failed: {e}")

    print('\n-- Generating per-problem grid ...')
    try:
        plot_per_problem_grid(all_medians, all_stds, 'exec_time_grid.png')
    except Exception as e:
        print(f"Grid failed: {e}")

    print('\n' + '=' * 62)
    print('  SUMMARY -- Fastest Algorithm per Problem')
    print('  ' + '-' * 58)
    for pi, prob in enumerate(PROB_LABELS):
        best_ai  = int(np.argmin(time_matrix[pi]))
        worst_ai = int(np.argmax(time_matrix[pi]))
        print(f'  {prob:<14}  fastest: {ALGO_LABELS[best_ai]:<5}'
              f'  ({time_matrix[pi, best_ai]:.3f}s)'
              f'   slowest: {ALGO_LABELS[worst_ai]:<5}'
              f'  ({time_matrix[pi, worst_ai]:.3f}s)')

    elapsed = time.time() - t0_total
    print(f'\n  Total benchmark time: {elapsed:.1f}s')
    print(f'  Output -> {OUT_DIR}')
    print('=' * 62)

if __name__ == '__main__':
    main()
