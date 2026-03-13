"""
Robustness Analysis — Continuous Problems (30 independent runs)
==============================================================
Mục đích: Đánh giá độ ổn định của 9 thuật toán tối ưu liên tục
          qua 30 lần chạy độc lập, biểu thị bằng Boxplot.

Thuật toán: GA, PSO, DE, SA, HC, ABC, CS, FA, TLBO
Bài toán  : Sphere, Rastrigin, Ackley, Rosenbrock, Griewank

Output: results/robustness_continuous/
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

OUT_DIR = os.path.join(BASE_DIR, 'results', 'robustness_continuous')
os.makedirs(OUT_DIR, exist_ok=True)

DIM        = 10
ITERATIONS = 100    
N_RUNS     = 30

BG_DARK  = '#0d1117'
AX_DARK  = '#161b22'
TEXT_CLR = '#e6edf3'
GRID_CLR = '#30363d'
MID_CLR  = '#8b949e'

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
ALGO_LABELS = list(ALGO_COLORS.keys())

def _style_ax(ax):
    ax.set_facecolor(AX_DARK)
    ax.tick_params(colors=MID_CLR, labelsize=9)
    ax.xaxis.label.set_color(MID_CLR)
    ax.yaxis.label.set_color(MID_CLR)
    ax.title.set_color(TEXT_CLR)
    for sp in ax.spines.values():
        sp.set_color(GRID_CLR)
    ax.grid(axis='y', color=GRID_CLR, linestyle='--', linewidth=0.5, alpha=0.6)

def _savefig(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved -> {p}')
    return p

def run_algo(algo_key, problem):
    if algo_key == 'GA':
        from algorithms.evolution.GA import GeneticAlgorithm
        alg = GeneticAlgorithm(problem, params={'pop_size': 40})
    elif algo_key == 'PSO':
        from algorithms.biology.PSO import ParticleSwarmOptimization
        alg = ParticleSwarmOptimization(problem, params={'pop_size': 40})
    elif algo_key == 'DE':
        from algorithms.evolution.DE import DifferentialEvolution
        alg = DifferentialEvolution(problem, params={'pop_size': 40})
    elif algo_key == 'SA':
        from algorithms.physics.SA import SimulatedAnnealing
        alg = SimulatedAnnealing(problem, params={'T0': 1.0, 'alpha': 0.95})
    elif algo_key == 'HC':
        from algorithms.classical.Hill_climbing import HillClimbing
        alg = HillClimbing(problem, params={'restarts': 5})
    elif algo_key == 'ABC':
        from algorithms.biology.ABC import ArtificialBeeColony
        alg = ArtificialBeeColony(problem, params={'pop_size': 40, 'limit': 20})
    elif algo_key == 'CS':
        from algorithms.biology.CS import CuckooSearch
        alg = CuckooSearch(problem, params={'n': 40})
    elif algo_key == 'FA':
        from algorithms.biology.FA import FireflyAlgorithm
        alg = FireflyAlgorithm(problem, params={'pop_size': 20})
    elif algo_key == 'TLBO':
        from algorithms.human.TLBO import TLBO
        alg = TLBO(problem, params={'pop_size': 40})
    else:
        raise ValueError()

    alg.solve(iterations=ITERATIONS)
    return alg.best_fitness

def draw_boxplot(ax, data_dict, ylabel, title, higher_is_better=False, log_scale=False):
    labels = list(data_dict.keys())
    arrays = [data_dict[l] for l in labels]
    colors = [ALGO_COLORS[l] for l in labels]
    x_pos  = np.arange(1, len(labels) + 1)

    _style_ax(ax)

    if log_scale:
        ax.set_yscale('log')

    bp = ax.boxplot(
        arrays,
        positions=x_pos,
        patch_artist=True,
        widths=0.5,
        medianprops={'color': '#ffffff', 'linewidth': 2.0},
        whiskerprops={'color': MID_CLR, 'linewidth': 1.2},
        capprops={'color': MID_CLR, 'linewidth': 1.2},
        flierprops={'marker': '', 'markersize': 0}, 
    )

    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor('#ffffff')
        patch.set_linewidth(1.0)

    rng_jitter = np.random.default_rng(0)
    for xi, (arr, c) in enumerate(zip(arrays, colors), start=1):
        jitter = rng_jitter.uniform(-0.15, 0.15, size=len(arr))
        # Remove inf/nan for scatter
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            ax.scatter(xi + jitter[:len(valid)], valid, color=c, alpha=0.4, s=12, zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

def main():
    t0 = time.time()
    print('=' * 60)
    print(f'  Robustness Analysis -- Continuous Problems ({N_RUNS} runs)')
    print('=' * 60)

    from problems.continuous.Sphere     import Sphere
    from problems.continuous.Rastrigin  import Rastrigin
    from problems.continuous.Ackley     import Ackley
    from problems.continuous.Rosenbrock import Rosenbrock
    from problems.continuous.Griewank   import Griewank

    probs = {
        'Sphere': Sphere(dim=DIM),
        'Rastrigin': Rastrigin(dim=DIM),
        'Ackley': Ackley(dim=DIM),
        'Rosenbrock': Rosenbrock(dim=DIM),
        'Griewank': Griewank(dim=DIM),
    }

    all_data = {}

    for p_name, prob in probs.items():
        print(f'\n-> Running {p_name} ({N_RUNS} runs per algo)...')
        all_data[p_name] = {}
        
        for algo in ALGO_LABELS:
            print(f'   {algo:4} ', end='', flush=True)
            vals = []
            tt = time.time()
            for _ in range(N_RUNS):
                v = run_algo(algo, prob)
                vals.append(v)
            arr = np.array(vals)
            all_data[p_name][algo] = arr
            print(f'done ({time.time()-tt:.1f}s) | med={np.median(arr):.3e} std={arr.std():.3e}')

        # Save individual chart
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_DARK)
        draw_boxplot(
            ax, all_data[p_name],
            ylabel='Final Best Fitness (Log scale)' if p_name != 'Rastrigin' else 'Final Best Fitness',
            title=f'{p_name} Robustness -- {N_RUNS} runs (dim={DIM}, {ITERATIONS} iters)',
            higher_is_better=False,
            log_scale=True # Use log scale for most continuous to see small differences
        )
        # Check if min fitness is <= 0 for log scale issues
        min_fit = min([min(arr) for arr in all_data[p_name].values()])
        if min_fit <= 0:
            ax.set_yscale('linear') # fallback
            ax.set_ylabel('Final Best Fitness')
        
        _savefig(fig, f'robustness_cont_{p_name.lower()}.png')

    # Grid 
    print('\n-> Generating 2x3 Grid Dashboard...')
    fig = plt.figure(figsize=(22, 12), facecolor=BG_DARK)
    fig.suptitle(
        f'Robustness Analysis -- Continuous Problems (dim={DIM}, {ITERATIONS} iters, {N_RUNS} runs)',
        fontsize=18, fontweight='bold', color=TEXT_CLR, y=1.02
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.25)
    
    for idx, p_name in enumerate(probs.keys()):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        
        min_fit = min([min(arr) for arr in all_data[p_name].values()])
        use_log = (min_fit > 0 and p_name != 'Rastrigin')
        
        draw_boxplot(ax, all_data[p_name], 
                     ylabel='Fitness', title=p_name, log_scale=use_log)

    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.set_facecolor(AX_DARK)
    for sp in ax_leg.spines.values(): sp.set_color(GRID_CLR)
    ax_leg.set_xticks([]); ax_leg.set_yticks([])
    ax_leg.text(0.5, 0.5, 'Boxplots highlight variance across runs.\nNarrow box = Highly robust/stable.',
                ha='center', va='center', color=TEXT_CLR, fontsize=14, linespacing=1.8)

    _savefig(fig, 'dashboard_robustness_continuous.png')
    
    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
