"""
Parameter Sensitivity Analysis - Continuous Problems
======================================================
  Sphere      + PSO  : trọng số quán tính w         = [0.2, 0.7, 1.2]
  Rastrigin   + ABC  : giới hạn từ bỏ    limit      = [10,  50,  100]
  Ackley      + FA   : hệ số hấp thụ    γ (gamma)   = [0.01, 0.1, 1.0]
  Rosenbrock  + DE   : hệ số đột biến   F            = [0.2,  0.5, 0.9]
  Griewank    + TLBO : kích thước quần thể pop_size  = [10,   30,  100]

Output: results/param_sensitivity_continuous/
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Ensure repo root is importable ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUT_DIR = os.path.join(BASE_DIR, 'results', 'param_sensitivity_continuous')
os.makedirs(OUT_DIR, exist_ok=True)

SEED      = 42
DIM       = 10       # dimension for all continuous problems
ITERATIONS = 100
N_RUNS    = 3        # independent runs per config

# ══════════════════════════════════════════════════════════════════════
#  Visual style
# ══════════════════════════════════════════════════════════════════════
COLORS   = ['#58a6ff', '#ffd93d', '#ff6b6b', '#6bcb77', '#a371f7', '#f0883e', '#00d4aa']
BG_DARK  = '#0d1117'
AX_DARK  = '#161b22'
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
#  Experiment runners
# ══════════════════════════════════════════════════════════════════════

def run_pso_sphere(prob, w_val, n_runs=N_RUNS):
    from algorithms.biology.PSO import ParticleSwarmOptimization
    curves = []
    for _ in range(n_runs):
        params = {'pop_size': 40, 'w': w_val, 'c1': 1.5, 'c2': 1.5}
        alg = ParticleSwarmOptimization(prob, params=params)
        alg.solve(iterations=ITERATIONS)
        curves.append([h['global_best_fit'] for h in alg.history])
    return curves


def run_abc_rastrigin(prob, limit_val, n_runs=N_RUNS):
    from algorithms.biology.ABC import ArtificialBeeColony
    curves = []
    for _ in range(n_runs):
        params = {'pop_size': 40, 'limit': limit_val}
        alg = ArtificialBeeColony(prob, params=params)
        alg.solve(iterations=ITERATIONS)
        curves.append([h['global_best_fit'] for h in alg.history])
    return curves


def run_fa_ackley(prob, gamma_val, n_runs=N_RUNS):
    from algorithms.biology.FA import FireflyAlgorithm
    curves = []
    for _ in range(n_runs):
        # FA is O(N²) so keep pop small for speed
        params = {'pop_size': 20, 'alpha': 0.5, 'beta0': 1.0, 'gamma': gamma_val}
        alg = FireflyAlgorithm(prob, params=params)
        alg.solve(iterations=ITERATIONS)
        curves.append([h['global_best_fit'] for h in alg.history])
    return curves


def run_de_rosenbrock(prob, F_val, n_runs=N_RUNS):
    from algorithms.evolution.DE import DifferentialEvolution
    curves = []
    for _ in range(n_runs):
        params = {'pop_size': 40, 'F': F_val, 'CR': 0.9}
        alg = DifferentialEvolution(prob, params=params)
        alg.solve(iterations=ITERATIONS)
        curves.append([h['global_best_fit'] for h in alg.history])
    return curves


def run_tlbo_griewank(prob, pop_size_val, n_runs=N_RUNS):
    from algorithms.human.TLBO import TLBO
    curves = []
    for _ in range(n_runs):
        params = {'pop_size': pop_size_val}
        alg = TLBO(prob, params=params)
        alg.solve(iterations=ITERATIONS)
        curves.append([h['global_best_fit'] for h in alg.history])
    return curves


# ══════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════

def plot_sensitivity(experiments, title, ylabel, param_name, param_values,
                     filename, log_scale=False):
    """
    experiments : list[list[list[float]]]  # [param_idx][run_idx][iter]
    """
    n_params = len(param_values)
    iters    = np.arange(1, ITERATIONS + 1)

    fig = plt.figure(figsize=(18, 12), facecolor=BG_DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    # ── Top row: per-param convergence plots ────────────────────────
    ax_top = [fig.add_subplot(gs[0, i]) for i in range(n_params)]

    for pi, (pv, curves) in enumerate(zip(param_values, experiments)):
        ax = ax_top[pi]
        _style_ax(ax)
        arr   = np.array(curves)             # (n_runs, ITERATIONS)
        mean_ = arr.mean(axis=0)
        std_  = arr.std(axis=0)

        ax.fill_between(iters, mean_ - std_, mean_ + std_,
                         alpha=0.20, color=COLORS[pi], zorder=1)
        for run_curve in curves:
            ax.plot(iters, run_curve, color=COLORS[pi], alpha=0.30,
                    linewidth=0.9, zorder=2)
        ax.plot(iters, mean_, color=COLORS[pi], linewidth=2.2, zorder=3,
                label='Mean ± Std')

        if log_scale:
            ax.set_yscale('symlog', linthresh=1e-6)

        ax.set_title(f'{param_name} = {pv}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, facecolor=AX_DARK, labelcolor=TEXT_CLR,
                  edgecolor=GRID_CLR)

    # ── Bottom-left: overlay comparison ─────────────────────────────
    ax_ov = fig.add_subplot(gs[1, 0:2])
    _style_ax(ax_ov)
    if log_scale:
        ax_ov.set_yscale('symlog', linthresh=1e-6)

    legend_handles = []
    for pi, (pv, curves) in enumerate(zip(param_values, experiments)):
        arr   = np.array(curves)
        mean_ = arr.mean(axis=0)
        std_  = arr.std(axis=0)
        ax_ov.fill_between(iters, mean_ - std_, mean_ + std_,
                             alpha=0.12, color=COLORS[pi])
        line, = ax_ov.plot(iters, mean_, color=COLORS[pi], linewidth=2.2,
                            label=f'{param_name} = {pv}')
        legend_handles.append(line)

    ax_ov.set_title(f'Convergence Comparison - {param_name}', fontsize=12,
                     fontweight='bold')
    ax_ov.set_xlabel('Iteration')
    ax_ov.set_ylabel(ylabel)
    ax_ov.legend(handles=legend_handles, fontsize=9,
                  facecolor=AX_DARK, labelcolor=TEXT_CLR, edgecolor=GRID_CLR)

    # ── Bottom-right: final distribution (box plot) ─────────────────
    ax_bx = fig.add_subplot(gs[1, 2])
    _style_ax(ax_bx)

    finals = [np.array(curves)[:, -1] for curves in experiments]
    bp = ax_bx.boxplot(finals,
                        patch_artist=True,
                        medianprops={'color': '#ffffff', 'linewidth': 2.0},
                        whiskerprops={'color': '#8b949e'},
                        capprops={'color': '#8b949e'},
                        flierprops={'markerfacecolor': '#8b949e',
                                    'marker': 'o', 'markersize': 4})
    for patch, c in zip(bp['boxes'], COLORS[:n_params]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax_bx.set_xticks(range(1, n_params + 1))
    ax_bx.set_xticklabels([str(v) for v in param_values], fontsize=9)
    ax_bx.set_xlabel(param_name)
    ax_bx.set_ylabel(f'Final {ylabel}')
    ax_bx.set_title('Final Value Distribution', fontsize=11, fontweight='bold')
    if log_scale:
        ax_bx.set_yscale('symlog', linthresh=1e-6)

    fig.suptitle(title, fontsize=15, fontweight='bold', color=TEXT_CLR, y=1.01)
    return _savefig(fig, filename)


def plot_summary_dashboard(all_results, filename='summary_continuous.png'):
    """Bar chart dashboard across all 5 experiments."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 13), facecolor=BG_DARK)
    fig.suptitle('Parameter Sensitivity - Continuous Problems Dashboard',
                 fontsize=17, fontweight='bold', color=TEXT_CLR, y=1.01)

    configs = [
        ('Sphere + PSO',      'w (Inertia)',         all_results[0], False),
        ('Rastrigin + ABC',   'limit (Abandonment)', all_results[1], False),
        ('Ackley + FA',       'γ (Absorption)',      all_results[2], False),
        ('Rosenbrock + DE',   'F (Mutation)',         all_results[3], False),
        ('Griewank + TLBO',   'pop_size',             all_results[4], False),
    ]

    for idx, (name, param_label, res, higher) in enumerate(configs):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        _style_ax(ax)

        pvs   = res['param_values']
        means = np.array(res['means_final'])
        stds  = np.array(res['stds_final'])
        bests = np.array(res['best_finals'])
        x     = np.arange(len(pvs))

        ax.bar(x, means, color=COLORS[:len(pvs)], alpha=0.75,
               edgecolor='white', linewidth=0.6, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt='none',
                    color='#ffffff', capsize=5, linewidth=1.5, zorder=4)

        scale = max(means) - min(means) if max(means) != min(means) else 1.0
        for xi, (mn, be, sd) in enumerate(zip(means, bests, stds)):
            sym = '▲' if higher else '▼'
            ax.annotate(
                f'{sym}{be:.3g}\nμ={mn:.3g}',
                xy=(xi, mn + sd + scale * 0.05),
                ha='center', va='bottom', fontsize=8, color=TEXT_CLR
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f'{param_label}={v}' for v in pvs],
                            fontsize=7, rotation=10)
        ax.set_ylabel('Final Best Fitness', fontsize=9)
        ax.set_title(name, fontsize=12, fontweight='bold')

    # Hide the unused 6th subplot
    axes[1][2].axis('off')
    axes[1][2].set_facecolor(BG_DARK)

    plt.tight_layout()
    return _savefig(fig, filename)


# ══════════════════════════════════════════════════════════════════════
#  Console report
# ══════════════════════════════════════════════════════════════════════

def print_analysis(label, param_name, param_values, experiments):
    print(f'\n{"─"*60}')
    print(f'  {label}  |  param: {param_name}')
    print(f'{"─"*60}')
    finals = []
    for pv, curves in zip(param_values, experiments):
        arr = np.array(curves)[:, -1]
        mu = arr.mean(); sd = arr.std(); be = arr.min()
        finals.append(mu)
        print(f'  {param_name}={str(pv):>6}  mean={mu:12.4g}  std={sd:10.4g}'
              f'  best={be:12.4g}')
    best_idx = int(np.argmin(finals))
    print(f'\n  → Best setting: {param_name} = {param_values[best_idx]}'
          f'  (mean final = {finals[best_idx]:.4g})')


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print('=' * 60)
    print('  Parameter Sensitivity - Continuous Problems')
    print('=' * 60)

    summary = []

    # ──────────────────────────────────────────────────────────────
    # 1.  Sphere  +  PSO  :  w  ∈ {0.2, 0.7, 1.2}
    # ──────────────────────────────────────────────────────────────
    print('\n[1/5] Sphere + PSO - inertia weight w')
    from problems.continuous.Sphere import Sphere
    sphere = Sphere(dim=DIM)

    w_values = [0.2, 0.7, 1.2]
    pso_exp  = []
    for w in w_values:
        print(f'       w = {w} …', end=' ', flush=True)
        t1 = time.time()
        crv = run_pso_sphere(sphere, w)
        pso_exp.append(crv)
        print(f'done ({time.time()-t1:.1f}s)')

    plot_sensitivity(
        pso_exp,
        title='Sphere Function + PSO - Inertia Weight w',
        ylabel='Best Fitness (log)',
        param_name='w',
        param_values=w_values,
        filename='sphere_pso_w.png',
        log_scale=True,
    )
    print_analysis('Sphere + PSO', 'w', w_values, pso_exp)

    finals_pso = [np.array(c)[:, -1] for c in pso_exp]
    summary.append({
        'param_values': w_values,
        'means_final':  [f.mean() for f in finals_pso],
        'stds_final':   [f.std()  for f in finals_pso],
        'best_finals':  [f.min()  for f in finals_pso],
    })

    # ──────────────────────────────────────────────────────────────
    # 2.  Rastrigin  +  ABC  :  limit  ∈ {10, 50, 100}
    # ──────────────────────────────────────────────────────────────
    print('\n[2/5] Rastrigin + ABC - abandonment limit')
    from problems.continuous.Rastrigin import Rastrigin
    rastrigin = Rastrigin(dim=DIM)

    limit_values = [10, 50, 100]
    abc_exp      = []
    for lim in limit_values:
        print(f'       limit = {lim} …', end=' ', flush=True)
        t1 = time.time()
        crv = run_abc_rastrigin(rastrigin, lim)
        abc_exp.append(crv)
        print(f'done ({time.time()-t1:.1f}s)')

    plot_sensitivity(
        abc_exp,
        title='Rastrigin Function + ABC - Scout Abandonment Limit',
        ylabel='Best Fitness',
        param_name='limit',
        param_values=limit_values,
        filename='rastrigin_abc_limit.png',
        log_scale=False,
    )
    print_analysis('Rastrigin + ABC', 'limit', limit_values, abc_exp)

    finals_abc = [np.array(c)[:, -1] for c in abc_exp]
    summary.append({
        'param_values': limit_values,
        'means_final':  [f.mean() for f in finals_abc],
        'stds_final':   [f.std()  for f in finals_abc],
        'best_finals':  [f.min()  for f in finals_abc],
    })

    # ──────────────────────────────────────────────────────────────
    # 3.  Ackley  +  FA  :  gamma  ∈ {0.01, 0.1, 1.0}
    # ──────────────────────────────────────────────────────────────
    print('\n[3/5] Ackley + FA - absorption coefficient γ')
    from problems.continuous.Ackley import Ackley
    ackley = Ackley(dim=DIM)

    gamma_values = [0.01, 0.1, 1.0]
    fa_exp       = []
    for gamma in gamma_values:
        print(f'       γ = {gamma} …', end=' ', flush=True)
        t1 = time.time()
        crv = run_fa_ackley(ackley, gamma)
        fa_exp.append(crv)
        print(f'done ({time.time()-t1:.1f}s)')

    plot_sensitivity(
        fa_exp,
        title='Ackley Function + Firefly Algorithm - Absorption Coefficient γ',
        ylabel='Best Fitness',
        param_name='γ',
        param_values=gamma_values,
        filename='ackley_fa_gamma.png',
        log_scale=False,
    )
    print_analysis('Ackley + FA', 'γ', gamma_values, fa_exp)

    finals_fa = [np.array(c)[:, -1] for c in fa_exp]
    summary.append({
        'param_values': gamma_values,
        'means_final':  [f.mean() for f in finals_fa],
        'stds_final':   [f.std()  for f in finals_fa],
        'best_finals':  [f.min()  for f in finals_fa],
    })

    # ──────────────────────────────────────────────────────────────
    # 4.  Rosenbrock  +  DE  :  F  ∈ {0.2, 0.5, 0.9}
    # ──────────────────────────────────────────────────────────────
    print('\n[4/5] Rosenbrock + DE - mutation factor F')
    from problems.continuous.Rosenbrock import Rosenbrock
    rosenbrock = Rosenbrock(dim=DIM)

    F_values = [0.2, 0.5, 0.9]
    de_exp   = []
    for F in F_values:
        print(f'       F = {F} …', end=' ', flush=True)
        t1 = time.time()
        crv = run_de_rosenbrock(rosenbrock, F)
        de_exp.append(crv)
        print(f'done ({time.time()-t1:.1f}s)')

    plot_sensitivity(
        de_exp,
        title='Rosenbrock Function + Differential Evolution - Mutation Factor F',
        ylabel='Best Fitness (log)',
        param_name='F',
        param_values=F_values,
        filename='rosenbrock_de_F.png',
        log_scale=True,
    )
    print_analysis('Rosenbrock + DE', 'F', F_values, de_exp)

    finals_de = [np.array(c)[:, -1] for c in de_exp]
    summary.append({
        'param_values': F_values,
        'means_final':  [f.mean() for f in finals_de],
        'stds_final':   [f.std()  for f in finals_de],
        'best_finals':  [f.min()  for f in finals_de],
    })

    # ──────────────────────────────────────────────────────────────
    # 5.  Griewank  +  TLBO  :  pop_size  ∈ {10, 30, 100}
    # ──────────────────────────────────────────────────────────────
    print('\n[5/5] Griewank + TLBO - population size (TLBO is parameter-free)')
    from problems.continuous.Griewank import Griewank
    griewank = Griewank(dim=DIM)

    pop_values = [10, 30, 100]
    tlbo_exp   = []
    for ps in pop_values:
        print(f'       pop_size = {ps} …', end=' ', flush=True)
        t1 = time.time()
        crv = run_tlbo_griewank(griewank, ps)
        tlbo_exp.append(crv)
        print(f'done ({time.time()-t1:.1f}s)')

    plot_sensitivity(
        tlbo_exp,
        title='Griewank Function + TLBO - Population Size\n'
              '(TLBO is parameter-free; pop_size is the only tunable hyperparameter)',
        ylabel='Best Fitness (log)',
        param_name='pop_size',
        param_values=pop_values,
        filename='griewank_tlbo_popsize.png',
        log_scale=True,
    )
    print_analysis('Griewank + TLBO', 'pop_size', pop_values, tlbo_exp)

    finals_tl = [np.array(c)[:, -1] for c in tlbo_exp]
    summary.append({
        'param_values': pop_values,
        'means_final':  [f.mean() for f in finals_tl],
        'stds_final':   [f.std()  for f in finals_tl],
        'best_finals':  [f.min()  for f in finals_tl],
    })

    # ──────────────────────────────────────────────────────────────
    # 6.  Combined dashboard
    # ──────────────────────────────────────────────────────────────
    print('\n[6/6] Generating combined summary dashboard …')
    plot_summary_dashboard(summary)

    elapsed = time.time() - t0
    print(f'\n{"═"*60}')
    print(f'  All done in {elapsed:.1f}s')
    print(f'  Output → {OUT_DIR}')
    print(f'{"═"*60}')


if __name__ == '__main__':
    main()
