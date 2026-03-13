import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

BG = '#0d1117'
PANEL = '#161b22'
GRID_C = '#30363d'
TXT = '#c9d1d9'
BLUE = '#58a6ff'
CYAN = '#39c5cf'
PURPLE = '#bc8cff'
GREEN = '#3fb950'
RED = '#ff7b72'
YELLOW = '#d29922'
ORANGE = '#f0883e'

def _style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    if title:
        ax.set_title(title, color=TXT, fontsize=11, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)

def visualize_knapsack_optimization(problem, adapter, AlgClass, algo_name='GA',
                                    params=None, iterations=150, interval=200,
                                    save_path=None, show=True, max_frames=None):
    algo = AlgClass(adapter, params=params or {})
    algo.solve(iterations=iterations)
    history = algo.history
    if not history:
        raise ValueError('Algorithm produced no history snapshots')

    n_history = len(history)
    hold = 15
    total_frames = n_history + hold
    if max_frames is not None:
        if n_history > max_frames - hold:
            indices = np.linspace(0, n_history - 1, max_frames - hold,
                                  dtype=int).tolist()
            history = [history[i] for i in indices]
            n_history = len(history)
            total_frames = n_history + hold

    fig = plt.figure(figsize=(22, 13))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22,
                          height_ratios=[1, 1], width_ratios=[1.2, 1])
    # Give table full height, remove description panel
    ax_table = fig.add_subplot(gs[:, 0])
    ax_bar   = fig.add_subplot(gs[0, 1])
    ax_conv  = fig.add_subplot(gs[1, 1])

    # ── Convergence (static setup) ───────────────────────────────
    conv_data = []
    _style_ax(ax_conv, title='Convergence (Total Value)',
              xlabel='Iteration', ylabel='Value')
    ax_conv.grid(True, alpha=0.15, color=GRID_C)
    conv_line, = ax_conv.plot([], [], color=BLUE, linewidth=1.5)

    # ── Pre-compute helpers ──────────────────────────────────────
    item_names  = [it.name[:12] for it in problem.items]
    item_w      = [it.weight for it in problem.items]
    item_v      = [it.volume for it in problem.items]
    item_pw     = [it.power  for it in problem.items]
    item_val    = [it.value  for it in problem.items]
    item_pri    = [it.priority_class[:4] for it in problem.items]

    def update(frame):
        ax_bar.clear()
        ax_table.clear()
        ax_bar.set_facecolor(PANEL)
        ax_table.set_facecolor(PANEL)
        ax_table.axis('off')

        idx = min(frame, n_history - 1)
        snap = history[idx]

        best_sol = snap.get('global_best_sol')
        bfit = snap.get('global_best_fit', 0)
        display_val = -bfit if bfit is not None else 0

        if best_sol is not None:
            sol_arr = np.asarray(best_sol)
            sel_ids = set(int(i) for i in range(len(sol_arr)) if sol_arr[i] > 0.5)
        else:
            sel_ids = set()

        sel_bin = [1 if i in sel_ids else 0 for i in range(problem.n_items)]
        result = problem.evaluate(sel_bin)

        # ── Resource bars ────────────────────────────────────────
        names = ['Weight', 'Volume', 'Power']
        caps  = [problem.capacity_weight, problem.capacity_volume, problem.capacity_power]
        used  = [result['weight']['used'], result['volume']['used'], result['power']['used']]
        colors = [BLUE, PURPLE, ORANGE]
        yp = np.arange(3)

        ax_bar.barh(yp, caps, height=0.5, color='#21262d', edgecolor=GRID_C)
        bc = [RED if used[i] > caps[i] else colors[i] for i in range(3)]
        ax_bar.barh(yp, used, height=0.5, color=bc, alpha=0.85)

        for i in range(3):
            pct = used[i] / caps[i] * 100 if caps[i] > 0 else 0
            status = '!! OVER' if used[i] > caps[i] else 'OK'
            ax_bar.text(max(used[i], caps[i]) + 2, yp[i],
                        f'{used[i]:.0f}/{caps[i]:.0f} ({pct:.0f}%) {status}',
                        va='center', color=RED if used[i] > caps[i] else '#c9d1d9',
                        fontsize=9, fontweight='bold' if used[i] > caps[i] else 'normal')

        ax_bar.set_yticks(yp)
        ax_bar.set_yticklabels(names, color='#c9d1d9')
        ax_bar.tick_params(colors='#8b949e', labelsize=10)
        for sp in ax_bar.spines.values():
            sp.set_color(GRID_C)
        ax_bar.set_title(f'Resource Usage  |  {len(sel_ids)} items selected',
                         color=TXT, fontsize=11, fontweight='bold', pad=8)

        # ── Item table ───────────────────────────────────────────
        hdr = (
            f"  Item Selection Table\n"
            f"  {'─'*60}\n"
            f"  {'ID':>3s} {'Name':<14s} {'W':>6s} {'V':>6s} {'P':>6s}"
            f" {'Val':>6s} {'Pri':>5s} {'Sel':>3s}\n"
            f"  {'─'*60}"
        )
        rows = []
        for j in range(problem.n_items):
            sel_mark = '✓' if j in sel_ids else '·'
            color_mark = '★' if item_pri[j] == 'crit' else ' '
            rows.append(
                f"  {j:>3d} {item_names[j]:<14s} {item_w[j]:>6.0f}"
                f" {item_v[j]:>6.0f} {item_pw[j]:>6.0f}"
                f" {item_val[j]:>6.0f} {item_pri[j]:>5s} {sel_mark}{color_mark}")

        if len(rows) > 35:
            shown = rows[:35]
            shown.append(f'  ... (+{len(rows)-35} more items)')
        else:
            shown = rows

        syn_active = result.get('active_synergies', [])
        viol = result.get('violations', {})
        conflict_viol = viol.get('conflicts', [])

        footer_parts = [f"  {'─'*60}"]
        footer_parts.append(
            f"  Base: {result['base_value']:.0f}  "
            f"Synergy: +{result.get('synergy_bonus', 0):.0f}  "
            f"Penalty: -{result['penalty']:.0f}  "
            f"= {result['total_value']:.0f}")

        if syn_active:
            footer_parts.append(f"  Active synergies: {len(syn_active)}")
        if conflict_viol:
            footer_parts.append(f"  !! Conflicts: {len(conflict_viol)}")
        if 'critical_items' in viol:
            ci = viol['critical_items']
            footer_parts.append(
                f"  !! Critical items: {ci['selected']}/{ci['required']}")

        table_text = hdr + '\n' + '\n'.join(shown) + '\n' + '\n'.join(footer_parts)
        ax_table.text(
            0.05, 0.98, table_text, transform=ax_table.transAxes,
            va='top', ha='left', fontfamily='monospace', fontsize=8.5,
            color='#c9d1d9',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#21262d', edgecolor=GRID_C))

        if frame < n_history:
            conv_data.append(display_val)
        conv_line.set_data(range(1, len(conv_data) + 1), conv_data)
        ax_conv.relim()
        ax_conv.autoscale_view()

        it_s = min(frame + 1, n_history)
        feasible = '✓ Feasible' if result['feasible'] else '✗ VIOLATED'
        fig.suptitle(
            f'{algo_name} — Knapsack  |  Iter {it_s}/{n_history}  |  '
            f'Value: {display_val:.1f}  |  Items: {len(sel_ids)}  |  {feasible}',
            color=TXT, fontsize=14, fontweight='bold', y=0.99)
        return []

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=interval, blit=False, repeat=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        try:
            ani.save(save_path, writer=animation.PillowWriter(fps=max(1, 1000 // interval)), dpi=100)
        except Exception as e:
            pass
    if show:
        plt.show()
    else:
        plt.close(fig)
    return ani
