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

def visualize_graph_coloring_optimization(problem, adapter, AlgClass,
                                          algo_name='GA',
                                          params=None, iterations=150,
                                          interval=200,
                                          save_path=None, show=True,
                                          max_frames=None):
    PALETTE = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#a371f7',
               '#f0883e', '#00d4aa', '#ff9ff3', '#54a0ff', '#5f27cd',
               '#01a3a4', '#ee5a24', '#778beb', '#f8a5c2', '#63cdda',
               '#cf6a87']

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
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22,
                          height_ratios=[1.2, 1], width_ratios=[1.3, 1])
    # Give the Graph the whole left side
    ax_graph = fig.add_subplot(gs[:, 0])
    ax_asgn  = fig.add_subplot(gs[0, 1])
    ax_conv  = fig.add_subplot(gs[1, 1])

    _style_ax(ax_conv, title='Convergence (Penalty)',
              xlabel='Iteration', ylabel='Penalty')
    ax_conv.grid(True, alpha=0.15, color=GRID_C)
    conv_data = []
    conv_line, = ax_conv.plot([], [], color=PURPLE, linewidth=1.5)

    def update(frame):
        ax_graph.clear()
        ax_asgn.clear()
        ax_graph.set_facecolor(PANEL)
        ax_asgn.set_facecolor(PANEL)
        ax_asgn.axis('off')

        idx = min(frame, n_history - 1)
        snap = history[idx]

        best_sol = snap.get('global_best_sol')
        bfit = snap.get('global_best_fit', 0)

        if best_sol is not None:
            sol_arr = np.asarray(best_sol)
            coloring = {i: max(0, min(int(v), problem.n_colors - 1))
                        for i, v in enumerate(sol_arr)}
        else:
            coloring = {i: 0 for i in range(problem.n_nodes)}

        result = problem.evaluate(coloring)

        # ── Draw edges ───────────────────────────────────────────
        for edge in problem.edges:
            pu = problem.node_positions[edge.u]
            pv = problem.node_positions[edge.v]
            violated = coloring[edge.u] == coloring[edge.v]
            if violated:
                c, lw, a = '#ff4444', 3.0, 0.9
            elif edge.is_hard:
                c, lw, a = '#484f58', 0.8 + edge.weight * 0.3, 0.4
            else:
                c, lw, a = '#30363d', 0.6, 0.25
            ax_graph.plot([pu[0], pv[0]], [pu[1], pv[1]],
                          color=c, lw=lw, alpha=a, zorder=1)
            # mark violated edges with X
            if violated:
                mx = (pu[0] + pv[0]) / 2
                my = (pu[1] + pv[1]) / 2
                ax_graph.text(mx, my, '✗', color='#ff4444',
                              fontsize=10, fontweight='bold',
                              ha='center', va='center', zorder=8)

        # ── Draw forbidden pairs ─────────────────────────────────
        for fp in problem.forbidden_pairs:
            pu = problem.node_positions[fp.u]
            pv = problem.node_positions[fp.v]
            ax_graph.plot([pu[0], pv[0]], [pu[1], pv[1]],
                          color=YELLOW, lw=1.2, linestyle=':',
                          alpha=0.35, zorder=1)

        # ── Draw nodes with event names ──────────────────────────
        for ev in problem.events:
            pos = problem.node_positions[ev.id]
            sz = 120 + ev.popularity * 15
            co = PALETTE[coloring[ev.id] % len(PALETTE)]
            ec = YELLOW if ev.pre_assigned_color is not None else 'white'
            ew = 2.5 if ev.pre_assigned_color is not None else 0.8
            ax_graph.scatter(pos[0], pos[1], s=sz, c=co, zorder=5,
                             edgecolors=ec, linewidths=ew)
            nm = ev.name[:8] if hasattr(ev, 'name') and ev.name else str(ev.id)
            ax_graph.annotate(
                f'{ev.id}:{nm}', pos,
                textcoords='offset points', xytext=(8, 8),
                fontsize=8, color='#c9d1d9', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', alpha=0.7, edgecolor='none'))

        ax_graph.set_aspect('equal')
        ax_graph.axis('off')

        n_violations = result['n_violations']
        colors_used  = result['colors_used']
        feasible     = '✓ OK' if result['feasible'] else '✗ VIOLATED'

        it_s = min(frame + 1, n_history)
        ax_graph.set_title(
            f'{algo_name}  |  Iter {it_s}/{n_history}  |  '
            f'Colors: {colors_used}  |  Violations: {n_violations}'
            f'  |  {feasible}',
            color=TXT, fontsize=12, fontweight='bold', pad=10)

        # ── Color legend + assignment table ──────────────────────
        legend_parts = ['  Color Legend (timeslot)']
        legend_parts.append(f"  {'─'*50}")
        for ci in range(problem.n_colors):
            sym = '█'*2
            nodes_in = [n for n in range(problem.n_nodes) if coloring[n] == ci]
            legend_parts.append(
                f"  {sym} Slot {ci}: "
                f"{len(nodes_in)} events")
        legend_parts.append(f"\n  {'─'*50}")
        legend_parts.append(f"  Assignment Table")
        legend_parts.append(f"  {'─'*50}")
        legend_parts.append(f"  {'ID':>3s} {'Event':<14s} {'Slot':>4s} {'Pop':>4s} {'Fix':>3s}")
        legend_parts.append(f"  {'─'*50}")

        shown = 0
        for ev in problem.events:
            if shown > 15:
                legend_parts.append(f"  ... (+{len(problem.events)-15} more)")
                break
            nm = ev.name[:14] if hasattr(ev, 'name') and ev.name else str(ev.id)
            sl = coloring[ev.id]
            fx = ' Y ' if ev.pre_assigned_color is not None else ' - '
            legend_parts.append(f"  {ev.id:>3d} {nm:<14s} {sl:>4d} {ev.popularity:>4.1f} {fx:>3s}")
            shown += 1

        legend_text = '\n'.join(legend_parts)
        ax_asgn.text(0.05, 0.98, legend_text, transform=ax_asgn.transAxes,
                     va='top', ha='left', fontfamily='monospace', fontsize=8.5,
                     color='#c9d1d9',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor=GRID_C))

        if frame < n_history:
            conv_data.append(bfit)
        conv_line.set_data(range(1, len(conv_data) + 1), conv_data)
        ax_conv.relim()
        ax_conv.autoscale_view()

        fig.suptitle(
            f'{algo_name} — Graph Coloring  |  Iter {it_s}/{n_history}  |  '
            f'Penalty: {bfit:.1f}  |  {feasible}',
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
