"""
Visualization module for comparing search algorithms across categories.

All visualizations run in **real-time** with side-by-side parallel animation
so you can watch every algorithm explore simultaneously.

Categories:
  1. Uninformed Search  (BFS, DFS)        -- grid exploration without heuristic
  2. Informed Search    (A*)              -- grid exploration with heuristic
  3. Local Search       (HC, SA, GA, DE, PSO, ABC, CS, FA, TLBO) -- continuous optimization

Provides:
  - visualize_uninformed_search : real-time parallel BFS/DFS on grid
  - visualize_informed_search   : real-time A* on grid
  - visualize_local_search      : real-time parallel metaheuristics on 2D contour
  - animate_grid_search         : single algorithm grid animation
  - animate_local_search        : single algorithm contour animation
  - visualize_all_categories    : real-time dashboard with all three categories
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from typing import List, Optional, Dict, Tuple

# ── Dark theme constants (consistent with the rest of SNIA) ──────────
BG     = '#0d1117'
PANEL  = '#161b22'
TXT    = '#e6edf3'
GRID_C = '#30363d'
CYAN   = '#00d4aa'
BLUE   = '#58a6ff'
YELLOW = '#ffd93d'
RED    = '#ff6b6b'
PURPLE = '#a371f7'
ORANGE = '#f0883e'
PINK   = '#ff9ff3'

CATEGORY_COLORS = {
    'uninformed': BLUE,
    'informed':   PURPLE,
    'local':      CYAN,
}

ALGO_COLORS = {
    'BFS': '#ff6b6b', 'DFS': '#54a0ff',
    'A*':  '#a371f7',
    'HC':  '#54a0ff', 'SA':  '#ff9ff3',
    'GA':  '#ff6b6b', 'DE':  '#f0883e',
    'PSO': '#58a6ff', 'ABC': '#00d4aa',
    'CS':  '#a371f7', 'FA':  '#f0883e',
    'TLBO': '#ffd93d',
}

TERRAIN_COLORS_MAP = {0: '#c2b280', 1: '#808080', 2: '#a8d8ea', 3: '#ff4444'}


def _style_ax(ax, title='', xlabel='', ylabel=''):
    """Apply dark theme to an axis."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    if title:
        ax.set_title(title, color=TXT, fontsize=11, fontweight='bold', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)


def _draw_grid_base(ax, problem):
    """Draw the ShortestPath grid background (elevation + terrain + hazards + waypoints)."""
    ax.imshow(problem.elevation, cmap='terrain', alpha=0.6, origin='upper')

    # Terrain overlay
    terrain_overlay = np.zeros((*problem.terrain.shape, 4))
    for t_type, color in TERRAIN_COLORS_MAP.items():
        mask = problem.terrain == t_type
        r_val = int(color[1:3], 16) / 255
        g_val = int(color[3:5], 16) / 255
        b_val = int(color[5:7], 16) / 255
        terrain_overlay[mask] = [r_val, g_val, b_val, 0.3]
    lava_mask = problem.terrain == 3  # TERRAIN_LAVA
    terrain_overlay[lava_mask, 3] = 0.7
    ax.imshow(terrain_overlay, origin='upper')

    # Hazard zones
    for hz in problem.hazard_zones:
        circle = patches.Circle(
            (hz.center_col, hz.center_row), hz.radius,
            facecolor=RED, alpha=0.12, linewidth=1.5,
            edgecolor='#ff4444', linestyle='--')
        ax.add_patch(circle)

    # Waypoints
    for wp in problem.waypoints:
        ax.scatter(wp.col, wp.row, s=150, c=YELLOW, marker='*',
                   edgecolors='white', linewidths=0.8, zorder=10)

    # Start and Goal
    ax.scatter(problem.start[1], problem.start[0], s=200, c=CYAN,
               marker='o', edgecolors='white', linewidths=2, zorder=12)
    ax.annotate('S', (problem.start[1], problem.start[0]),
                ha='center', va='center', fontsize=8, color='black',
                fontweight='bold', zorder=13)

    ax.scatter(problem.goal[1], problem.goal[0], s=200, c=RED,
               marker='s', edgecolors='white', linewidths=2, zorder=12)
    ax.annotate('G', (problem.goal[1], problem.goal[0]),
                ha='center', va='center', fontsize=8, color='white',
                fontweight='bold', zorder=13)


# =====================================================================
#  1) UNINFORMED SEARCH — REAL-TIME PARALLEL ANIMATION (BFS, DFS)
# =====================================================================

def visualize_uninformed_search(problem, algorithms: Dict[str, object],
                                iterations=None, interval=30,
                                save_path=None, show=True):
    """
    Real-time side-by-side animation of uninformed search algorithms.

    All algorithms run simultaneously — each subplot updates on the same frame.
    """
    algo_names = list(algorithms.keys())
    n = len(algo_names)

    # Run all algorithms first to collect history
    algo_data = {}
    for name, AlgClass in algorithms.items():
        algo = AlgClass(problem)
        path, cost = algo.solve(iterations=iterations)
        algo_data[name] = {
            'history': algo.history,
            'path': path,
            'cost': cost,
        }

    # Determine total frames (max across all algorithms + path reveal + hold)
    max_explore = max(len(d['history']) for d in algo_data.values())
    max_path = max(len(d['path']) if d['path'] else 0 for d in algo_data.values())
    hold_frames = 20
    total_frames = max_explore + max_path + hold_frames

    # Build figure
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = [axes]
    fig.suptitle('⚡ Uninformed Search — Real-time Parallel', color=TXT,
                 fontsize=16, fontweight='bold', y=0.98)

    # Per-algo state
    states = {}
    for i, name in enumerate(algo_names):
        ax = axes[i]
        _draw_grid_base(ax, problem)
        _style_ax(ax, title=name)

        color = ALGO_COLORS.get(name, BLUE)
        current_marker = ax.scatter([], [], s=120, c=YELLOW, marker='D',
                                    edgecolors='white', linewidths=1.5, zorder=15)
        path_line, = ax.plot([], [], color=color, linewidth=2.5, alpha=0.9, zorder=8)

        states[name] = {
            'ax': ax,
            'color': color,
            'current_marker': current_marker,
            'path_line': path_line,
            'drawn_visited': set(),
            'history': algo_data[name]['history'],
            'path': algo_data[name]['path'],
            'cost': algo_data[name]['cost'],
            'n_explore': len(algo_data[name]['history']),
        }

    def update(frame):
        for name in algo_names:
            s = states[name]
            ax = s['ax']
            n_exp = s['n_explore']
            path = s['path']
            cost = s['cost']

            if frame < n_exp:
                # Exploration phase
                snap = s['history'][frame]
                visited = snap.get('visited_nodes', [])
                current = snap.get('current_node', None)

                for (r, c) in visited:
                    if (r, c) not in s['drawn_visited']:
                        s['drawn_visited'].add((r, c))
                        rect = patches.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            facecolor=s['color'], alpha=0.35, zorder=3)
                        ax.add_patch(rect)

                if current is not None:
                    s['current_marker'].set_offsets([[current[1], current[0]]])
                else:
                    s['current_marker'].set_offsets(np.empty((0, 2)))

                ax.set_title(
                    f'{name}  |  Step {frame+1}/{n_exp}  |  Visited: {len(s["drawn_visited"])}',
                    color=TXT, fontsize=11, fontweight='bold', pad=8)

            elif frame < n_exp + (len(path) if path else 0):
                # Path reveal phase
                s['current_marker'].set_offsets(np.empty((0, 2)))
                idx = frame - n_exp + 1
                sub = path[:idx]
                if len(sub) >= 2:
                    cols = [p[1] for p in sub]
                    rows = [p[0] for p in sub]
                    s['path_line'].set_data(cols, rows)

                cost_str = f'{cost:.1f}' if cost and cost != float('inf') else 'N/A'
                ax.set_title(
                    f'{name}  ✓  |  Cost: {cost_str}  |  Path: {len(path)-1}',
                    color=CYAN, fontsize=11, fontweight='bold', pad=8)

            elif frame == n_exp + (len(path) if path else 0):
                # Show final path fully
                s['current_marker'].set_offsets(np.empty((0, 2)))
                if path and len(path) >= 2:
                    cols = [p[1] for p in path]
                    rows = [p[0] for p in path]
                    s['path_line'].set_data(cols, rows)
                cost_str = f'{cost:.1f}' if cost and cost != float('inf') else 'N/A'
                plen = len(path) - 1 if path else 0
                ax.set_title(
                    f'{name}  ✓  |  Cost: {cost_str}  |  Path: {plen}  |  Visited: {len(s["drawn_visited"])}',
                    color=CYAN, fontsize=11, fontweight='bold', pad=8)

        return []

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=interval, blit=False, repeat=False)
    plt.tight_layout()

    if save_path:
        try:
            ani.save(save_path, writer=animation.PillowWriter(fps=max(1, 1000 // interval)), dpi=100)
        except Exception as e:
            print(f'Warning: failed to save animation: {e}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani


# =====================================================================
#  2) INFORMED SEARCH — REAL-TIME ANIMATION (A*)
# =====================================================================

def visualize_informed_search(problem, algorithms: Dict[str, object],
                              iterations=None, interval=30,
                              save_path=None, show=True):
    """
    Real-time side-by-side animation of informed search algorithms.

    Shows gradient coloring (dark→bright) indicating exploration order
    with heuristic guidance.
    """
    algo_names = list(algorithms.keys())
    n = len(algo_names)

    algo_data = {}
    for name, AlgClass in algorithms.items():
        algo = AlgClass(problem)
        path, cost = algo.solve(iterations=iterations)
        algo_data[name] = {
            'history': algo.history,
            'path': path,
            'cost': cost,
        }

    max_explore = max(len(d['history']) for d in algo_data.values())
    max_path = max(len(d['path']) if d['path'] else 0 for d in algo_data.values())
    hold_frames = 20
    total_frames = max_explore + max_path + hold_frames

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = [axes]
    fig.suptitle('⚡ Informed Search — Real-time', color=TXT,
                 fontsize=16, fontweight='bold', y=0.98)

    states = {}
    for i, name in enumerate(algo_names):
        ax = axes[i]
        _draw_grid_base(ax, problem)
        _style_ax(ax, title=name)

        current_marker = ax.scatter([], [], s=120, c=YELLOW, marker='D',
                                    edgecolors='white', linewidths=1.5, zorder=15)
        path_line, = ax.plot([], [], color=PURPLE, linewidth=2.5, alpha=0.9, zorder=8)

        states[name] = {
            'ax': ax,
            'current_marker': current_marker,
            'path_line': path_line,
            'drawn_visited': set(),
            'visit_order': 0,
            'history': algo_data[name]['history'],
            'path': algo_data[name]['path'],
            'cost': algo_data[name]['cost'],
            'n_explore': len(algo_data[name]['history']),
        }

    def _gradient_color(order, total):
        frac = order / max(total - 1, 1)
        r = int(0x3b + frac * (0xa3 - 0x3b))
        g = int(0x20 + frac * (0x71 - 0x20))
        b = int(0x96 + frac * (0xf7 - 0x96))
        return f'#{r:02x}{g:02x}{b:02x}'

    def update(frame):
        for name in algo_names:
            s = states[name]
            ax = s['ax']
            n_exp = s['n_explore']
            path = s['path']
            cost = s['cost']
            total_visited_final = len(s['history'][-1].get('visited_nodes', [])) if s['history'] else 1

            if frame < n_exp:
                snap = s['history'][frame]
                visited = snap.get('visited_nodes', [])
                current = snap.get('current_node', None)

                for (r, c) in visited:
                    if (r, c) not in s['drawn_visited']:
                        s['drawn_visited'].add((r, c))
                        clr = _gradient_color(s['visit_order'], total_visited_final)
                        s['visit_order'] += 1
                        rect = patches.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            facecolor=clr, alpha=0.4, zorder=3)
                        ax.add_patch(rect)

                if current is not None:
                    s['current_marker'].set_offsets([[current[1], current[0]]])
                else:
                    s['current_marker'].set_offsets(np.empty((0, 2)))

                ax.set_title(
                    f'{name}  |  Step {frame+1}/{n_exp}  |  Visited: {len(s["drawn_visited"])}',
                    color=TXT, fontsize=11, fontweight='bold', pad=8)

            elif frame < n_exp + (len(path) if path else 0):
                s['current_marker'].set_offsets(np.empty((0, 2)))
                idx = frame - n_exp + 1
                sub = path[:idx]
                if len(sub) >= 2:
                    cols = [p[1] for p in sub]
                    rows = [p[0] for p in sub]
                    s['path_line'].set_data(cols, rows)

                cost_str = f'{cost:.1f}' if cost and cost != float('inf') else 'N/A'
                ax.set_title(
                    f'{name}  ✓  |  Cost: {cost_str}  |  Path: {len(path)-1}',
                    color=CYAN, fontsize=11, fontweight='bold', pad=8)

            elif frame == n_exp + (len(path) if path else 0):
                s['current_marker'].set_offsets(np.empty((0, 2)))
                if path and len(path) >= 2:
                    cols = [p[1] for p in path]
                    rows = [p[0] for p in path]
                    s['path_line'].set_data(cols, rows)
                cost_str = f'{cost:.1f}' if cost and cost != float('inf') else 'N/A'
                plen = len(path) - 1 if path else 0
                ax.set_title(
                    f'{name}  ✓  |  Cost: {cost_str}  |  Path: {plen}  |  Visited: {len(s["drawn_visited"])}',
                    color=CYAN, fontsize=11, fontweight='bold', pad=8)

        return []

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=interval, blit=False, repeat=False)
    plt.tight_layout()

    if save_path:
        try:
            ani.save(save_path, writer=animation.PillowWriter(fps=max(1, 1000 // interval)), dpi=100)
        except Exception as e:
            print(f'Warning: failed to save animation: {e}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani


# =====================================================================
#  3) LOCAL SEARCH — REAL-TIME PARALLEL ANIMATION (Metaheuristics)
# =====================================================================

def visualize_local_search(problem, algorithms: Dict[str, object],
                           iterations=200, dim=2, interval=50,
                           save_path=None, show=True):
    """
    Real-time side-by-side animation of metaheuristic algorithms
    on a 2D continuous landscape.

    Each subplot shows population movement; a shared convergence plot
    updates live.
    """
    algo_names = list(algorithms.keys())
    n = len(algo_names)

    # Run all algorithms first to collect history
    algo_data = {}
    for name, (AlgClass, params) in algorithms.items():
        prob_inst = problem.__class__(dim=dim) if hasattr(problem, 'dim') else problem
        algo = AlgClass(prob_inst, params=params)
        algo.solve(iterations=iterations)
        algo_data[name] = {
            'history': algo.history,
            'best_fitness': algo.best_fitness,
            'best_solution': algo.best_solution,
        }

    max_frames = max(len(d['history']) for d in algo_data.values())
    plot_data = problem.get_plotting_data(points=100)

    # Layout: n population subplots + 1 convergence subplot
    fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 6))
    fig.patch.set_facecolor(BG)
    if n + 1 == 1:
        axes = [axes]
    fig.suptitle(f'⚡ Local Search on {problem.name} — Real-time Parallel', color=TXT,
                 fontsize=15, fontweight='bold', y=0.98)

    conv_ax = axes[-1]
    _style_ax(conv_ax, title='Convergence', xlabel='Iteration', ylabel='Best Fitness')
    conv_ax.set_yscale('log')
    conv_ax.grid(True, alpha=0.15, color=GRID_C)

    states = {}
    for i, name in enumerate(algo_names):
        ax = axes[i]

        # Draw contour background
        if plot_data is not None:
            X, Y, Z = plot_data
            Z_show = np.log1p(Z) if Z.max() > 1000 else Z
            ax.contourf(X, Y, Z_show, levels=30, cmap='plasma', alpha=0.6)
            ax.contour(X, Y, Z_show, levels=15, colors='#30363d', linewidths=0.3, alpha=0.5)

        color = ALGO_COLORS.get(name, BLUE)
        scat = ax.scatter([], [], s=30, c=color, edgecolors='white',
                          linewidths=0.3, alpha=0.8, zorder=5)
        best_scat = ax.scatter([], [], s=200, c=CYAN, marker='*',
                               edgecolors='white', linewidths=1.5, zorder=10)
        conv_line, = conv_ax.plot([], [], color=color, linewidth=1.5, label=name)
        _style_ax(ax, title=name)

        states[name] = {
            'ax': ax,
            'scat': scat,
            'best_scat': best_scat,
            'conv_line': conv_line,
            'conv_data': [],
            'history': algo_data[name]['history'],
            'color': color,
        }

    conv_ax.legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TXT,
                   fontsize=8, loc='upper right')

    def update(frame):
        for name in algo_names:
            s = states[name]
            history = s['history']
            ax = s['ax']

            if frame >= len(history):
                # This algo finished; show last state
                snap = history[-1]
            else:
                snap = history[frame]

            positions = np.asarray(snap.get('positions', []))
            if positions.ndim == 2 and positions.shape[1] >= 2:
                s['scat'].set_offsets(positions[:, :2])
            elif positions.ndim == 1 and positions.size >= 2:
                s['scat'].set_offsets(positions[:2].reshape(1, 2))
            else:
                s['scat'].set_offsets(np.empty((0, 2)))

            bsol = snap.get('global_best_sol')
            if bsol is not None:
                b = np.asarray(bsol)
                if b.size >= 2:
                    s['best_scat'].set_offsets([b[:2]])

            bfit = snap.get('global_best_fit', None)
            if frame < len(history):
                s['conv_data'].append(bfit)
            s['conv_line'].set_data(range(1, len(s['conv_data']) + 1), s['conv_data'])

            fit_str = f'{bfit:.6f}' if bfit is not None else '?'
            iter_str = min(frame + 1, len(history))
            ax.set_title(f'{name}  |  Iter {iter_str}/{len(history)}  |  Best: {fit_str}',
                         color=TXT, fontsize=10, fontweight='bold', pad=8)

        conv_ax.relim()
        conv_ax.autoscale_view()
        return []

    ani = animation.FuncAnimation(fig, update, frames=max_frames,
                                  interval=interval, blit=False, repeat=False)
    plt.tight_layout()

    if save_path:
        try:
            ani.save(save_path, writer=animation.PillowWriter(fps=max(1, 1000 // interval)), dpi=100)
        except Exception as e:
            print(f'Warning: failed to save animation: {e}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani


# =====================================================================
#  4) ANIMATED GRID SEARCH — single algorithm (BFS/DFS/A*)
# =====================================================================

def animate_grid_search(problem, AlgClass, algo_name='Search',
                        iterations=None, interval=50,
                        save_path=None, show=True):
    """
    Create a real-time animation of a single grid search algorithm.
    """
    algo = AlgClass(problem)
    path, cost = algo.solve(iterations=iterations)
    history = algo.history

    if not history:
        raise ValueError('Algorithm produced no history snapshots')

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(BG)
    _draw_grid_base(ax, problem)

    color = ALGO_COLORS.get(algo_name, BLUE)
    current_marker = ax.scatter([], [], s=120, c=YELLOW, marker='D',
                                edgecolors='white', linewidths=1.5, zorder=15)
    path_line, = ax.plot([], [], color=CYAN, linewidth=2.5, alpha=0.9, zorder=8)

    n_explore = len(history)
    path_cells = path if path else []
    n_path = len(path_cells)
    hold_frames = 15
    total_frames = n_explore + n_path + hold_frames

    drawn_visited = set()

    def update(frame):
        nonlocal drawn_visited

        if frame < n_explore:
            snap = history[frame]
            visited = snap.get('visited_nodes', [])
            current = snap.get('current_node', None)

            for (r, c) in visited:
                if (r, c) not in drawn_visited:
                    drawn_visited.add((r, c))
                    rect = patches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        facecolor=color, alpha=0.35, zorder=3)
                    ax.add_patch(rect)

            if current is not None:
                current_marker.set_offsets([[current[1], current[0]]])
            else:
                current_marker.set_offsets(np.empty((0, 2)))

            ax.set_title(
                f'{algo_name}  |  Step {frame+1}/{n_explore}  |  '
                f'Visited: {len(drawn_visited)}',
                color=TXT, fontsize=12, fontweight='bold')

        elif frame < n_explore + n_path:
            current_marker.set_offsets(np.empty((0, 2)))
            idx = frame - n_explore + 1
            sub = path_cells[:idx]
            if len(sub) >= 2:
                cols = [p[1] for p in sub]
                rows = [p[0] for p in sub]
                path_line.set_data(cols, rows)

            cost_str = f'{cost:.1f}' if cost and cost != float('inf') else 'N/A'
            ax.set_title(
                f'{algo_name}  |  Path reveal  |  Cost: {cost_str}  |  '
                f'Length: {len(path_cells)-1}',
                color=TXT, fontsize=12, fontweight='bold')

        return []

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=interval, blit=False, repeat=False)

    if save_path:
        try:
            ani.save(save_path, writer=animation.PillowWriter(fps=max(1, 1000 // interval)), dpi=100)
        except Exception as e:
            print(f'Warning: failed to save animation: {e}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani


# =====================================================================
#  5) ANIMATED LOCAL SEARCH — single algorithm on 2D contour
# =====================================================================

def animate_local_search(problem, AlgClass, algo_name='Optimizer',
                         params=None, iterations=200, interval=100,
                         save_path=None, show=True):
    """
    Real-time animation of a single metaheuristic on a 2D landscape.
    """
    algo = AlgClass(problem, params=params or {})
    algo.solve(iterations=iterations)
    history = algo.history

    if not history:
        raise ValueError('Algorithm produced no history snapshots')

    plot_data = problem.get_plotting_data(points=100)
    fig, (ax_pop, ax_conv) = plt.subplots(1, 2, figsize=(14, 6),
                                           gridspec_kw={'width_ratios': [1.2, 1]})
    fig.patch.set_facecolor(BG)

    if plot_data is not None:
        X, Y, Z = plot_data
        Z_show = np.log1p(Z) if Z.max() > 1000 else Z
        ax_pop.contourf(X, Y, Z_show, levels=30, cmap='plasma', alpha=0.6)
        ax_pop.contour(X, Y, Z_show, levels=15, colors='#30363d', linewidths=0.3, alpha=0.5)

    color = ALGO_COLORS.get(algo_name, BLUE)
    scat = ax_pop.scatter([], [], s=30, c=color, edgecolors='white',
                          linewidths=0.3, alpha=0.8, zorder=5)
    best_scat = ax_pop.scatter([], [], s=200, c=CYAN, marker='*',
                               edgecolors='white', linewidths=1.5, zorder=10)
    _style_ax(ax_pop, title=algo_name)

    conv_line, = ax_conv.plot([], [], color=color, linewidth=1.5)
    _style_ax(ax_conv, title='Convergence', xlabel='Iteration', ylabel='Best Fitness')
    ax_conv.set_yscale('log')
    ax_conv.grid(True, alpha=0.15, color=GRID_C)

    conv_data = []

    def update(frame):
        snap = history[frame]
        positions = np.asarray(snap.get('positions', []))

        if positions.ndim == 2 and positions.shape[1] >= 2:
            scat.set_offsets(positions[:, :2])
        elif positions.ndim == 1 and positions.size >= 2:
            scat.set_offsets(positions[:2].reshape(1, 2))
        else:
            scat.set_offsets(np.empty((0, 2)))

        bsol = snap.get('global_best_sol')
        if bsol is not None:
            b = np.asarray(bsol)
            if b.size >= 2:
                best_scat.set_offsets([b[:2]])

        bfit = snap.get('global_best_fit', None)
        conv_data.append(bfit)
        conv_line.set_data(range(1, len(conv_data) + 1), conv_data)
        ax_conv.relim()
        ax_conv.autoscale_view()

        fit_str = f'{bfit:.6f}' if bfit is not None else '?'
        ax_pop.set_title(f'{algo_name}  |  Iter {frame+1}/{len(history)}  |  Best: {fit_str}',
                         color=TXT, fontsize=11, fontweight='bold')
        return scat, best_scat, conv_line

    ani = animation.FuncAnimation(fig, update, frames=len(history),
                                  interval=interval, blit=False, repeat=False)
    plt.tight_layout()

    if save_path:
        try:
            ani.save(save_path, writer=animation.PillowWriter(fps=max(1, 1000 // interval)), dpi=100)
        except Exception as e:
            print(f'Warning: failed to save animation: {e}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani


# =====================================================================
#  6) ALL CATEGORIES — REAL-TIME PARALLEL DASHBOARD
# =====================================================================

def visualize_all_categories(grid_problem, continuous_problem,
                             uninformed_algos=None, informed_algos=None,
                             local_search_algos=None, iterations_grid=None,
                             iterations_local=200, interval=40,
                             save_path=None, show=True):
    """
    Real-time dashboard with ALL three categories animating simultaneously.

    Row 1: BFS, DFS, A* exploring a grid side-by-side
    Row 2: PSO, GA, DE, etc. on a 2D contour + live convergence
    """
    uninformed_algos = uninformed_algos or {}
    informed_algos = informed_algos or {}
    local_search_algos = local_search_algos or {}

    # ── Run all algorithms upfront ─────────────────────────────────
    grid_data = {}
    for name, AlgClass in {**uninformed_algos, **informed_algos}.items():
        algo = AlgClass(grid_problem)
        path, cost = algo.solve(iterations=iterations_grid)
        grid_data[name] = {
            'history': algo.history,
            'path': path,
            'cost': cost,
            'is_informed': name in informed_algos,
        }

    local_data = {}
    for name, (AlgClass, params) in local_search_algos.items():
        algo = AlgClass(continuous_problem, params=params)
        algo.solve(iterations=iterations_local)
        local_data[name] = {
            'history': algo.history,
            'best_fitness': algo.best_fitness,
            'best_solution': algo.best_solution,
        }

    # ── Calculate total frames ─────────────────────────────────────
    grid_max_explore = max((len(d['history']) for d in grid_data.values()), default=0)
    grid_max_path = max((len(d['path']) if d['path'] else 0 for d in grid_data.values()), default=0)
    local_max = max((len(d['history']) for d in local_data.values()), default=0)
    hold = 20
    total_frames = max(grid_max_explore + grid_max_path, local_max) + hold

    # ── Build figure ───────────────────────────────────────────────
    n_grid = len(grid_data)
    n_local = len(local_data)
    n_cols = max(n_grid + 1, n_local + 1, 2)

    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('⚡ Search Algorithm Dashboard — Real-time Parallel', color=TXT,
                 fontsize=18, fontweight='bold', y=0.99)

    for row in axes:
        for ax in row:
            ax.set_visible(False)

    # ── ROW 1 setup: grid algorithms ──────────────────────────────
    grid_names = list(grid_data.keys())
    grid_states = {}

    for col, name in enumerate(grid_names):
        if col >= n_cols - 1:
            break
        ax = axes[0][col]
        ax.set_visible(True)
        _draw_grid_base(ax, grid_problem)

        is_inf = grid_data[name]['is_informed']
        color = ALGO_COLORS.get(name, PURPLE if is_inf else BLUE)
        current_marker = ax.scatter([], [], s=100, c=YELLOW, marker='D',
                                    edgecolors='white', linewidths=1, zorder=15)
        path_line, = ax.plot([], [], color=color, linewidth=2.5, alpha=0.9, zorder=8)
        tag = '[Informed]' if is_inf else '[Uninformed]'
        _style_ax(ax, title=f'{name} {tag}')

        n_exp = len(grid_data[name]['history'])
        total_visited_final = len(grid_data[name]['history'][-1].get('visited_nodes', [])) if grid_data[name]['history'] else 1

        grid_states[name] = {
            'ax': ax, 'color': color, 'is_informed': is_inf,
            'current_marker': current_marker, 'path_line': path_line,
            'drawn_visited': set(), 'visit_order': 0,
            'total_visited_final': total_visited_final,
            'history': grid_data[name]['history'],
            'path': grid_data[name]['path'],
            'cost': grid_data[name]['cost'],
            'n_explore': n_exp,
            'tag': tag,
        }

    # Grid summary panel (static — shows at end)
    ax_gs = axes[0][n_cols - 1]
    ax_gs.set_visible(True)
    _style_ax(ax_gs, title='Grid Search Summary')
    ax_gs.axis('off')
    summary_text_obj = ax_gs.text(0.05, 0.95, '', transform=ax_gs.transAxes,
                                   fontsize=9, verticalalignment='top',
                                   fontfamily='monospace', color=TXT,
                                   bbox=dict(boxstyle='round,pad=0.5',
                                             facecolor=PANEL, edgecolor=GRID_C))

    # ── ROW 2 setup: local search ─────────────────────────────────
    local_names = list(local_data.keys())
    local_states = {}
    plot_data = continuous_problem.get_plotting_data(points=100)

    for col, name in enumerate(local_names):
        if col >= n_cols - 1:
            break
        ax = axes[1][col]
        ax.set_visible(True)

        if plot_data is not None:
            X, Y, Z = plot_data
            Z_show = np.log1p(Z) if Z.max() > 1000 else Z
            ax.contourf(X, Y, Z_show, levels=30, cmap='plasma', alpha=0.6)
            ax.contour(X, Y, Z_show, levels=15, colors='#30363d', linewidths=0.3, alpha=0.5)

        color = ALGO_COLORS.get(name, BLUE)
        scat = ax.scatter([], [], s=25, c=color, edgecolors='white',
                          linewidths=0.3, alpha=0.8, zorder=5)
        best_scat = ax.scatter([], [], s=180, c=CYAN, marker='*',
                               edgecolors='white', linewidths=1.5, zorder=10)
        _style_ax(ax, title=f'{name} [Local]')

        local_states[name] = {
            'ax': ax, 'scat': scat, 'best_scat': best_scat,
            'history': local_data[name]['history'],
            'conv_data': [], 'color': color,
        }

    # Convergence panel
    ax_conv = axes[1][n_cols - 1]
    ax_conv.set_visible(True)
    _style_ax(ax_conv, title='Local Search Convergence', xlabel='Iteration', ylabel='Best Fitness')
    ax_conv.grid(True, alpha=0.15, color=GRID_C)

    conv_lines = {}
    for name in local_names:
        clr = ALGO_COLORS.get(name, BLUE)
        line, = ax_conv.plot([], [], color=clr, linewidth=1.5, label=name)
        conv_lines[name] = line
    if local_names:
        ax_conv.set_yscale('log')
        ax_conv.legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TXT,
                       fontsize=7, loc='upper right')

    def _gradient_color(order, total):
        frac = order / max(total - 1, 1)
        r = int(0x3b + frac * (0xa3 - 0x3b))
        g = int(0x20 + frac * (0x71 - 0x20))
        b = int(0x96 + frac * (0xf7 - 0x96))
        return f'#{r:02x}{g:02x}{b:02x}'

    def update(frame):
        # ── Update grid algorithms ─────────────────────────────
        for name in grid_names:
            s = grid_states.get(name)
            if s is None:
                continue
            ax = s['ax']
            n_exp = s['n_explore']
            path = s['path']
            cost = s['cost']

            if frame < n_exp:
                snap = s['history'][frame]
                visited = snap.get('visited_nodes', [])
                current = snap.get('current_node', None)

                for (r, c) in visited:
                    if (r, c) not in s['drawn_visited']:
                        s['drawn_visited'].add((r, c))
                        if s['is_informed']:
                            clr = _gradient_color(s['visit_order'], s['total_visited_final'])
                        else:
                            clr = s['color']
                        s['visit_order'] += 1
                        rect = patches.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            facecolor=clr, alpha=0.35, zorder=3)
                        ax.add_patch(rect)

                if current is not None:
                    s['current_marker'].set_offsets([[current[1], current[0]]])
                else:
                    s['current_marker'].set_offsets(np.empty((0, 2)))

                ax.set_title(
                    f'{name} {s["tag"]}  |  Step {frame+1}  |  Visited: {len(s["drawn_visited"])}',
                    color=TXT, fontsize=10, fontweight='bold', pad=6)

            elif frame < n_exp + (len(path) if path else 0):
                s['current_marker'].set_offsets(np.empty((0, 2)))
                idx = frame - n_exp + 1
                sub = path[:idx]
                if len(sub) >= 2:
                    cols = [p[1] for p in sub]
                    rows = [p[0] for p in sub]
                    s['path_line'].set_data(cols, rows)

                cost_str = f'{cost:.1f}' if cost and cost != float('inf') else 'N/A'
                ax.set_title(
                    f'{name} {s["tag"]}  ✓  Cost: {cost_str}  Path: {len(path)-1}',
                    color=CYAN, fontsize=10, fontweight='bold', pad=6)

            elif frame == n_exp + (len(path) if path else 0):
                s['current_marker'].set_offsets(np.empty((0, 2)))
                if path and len(path) >= 2:
                    cols = [p[1] for p in path]
                    rows = [p[0] for p in path]
                    s['path_line'].set_data(cols, rows)

                cost_str = f'{cost:.1f}' if cost and cost != float('inf') else 'N/A'
                plen = len(path) - 1 if path else 0
                ax.set_title(
                    f'{name} {s["tag"]}  ✓  Cost: {cost_str}  Path: {plen}  Vis: {len(s["drawn_visited"])}',
                    color=CYAN, fontsize=10, fontweight='bold', pad=6)

        # ── Update summary when all grid algos done ────────────
        all_grid_done = all(
            frame >= gs['n_explore'] + (len(gs['path']) if gs['path'] else 0)
            for gs in grid_states.values()
        )
        if all_grid_done and grid_states:
            lines = "GRID SEARCH COMPARISON\n" + "=" * 32 + "\n\n"
            lines += f"{'Algo':<6} {'Type':<12} {'Steps':>6} {'Path':>5} {'Cost':>8}\n"
            lines += "-" * 40 + "\n"
            for nm, gs in grid_states.items():
                tag = 'Informed' if gs['is_informed'] else 'Uninformed'
                cs = f'{gs["cost"]:.1f}' if gs['cost'] and gs['cost'] != float('inf') else 'N/A'
                plen = len(gs['path']) - 1 if gs['path'] else 0
                lines += f'{nm:<6} {tag:<12} {gs["n_explore"]:>6} {plen:>5} {cs:>8}\n'
            summary_text_obj.set_text(lines)

        # ── Update local search algorithms ─────────────────────
        for name in local_names:
            s = local_states.get(name)
            if s is None:
                continue
            ax = s['ax']
            history = s['history']

            if frame >= len(history):
                snap = history[-1]
                f_idx = len(history)
            else:
                snap = history[frame]
                f_idx = frame + 1

            positions = np.asarray(snap.get('positions', []))
            if positions.ndim == 2 and positions.shape[1] >= 2:
                s['scat'].set_offsets(positions[:, :2])
            elif positions.ndim == 1 and positions.size >= 2:
                s['scat'].set_offsets(positions[:2].reshape(1, 2))
            else:
                s['scat'].set_offsets(np.empty((0, 2)))

            bsol = snap.get('global_best_sol')
            if bsol is not None:
                b = np.asarray(bsol)
                if b.size >= 2:
                    s['best_scat'].set_offsets([b[:2]])

            bfit = snap.get('global_best_fit', None)
            if frame < len(history):
                s['conv_data'].append(bfit)
            conv_lines[name].set_data(range(1, len(s['conv_data']) + 1), s['conv_data'])

            fit_str = f'{bfit:.6f}' if bfit is not None else '?'
            ax.set_title(f'{name} [Local] | Iter {f_idx}/{len(history)} | Best: {fit_str}',
                         color=TXT, fontsize=10, fontweight='bold', pad=6)

        if local_states:
            ax_conv.relim()
            ax_conv.autoscale_view()

        return []

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=interval, blit=False, repeat=False)
    plt.tight_layout()

    if save_path:
        try:
            ani.save(save_path, writer=animation.PillowWriter(fps=max(1, 1000 // interval)), dpi=100)
        except Exception as e:
            print(f'Warning: failed to save animation: {e}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani

from visualization.viz_tsp import visualize_tsp_optimization
from visualization.viz_knapsack import visualize_knapsack_optimization
from visualization.viz_graph_coloring import visualize_graph_coloring_optimization
