"""
Generate animated GIF visualizations for each problem in SNIA.

Continuous problems: 3D rotating surface with marked global minimum
Discrete problems:   Step-by-step animated solution construction

Results saved to results/gif/

Usage:
    python scripts/generate_problem_gifs.py
"""

import os
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.colors import LightSource
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import heapq

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'results' / 'gif'
OUT.mkdir(parents=True, exist_ok=True)

# -- Dark theme constants --
BG   = '#0d1117'
PANEL = '#161b22'
TXT  = '#e6edf3'
GRID = '#30363d'
CYAN = '#00d4aa'
BLUE = '#58a6ff'
YELLOW = '#ffd93d'
RED  = '#ff6b6b'
PURPLE = '#a371f7'


def dark_3d(fig, ax, title=''):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor(PANEL)
        pane.set_edgecolor(GRID)
    ax.tick_params(colors='#8b949e', labelsize=7)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color('#8b949e')
        axis._axinfo['grid']['color'] = GRID
        axis._axinfo['grid']['linewidth'] = 0.3
    if title:
        ax.set_title(title, color=TXT, fontsize=14, fontweight='bold', pad=18)


def dark_2d(fig, ax, title=''):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#8b949e')
    for sp in ax.spines.values():
        sp.set_color(GRID)
    if title:
        ax.set_title(title, color=TXT, fontsize=13, fontweight='bold', pad=12)


# =====================================================================
#  1) CONTINUOUS PROBLEMS -- Rotating 3D surface
# =====================================================================

def gif_continuous(ProblemClass, name, points=100, n_frames=72, fps=18):
    prob = ProblemClass(dim=2)
    data = prob.get_plotting_data(points=points)
    if data is None:
        print(f'  [SKIP] {name}: no 2D plot data')
        return

    X, Y, Z = data
    Z_show = np.log1p(Z) if Z.max() > 1000 else Z

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    dark_3d(fig, ax, title=f'{name} Function')

    ls = LightSource(azdeg=315, altdeg=45)
    norm = plt.Normalize(Z_show.min(), Z_show.max())
    fc = cm.plasma(norm(Z_show))

    ax.plot_surface(X, Y, Z_show, facecolors=fc,
                    rstride=1, cstride=1, alpha=0.92,
                    antialiased=True, shade=True, lightsource=ls)

    # Mark global min
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    gx, gy, gz = X[min_idx], Y[min_idx], Z_show[min_idx]
    ax.scatter([gx], [gy], [gz], color=CYAN, s=120, marker='*',
               edgecolors='white', linewidths=1.5, zorder=10, depthshade=False)

    ax.set_xlabel('x1', fontsize=10)
    ax.set_ylabel('x2', fontsize=10)
    ax.set_zlabel('f(x)', fontsize=10)
    ax.view_init(elev=35, azim=0)

    fig.text(0.02, 0.02,
             f'Bounds: {prob.bounds}  |  Global min: {prob.min_val}',
             color='#8b949e', fontsize=9, fontfamily='monospace')
    plt.tight_layout()

    def update(frame):
        azim = (frame * 360 / n_frames) % 360
        elev = 30 + 15 * np.sin(2 * np.pi * frame / n_frames)
        ax.view_init(elev=elev, azim=azim)
        return []

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    path = OUT / f'{name.lower()}.gif'
    print(f'  Saving {path.name} ...', end='', flush=True)
    try:
        ani.save(str(path), writer=animation.PillowWriter(fps=fps), dpi=100)
        print(f' OK ({path.stat().st_size // 1024} KB)')
    except Exception as e:
        print(f' FAIL: {e}')
    finally:
        plt.close(fig)


# =====================================================================
#  2) TSP -- Route construction
# =====================================================================

def gif_tsp(fps=6):
    from problems.discrete.TSP import TSPProblem

    prob = TSPProblem.generate(n_cities=20, difficulty='medium', seed=42)

    # Nearest-neighbor route
    route = [0]
    visited = {0}
    for _ in range(prob.n_cities - 1):
        last = route[-1]
        dists = prob.distance_matrix[last].copy()
        dists[list(visited)] = np.inf
        nxt = int(np.argmin(dists))
        route.append(nxt)
        visited.add(nxt)

    fig, ax = plt.subplots(figsize=(10, 8))
    dark_2d(fig, ax, title='TSP: Storm Chaser Delivery Network')

    # Storm zones
    for storm in prob.storm_zones:
        c = mpatches.Circle((storm.cx, storm.cy), storm.radius,
                            color=RED, alpha=0.12, linewidth=1.5,
                            edgecolor='#ff4444', linestyle='--')
        ax.add_patch(c)

    # Cities
    for city in prob.cities:
        mk, sz, co = ('s', 120, CYAN) if city.is_depot else ('o', 60, BLUE)
        ax.scatter(city.x, city.y, s=sz, c=co, marker=mk,
                   zorder=5, edgecolors='white', linewidths=0.8)
        ax.annotate(str(city.id), (city.x, city.y),
                    textcoords='offset points', xytext=(6, 6),
                    fontsize=7, color='#c9d1d9', zorder=6)

    coords = [(c.x, c.y) for c in prob.cities]
    xs, ys = zip(*coords)
    ax.set_xlim(min(xs) - 8, max(xs) + 8)
    ax.set_ylim(min(ys) - 8, max(ys) + 8)
    ax.set_aspect('equal')

    lines = []
    full_route = route + [route[0]]
    n_edges = len(full_route) - 1
    hold = 10
    title_obj = ax.title

    def update(frame):
        if frame < n_edges:
            a, b = full_route[frame], full_route[frame + 1]
            ca, cb = prob.cities[a], prob.cities[b]
            t = prob.terrain_multiplier[a][b]
            if t < 0.9:
                co, lw = CYAN, 2.5
            elif t > 1.5:
                co, lw = RED, 1.8
            else:
                co, lw = YELLOW, 2.0
            ln, = ax.plot([ca.x, cb.x], [ca.y, cb.y],
                          color=co, linewidth=lw, alpha=0.85, zorder=2)
            lines.append(ln)
            title_obj.set_text(f'TSP: Building Route  |  Step {frame+1}/{n_edges}')
        elif frame == n_edges:
            res = prob.evaluate(route)
            feas = 'FEASIBLE' if res['feasible'] else 'INFEASIBLE'
            title_obj.set_text(f'TSP: Complete  |  Cost: {res["total_cost"]:.1f}  |  {feas}')
        return lines

    ani = animation.FuncAnimation(fig, update, frames=n_edges + hold,
                                  interval=1000 // fps, blit=False, repeat=True)
    path = OUT / 'tsp.gif'
    print(f'  Saving {path.name} ...', end='', flush=True)
    try:
        ani.save(str(path), writer=animation.PillowWriter(fps=fps), dpi=100)
        print(f' OK ({path.stat().st_size // 1024} KB)')
    except Exception as e:
        print(f' FAIL: {e}')
    finally:
        plt.close(fig)


# =====================================================================
#  3) KNAPSACK -- Greedy item selection
# =====================================================================

def gif_knapsack(fps=3):
    from problems.discrete.Knapsack import KnapsackProblem

    prob = KnapsackProblem.generate(n_items=20, difficulty='medium', seed=42)

    # Greedy by value/weight ratio
    ratios = sorted(
        [(it.value / (it.weight + 1e-6), it.id) for it in prob.items],
        reverse=True)

    steps = []
    sel = [0] * prob.n_items
    w = 0
    for ratio, idx in ratios:
        if w + prob.items[idx].weight <= prob.capacity_weight:
            sel[idx] = 1
            w += prob.items[idx].weight
            steps.append(list(sel))

    n_steps = len(steps)
    hold = 8

    fig, (ax_bar, ax_sc) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    def draw(frame):
        ax_bar.clear()
        ax_sc.clear()
        dark_2d(fig, ax_bar, title='Resource Usage')
        dark_2d(fig, ax_sc, title='Items (Value vs Weight)')

        step = min(frame, n_steps - 1)
        cur = steps[step]
        res = prob.evaluate(cur)
        sel_ids = {i for i, s in enumerate(cur) if s == 1}

        # Resource bars
        names = ['Weight', 'Volume', 'Power']
        caps = [prob.capacity_weight, prob.capacity_volume, prob.capacity_power]
        used = [res['weight']['used'], res['volume']['used'], res['power']['used']]
        colors = [BLUE, PURPLE, '#f0883e']
        yp = np.arange(3)

        ax_bar.barh(yp, caps, height=0.5, color='#21262d', edgecolor=GRID)
        bc = [RED if used[i] > caps[i] else colors[i] for i in range(3)]
        ax_bar.barh(yp, used, height=0.5, color=bc, alpha=0.85)
        for i in range(3):
            pct = used[i] / caps[i] * 100 if caps[i] > 0 else 0
            ax_bar.text(max(used[i], caps[i]) + 2, yp[i],
                        f'{used[i]:.0f}/{caps[i]:.0f} ({pct:.0f}%)',
                        va='center', color='#c9d1d9', fontsize=9)
        ax_bar.set_yticks(yp)
        ax_bar.set_yticklabels(names, color='#c9d1d9')

        # Item scatter
        for it in prob.items:
            is_sel = it.id in sel_ids
            co = YELLOW if is_sel else '#484f58'
            al = 1.0 if is_sel else 0.35
            sz = 40 + it.fragility * 60
            ec = RED if it.priority_class == 'critical' else 'none'
            ew = 2 if it.priority_class == 'critical' else 0
            ax_sc.scatter(it.weight, it.value, s=sz, c=co,
                          alpha=al, edgecolors=ec, linewidths=ew, zorder=3)
        ax_sc.set_xlabel('Weight', color='#8b949e')
        ax_sc.set_ylabel('Value', color='#8b949e')

        fig.suptitle(
            f'Knapsack: Space Cargo  |  '
            f'Step {step+1}/{n_steps}  |  '
            f'Selected: {res["n_selected"]}  |  '
            f'Value: {res["total_value"]:.1f}',
            color=TXT, fontsize=13, fontweight='bold', y=0.98)

    def update(frame):
        draw(frame)
        return []

    ani = animation.FuncAnimation(fig, update, frames=n_steps + hold,
                                  interval=1000 // fps, blit=False)
    path = OUT / 'knapsack.gif'
    print(f'  Saving {path.name} ...', end='', flush=True)
    try:
        ani.save(str(path), writer=animation.PillowWriter(fps=fps), dpi=100)
        print(f' OK ({path.stat().st_size // 1024} KB)')
    except Exception as e:
        print(f' FAIL: {e}')
    finally:
        plt.close(fig)


# =====================================================================
#  4) GRAPH COLORING -- Node-by-node coloring
# =====================================================================

def gif_graph_coloring(fps=3):
    from problems.discrete.GraphColoring import GraphColoringProblem

    prob = GraphColoringProblem.generate(
        n_nodes=25, n_colors=5, topology='clustered',
        difficulty='medium', seed=42)

    palette = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#a371f7',
               '#f0883e', '#00d4aa', '#ff9ff3', '#54a0ff', '#5f27cd']

    # Greedy coloring sequence
    coloring_steps = []
    coloring = {}
    for i in range(prob.n_nodes):
        if prob.events[i].pre_assigned_color is not None:
            coloring[i] = prob.events[i].pre_assigned_color
        else:
            neigh_cols = {coloring[j] for j in prob.adjacency[i] if j in coloring}
            coloring[i] = next((c for c in range(prob.n_colors) if c not in neigh_cols), 0)
        coloring_steps.append(dict(coloring))

    n_steps = len(coloring_steps)
    hold = 10

    fig, ax = plt.subplots(figsize=(10, 8))
    dark_2d(fig, ax)

    def draw(frame):
        ax.clear()
        ax.set_facecolor(PANEL)
        step = min(frame, n_steps - 1)
        cur = coloring_steps[step]

        # Edges
        for edge in prob.edges:
            pu = prob.node_positions[edge.u]
            pv = prob.node_positions[edge.v]
            violated = (edge.u in cur and edge.v in cur and cur[edge.u] == cur[edge.v])
            if violated:
                c, lw, a = '#ff4444', 2.5, 0.9
            elif edge.is_hard:
                c, lw, a = '#484f58', 0.8, 0.4
            else:
                c, lw, a = '#30363d', 0.5, 0.25
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color=c, lw=lw, alpha=a, zorder=1)

        # Nodes
        for ev in prob.events:
            pos = prob.node_positions[ev.id]
            sz = 60 + ev.popularity * 12
            if ev.id in cur:
                co = palette[cur[ev.id] % len(palette)]
                al = 1.0
            else:
                co = '#484f58'
                al = 0.4
            ec, ew = (YELLOW, 2.5) if ev.pre_assigned_color is not None else ('white', 0.8)
            ax.scatter(pos[0], pos[1], s=sz, c=co, zorder=5,
                       edgecolors=ec, linewidths=ew, alpha=al)
            ax.annotate(str(ev.id), pos, textcoords='offset points',
                        xytext=(5, 5), fontsize=7, color='#c9d1d9', zorder=6)

        ax.set_aspect('equal')
        ax.axis('off')

        full = dict(cur)
        for j in range(prob.n_nodes):
            if j not in full:
                full[j] = 0
        res = prob.evaluate(full)

        ax.set_title(
            f'Graph Coloring  |  Node {step+1}/{n_steps}  |  '
            f'Colors: {len(set(cur.values()))}  |  '
            f'Violations: {res["n_violations"]}',
            color=TXT, fontsize=12, fontweight='bold', pad=12)

    def update(frame):
        draw(frame)
        return []

    ani = animation.FuncAnimation(fig, update, frames=n_steps + hold,
                                  interval=1000 // fps, blit=False)
    path = OUT / 'graph_coloring.gif'
    print(f'  Saving {path.name} ...', end='', flush=True)
    try:
        ani.save(str(path), writer=animation.PillowWriter(fps=fps), dpi=100)
        print(f' OK ({path.stat().st_size // 1024} KB)')
    except Exception as e:
        print(f' FAIL: {e}')
    finally:
        plt.close(fig)


# =====================================================================
#  5) SHORTEST PATH -- Dijkstra exploration + path reveal
# =====================================================================

def gif_shortest_path(fps=8):
    from problems.discrete.ShortestPath import (
        ShortestPathProblem, TERRAIN_LAVA, TERRAIN_SAND,
        TERRAIN_ROCK, TERRAIN_SMOOTH)

    prob = ShortestPathProblem.generate(grid_size=25, difficulty='medium', seed=42)

    start = prob.start
    goal = prob.goal
    dist = {start: 0.0}
    prev = {}
    visited_order = []
    pq = [(0.0, start)]

    while pq:
        d, (r, c) = heapq.heappop(pq)
        if (r, c) in visited_order:
            continue
        visited_order.append((r, c))
        if (r, c) == goal:
            break
        for nr, nc in prob.neighbors(r, c):
            nd = d + prob.edge_cost(r, c, nr, nc)
            if (nr, nc) not in dist or nd < dist[(nr, nc)]:
                dist[(nr, nc)] = nd
                prev[(nr, nc)] = (r, c)
                heapq.heappush(pq, (nd, (nr, nc)))

    path_nodes = []
    node = goal
    while node in prev:
        path_nodes.append(node)
        node = prev[node]
    path_nodes.append(start)
    path_nodes.reverse()

    # Simple color grid
    gs = prob.grid_size
    grid_img = np.ones((gs, gs, 3))
    for r in range(gs):
        for c in range(gs):
            t = prob.terrain[r, c]
            if t == TERRAIN_LAVA:
                grid_img[r, c] = [0.25, 0.25, 0.28]
            elif t == TERRAIN_ROCK:
                grid_img[r, c] = [0.82, 0.80, 0.76]
            elif t == TERRAIN_SAND:
                grid_img[r, c] = [0.93, 0.91, 0.85]
            else:
                grid_img[r, c] = [0.96, 0.97, 0.98]

    chunk = max(1, len(visited_order) // 40)
    explore_frames = []
    for i in range(0, len(visited_order), chunk):
        explore_frames.append(visited_order[:i + chunk])
    n_explore = len(explore_frames)

    pstep = max(1, len(path_nodes) // 25)
    path_frames = []
    for i in range(0, len(path_nodes), pstep):
        path_frames.append(path_nodes[:i + pstep])
    n_path = len(path_frames)
    hold = 12

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('white')

    def draw(frame):
        ax.clear()
        ax.set_facecolor('white')
        ax.imshow(grid_img, origin='upper', interpolation='nearest')

        phase = ''
        if frame < n_explore:
            cells = explore_frames[frame]
            if cells:
                rows = [c[0] for c in cells]
                cols = [c[1] for c in cells]
                ax.scatter(cols, rows, s=6, c='#4d96ff', alpha=0.45,
                           marker='s', linewidths=0, zorder=5)
            pct = int(len(cells) / len(visited_order) * 100)
            phase = f'Exploring... {pct}%'

        elif frame < n_explore + n_path:
            ar = [c[0] for c in visited_order]
            ac = [c[1] for c in visited_order]
            ax.scatter(ac, ar, s=4, c='#4d96ff', alpha=0.15,
                       marker='s', linewidths=0, zorder=5)
            pidx = frame - n_explore
            partial = path_frames[min(pidx, n_path - 1)]
            if len(partial) >= 2:
                pr = [p[0] for p in partial]
                pc = [p[1] for p in partial]
                ax.plot(pc, pr, color='#00c896', linewidth=3,
                        solid_capstyle='round', zorder=8)
                ax.scatter([pc[-1]], [pr[-1]], s=60, c='#00c896',
                           marker='o', edgecolors='white', linewidths=1.5, zorder=9)
            phase = f'Path found ({len(partial)} steps)'

        else:
            ar = [c[0] for c in visited_order]
            ac = [c[1] for c in visited_order]
            ax.scatter(ac, ar, s=4, c='#4d96ff', alpha=0.1,
                       marker='s', linewidths=0, zorder=5)
            pr = [p[0] for p in path_nodes]
            pc = [p[1] for p in path_nodes]
            ax.plot(pc, pr, color='#00c896', linewidth=3,
                    solid_capstyle='round', zorder=8)
            res = prob.evaluate(path_nodes)
            phase = f'Done! Cost: {res["total_cost"]:.0f} | {res["path_length"]} steps'

        ax.scatter(start[1], start[0], s=180, c='#00c896', marker='o',
                   edgecolors='white', linewidths=2, zorder=12)
        ax.text(start[1], start[0] - 1.6, 'S', color='#00c896',
                fontsize=11, fontweight='bold', ha='center', va='bottom', zorder=12)
        ax.scatter(goal[1], goal[0], s=180, c='#e74c3c', marker='s',
                   edgecolors='white', linewidths=2, zorder=12)
        ax.text(goal[1], goal[0] - 1.6, 'G', color='#e74c3c',
                fontsize=11, fontweight='bold', ha='center', va='bottom', zorder=12)

        ax.set_title(f'Shortest Path   {phase}',
                     fontsize=14, fontweight='bold', color='#333', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#ddd')

    def update(frame):
        draw(frame)
        return []

    total = n_explore + n_path + hold
    ani = animation.FuncAnimation(fig, update, frames=total,
                                  interval=1000 // fps, blit=False)
    path = OUT / 'shortest_path.gif'
    print(f'  Saving {path.name} ...', end='', flush=True)
    try:
        ani.save(str(path), writer=animation.PillowWriter(fps=fps), dpi=100)
        print(f' OK ({path.stat().st_size // 1024} KB)')
    except Exception as e:
        print(f' FAIL: {e}')
    finally:
        plt.close(fig)


# =====================================================================
#  Main
# =====================================================================

def main():
    print('=' * 55)
    print('  SNIA -- Problem GIF Generator')
    print('=' * 55)
    print(f'  Output: {OUT}\n')

    # -- Continuous --
    from problems.continuous.Sphere import Sphere
    from problems.continuous.Rastrigin import Rastrigin
    from problems.continuous.Rosenbrock import Rosenbrock
    from problems.continuous.Ackley import Ackley
    from problems.continuous.Griewank import Griewank

    print('--- Continuous Problems (3D rotating surfaces) ---')
    for cls, name in [
        (Sphere,     'Sphere'),
        (Rastrigin,  'Rastrigin'),
        (Rosenbrock, 'Rosenbrock'),
        (Ackley,     'Ackley'),
        (Griewank,   'Griewank'),
    ]:
        print(f'\n  [{name}]')
        gif_continuous(cls, name, points=100, n_frames=72, fps=18)

    # -- Discrete --
    print('\n--- Discrete Problems (step-by-step solutions) ---')

    print('\n  [TSP - Storm Chaser Delivery]')
    gif_tsp(fps=6)

    print('\n  [Knapsack - Space Cargo Loading]')
    gif_knapsack(fps=3)

    print('\n  [Graph Coloring - Festival Scheduling]')
    gif_graph_coloring(fps=3)

    print('\n  [Shortest Path - Mars Rover Navigation]')
    gif_shortest_path(fps=8)

    print(f'\n{"=" * 55}')
    n = len(list(OUT.glob('*.gif')))
    print(f'  Done! {n} GIFs saved to {OUT}')
    print(f'{"=" * 55}')


if __name__ == '__main__':
    main()
