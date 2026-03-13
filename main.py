"""SNIA desktop app.

Run:
  python main.py
"""

import io
import os
import multiprocessing as mp
import shutil
import threading
import time
import traceback
from contextlib import redirect_stdout
from queue import Empty

import base64

import flet as ft
from PIL import Image as PILImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Worker functions (run in subprocess) ─────────────────────────────

def run_compare():
    import runpy
    runpy.run_path(os.path.join(BASE_DIR, 'scripts', 'compare_all.py'), run_name='__main__')
    out_dir = os.path.join(BASE_DIR, 'results', 'compare')
    images = []
    for root, _, files in os.walk(out_dir):
        for f in sorted(files):
            if f.endswith('.png'):
                images.append(os.path.join(root, f))
    return images


def run_gif():
    import runpy
    runpy.run_path(os.path.join(BASE_DIR, 'scripts', 'generate_problem_gifs.py'), run_name='__main__')
    out_dir = os.path.join(BASE_DIR, 'results', 'gif')
    images = []
    if os.path.isdir(out_dir):
        for f in sorted(os.listdir(out_dir)):
            if f.endswith(('.gif', '.png')):
                images.append(os.path.join(out_dir, f))
    return images


# ── Compatibility map ────────────────────────────────────────────────

CONTINUOUS_PROBLEMS = ['Sphere', 'Rastrigin', 'Rosenbrock', 'Ackley', 'Griewank']
DISCRETE_PROBLEMS = ['ShortestPath', 'TSP', 'Knapsack', 'GraphColoring']
ALL_PROBLEMS = CONTINUOUS_PROBLEMS + DISCRETE_PROBLEMS

CONTINUOUS_ALGOS = ['GA', 'PSO', 'DE', 'SA', 'HC', 'ABC', 'CS', 'FA', 'TLBO']
GRAPH_SEARCH_ALGOS = ['BFS', 'DFS', 'A*']
TSP_ALGOS = ['ACO']
KNAPSACK_ALGOS = ['GA', 'ABC']
GRAPH_COLORING_ALGOS = ['GA', 'SA']
ALL_ALGOS = list(dict.fromkeys(
    CONTINUOUS_ALGOS + GRAPH_SEARCH_ALGOS + TSP_ALGOS + KNAPSACK_ALGOS + GRAPH_COLORING_ALGOS
))

# Returns list of valid algorithm keys for a given problem key
def compatible_algos(prob_key):
    if prob_key in CONTINUOUS_PROBLEMS:
        return CONTINUOUS_ALGOS
    if prob_key == 'ShortestPath':
        return GRAPH_SEARCH_ALGOS
    if prob_key == 'TSP':
        return TSP_ALGOS
    if prob_key == 'Knapsack':
        return KNAPSACK_ALGOS
    if prob_key == 'GraphColoring':
        return GRAPH_COLORING_ALGOS
    return []


def get_problem_info(prob):
    """Extract detailed info string from a problem instance."""
    lines = []
    # __str__ gives the summary
    lines.append(str(prob))
    # complexity_class gives detailed metrics
    if hasattr(prob, 'complexity_class'):
        cx = prob.complexity_class
        lines.append('')
        lines.append(f"Complexity: {cx.get('class', '?')}")
        if 'search_space_log10' in cx:
            lines.append(f"Search space: ~10^{cx['search_space_log10']:.1f}")
        if 'brute_force' in cx:
            lines.append(f"Brute force: {cx['brute_force']}")
        if 'bfs_complexity' in cx:
            lines.append(f"BFS: {cx['bfs_complexity']}")
        if 'dijkstra_complexity' in cx:
            lines.append(f"Dijkstra: {cx['dijkstra_complexity']}")
        if 'astar_note' in cx:
            lines.append(f"A*: {cx['astar_note']}")
        if 'grid_size' in cx:
            lines.append(f"Grid: {cx['grid_size']}")
        if 'hazard_zones' in cx:
            lines.append(f"Hazard zones: {cx['hazard_zones']}")
        if cx.get('n'):
            lines.append(f"n = {cx['n']}")
    return '\n'.join(lines)


def build_problem_description(category, difficulty='easy'):
    """Build a detailed, human-readable problem description for the UI tab."""
    if category in ('uninformed', 'informed'):
        return _desc_shortest_path(difficulty)
    if category == 'tsp':
        return _desc_tsp(difficulty)
    if category == 'knapsack':
        return _desc_knapsack(difficulty)
    if category == 'graph_coloring':
        return _desc_graph_coloring(difficulty)
    if category == 'local':
        return _desc_continuous()
    if category == 'all':
        return _desc_continuous()
    return ''


def _desc_shortest_path(difficulty):
    from problems.discrete.ShortestPath import ShortestPathProblem
    factory = getattr(ShortestPathProblem, difficulty, ShortestPathProblem.easy)
    p = factory(seed=42)
    cx = p.complexity_class
    n_wp = len(p.waypoints)
    n_hz = len(p.hazard_zones)
    return (
        '=' * 56 + '\n'
        '  SHORTEST PATH: "Mars Rover Navigation"\n'
        '=' * 56 + '\n\n'
        'SCENARIO\n'
        '  Navigate a rover across Mars terrain from Start to\n'
        '  Goal. The terrain is a 2D grid with elevation and\n'
        '  different ground types.\n\n'
        'OBJECTIVE\n'
        '  Find the path with MINIMUM total energy cost,\n'
        '  visiting all required waypoints (science stations).\n\n'
        'CONSTRAINTS\n'
        f'  - Grid size: {p.grid_size}x{p.grid_size}\n'
        f'  - Terrain types: Sand (x1.5), Rock (x1.0),\n'
        f'    Smooth (x0.5), Lava (impassable)\n'
        f'  - Elevation changes add extra cost\n'
        f'  - Hazard zones: {n_hz} radiation areas (+penalty)\n'
        f'  - Waypoints: {n_wp} science stations to visit\n'
        f'  - 8-directional movement (diagonal = sqrt(2) cost)\n\n'
        'COST FORMULA\n'
        '  cost(A->B) = base_terrain_cost(B)\n'
        '             x terrain_multiplier(B)\n'
        '             + |elevation(A) - elevation(B)|\n'
        '             + hazard_penalty (if in zone)\n'
        '  Total = Sum of all edge costs along path\n\n'
        'COMPLEXITY\n'
        f'  Without waypoints: P -- Dijkstra O(V log V)\n'
        f'  With {n_wp} waypoint(s): '
        + (f'NP-hard (Steiner path)\n' if n_wp >= 2 else 'P (polynomial)\n') +
        f'  Class: {cx.get("class", "?")}\n\n'
        'DETAILS\n'
        '  The problem models a rover navigating an alien\n'
        '  surface avoiding hazards and reaching checkpoints.\n'
        '  Used for graph search algorithms (BFS, DFS, A*).\n' 
    )


def _desc_tsp(difficulty):
    from problems.discrete.TSP import TSPProblem
    factory = getattr(TSPProblem, difficulty, TSPProblem.easy)
    p = factory(seed=42)
    cx = p.complexity_class
    n_depots = sum(1 for c in p.cities if c.is_depot)
    return (
        '=' * 56 + '\n'
        '  TSP: "Storm Chaser Delivery Network"\n'
        '=' * 56 + '\n\n'
        'SCENARIO\n'
        '  A delivery driver must visit all cities and return\n'
        '  to start. Routes cross different terrain and may\n'
        '  pass through dangerous storm zones.\n\n'
        'OBJECTIVE\n'
        f'  Visit ALL {p.n_cities} cities once, return to origin,\n'
        '  with MINIMUM total travel cost.\n\n'
        'CONSTRAINTS\n'
        f'  - Time windows: each city has [open, close]\n'
        f'    Arriving late => heavy penalty (x100)\n'
        f'  - Terrain costs per edge:\n'
        f'    Highway = x0.7 (fast)  Normal = x1.0\n'
        f'    Mountain = x2.0 (slow)\n'
        f'  - Storm zones: {len(p.storm_zones)} areas on the map\n'
        f'    Crossing adds extra penalty to that edge\n'
        f'  - Fuel: max capacity = {p.fuel_capacity:.0f}\n'
        f'    Must refuel at depot cities ({n_depots} depots)\n\n'
        'COST FORMULA\n'
        '  cost(i->j) = distance(i,j) x terrain_multiplier(i,j)\n'
        '             + storm_penalty(i,j)\n\n'
        '  TOTAL = Sum cost(i->j) for all edges in tour\n'
        '        + 100 x Sum(time_window_late)\n'
        '        + 100 x Sum(fuel_shortage)\n\n'
        'COMPLEXITY\n'
        f'  Class: {cx["class"]}\n'
        f'  Subclass: TSP with Time Windows -- strongly NP-hard\n'
        f'  Search space: ({p.n_cities}-1)!/2\n'
        f'    = ~10^{cx["search_space_log10"]:.1f} permutations\n'
        f'  Exact DP: O(n^2 * 2^n) (Held-Karp)\n'
        f'  Approximation: 1.5x optimal (Christofides)\n\n'
        'DETAILS\n'
        '  A complex variation of the standard Traveling\n'
        '  Salesperson Problem featuring fuel constraints and\n'
        '  dynamic terrain multipliers. Highly challenging.\n' 
    )


def _desc_knapsack(difficulty):
    from problems.discrete.Knapsack import KnapsackProblem
    factory = getattr(KnapsackProblem, difficulty, KnapsackProblem.easy)
    p = factory(seed=42)
    cx = p.complexity_class
    n_crit = sum(1 for it in p.items if it.priority_class == 'critical')
    return (
        '=' * 56 + '\n'
        '  KNAPSACK: "Space Cargo Loading"\n'
        '=' * 56 + '\n\n'
        'SCENARIO\n'
        '  Load cargo onto a spacecraft. Each item has weight,\n'
        '  volume, power demand, and mission value. Some items\n'
        '  work better together (synergy), others conflict.\n\n'
        'OBJECTIVE\n'
        '  Select items to MAXIMIZE total value\n'
        '  without exceeding resource limits.\n\n'
        'CONSTRAINTS (3-dimensional)\n'
        f'  - Weight capacity:  {p.capacity_weight:.0f}\n'
        f'  - Volume capacity:  {p.capacity_volume:.0f}\n'
        f'  - Power capacity:   {p.capacity_power:.0f}\n'
        f'  - Must include >= {p.min_critical_items} critical items\n'
        f'    (there are {n_crit} critical items available)\n'
        f'  - Conflicts: {len(p.conflicts)} forbidden item pairs\n'
        f'  - Synergies: {len(p.synergies)} bonus item pairs\n\n'
        'VALUE FORMULA\n'
        f'  Choose x in {{0,1}}^{p.n_items}  (select/skip each item)\n\n'
        '  value = Sum(item_value_i * x_i)\n'
        '        + Sum(synergy_bonus for selected pairs)\n'
        '        - penalty_for_violations\n\n'
        '  penalty = 50 * max(0, weight_used - W_cap)\n'
        '          + 50 * max(0, volume_used - V_cap)\n'
        '          + 50 * max(0, power_used  - P_cap)\n'
        '          + 200 * number_of_conflict_violations\n'
        '          + 300 * max(0, min_critical - critical_selected)\n\n'
        'COMPLEXITY\n'
        f'  Class: {cx["class"]}\n'
        f'  Dimensions: {cx.get("dimensions", 3)}\n'
        f'  Search space: 2^{p.n_items} subsets\n'
        f'    = ~10^{cx["search_space_log10"]:.1f} subsets\n'
        f'  Standard 0/1 KP: pseudo-poly DP O(n*W)\n'
        f'  Multi-dim KP (d>=2): strongly NP-hard\n\n'
        'DETAILS\n'
        '  Multi-dimensional knapsack problem where items\n'
        '  have interdependent values (synergy/conflicts).\n' 
    )


def _desc_graph_coloring(difficulty):
    from problems.discrete.GraphColoring import GraphColoringProblem
    factory = getattr(GraphColoringProblem, difficulty, GraphColoringProblem.easy)
    p = factory(seed=42)
    cx = p.complexity_class
    n_hard = sum(1 for e in p.edges if e.is_hard)
    n_soft = len(p.edges) - n_hard
    n_pre = sum(1 for ev in p.events if ev.pre_assigned_color is not None)
    return (
        '=' * 56 + '\n'
        '  GRAPH COLORING: "Festival Scheduling"\n'
        '=' * 56 + '\n\n'
        'SCENARIO\n'
        '  Schedule festival events into time slots (colors).\n'
        '  Events sharing resources or audience should not\n'
        '  overlap. Some constraints are hard (must obey),\n'
        '  others are soft (penalty if violated).\n\n'
        'OBJECTIVE\n'
        f'  Assign {p.n_nodes} events to {p.n_colors} time slots\n'
        '  so that no conflicting events share a slot.\n'
        '  MINIMIZE total penalty.\n\n'
        'CONSTRAINTS\n'
        f'  - Hard edges: {n_hard} event pairs MUST NOT\n'
        f'    share the same time slot\n'
        f'  - Soft edges: {n_soft} pairs that SHOULD avoid\n'
        f'    overlap (penalty = weight x popularity)\n'
        f'  - Forbidden pairs: {len(p.forbidden_pairs)} pairs\n'
        f'    that cannot be in adjacent slots either\n'
        f'  - Pre-assigned: {n_pre} events already locked\n'
        f'    to specific time slots\n\n'
        'PENALTY FORMULA\n'
        '  P = Sum(weight x popularity) for same-color pairs\n'
        '    + 500 x number_of_hard_violations\n'
        '    + 300 x number_of_forbidden_violations\n'
        '    + 1000 x number_of_preassign_violations\n\n'
        'COMPLEXITY\n'
        f'  Class: {cx["class"]}\n'
        f'  k-Coloring (k>=3): NP-complete\n'
        f'  Chromatic number: NP-hard to compute\n'
        f'  Max degree: {cx.get("max_degree", "?")}\n'
        f'  Graph density: {cx.get("graph_density", "?")}\n'
        f'  Search space: {p.n_colors}^{p.n_nodes} possibilities\n'
        f'    = ~10^{cx["search_space_log10"]:.1f}\n\n'
        'DETAILS\n'
        '  Graph coloring with weighted edges, a classic\n'
        '  scheduling optimization challenge.\n' 
    )


def _desc_continuous():
    return (
        '=' * 56 + '\n'
        '  CONTINUOUS OPTIMIZATION BENCHMARK\n'
        '=' * 56 + '\n\n'
        'SCENARIO\n'
        '  Optimize benchmark mathematical functions in\n'
        '  continuous search space. Algorithms compete to\n'
        '  find the global minimum.\n\n'
        'AVAILABLE FUNCTIONS\n'
        '  Sphere     -- Convex, unimodal. f*=0 at origin.\n'
        '  Rastrigin  -- Highly multimodal (cosine landscape).\n'
        '  Rosenbrock -- Narrow curved valley. Hard to optimize.\n'
        '  Ackley     -- Many local minima + global minimum.\n'
        '  Griewank   -- Wide search space, multimodal.\n\n'
        'DETAILS\n'
        '  Used for evaluating nature-inspired metaheuristics\n'
        '  like GA, PSO, SA, DE, ABC, CS, and FA on difficult\n'
        '  mathematical surfaces.\n' 
    )


def _draw_problem_map(ax, prob, problem_key, path=None, route=None,
                      selection=None, coloring=None, difficulty='easy'):
    """Draw the problem-specific map/graph on the given matplotlib axis."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if problem_key == 'ShortestPath':
        ax.imshow(prob.elevation, cmap='terrain', alpha=0.6, origin='upper')
        import numpy as np
        terrain_colors = {0: '#c2b280', 1: '#808080', 2: '#a8d8ea', 3: '#ff4444'}
        overlay = np.zeros((*prob.terrain.shape, 4))
        for t, color in terrain_colors.items():
            mask = prob.terrain == t
            r = int(color[1:3], 16) / 255
            g = int(color[3:5], 16) / 255
            b = int(color[5:7], 16) / 255
            overlay[mask] = [r, g, b, 0.3]
        overlay[prob.terrain == 3, 3] = 0.7
        ax.imshow(overlay, origin='upper')
        for hz in prob.hazard_zones:
            c = mpatches.Circle((hz.center_col, hz.center_row), hz.radius,
                                color='#ff6b6b', alpha=0.15, lw=1.5,
                                edgecolor='#ff4444', linestyle='--')
            ax.add_patch(c)
        for wp in prob.waypoints:
            ax.scatter(wp.col, wp.row, s=120, c='#ffd93d', marker='*',
                       edgecolors='white', linewidths=0.8, zorder=10)
        if path:
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            ax.plot(cols, rows, color='#00d4aa', lw=2, alpha=0.9, zorder=8)
        ax.scatter(prob.start[1], prob.start[0], s=180, c='#00d4aa',
                   marker='o', edgecolors='white', lw=1.5, zorder=12)
        ax.scatter(prob.goal[1], prob.goal[0], s=180, c='#ff6b6b',
                   marker='s', edgecolors='white', lw=1.5, zorder=12)
        ax.set_title(f'Map ({difficulty})', fontsize=11, fontweight='bold')

    elif problem_key == 'TSP':
        for storm in prob.storm_zones:
            c = mpatches.Circle((storm.cx, storm.cy), storm.radius,
                                color='#ff6b6b', alpha=0.15, lw=1.5,
                                edgecolor='#ff4444', linestyle='--')
            ax.add_patch(c)
        if route is not None:
            full = list(route) + [route[0]]
            for i in range(len(full) - 1):
                a, b = full[i], full[i + 1]
                ca, cb = prob.cities[a], prob.cities[b]
                t = prob.terrain_multiplier[a][b]
                color = '#00d4aa' if t < 0.9 else ('#ff6b6b' if t > 1.5 else '#ffd93d')
                ax.plot([ca.x, cb.x], [ca.y, cb.y], color=color, lw=1.5, alpha=0.7, zorder=2)
        for city in prob.cities:
            mk, sz, cl = ('s', 100, '#00d4aa') if city.is_depot else ('o', 50, '#58a6ff')
            ax.scatter(city.x, city.y, s=sz, c=cl, marker=mk, zorder=5,
                       edgecolors='white', lw=0.6)
        ax.set_title(f'TSP Map ({difficulty})', fontsize=11, fontweight='bold')

    elif problem_key == 'Knapsack':
        # Show items as scatter: x=weight, y=value, size=volume
        weights = [it.weight for it in prob.items]
        values = [it.value for it in prob.items]
        volumes = [max(it.volume * 15, 20) for it in prob.items]
        colors = ['#00d4aa' if (selection and it.id in selection) else '#58a6ff'
                  for it in prob.items]
        ax.scatter(weights, values, s=volumes, c=colors, alpha=0.7, edgecolors='white', lw=0.5)
        ax.set_xlabel('Weight', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.axvline(prob.capacity_weight, color='#ff6b6b', ls='--', lw=1, alpha=0.6, label='W cap')
        ax.legend(fontsize=8, facecolor='#f0f0f0')
        ax.set_title(f'Knapsack Items ({difficulty})', fontsize=11, fontweight='bold')

    elif problem_key == 'GraphColoring':
        pos = prob.node_positions
        for edge in prob.edges:
            x = [pos[edge.u][0], pos[edge.v][0]]
            y = [pos[edge.u][1], pos[edge.v][1]]
            ax.plot(x, y, color='#8b949e', lw=0.6, alpha=0.5)
        xs = [pos[n][0] for n in pos]
        ys = [pos[n][1] for n in pos]
        if coloring:
            cmap = plt.cm.get_cmap('tab10')
            cs = [cmap(coloring.get(n, 0) % 10) for n in pos]
        else:
            cs = '#58a6ff'
        ax.scatter(xs, ys, s=60, c=cs, edgecolors='white', lw=0.5, zorder=5)
        ax.set_title(f'Graph ({difficulty})', fontsize=11, fontweight='bold')

    ax.tick_params(labelsize=7)


# ── Adapter classes for discrete → continuous metaheuristic ──────

class _KnapsackAdapter:
    """Wrap KnapsackProblem so continuous metaheuristics can solve it."""

    def __init__(self, prob):
        import numpy as np
        self._prob = prob
        self.name = prob.name
        self.dim = prob.n_items
        self.bounds = [0.0, 1.0]
        self.min_val = None
        self.opt_type = 'min'

    def evaluate(self, x):
        import numpy as np
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            selection = (x > 0.5).astype(int).tolist()
            result = self._prob.evaluate(selection)
            return -result['total_value']
        return np.array([self.evaluate(row) for row in x])

    def get_plotting_data(self, points=100):
        return None


class _GraphColoringAdapter:
    """Wrap GraphColoringProblem so continuous metaheuristics can solve it."""

    def __init__(self, prob):
        self._prob = prob
        self.name = prob.name
        self.dim = prob.n_nodes
        self.bounds = [0.0, float(prob.n_colors)]
        self.min_val = 0.0
        self.opt_type = 'min'

    def evaluate(self, x):
        import numpy as np
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            coloring = {i: min(int(xi), self._prob.n_colors - 1)
                        for i, xi in enumerate(x)}
            result = self._prob.evaluate(coloring)
            return result['total_penalty']
        return np.array([self.evaluate(row) for row in x])

    def get_plotting_data(self, points=100):
        return None


def run_single(problem, algo, iterations, pop_size, dim, save_result, difficulty='easy'):
    """Run a single algorithm.  Always render preview; save to results/ only if requested."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Problem registry ─────────────────────────────────────────
    continuous_problems = {
        'Sphere': ('problems.continuous.Sphere', 'Sphere'),
        'Rastrigin': ('problems.continuous.Rastrigin', 'Rastrigin'),
        'Rosenbrock': ('problems.continuous.Rosenbrock', 'Rosenbrock'),
        'Ackley': ('problems.continuous.Ackley', 'Ackley'),
        'Griewank': ('problems.continuous.Griewank', 'Griewank'),
    }
    discrete_problems = {
        'ShortestPath': ('problems.discrete.ShortestPath', 'ShortestPathProblem'),
        'TSP': ('problems.discrete.TSP', 'TSPProblem'),
        'Knapsack': ('problems.discrete.Knapsack', 'KnapsackProblem'),
        'GraphColoring': ('problems.discrete.GraphColoring', 'GraphColoringProblem'),
    }

    # ── Algorithm registry ───────────────────────────────────────
    algos = {
        'GA': ('algorithms.evolution.GA', 'GeneticAlgorithm'),
        'PSO': ('algorithms.biology.PSO', 'ParticleSwarmOptimization'),
        'DE': ('algorithms.evolution.DE', 'DifferentialEvolution'),
        'SA': ('algorithms.physics.SA', 'SimulatedAnnealing'),
        'HC': ('algorithms.classical.Hill_climbing', 'HillClimbing'),
        'ABC': ('algorithms.biology.ABC', 'ArtificialBeeColony'),
        'CS': ('algorithms.biology.CS', 'CuckooSearch'),
        'FA': ('algorithms.biology.FA', 'FireflyAlgorithm'),
        'TLBO': ('algorithms.human.TLBO', 'TLBO'),
        'ACO': ('algorithms.biology.ACO', 'AntColonyOptimization'),
        'BFS': ('algorithms.classical.BFS', 'BreadthFirstSearch'),
        'DFS': ('algorithms.classical.DFS', 'DepthFirstSearch'),
        'A*': ('algorithms.classical.A_star', 'AStar'),
    }

    is_continuous = problem in continuous_problems
    is_graph_search = algo in ('BFS', 'DFS', 'A*')
    is_aco = algo == 'ACO'

    # ── Load algorithm class ─────────────────────────────────────
    moda, clsnamea = algos[algo]
    ma = __import__(moda, fromlist=[clsnamea])
    alg_cls = getattr(ma, clsnamea)

    preview_dir = os.path.join(BASE_DIR, 'results', 'visualize', '.preview')
    os.makedirs(preview_dir, exist_ok=True)

    # ── Branch: continuous problem ───────────────────────────────
    if is_continuous:
        modp, clsnamep = continuous_problems[problem]
        m = __import__(modp, fromlist=[clsnamep])
        prob_cls = getattr(m, clsnamep)
        prob = prob_cls(dim=dim)
        params = {'pop_size': pop_size}
        alg = alg_cls(prob, params=params)
        alg.solve(iterations=iterations)

        vals = [h.get('global_best_fit') for h in alg.history]
        fig, (ax_conv, ax_info) = plt.subplots(1, 2, figsize=(14, 5),
                                                gridspec_kw={'width_ratios': [2, 1]})
        ax_conv.plot(range(1, len(vals) + 1), vals, marker='o', markersize=2)
        ax_conv.set_title(f'{algo} on {problem} (dim={dim})', fontsize=12, fontweight='bold')
        ax_conv.set_xlabel('Iteration')
        ax_conv.set_ylabel('Global Best Fitness')
        ax_conv.grid(alpha=0.25)
        best = vals[-1] if vals else '?'

        # Problem info panel
        info = (f'{problem} (dim={dim})\n'
                f'Bounds: {prob.bounds}\n'
                f'Global min: {prob.min_val}\n'
                f'Opt type: {prob.opt_type}\n\n'
                f'Algorithm: {algo}\n'
                f'Pop size: {pop_size}\n'
                f'Iterations: {iterations}\n'
                f'Best found: {best}')
        ax_info.axis('off')
        ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                     verticalalignment='top', fontsize=10, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#ccc'))

    # ── Branch: graph search (BFS / DFS / A*) ───────────────────
    elif is_graph_search:
        modp, clsnamep = discrete_problems[problem]
        m = __import__(modp, fromlist=[clsnamep])
        prob_cls = getattr(m, clsnamep)
        factory = getattr(prob_cls, difficulty, prob_cls.easy)
        prob = factory(seed=42)
        alg = alg_cls(prob)
        alg.solve()

        visited_counts = [len(h.get('visited_nodes', [])) for h in alg.history]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                                 gridspec_kw={'width_ratios': [2, 2, 1]})
        ax_conv, ax_map, ax_info = axes
        ax_conv.plot(range(1, len(visited_counts) + 1), visited_counts,
                     marker='o', markersize=2, color='#58a6ff')
        ax_conv.set_title(f'{algo} on {problem} ({difficulty})', fontsize=12, fontweight='bold')
        ax_conv.set_xlabel('Step')
        ax_conv.set_ylabel('Visited Nodes')
        ax_conv.grid(alpha=0.25)
        best = alg.best_fitness if alg.best_fitness is not None else '?'

        # Map panel: show grid with solution path
        path = alg.best_solution
        _draw_problem_map(ax_map, prob, problem, path=path, difficulty=difficulty)

        # Problem info panel with detailed difficulty info
        info = get_problem_info(prob)
        info += (f'\n\nAlgorithm: {algo}\n'
                 f'Difficulty: {difficulty}\n'
                 f'Total steps: {len(alg.history)}\n'
                 f'Total visited: {len(alg.history[-1].get("visited_nodes", []))}\n'
                 f'Path cost: {best}')
        ax_info.axis('off')
        ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                     verticalalignment='top', fontsize=9, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#ccc'))

    # ── Branch: ACO / combinatorial ──────────────────────────────
    elif is_aco:
        modp, clsnamep = discrete_problems[problem]
        m = __import__(modp, fromlist=[clsnamep])
        prob_cls = getattr(m, clsnamep)
        factory = getattr(prob_cls, difficulty, prob_cls.easy)
        prob = factory(seed=42)
        params = {'n_ants': pop_size}
        alg = alg_cls(prob, params=params)
        alg.solve(iterations=iterations)

        vals = [h.get('global_best_fit') for h in alg.history]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                                 gridspec_kw={'width_ratios': [2, 2, 1]})
        ax_conv, ax_map, ax_info = axes
        ax_conv.plot(range(1, len(vals) + 1), vals, marker='o', markersize=2, color='#00d4aa')
        ax_conv.set_title(f'{algo} on {problem} ({difficulty})', fontsize=12, fontweight='bold')
        ax_conv.set_xlabel('Iteration')
        ax_conv.set_ylabel('Best Tour Cost')
        ax_conv.grid(alpha=0.25)
        best = vals[-1] if vals else '?'

        # Map panel: show cities with best route
        route = alg.best_solution
        _draw_problem_map(ax_map, prob, problem, route=route, difficulty=difficulty)

        # Problem info panel
        info = get_problem_info(prob)
        info += (f'\n\nAlgorithm: {algo}\n'
                 f'Difficulty: {difficulty}\n'
                 f'Ants: {pop_size}\n'
                 f'Iterations: {iterations}\n'
                 f'Best cost: {best}')
        ax_info.axis('off')
        ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                     verticalalignment='top', fontsize=9, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#ccc'))

    # ── Branch: Knapsack (metaheuristic via adapter) ─────────────
    elif problem == 'Knapsack':
        modp, clsnamep = discrete_problems[problem]
        m = __import__(modp, fromlist=[clsnamep])
        prob_cls = getattr(m, clsnamep)
        factory = getattr(prob_cls, difficulty, prob_cls.easy)
        prob = factory(seed=42)
        adapter = _KnapsackAdapter(prob)
        params = {'pop_size': pop_size}
        alg = alg_cls(adapter, params=params)
        alg.solve(iterations=iterations)

        vals = [h.get('global_best_fit') for h in alg.history]
        display_vals = [-v if v is not None else 0 for v in vals]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                                 gridspec_kw={'width_ratios': [2, 2, 1]})
        ax_conv, ax_map, ax_info = axes
        ax_conv.plot(range(1, len(display_vals) + 1), display_vals,
                     marker='o', markersize=2, color='#58a6ff')
        ax_conv.set_title(f'{algo} on Knapsack ({difficulty})', fontsize=12, fontweight='bold')
        ax_conv.set_xlabel('Iteration')
        ax_conv.set_ylabel('Total Value')
        ax_conv.grid(alpha=0.25)

        import numpy as np
        best_sol = alg.best_solution
        selection = set(int(i) for i, v in enumerate(np.asarray(best_sol) > 0.5) if v) \
            if best_sol is not None else None
        _draw_problem_map(ax_map, prob, 'Knapsack', selection=selection, difficulty=difficulty)

        best = display_vals[-1] if display_vals else '?'
        info = get_problem_info(prob)
        info += (f'\n\nAlgorithm: {algo}\n'
                 f'Difficulty: {difficulty}\n'
                 f'Pop size: {pop_size}\n'
                 f'Iterations: {iterations}\n'
                 f'Best value: {best}')
        ax_info.axis('off')
        ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                     verticalalignment='top', fontsize=9, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#ccc'))

    # ── Branch: GraphColoring (metaheuristic via adapter) ────────
    elif problem == 'GraphColoring':
        modp, clsnamep = discrete_problems[problem]
        m = __import__(modp, fromlist=[clsnamep])
        prob_cls = getattr(m, clsnamep)
        factory = getattr(prob_cls, difficulty, prob_cls.easy)
        prob = factory(seed=42)
        adapter = _GraphColoringAdapter(prob)
        params = {'pop_size': pop_size}
        alg = alg_cls(adapter, params=params)
        alg.solve(iterations=iterations)

        vals = [h.get('global_best_fit') for h in alg.history]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                                 gridspec_kw={'width_ratios': [2, 2, 1]})
        ax_conv, ax_map, ax_info = axes
        ax_conv.plot(range(1, len(vals) + 1), vals,
                     marker='o', markersize=2, color='#a371f7')
        ax_conv.set_title(f'{algo} on GraphColoring ({difficulty})', fontsize=12, fontweight='bold')
        ax_conv.set_xlabel('Iteration')
        ax_conv.set_ylabel('Penalty')
        ax_conv.grid(alpha=0.25)

        import numpy as np
        best_sol = alg.best_solution
        coloring = {i: min(int(v), prob.n_colors - 1)
                    for i, v in enumerate(np.asarray(best_sol))} \
            if best_sol is not None else None
        _draw_problem_map(ax_map, prob, 'GraphColoring', coloring=coloring, difficulty=difficulty)

        best = vals[-1] if vals else '?'
        info = get_problem_info(prob)
        info += (f'\n\nAlgorithm: {algo}\n'
                 f'Difficulty: {difficulty}\n'
                 f'Pop size: {pop_size}\n'
                 f'Iterations: {iterations}\n'
                 f'Best penalty: {best}')
        ax_info.axis('off')
        ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                     verticalalignment='top', fontsize=9, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#ccc'))

    else:
        raise ValueError(f'Unsupported combination: {algo} on {problem}')

    preview = os.path.join(preview_dir, f'{problem}_{algo}_conv.png')
    fig.tight_layout()
    fig.savefig(preview, dpi=120, bbox_inches='tight')
    plt.close(fig)
    images = [preview]

    # Print detailed problem info to log
    if not is_continuous and hasattr(prob, 'complexity_class'):
        print(f'--- Problem Info ({difficulty}) ---')
        print(str(prob))

    # Optionally save to results/
    if save_result:
        out_dir = os.path.join(BASE_DIR, 'results')
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f'{problem}_{algo}_conv.png')
        shutil.copyfile(preview, out)
        images.append(out)
        print(f'Saved {out}')

    print(f'{algo} on {problem}: best = {best}')
    return images


def run_visualize(category, problem, iterations, speed, save_gif, difficulty='easy'):
    """Pop up real-time matplotlib visualization window. Save to results/ if requested."""
    import os
    from visualization.runner import run_realtime_viz
    
    viz_dir = os.path.join(BASE_DIR, 'results', 'visualize')
    
    print("Launching real-time visualization window...")
    run_realtime_viz(category, problem, iterations, speed, save_gif, difficulty, viz_dir)
    print("Visualization finished.")
    return []


# ── Worker entry ─────────────────────────────────────────────────────

def _worker_entry(job, result_queue):
    action = job.get('action')
    import matplotlib
    if action == 'visualize':
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')

    capture = io.StringIO()
    try:
        with redirect_stdout(capture):
            action = job['action']
            if action == 'compare':
                images = run_compare()
            elif action == 'gif':
                images = run_gif()
            elif action == 'run':
                images = run_single(
                    problem=job['problem'], algo=job['algo'],
                    iterations=job['iterations'], pop_size=job['pop_size'],
                    dim=job['dim'], save_result=job.get('save_result', False),
                    difficulty=job.get('difficulty', 'easy'),
                )
            elif action == 'visualize':
                images = run_visualize(
                    category=job['category'], problem=job['problem'],
                    iterations=job['iterations'], speed=job['speed'],
                    save_gif=job['save_gif'],
                    difficulty=job.get('difficulty', 'easy'),
                )
            else:
                raise ValueError(f'Unknown action: {action}')

        result_queue.put({
            'status': 'ok',
            'logs': capture.getvalue().strip(),
            'images': images if 'images' in locals() else [],
            'description': build_problem_description(
                job.get('category', ''), job.get('difficulty', 'easy')
            ) if action == 'visualize' else '',
        })
    except Exception:
        result_queue.put({
            'status': 'error',
            'logs': capture.getvalue().strip(),
            'error': traceback.format_exc(),
        })


# ── Flet UI ──────────────────────────────────────────────────────────

def launch_app(page: ft.Page):
    page.title = 'SNIA — Search & Nature-Inspired Algorithms'
    page.window.maximized = True
    page.padding = 20
    page.bgcolor = '#0d1117'
    page.theme_mode = ft.ThemeMode.DARK

    running = {'value': False}
    runtime = {'process': None, 'queue': None}

    # ── Controls ─────────────────────────────────────────────────────

    mode = ft.Dropdown(
        label='Workflow', value='visualize', width=200,
        options=[
            ft.dropdown.Option('run', 'Run Single'),
            ft.dropdown.Option('visualize', 'Visualize'),
            ft.dropdown.Option('compare', 'Compare All'),
            ft.dropdown.Option('gif', 'Generate GIFs'),
        ],
    )

    # Run-specific
    problem = ft.Dropdown(
        label='Problem', value='Sphere', width=160,
        options=[ft.dropdown.Option(p) for p in ALL_PROBLEMS],
    )
    algo = ft.Dropdown(
        label='Algorithm', value='PSO', width=160,
        options=[ft.dropdown.Option(a) for a in CONTINUOUS_ALGOS],
    )
    difficulty = ft.Dropdown(
        label='Difficulty', value='easy', width=120,
        options=[ft.dropdown.Option(d) for d in ['easy', 'medium', 'hard', 'extreme']],
        visible=False,
    )
    pop_size = ft.TextField(label='Pop Size', value='40', width=100)
    dim = ft.TextField(label='Dim', value='10', width=80)
    save_result = ft.Switch(label='Save', value=False)

    # Visualize-specific
    category = ft.Dropdown(
        label='Category', value='local', width=180,
        options=[
            ft.dropdown.Option('uninformed', 'Uninformed (BFS/DFS)'),
            ft.dropdown.Option('informed', 'Informed (A*)'),
            ft.dropdown.Option('local', 'Local Search'),
            ft.dropdown.Option('tsp', 'TSP (ACO)'),
            ft.dropdown.Option('knapsack', 'Knapsack (GA)'),
            ft.dropdown.Option('graph_coloring', 'Graph Coloring (GA)'),
            ft.dropdown.Option('all', 'Dashboard (All)'),
        ],
    )
    viz_problem = ft.Dropdown(
        label='Problem', value='Sphere', width=160,
        options=[ft.dropdown.Option(p) for p in CONTINUOUS_PROBLEMS],
    )
    viz_difficulty = ft.Dropdown(
        label='Difficulty', value='easy', width=120,
        options=[ft.dropdown.Option(d) for d in ['easy', 'medium', 'hard', 'extreme']],
        visible=False,
    )
    speed = ft.TextField(label='Speed (ms)', value='50', width=100)
    save_gif = ft.Switch(label='Save GIF', value=False)

    # Shared
    iterations = ft.TextField(label='Iterations', value='100', width=120)

    # Field groups
    run_fields = ft.Row(
        [problem, algo, iterations, pop_size, dim, difficulty, save_result],
        spacing=10, visible=False,
    )
    viz_fields = ft.Row(
        [category, viz_problem, viz_difficulty, iterations, speed, save_gif],
        spacing=10, visible=True,
    )

    # Buttons
    run_button = ft.ElevatedButton(
        'Run', icon=ft.Icons.PLAY_ARROW,
        style=ft.ButtonStyle(bgcolor='#238636', color='white'),
    )
    stop_button = ft.OutlinedButton('Stop', icon=ft.Icons.STOP, disabled=True)
    clear_button = ft.OutlinedButton('Clear', icon=ft.Icons.DELETE_OUTLINE)

    status = ft.Text('Ready', size=13, color='#3fb950', weight=ft.FontWeight.W_500)
    progress = ft.ProgressBar(width=300, visible=False, color='#58a6ff')

    # ── Preview area ─────────────────────────────────────────────────

    preview_image = ft.Image(
        src='', fit=ft.ImageFit.CONTAIN, visible=False,
    )

    no_preview = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.IMAGE_OUTLINED, size=48, color='#484f58'),
                ft.Text('Run a workflow to see results here',
                        color='#8b949e', size=14),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=8,
        ),
        alignment=ft.alignment.center,
        height=400,
    )

    # ── GIF player state & controls ──────────────────────────────────

    gif_state = {
        'frames': [],       # base64 PNG strings per frame
        'durations': [],    # ms per frame
        'index': 0,
        'playing': False,
        'timer': None,
    }

    gif_frame_text = ft.Text('', size=12, color='#8b949e')
    gif_play_btn = ft.IconButton(
        icon=ft.Icons.PLAY_ARROW, icon_color='#58a6ff',
        tooltip='Play / Pause (Space)', visible=False,
    )
    gif_prev_btn = ft.IconButton(
        icon=ft.Icons.SKIP_PREVIOUS, icon_color='#8b949e',
        tooltip='Previous frame', visible=False,
    )
    gif_next_btn = ft.IconButton(
        icon=ft.Icons.SKIP_NEXT, icon_color='#8b949e',
        tooltip='Next frame', visible=False,
    )
    gif_speed_text = ft.Text('1.0x', size=12, color='#8b949e')
    gif_speed_slider = ft.Slider(
        min=0.25, max=3.0, value=1.0, divisions=11,
        label='{value}x', width=120, visible=False,
    )
    gif_slider = ft.Slider(min=0, max=1, value=0, visible=False, expand=True)

    gif_controls = ft.Row(
        [
            gif_prev_btn, gif_play_btn, gif_next_btn,
            gif_frame_text,
            gif_slider,
            gif_speed_text, gif_speed_slider,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        visible=False,
        spacing=4,
    )

    # ── Problem description tab content ──────────────────────────────

    desc_text = ft.Text(
        value='', size=13, color='#c9d1d9',
        font_family='monospace', selectable=True,
    )
    desc_container = ft.Container(
        content=ft.Column(
            [desc_text],
            scroll=ft.ScrollMode.AUTO,
            spacing=0,
        ),
        bgcolor='#0d1117',
        padding=20,
        expand=True,
    )

    # ── Tab-based preview area ───────────────────────────────────────

    viz_tab_content = ft.Column(
        [no_preview, preview_image, gif_controls],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        scroll=ft.ScrollMode.AUTO,
    )

    preview_tabs = ft.Tabs(
        selected_index=0,
        animation_duration=200,
        tabs=[
            ft.Tab(
                text='Algorithm Visualization',
                content=ft.Container(
                    content=viz_tab_content,
                    bgcolor='#161b22',
                    padding=10,
                    expand=True,
                ),
            ),
            ft.Tab(
                text='Problem Description',
                content=ft.Container(
                    content=desc_container,
                    bgcolor='#161b22',
                    padding=10,
                    expand=True,
                ),
            ),
        ],
        expand=True,
    )

    preview_area = ft.Container(
        content=preview_tabs,
        bgcolor='#161b22',
        border_radius=8,
        border=ft.border.all(1, '#30363d'),
        padding=0,
        expand=True,
    )

    # ── Log (bottom) ─────────────────────────────────────────────────

    output = ft.TextField(
        label='Log', multiline=True, min_lines=3, max_lines=5,
        read_only=True, value='', text_size=12, border_color='#30363d',
    )

    # ── Helpers ──────────────────────────────────────────────────────

    def safe_update():
        try:
            page.update()
        except Exception:
            pass

    def append_log(text):
        output.value = (output.value + ('\n' if output.value else '') + text).strip()
        safe_update()

    # ── GIF player helpers ────────────────────────────────────────────

    def _extract_gif_frames(path):
        """Extract individual frames from a GIF as base64 PNG strings."""
        frames = []
        durations = []
        with PILImage.open(path) as img:
            for i in range(img.n_frames):
                img.seek(i)
                frame = img.convert('RGBA')
                buf = io.BytesIO()
                frame.save(buf, format='PNG')
                frames.append(base64.b64encode(buf.getvalue()).decode())
                durations.append(img.info.get('duration', 100) or 100)
        return frames, durations

    def _gif_show_frame(index):
        """Display a specific GIF frame by index."""
        if not gif_state['frames']:
            return
        index = index % len(gif_state['frames'])
        gif_state['index'] = index
        preview_image.src_base64 = gif_state['frames'][index]
        gif_frame_text.value = f'{index + 1} / {len(gif_state["frames"])}'
        gif_slider.value = index
        safe_update()

    def _gif_advance():
        """Auto-advance to the next frame (called by timer)."""
        if not gif_state['playing'] or not gif_state['frames']:
            return
        idx = (gif_state['index'] + 1) % len(gif_state['frames'])
        _gif_show_frame(idx)
        speed_mult = gif_speed_slider.value if gif_speed_slider.value else 1.0
        duration = gif_state['durations'][idx] / (1000.0 * speed_mult)
        t = threading.Timer(max(duration, 0.01), _gif_advance)
        t.daemon = True
        gif_state['timer'] = t
        t.start()

    def _gif_stop_timer():
        """Cancel the running playback timer."""
        gif_state['playing'] = False
        t = gif_state.get('timer')
        if t is not None:
            t.cancel()
            gif_state['timer'] = None

    def _gif_play_pause(_=None):
        if gif_state['playing']:
            _gif_stop_timer()
            gif_play_btn.icon = ft.Icons.PLAY_ARROW
        else:
            gif_state['playing'] = True
            gif_play_btn.icon = ft.Icons.PAUSE
            _gif_advance()
        safe_update()

    def _gif_prev(_=None):
        if gif_state['frames']:
            _gif_show_frame(gif_state['index'] - 1)

    def _gif_next(_=None):
        if gif_state['frames']:
            _gif_show_frame(gif_state['index'] + 1)

    def _gif_slider_change(e):
        if gif_state['frames']:
            _gif_show_frame(int(e.control.value))

    def _gif_speed_change(e):
        val = e.control.value
        gif_speed_text.value = f'{val:.2g}x'
        safe_update()

    gif_play_btn.on_click = _gif_play_pause
    gif_prev_btn.on_click = _gif_prev
    gif_next_btn.on_click = _gif_next
    gif_slider.on_change = _gif_slider_change
    gif_speed_slider.on_change = _gif_speed_change

    def _hide_gif_controls():
        gif_controls.visible = False
        gif_prev_btn.visible = False
        gif_play_btn.visible = False
        gif_next_btn.visible = False
        gif_slider.visible = False
        gif_speed_slider.visible = False
        gif_speed_text.visible = False
        gif_frame_text.visible = False

    def _show_gif_controls(n_frames):
        gif_slider.max = max(n_frames - 1, 1)
        gif_slider.value = 0
        gif_frame_text.value = f'1 / {n_frames}'
        gif_play_btn.icon = ft.Icons.PLAY_ARROW
        gif_speed_text.value = f'{gif_speed_slider.value:.2g}x'
        gif_controls.visible = True
        gif_prev_btn.visible = True
        gif_play_btn.visible = True
        gif_next_btn.visible = True
        gif_slider.visible = True
        gif_speed_slider.visible = True
        gif_speed_text.visible = True
        gif_frame_text.visible = True

    def show_preview(image_paths, description=''):
        # Update problem description tab
        if description:
            desc_text.value = description

        # Stop any existing GIF playback
        _gif_stop_timer()

        if not image_paths or not os.path.exists(image_paths[0]):
            preview_image.visible = False
            _hide_gif_controls()
            no_preview.content.controls[1].value = 'Visualization finished in external window.'
            no_preview.visible = True
            safe_update()
            return

        first = image_paths[0]
        if not os.path.exists(first):
            return

        # Update problem description tab
        if description:
            desc_text.value = description
        else:
            desc_text.value = ''

        # Stop any existing GIF playback
        _gif_stop_timer()

        ext = first.rsplit('.', 1)[-1].lower()

        if ext == 'gif':
            # Extract frames and initialise player
            frames, durations = _extract_gif_frames(first)
            gif_state['frames'] = frames
            gif_state['durations'] = durations
            gif_state['index'] = 0
            gif_state['playing'] = False

            _show_gif_controls(len(frames))

            # Show first frame
            preview_image.src_base64 = frames[0] if frames else ''
            preview_image.src = ''
            preview_image.visible = True
            no_preview.visible = False

            # Auto-play and switch to visualization tab
            gif_state['playing'] = True
            gif_play_btn.icon = ft.Icons.PAUSE
            preview_tabs.selected_index = 0
            safe_update()
            _gif_advance()
        else:
            # Static image — no GIF controls
            gif_state['frames'] = []
            _hide_gif_controls()

            with open(first, 'rb') as f:
                data = base64.b64encode(f.read()).decode()
            preview_image.src_base64 = data
            preview_image.src = ''
            preview_image.visible = True
            no_preview.visible = False
            preview_tabs.selected_index = 0
            safe_update()

    def parse_int(field, name):
        try:
            v = int(field.value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{name} must be an integer') from exc
        if v <= 0:
            raise ValueError(f'{name} must be > 0')
        return v

    def set_running(is_running, msg):
        running['value'] = is_running
        run_button.disabled = is_running
        stop_button.disabled = not is_running
        progress.visible = is_running
        status.value = msg
        status.color = '#58a6ff' if is_running else '#3fb950'
        safe_update()

    # ── Result collector ─────────────────────────────────────────────

    def collect_result(proc, rq):
        try:
            proc.join()
        except Exception:
            return

        try:
            payload = rq.get_nowait()
        except (Empty, EOFError, OSError):
            payload = None

        runtime['process'] = None
        runtime['queue'] = None

        if payload is None:
            set_running(False, 'Stopped')
            return

        if payload.get('logs'):
            append_log(payload['logs'])

        images = payload.get('images') or []
        description = payload.get('description', '')
        show_preview(images, description=description)

        if payload.get('status') == 'ok':
            msg = 'Done'
            if images:
                msg += f'  ({len(images)} file(s))'
            set_running(False, msg)
            return

        append_log('ERROR:\n' + payload.get('error', 'Unknown'))
        status.value = 'Failed'
        status.color = '#f85149'
        running['value'] = False
        run_button.disabled = False
        stop_button.disabled = True
        progress.visible = False
        safe_update()

    # ── Build job ────────────────────────────────────────────────────

    def build_job():
        sel = mode.value
        if sel == 'compare':
            return {'action': 'compare'}
        if sel == 'gif':
            return {'action': 'gif'}
        if sel == 'run':
            prob_val = problem.value
            algo_val = algo.value
            if not algo_val:
                raise ValueError(f'No compatible algorithm for {prob_val}')
            is_discrete = prob_val in DISCRETE_PROBLEMS
            is_graph = algo_val in ('BFS', 'DFS', 'A*')
            job = {
                'action': 'run',
                'problem': prob_val, 'algo': algo_val,
                'iterations': parse_int(iterations, 'Iterations'),
                'pop_size': parse_int(pop_size, 'Pop Size') if not is_graph else 1,
                'dim': parse_int(dim, 'Dim') if not is_discrete else 2,
                'save_result': save_result.value,
                'difficulty': difficulty.value if is_discrete else 'easy',
            }
            return job
        if sel == 'visualize':
            cat = category.value
            # All discrete categories get difficulty
            discrete_cats = ('uninformed', 'informed', 'all', 'tsp', 'knapsack', 'graph_coloring')
            diff = viz_difficulty.value if cat in discrete_cats else 'easy'
            return {
                'action': 'visualize',
                'category': cat, 'problem': viz_problem.value,
                'iterations': parse_int(iterations, 'Iterations'),
                'speed': parse_int(speed, 'Speed'),
                'save_gif': save_gif.value,
                'difficulty': diff,
            }
        raise ValueError(f'Unknown workflow: {sel}')

    # ── Visibility & compatibility ──────────────────────────────────

    def update_algo_options(_=None):
        """Update algorithm dropdown based on selected problem."""
        prob_val = problem.value
        valid = compatible_algos(prob_val)
        algo.options = [ft.dropdown.Option(a) for a in valid]
        if not valid:
            algo.value = None
            algo.disabled = True
        else:
            algo.disabled = False
            if algo.value not in valid:
                algo.value = valid[0]

        is_discrete = prob_val in DISCRETE_PROBLEMS
        is_graph = algo.value in ('BFS', 'DFS', 'A*') if algo.value else False
        # Show/hide fields based on problem type
        difficulty.visible = is_discrete
        dim.visible = not is_discrete
        pop_size.visible = not is_graph
        # Graph search doesn't use iterations
        iterations.visible = not is_graph or mode.value != 'run'
        # Disable run when no algorithm is available
        run_button.disabled = running['value'] or (not valid and mode.value == 'run')
        safe_update()

    problem.on_change = update_algo_options
    algo.on_change = update_algo_options

    def update_viz_problem(_=None):
        """Update viz problem options and difficulty visibility based on category."""
        cat = category.value
        if cat in ('uninformed', 'informed'):
            # ShortestPath only – fixed problem, user picks difficulty
            viz_problem.options = [ft.dropdown.Option('ShortestPath')]
            viz_problem.value = 'ShortestPath'
            viz_problem.visible = False
            viz_difficulty.visible = True
        elif cat == 'tsp':
            viz_problem.options = [ft.dropdown.Option('TSP')]
            viz_problem.value = 'TSP'
            viz_problem.visible = False
            viz_difficulty.visible = True
        elif cat == 'knapsack':
            viz_problem.options = [ft.dropdown.Option('Knapsack')]
            viz_problem.value = 'Knapsack'
            viz_problem.visible = False
            viz_difficulty.visible = True
        elif cat == 'graph_coloring':
            viz_problem.options = [ft.dropdown.Option('GraphColoring')]
            viz_problem.value = 'GraphColoring'
            viz_problem.visible = False
            viz_difficulty.visible = True
        elif cat == 'all':
            viz_problem.options = [ft.dropdown.Option(p) for p in CONTINUOUS_PROBLEMS]
            if viz_problem.value not in CONTINUOUS_PROBLEMS:
                viz_problem.value = 'Sphere'
            viz_problem.visible = True
            viz_difficulty.visible = True
        else:
            # local search – continuous problems only
            viz_problem.options = [ft.dropdown.Option(p) for p in CONTINUOUS_PROBLEMS]
            if viz_problem.value not in CONTINUOUS_PROBLEMS:
                viz_problem.value = 'Sphere'
            viz_problem.visible = True
            viz_difficulty.visible = False
        safe_update()

    category.on_change = update_viz_problem

    def update_visibility(_=None):
        sel = mode.value
        is_run = sel == 'run'
        is_viz = sel == 'visualize'

        run_fields.visible = is_run
        viz_fields.visible = is_viz

        # Move shared iterations to the active row
        if is_run:
            if iterations in viz_fields.controls:
                viz_fields.controls.remove(iterations)
            if iterations not in run_fields.controls:
                run_fields.controls.insert(2, iterations)
            update_algo_options()
        elif is_viz:
            if iterations in run_fields.controls:
                run_fields.controls.remove(iterations)
            if iterations not in viz_fields.controls:
                viz_fields.controls.insert(3, iterations)
            update_viz_problem()

        safe_update()

    mode.on_change = update_visibility

    # ── Handlers ─────────────────────────────────────────────────────

    def on_stop(_):
        proc = runtime.get('process')
        if not proc or not proc.is_alive():
            return
        proc.terminate()
        proc.join(timeout=2)
        runtime['process'] = None
        runtime['queue'] = None
        set_running(False, 'Stopped by user')
        append_log('Stopped.')

    def on_run(_):
        if running['value']:
            return
        try:
            job = build_job()
        except Exception as exc:
            append_log(f'ERROR: {exc}')
            return

        action = job['action']
        details = ''
        if action == 'visualize':
            details = f"  [{job.get('category','')}]"
        elif action == 'run':
            details = f"  [{job.get('algo','')} on {job.get('problem','')}]"
        append_log(f'\u25b6 {action}{details}')
        set_running(True, f'Running {action}...')

        rq = mp.Queue()
        proc = mp.Process(target=_worker_entry, args=(job, rq), daemon=True)
        runtime['queue'] = rq
        runtime['process'] = proc
        proc.start()

        threading.Thread(target=collect_result, args=(proc, rq), daemon=True).start()

    def on_clear(_):
        _gif_stop_timer()
        gif_state['frames'] = []
        _hide_gif_controls()
        output.value = ''
        preview_image.visible = False
        preview_image.src = ''
        preview_image.src_base64 = None
        no_preview.visible = True
        safe_update()

    run_button.on_click = on_run
    stop_button.on_click = on_stop
    clear_button.on_click = on_clear

    # ── Page layout ──────────────────────────────────────────────────

    update_visibility()

    page.add(
        # Header
        ft.Row(
            [
                ft.Text('SNIA', size=26, weight=ft.FontWeight.BOLD, color='#e6edf3'),
                ft.Text('Search & Nature-Inspired Algorithms',
                        size=13, color='#8b949e'),
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.END,
        ),
        ft.Divider(height=1, color='#21262d'),

        # Control bar
        ft.Row(
            [mode, run_fields, viz_fields],
            spacing=15,
            vertical_alignment=ft.CrossAxisAlignment.END,
            wrap=True,
        ),

        # Buttons + status
        ft.Row(
            [run_button, stop_button, clear_button,
             ft.Container(width=10),
             progress, status],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        ft.Divider(height=1, color='#21262d'),

        # Preview (expand)
        preview_area,

        # Log bottom
        output,
    )


if __name__ == '__main__':
    ft.app(target=launch_app)
