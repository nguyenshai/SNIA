"""
Shortest Path Problem: "Mars Rover Navigation"
===============================================
Navigate a rover across Mars terrain from start to goal:
  - Elevation map -- generated with diamond-square fractal noise
  - Terrain types -- sand (slow), rock (medium), smooth (fast), lava (impassable)
  - Energy cost -- depends on elevation change + terrain type
  - Hazard zones -- radiation areas with risk penalty (risk-reward shortcuts)
  - Waypoints -- must visit science stations before reaching goal
  - 8-directional movement with diagonal sqrt(2) cost multiplier

Complexity Class:
  - Standard shortest path (positive weights): P via Dijkstra O((V+E)logV)
  - With mandatory waypoints (Steiner path): NP-hard
  - k waypoints: O(k! * SP) via permutation + shortest-path subroutine
  - Grid graph: O(n^2 log n) for nxn grid with Dijkstra
  - Without waypoints: polynomial (Dijkstra/A*)
  - With waypoints: reduces to TSP on waypoint subset

Suitable for testing: BFS, DFS, A*
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# Terrain type constants
TERRAIN_SAND = 0     # slow (cost x1.5)
TERRAIN_ROCK = 1     # medium (cost x1.2)
TERRAIN_SMOOTH = 2   # fast (cost x0.8)
TERRAIN_LAVA = 3     # impassable

TERRAIN_NAMES = {0: 'sand', 1: 'rock', 2: 'smooth', 3: 'lava'}
TERRAIN_COSTS = {0: 1.5, 1: 1.2, 2: 0.8, 3: float('inf')}
TERRAIN_COLORS = {0: '#c2b280', 1: '#808080', 2: '#a8d8ea', 3: '#ff4444'}


@dataclass
class HazardZone:
    center_row: int
    center_col: int
    radius: float
    risk_level: float    # 0.0-1.0, probability of damage
    damage: float        # energy cost if hit


@dataclass
class Waypoint:
    row: int
    col: int
    reward: float        # bonus for visiting
    name: str


@dataclass
class ShortestPathProblem:
    """Grid-based shortest path with terrain, hazards, and waypoints.

    Complexity Classes:
        - Basic shortest path: P (Dijkstra: O((V+E)logV))
        - On grid (nxn): O(n^2 log n) via Dijkstra
        - With k mandatory waypoints: NP-hard (Steiner path)
        - Waypoint ordering: O(k! * n^2 log n) exact
        - Without waypoints: polynomial
        - BFS optimal for unweighted, Dijkstra for weighted
        - A* with admissible heuristic: optimal and often faster
    """

    name: str
    grid_size: int
    elevation: np.ndarray          # grid_size x grid_size
    terrain: np.ndarray            # grid_size x grid_size (int)
    start: Tuple[int, int]
    goal: Tuple[int, int]
    hazard_zones: List[HazardZone]
    waypoints: List[Waypoint]
    base_move_cost: float = 1.0

    # --- Complexity info -------------------------------------------------
    @property
    def complexity_class(self) -> Dict:
        n = self.grid_size
        k = len(self.waypoints)
        v = n * n
        e = v * 8  # approx for 8-directional
        lava_pct = float(np.sum(self.terrain == TERRAIN_LAVA)) / v * 100
        return {
            'problem': 'Shortest Path (Grid with Waypoints)',
            'class': 'NP-hard' if k >= 2 else 'P (polynomial)',
            'without_waypoints': f'P -- Dijkstra O(V log V) = O({v} log {v})',
            'with_waypoints': f'NP-hard (Steiner path), k={k} waypoints',
            'exact_waypoint': f'O(k! * Dijkstra) = O({math.factorial(k)} * {v} log {v})' if k > 0 else 'N/A',
            'grid_size': f'{n}x{n}',
            'vertices': v,
            'approx_edges': e,
            'bfs_complexity': f'O(V + E) = O({v + e})',
            'dijkstra_complexity': f'O((V+E) log V) = O({v + e} * log {v})',
            'astar_note': 'A* with Manhattan/Euclidean heuristic is optimal',
            'lava_coverage': f'{lava_pct:.1f}%',
            'hazard_zones': len(self.hazard_zones),
            'n': n, 'k': k,
        }

    # --- Neighbors -------------------------------------------------------
    def neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return valid 8-directional neighbors."""
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if self.terrain[nr, nc] != TERRAIN_LAVA:
                        nbrs.append((nr, nc))
        return nbrs

    # --- Edge cost -------------------------------------------------------
    def edge_cost(self, r1: int, c1: int, r2: int, c2: int) -> float:
        """Cost to move from (r1,c1) to (r2,c2)."""
        # Terrain cost at destination
        t = int(self.terrain[r2, c2])
        if t == TERRAIN_LAVA:
            return float('inf')
        terrain_mult = TERRAIN_COSTS[t]

        # Elevation change penalty
        elev_diff = abs(float(self.elevation[r2, c2]) - float(self.elevation[r1, c1]))
        elev_cost = 1.0 + elev_diff * 2.0

        # Diagonal multiplier
        is_diagonal = (r1 != r2) and (c1 != c2)
        diag_mult = math.sqrt(2) if is_diagonal else 1.0

        return self.base_move_cost * terrain_mult * elev_cost * diag_mult

    # --- Cell risk -------------------------------------------------------
    def cell_risk(self, r: int, c: int) -> float:
        """Total risk at cell (r,c) from all hazard zones."""
        risk = 0.0
        for hz in self.hazard_zones:
            dist = math.sqrt((r - hz.center_row)**2 + (c - hz.center_col)**2)
            if dist <= hz.radius:
                # Risk increases closer to center
                intensity = 1.0 - (dist / (hz.radius + 1e-6))
                risk += hz.risk_level * intensity * hz.damage
        return risk

    # --- Evaluate --------------------------------------------------------
    def evaluate(self, path: List[Tuple[int, int]]) -> Dict:
        """
        Evaluate a path (list of (row, col) tuples from start to goal).
        Returns total_cost, energy, risk, waypoint info, feasibility.
        """
        if not path:
            return {"error": "Empty path", "total_cost": float('inf'), "feasible": False}

        if path[0] != self.start:
            return {"error": f"Path must start at {self.start}",
                    "total_cost": float('inf'), "feasible": False}
        if path[-1] != self.goal:
            return {"error": f"Path must end at {self.goal}",
                    "total_cost": float('inf'), "feasible": False}

        total_energy = 0.0
        total_risk = 0.0
        invalid_moves = []

        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]

            # Bounds check
            if not (0 <= r2 < self.grid_size and 0 <= c2 < self.grid_size):
                invalid_moves.append({'step': i, 'from': (r1, c1), 'to': (r2, c2), 'reason': 'out of bounds'})
                continue

            # Adjacency check (8-directional)
            if abs(r2 - r1) > 1 or abs(c2 - c1) > 1:
                invalid_moves.append({'step': i, 'from': (r1, c1), 'to': (r2, c2), 'reason': 'not adjacent'})
                continue

            # Lava check
            if self.terrain[r2, c2] == TERRAIN_LAVA:
                invalid_moves.append({'step': i, 'from': (r1, c1), 'to': (r2, c2), 'reason': 'lava (impassable)'})
                continue

            cost = self.edge_cost(r1, c1, r2, c2)
            total_energy += cost
            total_risk += self.cell_risk(r2, c2)

        # Waypoint checking
        visited_cells = set(path)
        waypoints_visited = []
        waypoints_missed = []
        waypoint_reward = 0.0

        for wp in self.waypoints:
            if (wp.row, wp.col) in visited_cells:
                waypoints_visited.append(wp.name)
                waypoint_reward += wp.reward
            else:
                waypoints_missed.append(wp.name)

        # Missed waypoint penalty
        missed_penalty = len(waypoints_missed) * 100.0

        feasible = (len(invalid_moves) == 0 and len(waypoints_missed) == 0)

        total_cost = total_energy + total_risk + missed_penalty - waypoint_reward

        return {
            'total_cost': round(total_cost, 2),
            'energy': round(total_energy, 2),
            'risk': round(total_risk, 2),
            'waypoint_reward': round(waypoint_reward, 2),
            'missed_penalty': round(missed_penalty, 2),
            'path_length': len(path) - 1,
            'waypoints_visited': waypoints_visited,
            'waypoints_missed': waypoints_missed,
            'waypoints_total': f'{len(waypoints_visited)}/{len(self.waypoints)}',
            'invalid_moves': invalid_moves,
            'feasible': feasible,
        }

    # --- Visualization ---------------------------------------------------
    def visualize(self, path=None, save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')

        # Elevation heatmap
        ax.imshow(self.elevation, cmap='terrain', alpha=0.6, origin='upper')

        # Terrain overlay
        terrain_overlay = np.zeros((*self.terrain.shape, 4))
        for t_type, color in TERRAIN_COLORS.items():
            mask = self.terrain == t_type
            r_val = int(color[1:3], 16) / 255
            g_val = int(color[3:5], 16) / 255
            b_val = int(color[5:7], 16) / 255
            terrain_overlay[mask] = [r_val, g_val, b_val, 0.3]
        # Lava is more opaque
        lava_mask = self.terrain == TERRAIN_LAVA
        terrain_overlay[lava_mask, 3] = 0.7
        ax.imshow(terrain_overlay, origin='upper')

        # Hazard zones
        for hz in self.hazard_zones:
            circle = patches.Circle(
                (hz.center_col, hz.center_row), hz.radius,
                color='#ff6b6b', alpha=0.15, linewidth=2,
                edgecolor='#ff4444', linestyle='--')
            ax.add_patch(circle)
            ax.annotate(f'RAD\nrisk:{hz.risk_level:.1f}',
                        (hz.center_col, hz.center_row),
                        color='#ff6b6b', fontsize=7, ha='center', va='center', alpha=0.8)

        # Waypoints
        for wp in self.waypoints:
            ax.scatter(wp.col, wp.row, s=200, c='#ffd93d', marker='*',
                       edgecolors='white', linewidths=1, zorder=10)
            ax.annotate(f'{wp.name}\n+{wp.reward:.0f}',
                        (wp.col, wp.row), textcoords="offset points",
                        xytext=(8, 8), fontsize=7, color='#ffd93d', zorder=11)

        # Path
        if path is not None:
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            ax.plot(cols, rows, color='#00d4aa', linewidth=2.5, alpha=0.9, zorder=8)
            # Arrow markers every few steps
            step = max(1, len(path) // 15)
            for i in range(0, len(path) - 1, step):
                dr = rows[i+1] - rows[i]
                dc = cols[i+1] - cols[i]
                ax.annotate('', xy=(cols[i+1], rows[i+1]),
                            xytext=(cols[i], rows[i]),
                            arrowprops=dict(arrowstyle='->', color='#00d4aa', lw=1.5),
                            zorder=9)

        # Start and goal
        ax.scatter(self.start[1], self.start[0], s=300, c='#00d4aa',
                   marker='o', edgecolors='white', linewidths=2, zorder=12)
        ax.annotate('START', (self.start[1], self.start[0]),
                    textcoords="offset points", xytext=(10, -10),
                    fontsize=9, color='#00d4aa', fontweight='bold', zorder=12)

        ax.scatter(self.goal[1], self.goal[0], s=300, c='#ff6b6b',
                   marker='s', edgecolors='white', linewidths=2, zorder=12)
        ax.annotate('GOAL', (self.goal[1], self.goal[0]),
                    textcoords="offset points", xytext=(10, -10),
                    fontsize=9, color='#ff6b6b', fontweight='bold', zorder=12)

        # Title
        title = f'{self.name}  |  {self.grid_size}x{self.grid_size}'
        if path:
            result = self.evaluate(path)
            title += (f'  |  Cost: {result["total_cost"]:.1f}'
                      f'  |  Energy: {result["energy"]:.1f}'
                      f'  |  {"OK" if result["feasible"] else "FAIL"}')
        ax.set_title(title, color='#e6edf3', fontsize=13, fontweight='bold', pad=15)

        # Legend
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00d4aa', markersize=10, linestyle='None', label='Start'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff6b6b', markersize=10, linestyle='None', label='Goal'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#ffd93d', markersize=12, linestyle='None', label='Waypoint'),
            patches.Patch(facecolor='#c2b280', alpha=0.5, label='Sand (1.5x)'),
            patches.Patch(facecolor='#808080', alpha=0.5, label='Rock (1.2x)'),
            patches.Patch(facecolor='#a8d8ea', alpha=0.5, label='Smooth (0.8x)'),
            patches.Patch(facecolor='#ff4444', alpha=0.5, label='Lava (blocked)'),
            patches.Patch(facecolor='#ff6b6b', alpha=0.2, label='Radiation zone'),
        ]
        ax.legend(handles=legend_elems, loc='upper left', facecolor='#21262d',
                  edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=8)

        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.show()

    # --- Diamond-square terrain gen --------------------------------------
    @staticmethod
    def _diamond_square(size, roughness, rng):
        # Find next power of 2
        n = 1
        while n < size:
            n *= 2
        n += 1

        terrain = np.zeros((n, n))
        terrain[0, 0] = rng.random()
        terrain[0, n-1] = rng.random()
        terrain[n-1, 0] = rng.random()
        terrain[n-1, n-1] = rng.random()

        step = n - 1
        scale = roughness

        while step > 1:
            half = step // 2

            # Diamond step
            for y in range(0, n - 1, step):
                for x in range(0, n - 1, step):
                    avg = (terrain[y, x] + terrain[y, x+step] +
                           terrain[y+step, x] + terrain[y+step, x+step]) / 4.0
                    terrain[y+half, x+half] = avg + rng.uniform(-scale, scale)

            # Square step
            for y in range(0, n, half):
                for x in range((y + half) % step, n, step):
                    vals = []
                    if y >= half:
                        vals.append(terrain[y-half, x])
                    if y + half < n:
                        vals.append(terrain[y+half, x])
                    if x >= half:
                        vals.append(terrain[y, x-half])
                    if x + half < n:
                        vals.append(terrain[y, x+half])
                    avg = sum(vals) / len(vals)
                    terrain[y, x] = avg + rng.uniform(-scale, scale)

            scale *= 0.5
            step = half

        # Crop and normalize
        terrain = terrain[:size, :size]
        t_min, t_max = terrain.min(), terrain.max()
        if t_max - t_min > 1e-6:
            terrain = (terrain - t_min) / (t_max - t_min)
        return terrain

    def __str__(self):
        lava_pct = float(np.sum(self.terrain == TERRAIN_LAVA)) / (self.grid_size**2) * 100
        cx = self.complexity_class
        return (f"ShortestPath: '{self.name}'\n"
                f"  Grid: {self.grid_size}x{self.grid_size}\n"
                f"  Start: {self.start} -> Goal: {self.goal}\n"
                f"  Waypoints: {len(self.waypoints)}\n"
                f"  Hazard zones: {len(self.hazard_zones)}\n"
                f"  Lava coverage: {lava_pct:.1f}%\n"
                f"  Complexity: {cx['class']}")

    @classmethod
    def generate(cls, grid_size=30, difficulty='medium', seed=None):
        rng = np.random.default_rng(seed)
        params = {
            'easy':    {'roughness': 0.3, 'lava_thresh': 0.95, 'n_hazards': 1, 'n_waypoints': 0, 'hazard_r': (3, 5)},
            'medium':  {'roughness': 0.5, 'lava_thresh': 0.90, 'n_hazards': 3, 'n_waypoints': 2, 'hazard_r': (3, 7)},
            'hard':    {'roughness': 0.7, 'lava_thresh': 0.85, 'n_hazards': 5, 'n_waypoints': 4, 'hazard_r': (4, 8)},
            'extreme': {'roughness': 0.9, 'lava_thresh': 0.80, 'n_hazards': 8, 'n_waypoints': 6, 'hazard_r': (5, 10)},
        }[difficulty]

        # Generate elevation via diamond-square
        elevation = cls._diamond_square(grid_size, params['roughness'], rng)

        # Generate terrain types
        terrain = np.zeros((grid_size, grid_size), dtype=int)
        for r in range(grid_size):
            for c in range(grid_size):
                e = elevation[r, c]
                if e > params['lava_thresh']:
                    terrain[r, c] = TERRAIN_LAVA
                elif e > 0.6:
                    terrain[r, c] = TERRAIN_ROCK
                elif e > 0.3:
                    terrain[r, c] = TERRAIN_SAND
                else:
                    terrain[r, c] = TERRAIN_SMOOTH

        # Place start and goal (ensure on passable terrain)
        margin = max(1, grid_size // 10)

        def find_passable(r_range, c_range):
            for _ in range(200):
                r = int(rng.integers(r_range[0], r_range[1]))
                c = int(rng.integers(c_range[0], c_range[1]))
                if terrain[r, c] != TERRAIN_LAVA:
                    return (r, c)
            return (r_range[0], c_range[0])

        start = find_passable((margin, grid_size//3), (margin, grid_size//3))
        goal = find_passable((2*grid_size//3, grid_size-margin),
                             (2*grid_size//3, grid_size-margin))

        # Ensure start/goal are passable
        terrain[start[0], start[1]] = TERRAIN_SMOOTH
        terrain[goal[0], goal[1]] = TERRAIN_SMOOTH

        # Hazard zones
        hazards = []
        for _ in range(params['n_hazards']):
            hr = int(rng.integers(margin, grid_size - margin))
            hc = int(rng.integers(margin, grid_size - margin))
            radius = float(rng.uniform(*params['hazard_r']))
            risk = float(rng.uniform(0.3, 0.9))
            damage = float(rng.uniform(5, 20))
            hazards.append(HazardZone(hr, hc, radius, round(risk, 2), round(damage, 1)))

        # Waypoints
        waypoints = []
        wp_names = ['Alpha Station', 'Beta Lab', 'Gamma Outpost', 'Delta Cache',
                    'Epsilon Relay', 'Zeta Observatory', 'Eta Depot', 'Theta Base']
        for i in range(params['n_waypoints']):
            for _ in range(100):
                wr = int(rng.integers(margin, grid_size - margin))
                wc = int(rng.integers(margin, grid_size - margin))
                if terrain[wr, wc] != TERRAIN_LAVA:
                    break
            reward = float(rng.uniform(10, 40))
            waypoints.append(Waypoint(wr, wc, round(reward, 1),
                                      wp_names[i % len(wp_names)]))

        name_map = {
            'easy': 'Olympus Plains Trek',
            'medium': 'Valles Marineris Crossing',
            'hard': 'Hellas Basin Expedition',
            'extreme': 'Tharsis Volcanic Traverse',
        }

        # Compute straight-line baseline
        sr, sc = start
        gr, gc = goal
        straight_dist = math.sqrt((gr - sr)**2 + (gc - sc)**2)

        return cls(
            name=name_map[difficulty],
            grid_size=grid_size,
            elevation=elevation,
            terrain=terrain,
            start=start,
            goal=goal,
            hazard_zones=hazards,
            waypoints=waypoints,
        )

    @classmethod
    def easy(cls, seed=None): return cls.generate(20, 'easy', seed)
    @classmethod
    def medium(cls, seed=None): return cls.generate(30, 'medium', seed)
    @classmethod
    def hard(cls, seed=None): return cls.generate(50, 'hard', seed)
    @classmethod
    def extreme(cls, seed=None): return cls.generate(80, 'extreme', seed)


if __name__ == '__main__':
    for diff in ['easy', 'medium', 'hard']:
        sz = {'easy': 20, 'medium': 30, 'hard': 50}[diff]
        p = ShortestPathProblem.generate(grid_size=sz, difficulty=diff, seed=42)
        print(p)

        # Simple diagonal walk
        path = [p.start]
        r, c = p.start
        gr, gc = p.goal
        while (r, c) != (gr, gc):
            dr = 1 if gr > r else (-1 if gr < r else 0)
            dc = 1 if gc > c else (-1 if gc < c else 0)
            nr, nc = r + dr, c + dc
            nr = max(0, min(nr, p.grid_size - 1))
            nc = max(0, min(nc, p.grid_size - 1))
            if (nr, nc) == (r, c):
                break
            r, c = nr, nc
            path.append((r, c))

        result = p.evaluate(path)
        cx = p.complexity_class
        print(f"  Energy: {result['energy']}")
        print(f"  Risk: {result['risk']}")
        print(f"  Path length: {result['path_length']} steps")
        print(f"  Waypoints: {result['waypoints_total']}")
        print(f"  Feasible: {result['feasible']}")
        print(f"  Complexity: {cx['class']}")
        print(f"  Dijkstra: {cx['dijkstra_complexity']}")
        print()
