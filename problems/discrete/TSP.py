"""
TSP Problem: "Storm Chaser Delivery Network"
=============================================
A delivery driver must visit all cities on a 2D map under multiple constraints:
  - Time windows per city (must arrive within [open, close])
  - Terrain cost multiplier per edge (mountains=2x, highways=0.7x)
  - Weather/storm zones that add time penalty when crossing
  - Fuel constraint -- max distance before needing refuel at depot cities

Complexity Class:
  - NP-hard (even basic TSP without constraints)
  - With time windows (TSPTW): strongly NP-hard
  - No polynomial-time exact algorithm exists unless P=NP
  - Best known exact: O(n^2 * 2^n) via dynamic programming (Held-Karp)
  - Approximation: Christofides gives 1.5x optimal for metric TSP

Suitable for testing: ACO, GA, SA

Usage:
    problem = TSPProblem.medium()
    print(problem)
    score = problem.evaluate(route)
    problem.visualize(route)
"""

import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# =============================================================================
#  Data classes
# =============================================================================
@dataclass
class City:
    id: int
    x: float
    y: float
    time_window: Tuple[float, float]   # (open, close)
    is_depot: bool = False             # can refuel here
    demand: float = 0.0                # delivery size


@dataclass
class StormZone:
    cx: float          # center x
    cy: float          # center y
    radius: float      # radius of the storm
    penalty: float     # extra time cost when crossing


@dataclass
class TSPProblem:
    """Multi-constrained Traveling Salesman Problem.

    Complexity Classes:
        - Decision version: NP-complete
        - Optimization version: NP-hard
        - With time windows: strongly NP-hard
        - Inapproximable within any constant factor for general TSP
        - Metric TSP: APX (1.5-approximable via Christofides)
        - Euclidean TSP: PTAS exists (Arora/Mitchell)
        - Search space size: (n-1)! / 2 for symmetric TSP
    """

    name: str
    n_cities: int
    cities: List[City]
    distance_matrix: np.ndarray        # raw Euclidean distances
    terrain_multiplier: np.ndarray     # multiplier per edge (nxn)
    storm_zones: List[StormZone]
    fuel_capacity: float               # max travel distance before refuel
    speed: float = 1.0                 # distance units per time unit

    # --- Complexity info -------------------------------------------------
    @property
    def complexity_class(self) -> Dict:
        """Return complexity classification for this problem instance."""
        n = self.n_cities
        search_space = math.factorial(n - 1) // 2 if n > 1 else 1
        return {
            'problem': 'Traveling Salesman Problem (TSP)',
            'class': 'NP-hard',
            'decision_class': 'NP-complete',
            'subclass': 'TSP with Time Windows (TSPTW) - strongly NP-hard',
            'search_space_size': search_space,
            'search_space_log10': math.log10(search_space) if search_space > 0 else 0,
            'exact_dp_complexity': f'O(n^2 * 2^n) = O({n}^2 * 2^{n})',
            'brute_force_complexity': f'O(n!) = O({n}!)',
            'best_approximation': '1.5x optimal (Christofides, metric TSP)',
            'euclidean_tsp': 'PTAS exists (Arora 1998)',
            'n': n,
            'constraints': {
                'time_windows': True,
                'terrain_costs': True,
                'storm_zones': len(self.storm_zones),
                'fuel_constraint': True,
            },
            'note': ('Adding time windows makes TSP strongly NP-hard. '
                     'No FPTAS exists unless P=NP.')
        }

    # --- Evaluate --------------------------------------------------------
    def evaluate(self, route: List[int]) -> Dict:
        """
        Evaluate a route (list of city indices, starting/ending at city 0).
        Returns dict with total_cost, time_violations, fuel_violations,
        storm_penalties, and overall feasibility.
        """
        if set(route) != set(range(self.n_cities)):
            return {"error": "Route must visit every city exactly once",
                    "total_cost": float('inf'), "feasible": False}

        total_distance = 0.0
        total_time = 0.0
        total_storm_penalty = 0.0
        time_violations = []
        fuel_violations = []
        fuel_remaining = self.fuel_capacity
        current_time = 0.0

        full_route = route + [route[0]]  # return to start

        for i in range(len(full_route) - 1):
            a, b = full_route[i], full_route[i + 1]

            # Base distance * terrain
            raw_dist = self.distance_matrix[a][b]
            terrain = self.terrain_multiplier[a][b]
            edge_cost = raw_dist * terrain

            # Storm penalty for this edge
            storm_pen = self._storm_penalty_for_edge(
                self.cities[a], self.cities[b])
            edge_cost += storm_pen
            total_storm_penalty += storm_pen

            total_distance += edge_cost
            travel_time = edge_cost / self.speed
            current_time += travel_time

            # Fuel check
            fuel_remaining -= raw_dist
            if fuel_remaining < 0:
                fuel_violations.append({
                    "at_city": b,
                    "deficit": -fuel_remaining
                })
            # Refuel at depots
            if self.cities[b].is_depot:
                fuel_remaining = self.fuel_capacity

            # Time window check (skip return-to-start)
            if i < len(full_route) - 2:
                city_b = self.cities[b]
                tw_open, tw_close = city_b.time_window
                if current_time < tw_open:
                    # Wait until opens
                    current_time = tw_open
                elif current_time > tw_close:
                    time_violations.append({
                        "city": b,
                        "arrived": round(current_time, 2),
                        "window": (tw_open, tw_close),
                        "late_by": round(current_time - tw_close, 2)
                    })

        # --- Scoring -----------------------------------------------------
        time_penalty = sum(v["late_by"] for v in time_violations)
        fuel_penalty = sum(v["deficit"] for v in fuel_violations)
        penalty_weight = 100.0

        total_cost = (total_distance
                      + penalty_weight * time_penalty
                      + penalty_weight * fuel_penalty)

        feasible = len(time_violations) == 0 and len(fuel_violations) == 0

        return {
            "total_cost": round(total_cost, 2),
            "total_distance": round(total_distance, 2),
            "total_time": round(current_time, 2),
            "storm_penalties": round(total_storm_penalty, 2),
            "time_violations": time_violations,
            "fuel_violations": fuel_violations,
            "feasible": feasible,
        }

    # --- Storm penalty helper --------------------------------------------
    def _storm_penalty_for_edge(self, c1: City, c2: City) -> float:
        """Check if the line segment (c1->c2) passes through any storm zone."""
        penalty = 0.0
        for storm in self.storm_zones:
            if self._segment_intersects_circle(
                    c1.x, c1.y, c2.x, c2.y,
                    storm.cx, storm.cy, storm.radius):
                penalty += storm.penalty
        return penalty

    @staticmethod
    def _segment_intersects_circle(x1, y1, x2, y2, cx, cy, r) -> bool:
        """Does line segment (x1,y1)->(x2,y2) intersect circle (cx,cy,r)?"""
        dx, dy = x2 - x1, y2 - y1
        fx, fy = x1 - cx, y1 - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        disc = b * b - 4 * a * c
        if disc < 0:
            return False
        disc = math.sqrt(disc)
        t1 = (-b - disc) / (2 * a + 1e-12)
        t2 = (-b + disc) / (2 * a + 1e-12)
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)

    # --- Visualization ---------------------------------------------------
    def visualize(self, route: Optional[List[int]] = None,
                  save_path: Optional[str] = None):
        """
        Render the TSP instance with matplotlib.
        Shows cities, storm zones, terrain legend, and optional route.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')

        # --- Storm zones (red translucent circles) -----------------------
        for storm in self.storm_zones:
            circle = patches.Circle(
                (storm.cx, storm.cy), storm.radius,
                color='#ff6b6b', alpha=0.15, linewidth=2,
                edgecolor='#ff4444', linestyle='--'
            )
            ax.add_patch(circle)
            ax.annotate(f'Storm\n+{storm.penalty:.0f}',
                        (storm.cx, storm.cy),
                        color='#ff6b6b', fontsize=8,
                        ha='center', va='center', alpha=0.7)

        # --- Route -------------------------------------------------------
        if route is not None:
            full = route + [route[0]]
            for i in range(len(full) - 1):
                a, b = full[i], full[i + 1]
                ca, cb = self.cities[a], self.cities[b]
                terrain = self.terrain_multiplier[a][b]
                # Color by terrain: green=highway, yellow=normal, red=mountain
                if terrain < 0.9:
                    color = '#00d4aa'
                    lw = 2.5
                elif terrain > 1.5:
                    color = '#ff6b6b'
                    lw = 1.5
                else:
                    color = '#ffd93d'
                    lw = 2.0
                ax.plot([ca.x, cb.x], [ca.y, cb.y],
                        color=color, linewidth=lw, alpha=0.7, zorder=2)
                # Arrow at midpoint
                mx = (ca.x + cb.x) / 2
                my = (ca.y + cb.y) / 2
                dx = cb.x - ca.x
                dy = cb.y - ca.y
                ax.annotate('', xy=(mx + dx * 0.01, my + dy * 0.01),
                            xytext=(mx, my),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=1.5), zorder=3)

        # --- Cities ------------------------------------------------------
        for city in self.cities:
            if city.is_depot:
                marker, size, color = 's', 150, '#00d4aa'
                label_color = '#00d4aa'
            else:
                marker, size, color = 'o', 80, '#58a6ff'
                label_color = '#c9d1d9'

            ax.scatter(city.x, city.y, s=size, c=color,
                       marker=marker, zorder=5, edgecolors='white',
                       linewidths=0.8)
            # Time window label
            tw = city.time_window
            ax.annotate(f'{city.id}\n[{tw[0]:.0f}-{tw[1]:.0f}]',
                        (city.x, city.y), textcoords="offset points",
                        xytext=(8, 8), fontsize=7, color=label_color,
                        zorder=6)

        # --- Legend ------------------------------------------------------
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#00d4aa', lw=2.5,
                   label='Highway (0.7x)'),
            Line2D([0], [0], color='#ffd93d', lw=2,
                   label='Normal road (1.0x)'),
            Line2D([0], [0], color='#ff6b6b', lw=1.5,
                   label='Mountain (2.0x)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#00d4aa',
                   markersize=10, label='Depot (refuel)', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#58a6ff',
                   markersize=8, label='City', linestyle='None'),
            patches.Patch(facecolor='#ff6b6b', alpha=0.3,
                          label='Storm zone'),
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                  facecolor='#21262d', edgecolor='#30363d',
                  labelcolor='#c9d1d9', fontsize=9)

        # --- Title & labels ----------------------------------------------
        title = f'{self.name}  |  {self.n_cities} cities'
        if route is not None:
            result = self.evaluate(route)
            title += (f'  |  Cost: {result["total_cost"]:.1f}'
                      f'  |  {"FEASIBLE" if result["feasible"] else "INFEASIBLE"}')
        ax.set_title(title, color='#e6edf3', fontsize=14,
                     fontweight='bold', pad=15)
        ax.set_xlabel('X', color='#8b949e')
        ax.set_ylabel('Y', color='#8b949e')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
        plt.show()

    # --- String representation -------------------------------------------
    def __str__(self):
        depots = sum(1 for c in self.cities if c.is_depot)
        cx = self.complexity_class
        return (f"TSP: '{self.name}'\n"
                f"  Cities: {self.n_cities} ({depots} depots)\n"
                f"  Storm zones: {len(self.storm_zones)}\n"
                f"  Fuel capacity: {self.fuel_capacity:.1f}\n"
                f"  Speed: {self.speed}\n"
                f"  Complexity: {cx['class']} | "
                f"Search space: ~10^{cx['search_space_log10']:.1f}")

    # --- Generators ------------------------------------------------------
    @classmethod
    def generate(cls, n_cities: int = 20, difficulty: str = 'medium',
                 seed: Optional[int] = None) -> 'TSPProblem':
        """
        Generate a TSP instance with configurable difficulty.

        difficulty levels:
            'easy'    -- wide time windows, few storms, generous fuel
            'medium'  -- moderate constraints
            'hard'    -- tight time windows, many storms, limited fuel
            'extreme' -- very tight, many storms, minimal fuel, more cities
        """
        rng = np.random.default_rng(seed)
        random.seed(seed)

        # Difficulty parameters
        params = {
            'easy':    {'tw_slack': 3.0, 'n_storms': 1, 'fuel_mult': 2.5,
                        'depot_ratio': 0.3, 'mountain_prob': 0.1},
            'medium':  {'tw_slack': 1.5, 'n_storms': 3, 'fuel_mult': 1.5,
                        'depot_ratio': 0.15, 'mountain_prob': 0.2},
            'hard':    {'tw_slack': 0.8, 'n_storms': 5, 'fuel_mult': 1.0,
                        'depot_ratio': 0.1, 'mountain_prob': 0.35},
            'extreme': {'tw_slack': 0.5, 'n_storms': 8, 'fuel_mult': 0.7,
                        'depot_ratio': 0.05, 'mountain_prob': 0.45},
        }[difficulty]

        map_size = 100.0

        # --- Generate cities ---------------------------------------------
        # Use clustered placement for realism
        n_clusters = max(2, n_cities // 5)
        cluster_centers = rng.uniform(15, map_size - 15, (n_clusters, 2))
        cities = []
        for i in range(n_cities):
            cc = cluster_centers[i % n_clusters]
            x = float(np.clip(cc[0] + rng.normal(0, 12), 2, map_size - 2))
            y = float(np.clip(cc[1] + rng.normal(0, 12), 2, map_size - 2))
            cities.append(City(id=i, x=x, y=y,
                               time_window=(0, 0)))  # filled below

        # Depot assignment
        n_depots = max(1, int(n_cities * params['depot_ratio']))
        depot_ids = set(rng.choice(n_cities, n_depots, replace=False))
        depot_ids.add(0)  # city 0 is always a depot (start)
        for did in depot_ids:
            cities[did].is_depot = True

        # --- Distance matrix ---------------------------------------------
        coords = np.array([(c.x, c.y) for c in cities])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        # --- Terrain multiplier ------------------------------------------
        terrain = np.ones((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(i + 1, n_cities):
                r = rng.random()
                if r < 0.15:  # highway
                    terrain[i][j] = terrain[j][i] = 0.7
                elif r < 0.15 + params['mountain_prob']:  # mountain
                    terrain[i][j] = terrain[j][i] = round(
                        1.5 + rng.random() * 0.8, 2)
                else:  # normal
                    terrain[i][j] = terrain[j][i] = round(
                        0.9 + rng.random() * 0.3, 2)

        # --- Time windows ------------------------------------------------
        # Compute rough arrival times via nearest-neighbor tour
        effective_dist = dist_matrix * terrain
        nn_order = [0]
        visited = {0}
        for _ in range(n_cities - 1):
            last = nn_order[-1]
            dists = effective_dist[last].copy()
            dists[list(visited)] = np.inf
            nxt = int(np.argmin(dists))
            nn_order.append(nxt)
            visited.add(nxt)

        cumulative_time = 0.0
        arrival_estimate = {0: 0.0}
        for i in range(len(nn_order) - 1):
            a, b = nn_order[i], nn_order[i + 1]
            cumulative_time += effective_dist[a][b]
            arrival_estimate[b] = cumulative_time

        total_span = cumulative_time
        slack = params['tw_slack']
        for city in cities:
            est = arrival_estimate[city.id]
            window_size = total_span * 0.1 * slack
            tw_open = max(0, est - window_size * 0.3)
            tw_close = est + window_size * 0.7
            city.time_window = (round(tw_open, 1), round(tw_close, 1))
        # Start city has wide window
        cities[0].time_window = (0, total_span * 3)

        # --- Storm zones -------------------------------------------------
        storms = []
        for _ in range(params['n_storms']):
            cx = float(rng.uniform(15, map_size - 15))
            cy = float(rng.uniform(15, map_size - 15))
            radius = float(rng.uniform(8, 20))
            penalty = float(rng.uniform(5, 25))
            storms.append(StormZone(cx, cy, radius, round(penalty, 1)))

        # --- Fuel capacity -----------------------------------------------
        avg_nn_dist = total_span / n_cities
        fuel_cap = avg_nn_dist * n_cities * params['fuel_mult'] / max(
            len(depot_ids), 1)
        fuel_cap = max(fuel_cap, avg_nn_dist * 3)  # at least 3 hops

        name_map = {
            'easy': 'Sunny Day Delivery',
            'medium': 'Storm Chaser Route',
            'hard': 'Hurricane Season Run',
            'extreme': 'Category 5 Gauntlet',
        }

        return cls(
            name=name_map[difficulty],
            n_cities=n_cities,
            cities=cities,
            distance_matrix=dist_matrix,
            terrain_multiplier=terrain,
            storm_zones=storms,
            fuel_capacity=round(fuel_cap, 1),
            speed=1.0,
        )

    # --- Preset difficulties ---------------------------------------------
    @classmethod
    def easy(cls, seed=None):
        return cls.generate(10, 'easy', seed)

    @classmethod
    def medium(cls, seed=None):
        return cls.generate(25, 'medium', seed)

    @classmethod
    def hard(cls, seed=None):
        return cls.generate(50, 'hard', seed)

    @classmethod
    def extreme(cls, seed=None):
        return cls.generate(100, 'extreme', seed)


# =============================================================================
#  Quick test
# =============================================================================
if __name__ == '__main__':
    for diff in ['easy', 'medium', 'hard']:
        p = TSPProblem.generate(difficulty=diff, seed=42,
                                n_cities={'easy': 10, 'medium': 25,
                                          'hard': 50}[diff])
        print(p)
        naive_route = list(range(p.n_cities))
        result = p.evaluate(naive_route)
        print(f"  Naive route cost: {result['total_cost']}")
        print(f"  Feasible: {result['feasible']}")
        print(f"  Time violations: {len(result['time_violations'])}")
        print(f"  Fuel violations: {len(result['fuel_violations'])}")

        cx = p.complexity_class
        print(f"  Complexity: {cx['class']}")
        print(f"  Search space: ~10^{cx['search_space_log10']:.1f}")
        print(f"  Exact DP: {cx['exact_dp_complexity']}")
        print()
