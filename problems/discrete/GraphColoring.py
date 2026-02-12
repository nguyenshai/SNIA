"""
Graph Coloring Problem: "Festival Scheduling"
==============================================
Schedule festival events into time slots (= colors) with:
  - Conflict graph -- events sharing resources/audience can't overlap
  - Weighted edges -- severity of conflict (soft vs hard constraints)
  - Forbidden color pairs -- certain events can't be in adjacent slots either
  - Pre-assigned nodes -- some events already locked to specific slots
  - Node weights -- popular events have higher penalty multiplier

Complexity Class:
  - Graph k-Coloring (k>=3): NP-complete
  - Chromatic number: NP-hard to compute
  - Inapproximable within n^(1-eps) unless P=NP
  - 2-coloring: P (bipartite check via BFS)
  - Planar graphs: always 4-colorable (Four Color Theorem)

Suitable for testing: GA, SA
"""

import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set


@dataclass
class Event:
    id: int
    name: str
    popularity: float
    pre_assigned_color: Optional[int] = None
    category: str = ''

@dataclass
class Edge:
    u: int
    v: int
    weight: float
    is_hard: bool = True

@dataclass
class ForbiddenPair:
    u: int
    v: int

@dataclass
class GraphColoringProblem:
    """Weighted Graph Coloring with forbidden pairs and pre-assignments."""

    name: str
    n_nodes: int
    n_colors: int
    events: List[Event]
    edges: List[Edge]
    adjacency: Dict[int, Set[int]]
    forbidden_pairs: List[ForbiddenPair]
    node_positions: Dict[int, Tuple[float, float]]

    @property
    def complexity_class(self) -> Dict:
        n = self.n_nodes
        k = self.n_colors
        max_deg = max(len(adj) for adj in self.adjacency.values()) if self.adjacency else 0
        return {
            'problem': 'Graph Coloring Problem (GCP)',
            'class': 'NP-hard' if k >= 3 else 'P (2-colorable = bipartite)',
            'decision_class': 'NP-complete (k>=3)',
            'search_space_log10': n * math.log10(k) if k > 0 else 0,
            'brute_force': f'O(k^n) = O({k}^{n})',
            'greedy_upper_bound': f'{max_deg + 1} colors (Brooks theorem)',
            'max_degree': max_deg,
            'graph_density': round((2 * len(self.edges)) / (n * (n - 1) + 1e-6), 4),
            'approximation': 'Inapproximable within n^(1-eps) for general graphs',
            'planar_note': 'If planar: always 4-colorable (Four Color Theorem)',
            'n': n, 'k': k,
        }

    def evaluate(self, coloring: Dict[int, int]) -> Dict:
        if set(coloring.keys()) != set(range(self.n_nodes)):
            return {"error": "Must assign a color to every node",
                    "total_penalty": float('inf'), "feasible": False}
        for node, color in coloring.items():
            if color < 0 or color >= self.n_colors:
                return {"error": f"Color {color} out of range [0, {self.n_colors})",
                        "total_penalty": float('inf'), "feasible": False}

        hard_violations, soft_violations = [], []
        total_penalty = 0.0

        for edge in self.edges:
            if coloring[edge.u] == coloring[edge.v]:
                pop_u = self.events[edge.u].popularity
                pop_v = self.events[edge.v].popularity
                penalty = edge.weight * (pop_u + pop_v) / 2.0
                total_penalty += penalty
                v = {'edge': (edge.u, edge.v), 'color': coloring[edge.u],
                     'weight': edge.weight, 'penalty': round(penalty, 2)}
                (hard_violations if edge.is_hard else soft_violations).append(v)

        forbidden_violations = []
        for fp in self.forbidden_pairs:
            if abs(coloring[fp.u] - coloring[fp.v]) <= 1:
                total_penalty += 15.0
                forbidden_violations.append({
                    'pair': (fp.u, fp.v),
                    'colors': (coloring[fp.u], coloring[fp.v]), 'penalty': 15.0})

        preassign_violations = []
        for event in self.events:
            if event.pre_assigned_color is not None:
                if coloring[event.id] != event.pre_assigned_color:
                    penalty = 50.0 * event.popularity
                    total_penalty += penalty
                    preassign_violations.append({
                        'node': event.id, 'assigned': event.pre_assigned_color,
                        'given': coloring[event.id], 'penalty': round(penalty, 2)})

        feasible = (len(hard_violations) == 0
                    and len(preassign_violations) == 0
                    and len(forbidden_violations) == 0)
        return {
            'total_penalty': round(total_penalty, 2),
            'colors_used': len(set(coloring.values())),
            'hard_violations': hard_violations,
            'soft_violations': soft_violations,
            'forbidden_violations': forbidden_violations,
            'preassign_violations': preassign_violations,
            'n_violations': len(hard_violations) + len(forbidden_violations) + len(preassign_violations),
            'feasible': feasible,
        }

    def visualize(self, coloring=None, save_path=None):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [3, 1]})
        fig.patch.set_facecolor('#0d1117')
        ax = axes[0]
        ax.set_facecolor('#161b22')

        palette = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#a371f7',
                   '#f0883e', '#00d4aa', '#ff9ff3', '#54a0ff', '#5f27cd',
                   '#01a3a4', '#ee5a24', '#778beb', '#f8a5c2', '#63cdda', '#cf6a87']

        for edge in self.edges:
            pu, pv = self.node_positions[edge.u], self.node_positions[edge.v]
            violated = coloring and coloring[edge.u] == coloring[edge.v]
            if violated:
                c, lw, a = '#ff4444', 2.5, 0.9
            elif edge.is_hard:
                c, lw, a = '#484f58', 1.0 + edge.weight * 0.5, 0.5
            else:
                c, lw, a = '#30363d', 0.8, 0.3
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color=c, lw=lw, alpha=a, zorder=1)
            if edge.weight > 1.5 or violated:
                mx, my = (pu[0]+pv[0])/2, (pu[1]+pv[1])/2
                ax.annotate(f'{edge.weight:.1f}', (mx, my), color='#8b949e', fontsize=6, ha='center', alpha=0.7, zorder=2)

        for fp in self.forbidden_pairs:
            pu, pv = self.node_positions[fp.u], self.node_positions[fp.v]
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color='#ffd93d', lw=1.5, linestyle=':', alpha=0.4, zorder=1)

        for event in self.events:
            pos = self.node_positions[event.id]
            size = 60 + event.popularity * 15
            color = palette[coloring[event.id] % len(palette)] if coloring and event.id in coloring else '#484f58'
            ec, ew = ('#ffd93d', 2.5) if event.pre_assigned_color is not None else ('white', 0.8)
            ax.scatter(pos[0], pos[1], s=size, c=color, zorder=5, edgecolors=ec, linewidths=ew)
            ax.annotate(str(event.id), pos, textcoords="offset points", xytext=(5, 5), fontsize=7, color='#c9d1d9', zorder=6)

        ax.set_aspect('equal'); ax.axis('off')
        title = f'{self.name}  |  {self.n_nodes} events, {self.n_colors} slots'
        if coloring:
            r = self.evaluate(coloring)
            title += f'  |  Penalty: {r["total_penalty"]:.1f}  |  {"OK" if r["feasible"] else "FAIL"}  |  {r["colors_used"]} colors'
        ax.set_title(title, color='#e6edf3', fontsize=13, fontweight='bold', pad=15)

        ax2 = axes[1]; ax2.set_facecolor('#161b22'); ax2.axis('off')
        info = [f"Edges: {len(self.edges)}", f"  Hard: {sum(1 for e in self.edges if e.is_hard)}",
                f"  Soft: {sum(1 for e in self.edges if not e.is_hard)}",
                f"Forbidden: {len(self.forbidden_pairs)}",
                f"Pre-assigned: {sum(1 for e in self.events if e.pre_assigned_color is not None)}"]
        if coloring:
            r = self.evaluate(coloring)
            info += ["", f"Penalty: {r['total_penalty']}", f"Violations: {r['n_violations']}",
                     f"{'FEASIBLE' if r['feasible'] else 'INFEASIBLE'}"]
        ax2.text(0.05, 0.95, '\n'.join(info), transform=ax2.transAxes, verticalalignment='top',
                 fontsize=10, color='#c9d1d9', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#30363d'))
        for i in range(min(self.n_colors, len(palette))):
            ax2.add_patch(plt.Rectangle((0.08, 0.92 - i*0.03 - 0.008), 0.04, 0.02,
                          transform=ax2.transAxes, facecolor=palette[i], edgecolor='white', lw=0.5))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.show()

    def __str__(self):
        hard = sum(1 for e in self.edges if e.is_hard)
        pre = sum(1 for e in self.events if e.pre_assigned_color is not None)
        cx = self.complexity_class
        return (f"GraphColoring: '{self.name}'\n"
                f"  Nodes: {self.n_nodes}, Colors: {self.n_colors}\n"
                f"  Edges: {len(self.edges)} ({hard} hard, {len(self.edges)-hard} soft)\n"
                f"  Forbidden pairs: {len(self.forbidden_pairs)}, Pre-assigned: {pre}\n"
                f"  Max degree: {cx['max_degree']} | Density: {cx['graph_density']}\n"
                f"  Complexity: {cx['class']} | Search: {self.n_colors}^{self.n_nodes} (~10^{cx['search_space_log10']:.1f})")

    @classmethod
    def generate(cls, n_nodes=30, n_colors=5, topology='clustered',
                 difficulty='medium', seed=None):
        rng = np.random.default_rng(seed)
        random.seed(seed)
        params = {
            'easy':    {'edge_density': 0.08, 'soft_ratio': 0.5, 'forbidden_ratio': 0.01, 'preassign_ratio': 0.05, 'pop_range': (1, 4)},
            'medium':  {'edge_density': 0.15, 'soft_ratio': 0.3, 'forbidden_ratio': 0.03, 'preassign_ratio': 0.1,  'pop_range': (1, 7)},
            'hard':    {'edge_density': 0.22, 'soft_ratio': 0.15,'forbidden_ratio': 0.05, 'preassign_ratio': 0.15, 'pop_range': (2, 10)},
            'extreme': {'edge_density': 0.3,  'soft_ratio': 0.1, 'forbidden_ratio': 0.08, 'preassign_ratio': 0.2,  'pop_range': (3, 10)},
        }[difficulty]

        positions = {}; edge_set = set()
        if topology == 'planar':
            side = int(math.ceil(math.sqrt(n_nodes)))
            for i in range(n_nodes):
                row, col = divmod(i, side)
                positions[i] = (col*2.0 + float(rng.normal(0, 0.3)), row*2.0 + float(rng.normal(0, 0.3)))
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    dx, dy = positions[i][0]-positions[j][0], positions[i][1]-positions[j][1]
                    if math.sqrt(dx*dx + dy*dy) < 2.8:
                        edge_set.add((i, j))
        elif topology == 'clustered':
            nc = max(2, n_nodes // 6)
            centers = [(float(rng.uniform(-8, 8)), float(rng.uniform(-8, 8))) for _ in range(nc)]
            ca = {}
            for i in range(n_nodes):
                cx, cy = centers[i % nc]
                positions[i] = (float(cx + rng.normal(0, 1.5)), float(cy + rng.normal(0, 1.5)))
                ca[i] = i % nc
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    p = params['edge_density'] * (3 if ca[i] == ca[j] else 0.3)
                    if rng.random() < p:
                        edge_set.add((i, j))
        elif topology == 'scale_free':
            m = min(3, n_nodes - 1)
            for i in range(m+1):
                for j in range(i+1, m+1):
                    edge_set.add((i, j))
            degree = {i: m for i in range(m+1)}
            for i in range(m+1, n_nodes):
                degree[i] = 0
                total_deg = sum(degree.values()) + 1
                targets = set(); att = 0
                while len(targets) < m and att < m*20:
                    for j in range(i):
                        if rng.random() < (degree[j]+1)/total_deg and j not in targets:
                            targets.add(j)
                            if len(targets) >= m: break
                    att += 1
                for t in targets:
                    edge_set.add((min(i, t), max(i, t))); degree[i] += 1; degree[t] += 1
            positions = cls._spring_layout(n_nodes, edge_set, rng)

        target_edges = int(n_nodes*(n_nodes-1)/2 * params['edge_density'])
        extras = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes) if (i, j) not in edge_set]
        rng.shuffle(extras)
        for pair in extras:
            if len(edge_set) >= target_edges: break
            edge_set.add(pair)

        edges = []
        for u, v in edge_set:
            soft = rng.random() < params['soft_ratio']
            w = float(rng.uniform(0.5, 1.0)) if soft else float(rng.uniform(1.0, 3.0))
            edges.append(Edge(u, v, round(w, 2), is_hard=not soft))

        adjacency = {i: set() for i in range(n_nodes)}
        for e in edges: adjacency[e.u].add(e.v); adjacency[e.v].add(e.u)

        names = ['Rock Concert','Jazz Night','DJ Set','Folk Session','Poetry Slam','Dance Show',
                 'Comedy Hour','Art Exhibit','Film Screening','Workshop','Lecture','Panel Discussion',
                 'Food Contest','Fashion Show','Magic Show','Acrobatics','Choir Recital','Drum Circle',
                 'Open Mic','Talent Show','Yoga Session','Gaming Tournament','Hackathon','Quiz Night',
                 'Karaoke Battle','Puppet Theater','Orchestra','Ballet','Street Performance','Fireworks','Parade','Ceremony']
        cats = ['music','art','education','food','entertainment','wellness','tech','sports']
        events = []
        for i in range(n_nodes):
            nm = names[i % len(names)] + (f" #{i//len(names)+1}" if i >= len(names) else "")
            events.append(Event(id=i, name=nm, popularity=round(float(rng.uniform(*params['pop_range'])), 1), category=cats[i % len(cats)]))

        for pid in rng.choice(n_nodes, min(int(n_nodes*params['preassign_ratio']), n_nodes), replace=False):
            events[pid].pre_assigned_color = int(rng.integers(0, n_colors))

        el = list(edge_set); rng.shuffle(el)
        forbidden = [ForbiddenPair(u, v) for u, v in el[:int(len(edge_set)*params['forbidden_ratio'])]]

        nm = {'easy':'Summer Picnic Festival','medium':'City Arts Festival','hard':'International Music Fest','extreme':'Mega Multi-Stage Festival'}
        return cls(name=nm[difficulty], n_nodes=n_nodes, n_colors=n_colors, events=events,
                   edges=edges, adjacency=adjacency, forbidden_pairs=forbidden, node_positions=positions)

    @staticmethod
    def _spring_layout(n_nodes, edge_set, rng, iterations=50):
        pos = {i: (float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5))) for i in range(n_nodes)}
        k = 2.0
        for _ in range(iterations):
            disp = {i: [0.0, 0.0] for i in range(n_nodes)}
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    dx, dy = pos[i][0]-pos[j][0], pos[i][1]-pos[j][1]
                    d = max(math.sqrt(dx*dx + dy*dy), 0.01)
                    f = k*k/d*0.1
                    disp[i][0] += dx/d*f; disp[i][1] += dy/d*f
                    disp[j][0] -= dx/d*f; disp[j][1] -= dy/d*f
            for u, v in edge_set:
                dx, dy = pos[u][0]-pos[v][0], pos[u][1]-pos[v][1]
                d = max(math.sqrt(dx*dx + dy*dy), 0.01)
                f = d/k*0.1
                disp[u][0] -= dx/d*f; disp[u][1] -= dy/d*f
                disp[v][0] += dx/d*f; disp[v][1] += dy/d*f
            for i in range(n_nodes):
                m = max(math.sqrt(disp[i][0]**2 + disp[i][1]**2), 0.01)
                s = min(m, 1.0)/m
                pos[i] = (pos[i][0]+disp[i][0]*s, pos[i][1]+disp[i][1]*s)
        return pos

    @classmethod
    def easy(cls, seed=None): return cls.generate(15, 4, 'planar', 'easy', seed)
    @classmethod
    def medium(cls, seed=None): return cls.generate(30, 5, 'clustered', 'medium', seed)
    @classmethod
    def hard(cls, seed=None): return cls.generate(50, 4, 'scale_free', 'hard', seed)
    @classmethod
    def extreme(cls, seed=None): return cls.generate(80, 4, 'clustered', 'extreme', seed)


if __name__ == '__main__':
    for diff, topo in [('easy', 'planar'), ('medium', 'clustered'), ('hard', 'scale_free')]:
        n = {'easy': 15, 'medium': 30, 'hard': 50}[diff]
        p = GraphColoringProblem.generate(n_nodes=n, n_colors=5, topology=topo, difficulty=diff, seed=42)
        print(p)
        coloring = {}
        for i in range(p.n_nodes):
            if p.events[i].pre_assigned_color is not None:
                coloring[i] = p.events[i].pre_assigned_color; continue
            nc = {coloring[j] for j in p.adjacency[i] if j in coloring}
            coloring[i] = next((c for c in range(p.n_colors) if c not in nc), 0)
        r = p.evaluate(coloring)
        cx = p.complexity_class
        print(f"  Greedy: penalty={r['total_penalty']}, colors={r['colors_used']}, feasible={r['feasible']}")
        print(f"  Complexity: {cx['class']} | Brooks: {cx['greedy_upper_bound']}")
        print()
