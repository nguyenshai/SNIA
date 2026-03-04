"""
Knapsack Problem: "Space Cargo Loading"
=======================================
Load cargo onto a spacecraft with multi-dimensional constraints:
  - 3 resource dimensions: Weight, Volume, Power consumption
  - Item synergies: certain item pairs give bonus value when co-selected
  - Item conflicts: some pairs cannot coexist
  - Fragility levels: fragile items reduce effective capacity
  - Priority classes: must include minimum number of mission-critical items

Complexity Class:
  - 0/1 Knapsack: NP-hard (weakly)
  - Multi-dimensional Knapsack (d>=2): strongly NP-hard
  - Standard 0/1 KP has pseudo-polynomial DP: O(n*W)
  - Multi-dimensional KP: no pseudo-polynomial algorithm unless P=NP
  - FPTAS exists for single-dimension 0/1 KP
  - With item conflicts (Independent Set): NP-hard to approximate

Suitable for testing: GA, ABC

Usage:
    problem = KnapsackProblem.medium()
    print(problem)
    score = problem.evaluate(selection)
    problem.visualize(selection)
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set


# =============================================================================
#  Data classes
# =============================================================================
@dataclass
class CargoItem:
    id: int
    name: str
    weight: float
    volume: float
    power: float
    value: float
    fragility: float        # 0.0 (robust) to 1.0 (extremely fragile)
    priority_class: str      # 'critical', 'science', 'comfort', 'backup'
    category: str            # for display grouping


@dataclass
class Synergy:
    item_a: int
    item_b: int
    bonus_value: float       # bonus when BOTH are selected
    description: str


@dataclass
class Conflict:
    item_a: int
    item_b: int
    description: str


@dataclass
class KnapsackProblem:
    """Multi-dimensional Knapsack with synergies and conflicts.

    Complexity Classes:
        - 0/1 Knapsack: NP-hard (weakly NP-hard)
        - Multi-dimensional (d>=2): strongly NP-hard
        - With synergies (quadratic KP): NP-hard
        - With conflicts (KP + Independent Set): NP-hard
        - Standard DP: O(n * W) pseudo-polynomial
        - Multi-dim DP: O(n * W1 * W2 * ... * Wd) -- intractable for large d
        - FPTAS: exists for 1D, does NOT exist for multi-dim (unless P=NP)
        - Search space: 2^n (all subsets)
    """

    name: str
    n_items: int
    items: List[CargoItem]
    capacity_weight: float
    capacity_volume: float
    capacity_power: float
    synergies: List[Synergy]
    conflicts: List[Conflict]
    min_critical_items: int   # must include at least this many 'critical' items
    fragility_penalty_rate: float  # how much fragility reduces effective capacity

    # --- Complexity info -------------------------------------------------
    @property
    def complexity_class(self) -> Dict:
        """Return complexity classification for this problem instance."""
        n = self.n_items
        search_space = 2 ** n
        n_dims = 3  # weight, volume, power
        return {
            'problem': 'Multi-Dimensional Knapsack Problem (MKP)',
            'class': 'NP-hard (strongly NP-hard for d>=2)',
            'decision_class': 'NP-complete',
            'dimensions': n_dims,
            'search_space_size': search_space,
            'search_space_log10': n * math.log10(2),
            'dp_complexity': (
                f'O(n*W1*W2*W3) -- intractable for large capacities'),
            'brute_force': f'O(2^n) = O(2^{n})',
            'with_synergies': 'Quadratic Knapsack -- NP-hard',
            'with_conflicts': 'KP + Independent Set -- NP-hard to approximate',
            'fptas': 'Exists for 1D only; not for multi-dimensional',
            'n': n,
            'constraints': {
                'resource_dimensions': n_dims,
                'synergies': len(self.synergies),
                'conflicts': len(self.conflicts),
                'priority_requirement': self.min_critical_items,
                'fragility': True,
            },
            'note': ('Multi-dimensional KP with synergies and conflicts '
                     'combines multiple NP-hard subproblems.')
        }

    # --- Evaluate --------------------------------------------------------
    def evaluate(self, selection: List[int]) -> Dict:
        """
        Evaluate a selection (list of 0/1 for each item, or list of selected item indices).
        Returns dict with total_value, resource usage, violations, feasibility.
        """
        # Normalize input
        if len(selection) == self.n_items and all(s in (0, 1) for s in selection):
            selected_ids = {i for i, s in enumerate(selection) if s == 1}
        else:
            selected_ids = set(selection)

        # Resource usage
        total_weight = sum(self.items[i].weight for i in selected_ids)
        total_volume = sum(self.items[i].volume for i in selected_ids)
        total_power = sum(self.items[i].power for i in selected_ids)

        # Base value
        base_value = sum(self.items[i].value for i in selected_ids)

        # Synergy bonus
        synergy_bonus = 0.0
        active_synergies = []
        for syn in self.synergies:
            if syn.item_a in selected_ids and syn.item_b in selected_ids:
                synergy_bonus += syn.bonus_value
                active_synergies.append(syn.description)

        # Fragility penalty -- fragile items effectively reduce volume capacity
        fragility_score = sum(
            self.items[i].fragility * self.items[i].volume
            for i in selected_ids)
        effective_volume_cap = self.capacity_volume * (
            1.0 - self.fragility_penalty_rate * fragility_score / (
                self.capacity_volume + 1e-6))
        effective_volume_cap = max(effective_volume_cap,
                                   self.capacity_volume * 0.5)

        # Violations
        violations = {}

        if total_weight > self.capacity_weight:
            violations['weight'] = {
                'used': round(total_weight, 2),
                'capacity': self.capacity_weight,
                'over': round(total_weight - self.capacity_weight, 2)
            }
        if total_volume > effective_volume_cap:
            violations['volume'] = {
                'used': round(total_volume, 2),
                'effective_capacity': round(effective_volume_cap, 2),
                'original_capacity': self.capacity_volume,
                'over': round(total_volume - effective_volume_cap, 2)
            }
        if total_power > self.capacity_power:
            violations['power'] = {
                'used': round(total_power, 2),
                'capacity': self.capacity_power,
                'over': round(total_power - self.capacity_power, 2)
            }

        # Conflict violations
        conflict_violations = []
        for conf in self.conflicts:
            if conf.item_a in selected_ids and conf.item_b in selected_ids:
                conflict_violations.append(conf.description)
        if conflict_violations:
            violations['conflicts'] = conflict_violations

        # Priority check
        critical_selected = sum(
            1 for i in selected_ids
            if self.items[i].priority_class == 'critical')
        if critical_selected < self.min_critical_items:
            violations['critical_items'] = {
                'selected': critical_selected,
                'required': self.min_critical_items,
                'missing': self.min_critical_items - critical_selected
            }

        # Penalty calculation
        penalty = 0.0
        if 'weight' in violations:
            penalty += violations['weight']['over'] * 50
        if 'volume' in violations:
            penalty += violations['volume']['over'] * 50
        if 'power' in violations:
            penalty += violations['power']['over'] * 50
        penalty += len(conflict_violations) * 200
        if 'critical_items' in violations:
            penalty += violations['critical_items']['missing'] * 300

        total_value = base_value + synergy_bonus - penalty
        feasible = len(violations) == 0

        return {
            'total_value': round(total_value, 2),
            'base_value': round(base_value, 2),
            'synergy_bonus': round(synergy_bonus, 2),
            'penalty': round(penalty, 2),
            'n_selected': len(selected_ids),
            'weight': {'used': round(total_weight, 2),
                       'capacity': self.capacity_weight},
            'volume': {'used': round(total_volume, 2),
                       'capacity': self.capacity_volume,
                       'effective': round(effective_volume_cap, 2)},
            'power': {'used': round(total_power, 2),
                      'capacity': self.capacity_power},
            'active_synergies': active_synergies,
            'violations': violations,
            'feasible': feasible,
        }

    # --- Visualization ---------------------------------------------------
    def visualize(self, selection: Optional[List[int]] = None,
                  save_path: Optional[str] = None):
        """
        Render the Knapsack instance with matplotlib.
        Shows resource bars, item grid, synergy graph.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#0d1117')
        gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # Normalize selection
        if selection is not None:
            if len(selection) == self.n_items and all(
                    s in (0, 1) for s in selection):
                selected_ids = {i for i, s in enumerate(selection)
                                if s == 1}
            else:
                selected_ids = set(selection)
            result = self.evaluate(selection)
        else:
            selected_ids = set()
            result = None

        # --- Resource usage bars (top-left) ------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#161b22')
        resources = ['Weight', 'Volume', 'Power']
        capacities = [self.capacity_weight, self.capacity_volume,
                      self.capacity_power]
        if result:
            used = [result['weight']['used'], result['volume']['used'],
                    result['power']['used']]
        else:
            used = [0, 0, 0]

        colors = ['#58a6ff', '#a371f7', '#f0883e']
        y_pos = np.arange(len(resources))

        # Capacity background
        ax1.barh(y_pos, capacities, height=0.5, color='#21262d',
                 edgecolor='#30363d')
        # Used
        bar_colors = []
        for i in range(3):
            bar_colors.append(
                '#ff6b6b' if used[i] > capacities[i] else colors[i])
        ax1.barh(y_pos, used, height=0.5, color=bar_colors, alpha=0.85)

        for i in range(3):
            pct = used[i] / capacities[i] * 100 if capacities[i] > 0 else 0
            ax1.text(max(used[i], capacities[i]) + 1, y_pos[i],
                     f'{used[i]:.0f}/{capacities[i]:.0f} ({pct:.0f}%)',
                     va='center', color='#c9d1d9', fontsize=9)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(resources, color='#c9d1d9')
        ax1.set_title('Resource Usage', color='#e6edf3',
                       fontweight='bold')
        ax1.tick_params(colors='#8b949e')
        for spine in ax1.spines.values():
            spine.set_color('#30363d')

        # --- Item value vs weight scatter (top-middle) -------------------
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#161b22')

        for item in self.items:
            is_sel = item.id in selected_ids
            color = '#ffd93d' if is_sel else '#8b949e'
            alpha = 1.0 if is_sel else 0.4
            size = 40 + item.fragility * 80
            edge = '#ff6b6b' if item.priority_class == 'critical' else 'none'
            edge_w = 2 if item.priority_class == 'critical' else 0
            ax2.scatter(item.weight, item.value, s=size, c=color,
                        alpha=alpha, edgecolors=edge, linewidths=edge_w,
                        zorder=3)

        ax2.set_xlabel('Weight', color='#8b949e')
        ax2.set_ylabel('Value', color='#8b949e')
        ax2.set_title('Value vs Weight (size=fragility)',
                       color='#e6edf3', fontweight='bold')
        ax2.tick_params(colors='#8b949e')
        for spine in ax2.spines.values():
            spine.set_color('#30363d')

        # --- Priority breakdown (top-right) ------------------------------
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor('#161b22')

        priorities = ['critical', 'science', 'comfort', 'backup']
        pri_colors = ['#ff6b6b', '#58a6ff', '#a371f7', '#8b949e']
        pri_counts = []
        pri_selected = []
        for pri in priorities:
            total_in_pri = sum(
                1 for it in self.items if it.priority_class == pri)
            sel_in_pri = sum(
                1 for it in self.items
                if it.priority_class == pri and it.id in selected_ids)
            pri_counts.append(total_in_pri)
            pri_selected.append(sel_in_pri)

        y_pos = np.arange(len(priorities))
        ax3.barh(y_pos, pri_counts, height=0.5, color='#21262d',
                 edgecolor='#30363d', label='Available')
        ax3.barh(y_pos, pri_selected, height=0.5, color=pri_colors,
                 alpha=0.85, label='Selected')

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(
            [p.capitalize() for p in priorities], color='#c9d1d9')
        ax3.set_title('Priority Classes', color='#e6edf3',
                       fontweight='bold')
        ax3.tick_params(colors='#8b949e')
        for spine in ax3.spines.values():
            spine.set_color('#30363d')

        # --- Synergy network (bottom-left, spanning 2 cols) --------------
        ax4 = fig.add_subplot(gs[1, 0:2])
        ax4.set_facecolor('#161b22')

        # Place items in a circle
        involved_items = set()
        for syn in self.synergies:
            involved_items.add(syn.item_a)
            involved_items.add(syn.item_b)
        for conf in self.conflicts:
            involved_items.add(conf.item_a)
            involved_items.add(conf.item_b)

        involved_list = sorted(involved_items)
        n_inv = len(involved_list)
        if n_inv > 0:
            angles = np.linspace(0, 2 * np.pi, n_inv, endpoint=False)
            positions = {
                item_id: (np.cos(a) * 0.8, np.sin(a) * 0.8)
                for item_id, a in zip(involved_list, angles)
            }

            # Draw synergy edges (green)
            for syn in self.synergies:
                pa = positions[syn.item_a]
                pb = positions[syn.item_b]
                active = (syn.item_a in selected_ids and
                          syn.item_b in selected_ids)
                color = '#00d4aa' if active else '#30363d'
                lw = 2.5 if active else 1.0
                alpha = 0.9 if active else 0.3
                ax4.plot([pa[0], pb[0]], [pa[1], pb[1]],
                         color=color, lw=lw, alpha=alpha, zorder=1)
                if active:
                    mx = (pa[0] + pb[0]) / 2
                    my = (pa[1] + pb[1]) / 2
                    ax4.annotate(f'+{syn.bonus_value:.0f}',
                                 (mx, my), color='#00d4aa',
                                 fontsize=7, ha='center', zorder=4)

            # Draw conflict edges (red dashed)
            for conf in self.conflicts:
                pa = positions[conf.item_a]
                pb = positions[conf.item_b]
                both = (conf.item_a in selected_ids and
                        conf.item_b in selected_ids)
                color = '#ff4444' if both else '#4a2020'
                lw = 2.5 if both else 1.0
                ax4.plot([pa[0], pb[0]], [pa[1], pb[1]],
                         color=color, lw=lw, linestyle='--',
                         alpha=0.7, zorder=1)

            # Draw nodes
            for item_id in involved_list:
                pos = positions[item_id]
                is_sel = item_id in selected_ids
                color = '#ffd93d' if is_sel else '#484f58'
                size = 80 if is_sel else 50
                ax4.scatter(pos[0], pos[1], s=size, c=color,
                            zorder=3, edgecolors='white', linewidths=0.5)
                ax4.annotate(str(item_id), pos, textcoords="offset points",
                             xytext=(5, 5), fontsize=7,
                             color='#c9d1d9', zorder=4)

        ax4.set_xlim(-1.2, 1.2)
        ax4.set_ylim(-1.2, 1.2)
        ax4.set_title('Synergies (green) & Conflicts (red dashed)',
                       color='#e6edf3', fontweight='bold')
        ax4.set_aspect('equal')
        ax4.axis('off')

        # --- Summary panel (bottom-right) --------------------------------
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_facecolor('#161b22')
        ax5.axis('off')

        summary_lines = [
            f"  {self.name}",
            f"",
            f"Items available: {self.n_items}",
            f"Items selected:  {len(selected_ids)}",
            f"Synergies:       {len(self.synergies)}",
            f"Conflicts:       {len(self.conflicts)}",
            f"Min critical:    {self.min_critical_items}",
        ]
        if result:
            summary_lines += [
                f"",
                f"--- Results ----------",
                f"Base value:    {result['base_value']}",
                f"Synergy bonus: +{result['synergy_bonus']}",
                f"Penalty:       -{result['penalty']}",
                f"=====================",
                f"Total value:   {result['total_value']}",
                f"",
                f"{'FEASIBLE' if result['feasible'] else 'INFEASIBLE'}",
            ]

        text = '\n'.join(summary_lines)
        ax5.text(0.05, 0.95, text, transform=ax5.transAxes,
                 verticalalignment='top', fontsize=10,
                 color='#c9d1d9', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d',
                           edgecolor='#30363d'))

        plt.suptitle(f'{self.name}', color='#e6edf3', fontsize=16,
                     fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
        plt.show()

    # --- String representation -------------------------------------------
    def __str__(self):
        critical = sum(
            1 for it in self.items if it.priority_class == 'critical')
        cx = self.complexity_class
        return (f"Knapsack: '{self.name}'\n"
                f"  Items: {self.n_items} ({critical} critical)\n"
                f"  Capacity -- W:{self.capacity_weight} "
                f"V:{self.capacity_volume} P:{self.capacity_power}\n"
                f"  Synergies: {len(self.synergies)}, "
                f"Conflicts: {len(self.conflicts)}\n"
                f"  Min critical required: {self.min_critical_items}\n"
                f"  Complexity: {cx['class']} | "
                f"Search space: 2^{self.n_items}")

    # --- Generators ------------------------------------------------------
    @classmethod
    def generate(cls, n_items: int = 30, difficulty: str = 'medium',
                 seed: Optional[int] = None) -> 'KnapsackProblem':
        """
        Generate a Knapsack instance with configurable difficulty.

        difficulty:
            'easy'    -- loose capacity, few synergies/conflicts
            'medium'  -- balanced
            'hard'    -- tight capacity, many synergies/conflicts
            'extreme' -- very tight, dense relationships, fragile items
        """
        rng = np.random.default_rng(seed)
        random.seed(seed)

        params = {
            'easy':    {'cap_ratio': 0.6, 'syn_density': 0.03,
                        'conf_density': 0.01, 'fragility_max': 0.2,
                        'critical_ratio': 0.1},
            'medium':  {'cap_ratio': 0.45, 'syn_density': 0.06,
                        'conf_density': 0.03, 'fragility_max': 0.5,
                        'critical_ratio': 0.15},
            'hard':    {'cap_ratio': 0.35, 'syn_density': 0.1,
                        'conf_density': 0.06, 'fragility_max': 0.7,
                        'critical_ratio': 0.2},
            'extreme': {'cap_ratio': 0.25, 'syn_density': 0.15,
                        'conf_density': 0.1, 'fragility_max': 0.9,
                        'critical_ratio': 0.25},
        }[difficulty]

        # --- Item generation ---------------------------------------------
        categories = {
            'propulsion':  ['Ion thruster', 'Chemical fuel', 'Plasma drive',
                            'RCS module', 'Emergency booster'],
            'life_support': ['O2 generator', 'Water recycler', 'CO2 scrubber',
                            'Food supply', 'Medical kit'],
            'science':     ['Spectrometer', 'Drill kit', 'Microscope',
                            'Sample container', 'Antenna array'],
            'power':       ['Solar panel', 'Battery pack', 'RTG unit',
                            'Fuel cell', 'Capacitor bank'],
            'structure':   ['Heat shield', 'Radiation shield', 'Hull patch',
                            'Docking adapter', 'Landing gear'],
            'electronics': ['Computer core', 'Sensor suite', 'Comm relay',
                            'Navigation AI', 'Data recorder'],
        }
        all_names = []
        for cat, names in categories.items():
            for name in names:
                all_names.append((name, cat))

        items = []
        priority_classes = ['critical', 'science', 'comfort', 'backup']

        for i in range(n_items):
            name, cat = all_names[i % len(all_names)]
            if i >= len(all_names):
                name = f"{name} Mk.{i // len(all_names) + 1}"

            # Greedy trap: high value items tend to be heavier
            base_val = float(rng.uniform(10, 100))
            weight = float(base_val * rng.uniform(0.5, 1.5) + rng.uniform(5, 20))
            volume = float(rng.uniform(5, 60))
            power = float(rng.uniform(1, 40))
            fragility = float(rng.uniform(0, params['fragility_max']))

            pri_roll = rng.random()
            if pri_roll < params['critical_ratio']:
                pri = 'critical'
            elif pri_roll < 0.4:
                pri = 'science'
            elif pri_roll < 0.7:
                pri = 'comfort'
            else:
                pri = 'backup'

            items.append(CargoItem(
                id=i, name=name, weight=round(weight, 1),
                volume=round(volume, 1), power=round(power, 1),
                value=round(base_val, 1), fragility=round(fragility, 2),
                priority_class=pri, category=cat,
            ))

        # --- Capacities (based on ratio of total) ------------------------
        total_w = sum(it.weight for it in items)
        total_v = sum(it.volume for it in items)
        total_p = sum(it.power for it in items)
        cap_w = round(total_w * params['cap_ratio'], 1)
        cap_v = round(total_v * params['cap_ratio'], 1)
        cap_p = round(total_p * params['cap_ratio'], 1)

        # --- Synergies ---------------------------------------------------
        n_possible_pairs = n_items * (n_items - 1) // 2
        n_synergies = int(n_possible_pairs * params['syn_density'])
        n_synergies = min(n_synergies, 50)

        synergy_pairs = set()
        synergies = []
        synergy_templates = [
            ("{a} + {b}: power boost", lambda rng: round(rng.uniform(5, 30), 1)),
            ("{a} synergizes with {b}", lambda rng: round(rng.uniform(8, 25), 1)),
            ("{a} & {b}: efficiency combo", lambda rng: round(rng.uniform(10, 35), 1)),
        ]

        attempts = 0
        while len(synergies) < n_synergies and attempts < n_synergies * 5:
            a, b = sorted(rng.choice(n_items, 2, replace=False))
            attempts += 1
            if (a, b) not in synergy_pairs:
                synergy_pairs.add((a, b))
                tmpl = synergy_templates[len(synergies) % len(synergy_templates)]
                desc = tmpl[0].format(a=items[a].name, b=items[b].name)
                bonus = tmpl[1](rng)
                synergies.append(Synergy(a, b, bonus, desc))

        # --- Conflicts ---------------------------------------------------
        n_conflicts = int(n_possible_pairs * params['conf_density'])
        n_conflicts = min(n_conflicts, 30)

        conflict_pairs = set()
        conflicts = []
        conflict_descs = [
            "{a} incompatible with {b}",
            "{a} and {b}: frequency interference",
            "{a} reacts with {b}: safety hazard",
        ]

        attempts = 0
        while len(conflicts) < n_conflicts and attempts < n_conflicts * 5:
            a, b = sorted(rng.choice(n_items, 2, replace=False))
            attempts += 1
            if ((a, b) not in conflict_pairs and
                    (a, b) not in synergy_pairs):
                conflict_pairs.add((a, b))
                desc = conflict_descs[len(conflicts) % len(conflict_descs)]
                desc = desc.format(a=items[a].name, b=items[b].name)
                conflicts.append(Conflict(a, b, desc))

        # --- Critical item requirement -----------------------------------
        n_critical = sum(1 for it in items if it.priority_class == 'critical')
        min_crit = max(1, int(n_critical * 0.6))

        name_map = {
            'easy':    'Orbital Supply Run',
            'medium':  'Deep Space Cargo Mission',
            'hard':    'Mars Colony Resupply',
            'extreme': 'Interstellar Ark Loading',
        }

        return cls(
            name=name_map[difficulty],
            n_items=n_items,
            items=items,
            capacity_weight=cap_w,
            capacity_volume=cap_v,
            capacity_power=cap_p,
            synergies=synergies,
            conflicts=conflicts,
            min_critical_items=min_crit,
            fragility_penalty_rate=0.3 if difficulty in ('hard', 'extreme') else 0.15,
        )

    # --- Presets ---------------------------------------------------------
    @classmethod
    def easy(cls, seed=None):
        return cls.generate(15, 'easy', seed)

    @classmethod
    def medium(cls, seed=None):
        return cls.generate(30, 'medium', seed)

    @classmethod
    def hard(cls, seed=None):
        return cls.generate(50, 'hard', seed)

    @classmethod
    def extreme(cls, seed=None):
        return cls.generate(80, 'extreme', seed)


import math

# =============================================================================
#  Quick test
# =============================================================================
if __name__ == '__main__':
    for diff in ['easy', 'medium', 'hard']:
        p = KnapsackProblem.generate(
            difficulty=diff, seed=42,
            n_items={'easy': 15, 'medium': 30, 'hard': 50}[diff])
        print(p)

        # Greedy by value-to-weight ratio
        ratios = [(it.value / (it.weight + 1e-6), it.id) for it in p.items]
        ratios.sort(reverse=True)
        sel = [0] * p.n_items
        w_used = 0
        for ratio, idx in ratios:
            if w_used + p.items[idx].weight <= p.capacity_weight:
                sel[idx] = 1
                w_used += p.items[idx].weight

        result = p.evaluate(sel)
        print(f"  Greedy value: {result['total_value']}")
        print(f"  Selected: {result['n_selected']} items")
        print(f"  Feasible: {result['feasible']}")
        if result['violations']:
            print(f"  Violations: {list(result['violations'].keys())}")

        cx = p.complexity_class
        print(f"  Complexity: {cx['class']}")
        print(f"  Search space: 2^{p.n_items} = ~10^{cx['search_space_log10']:.1f}")
        print()
