"""
Execution Time Analysis — Discrete Problems
==============================================
Mục đích: Đo và so sánh Median Execution Time của các thuật toán
          trên các bài toán Rời rạc (TSP, Knapsack, Graph Coloring, Shortest Path).

Mỗi cấu hình chạy 7 lần để lấy median thời gian.

Output: results/exec_time_discrete/
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUT_DIR = os.path.join(BASE_DIR, 'results', 'exec_time_discrete')
os.makedirs(OUT_DIR, exist_ok=True)

N_TIMING_RUNS = 7
ITERATIONS    = 100

BG_DARK  = '#0d1117'
AX_DARK  = '#161b22'
TEXT_CLR = '#e6edf3'
GRID_CLR = '#30363d'
MID_CLR  = '#8b949e'

ALGOS_DISCRETE = ['ACO', 'GA', 'ABC', 'CS', 'SA', 'BFS', 'DFS', 'A*']
ALGO_COLORS = {
    'ACO': '#ff6b6b', 'GA': '#58a6ff', 'ABC': '#ffd93d', 'CS': '#00d4aa',
    'SA': '#a371f7', 'BFS': '#808080', 'DFS': '#b0b0b0', 'A*': '#ffffff'
}

def _style_ax(ax):
    ax.set_facecolor(AX_DARK)
    ax.tick_params(colors=MID_CLR, labelsize=9)
    ax.title.set_color(TEXT_CLR)
    for sp in ax.spines.values(): sp.set_color(GRID_CLR)
    ax.grid(axis='y', color=GRID_CLR, linestyle='--', linewidth=0.5, alpha=0.6)

def _savefig(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

from problems.discrete.TSP import TSPProblem
from problems.discrete.Knapsack import KnapsackProblem
from problems.discrete.GraphColoring import GraphColoringProblem
from problems.discrete.ShortestPath import ShortestPathProblem
from scripts.robustness_discrete import KnapsackAdapter, GraphColoringAdapter, ShortestPathSAAdapter

def measure_times(runner):
    times = []
    for _ in range(N_TIMING_RUNS):
        t0 = time.perf_counter()
        runner()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.std(times))

def draw_bar(ax, medians, stds, title):
    algos = list(medians.keys())
    vals = [medians[a] for a in algos]
    errs = [stds[a] for a in algos]
    colors = [ALGO_COLORS[a] for a in algos]

    _style_ax(ax)
    x = np.arange(len(algos))
    bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor='w', linewidth=0.5)
    ax.errorbar(x, vals, yerr=errs, fmt='none', ecolor=TEXT_CLR, capsize=4)

    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e, f'{v:.3f}s', ha='center', va='bottom', color=TEXT_CLR, fontsize=9)

    if vals:
        best_i = int(np.argmin(vals))
        bars[best_i].set_edgecolor('#ffd93d')
        bars[best_i].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontweight='bold', fontsize=11)
    ax.set_ylabel('Median Time (s)')
    ax.set_title(title, fontweight='bold', pad=12)

def main():
    t0_global = time.time()
    print('=' * 60)
    print(f'  Execution Time -- Discrete Problems ({N_TIMING_RUNS} runs)')
    print('=' * 60)

    # 1. TSP
    tsp_prob = TSPProblem.easy(seed=42)
    def runner_tsp_aco():
        from algorithms.biology.ACO import AntColonyOptimization
        AntColonyOptimization(tsp_prob, params={'n_ants':20}).solve(iterations=ITERATIONS)
    print('TSP (ACO)...')
    m_aco, s_aco = measure_times(runner_tsp_aco)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG_DARK)
    draw_bar(ax, {'ACO': m_aco}, {'ACO': s_aco}, 'TSP (50 cities) Execution Time')
    _savefig(fig, 'exec_time_tsp.png')

    # 2. Knapsack (GA, ABC, CS)
    ks_prob = KnapsackProblem.easy(seed=42)
    ks_ad = KnapsackAdapter(ks_prob)
    from algorithms.evolution.GA import GeneticAlgorithm
    from algorithms.biology.ABC import ArtificialBeeColony
    from algorithms.biology.CS import CuckooSearch
    data_ks, err_ks = {}, {}
    print('Knapsack (GA, ABC, CS)...')
    m,s = measure_times(lambda: GeneticAlgorithm(ks_ad, params={'pop_size':40}).solve(iterations=ITERATIONS)); data_ks['GA']=m; err_ks['GA']=s
    m,s = measure_times(lambda: ArtificialBeeColony(ks_ad, params={'pop_size':40}).solve(iterations=ITERATIONS)); data_ks['ABC']=m; err_ks['ABC']=s
    m,s = measure_times(lambda: CuckooSearch(ks_ad, params={'n':40}).solve(iterations=ITERATIONS)); data_ks['CS']=m; err_ks['CS']=s
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_DARK)
    draw_bar(ax, data_ks, err_ks, 'Knapsack Execution Time')
    _savefig(fig, 'exec_time_knapsack.png')

    # 3. Graph Coloring (GA, SA)
    gc_prob = GraphColoringProblem.easy(seed=42)
    gc_ad = GraphColoringAdapter(gc_prob)
    from algorithms.physics.SA import SimulatedAnnealing
    data_gc, err_gc = {}, {}
    print('Graph Coloring (GA, SA)...')
    m,s = measure_times(lambda: GeneticAlgorithm(gc_ad, params={'pop_size':40}).solve(iterations=ITERATIONS)); data_gc['GA']=m; err_gc['GA']=s
    m,s = measure_times(lambda: SimulatedAnnealing(gc_ad, params={'T0':1.0}).solve(iterations=ITERATIONS)); data_gc['SA']=m; err_gc['SA']=s
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG_DARK)
    draw_bar(ax, data_gc, err_gc, 'Graph Coloring Execution Time')
    _savefig(fig, 'exec_time_graphcoloring.png')

    # 4. Shortest Path (BFS, DFS, A*, SA)
    sp_prob = ShortestPathProblem.easy(seed=42) # 20x20
    sp_ad = ShortestPathSAAdapter(sp_prob)
    from algorithms.classical.BFS import BreadthFirstSearch
    from algorithms.classical.DFS import DepthFirstSearch
    from algorithms.classical.A_star import AStar
    data_sp, err_sp = {}, {}
    print('Shortest Path (BFS, DFS, A*, SA)...')
    m,s = measure_times(lambda: BreadthFirstSearch(sp_prob).solve()); data_sp['BFS']=m; err_sp['BFS']=s
    m,s = measure_times(lambda: DepthFirstSearch(sp_prob).solve()); data_sp['DFS']=m; err_sp['DFS']=s
    m,s = measure_times(lambda: AStar(sp_prob).solve()); data_sp['A*']=m; err_sp['A*']=s
    m,s = measure_times(lambda: SimulatedAnnealing(sp_ad).solve(iterations=ITERATIONS)); data_sp['SA']=m; err_sp['SA']=s
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_DARK)
    draw_bar(ax, data_sp, err_sp, 'Shortest Path Execution Time')
    _savefig(fig, 'exec_time_shortestpath.png')

    # Grid 2x2
    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    fig.suptitle('Execution Time -- All Discrete Problems', fontsize=16, fontweight='bold', color=TEXT_CLR)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
    draw_bar(fig.add_subplot(gs[0,0]), {'ACO': m_aco}, {'ACO': s_aco}, 'TSP')
    draw_bar(fig.add_subplot(gs[0,1]), data_ks, err_ks, 'Knapsack')
    draw_bar(fig.add_subplot(gs[1,0]), data_gc, err_gc, 'Graph Coloring')
    
    # Exclude SA from classical comparison for visual scaling if it's too big, 
    # but let's keep it to show the difference.
    draw_bar(fig.add_subplot(gs[1,1]), data_sp, err_sp, 'Shortest Path')
    
    _savefig(fig, 'dashboard_exec_time_discrete.png')
    
    print(f'\nDone in {time.time()-t0_global:.1f}s.')

if __name__ == '__main__':
    main()
