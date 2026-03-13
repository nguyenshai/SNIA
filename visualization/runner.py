import os
import numpy as np

from problems.discrete.ShortestPath import ShortestPathProblem
from problems.discrete.TSP import TSPProblem
from problems.discrete.Knapsack import KnapsackProblem
from problems.discrete.GraphColoring import GraphColoringProblem
from problems.continuous.Sphere import Sphere
from problems.continuous.Rastrigin import Rastrigin
from problems.continuous.Rosenbrock import Rosenbrock
from problems.continuous.Ackley import Ackley
from problems.continuous.Griewank import Griewank

from algorithms.classical.BFS import BreadthFirstSearch
from algorithms.classical.DFS import DepthFirstSearch
from algorithms.classical.A_star import AStar
from algorithms.classical.Hill_climbing import HillClimbing
from algorithms.physics.SA import SimulatedAnnealing
from algorithms.evolution.GA import GeneticAlgorithm
from algorithms.evolution.DE import DifferentialEvolution
from algorithms.biology.PSO import ParticleSwarmOptimization
from algorithms.biology.ABC import ArtificialBeeColony
from algorithms.biology.CS import CuckooSearch
from algorithms.biology.FA import FireflyAlgorithm
from algorithms.biology.ACO import AntColonyOptimization
from algorithms.human.TLBO import TLBO

from visualization.visualize import (
    visualize_uninformed_search,
    visualize_informed_search,
    visualize_local_search,
    visualize_all_categories,
)
from visualization.viz_tsp import visualize_tsp_optimization
from visualization.viz_knapsack import visualize_knapsack_optimization
from visualization.viz_graph_coloring import visualize_graph_coloring_optimization


# ── Inline adapter classes (avoid circular import from main.py) ──────

class _KnapsackAdapter:
    def __init__(self, prob):
        self._prob = prob
        self.name = prob.name
        self.dim = prob.n_items
        self.bounds = [0.0, 1.0]
        self.min_val = None
        self.opt_type = 'min'

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            selection = (x > 0.5).astype(int).tolist()
            result = self._prob.evaluate(selection)
            return -result['total_value']
        return np.array([self.evaluate(row) for row in x])

    def get_plotting_data(self, points=100):
        return None


class _GraphColoringAdapter:
    def __init__(self, prob):
        self._prob = prob
        self.name = prob.name
        self.dim = prob.n_nodes
        self.bounds = [0.0, float(prob.n_colors)]
        self.min_val = 0.0
        self.opt_type = 'min'

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            coloring = {i: min(int(xi), self._prob.n_colors - 1)
                        for i, xi in enumerate(x)}
            result = self._prob.evaluate(coloring)
            return result['total_penalty']
        return np.array([self.evaluate(row) for row in x])

    def get_plotting_data(self, points=100):
        return None


def run_realtime_viz(category, problem, iterations, speed, save_gif, difficulty, out_dir):
    print(f"Starting realtime visualization for {category}...")

    save_path = None
    if save_gif:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{category}_visualization.gif")

    if category == 'uninformed':
        factory = getattr(ShortestPathProblem, difficulty, ShortestPathProblem.easy)
        grid_prob = factory(seed=42)
        algos = {'BFS': BreadthFirstSearch, 'DFS': DepthFirstSearch}
        visualize_uninformed_search(grid_prob, algos, interval=speed,
                                    save_path=save_path, show=True)

    elif category == 'informed':
        factory = getattr(ShortestPathProblem, difficulty, ShortestPathProblem.easy)
        grid_prob = factory(seed=42)
        algos = {'A*': AStar}
        visualize_informed_search(grid_prob, algos, interval=speed,
                                  save_path=save_path, show=True)

    elif category == 'local':
        prob_map = {
            'Sphere': Sphere, 'Rastrigin': Rastrigin, 'Ackley': Ackley,
            'Rosenbrock': Rosenbrock, 'Griewank': Griewank,
        }
        prob = prob_map.get(problem, Sphere)(dim=2)
        local_algos = {
            'PSO': (ParticleSwarmOptimization, {'pop_size': 30}),
            'GA': (GeneticAlgorithm, {'pop_size': 30}),
            'DE': (DifferentialEvolution, {'pop_size': 30}),
            'SA': (SimulatedAnnealing, {}),
            'HC': (HillClimbing, {}),
            'ABC': (ArtificialBeeColony, {'pop_size': 30}),
            'CS': (CuckooSearch, {'n': 30}),
            'FA': (FireflyAlgorithm, {'pop_size': 30}),
            'TLBO': (TLBO, {'pop_size': 30}),
        }
        visualize_local_search(prob, local_algos, iterations=iterations,
                               dim=2, interval=speed,
                               save_path=save_path, show=True)

    elif category == 'tsp':
        factory = getattr(TSPProblem, difficulty, TSPProblem.easy)
        prob = factory(seed=42)
        visualize_tsp_optimization(
            prob, AntColonyOptimization, algo_name='ACO',
            params={'n_ants': 30}, iterations=iterations, interval=speed,
            save_path=save_path, show=True)

    elif category == 'knapsack':
        factory = getattr(KnapsackProblem, difficulty, KnapsackProblem.easy)
        prob = factory(seed=42)
        adapter = _KnapsackAdapter(prob)
        visualize_knapsack_optimization(
            prob, adapter, GeneticAlgorithm, algo_name='GA',
            params={'pop_size': 40}, iterations=iterations, interval=speed,
            save_path=save_path, show=True)

    elif category == 'graph_coloring':
        factory = getattr(GraphColoringProblem, difficulty, GraphColoringProblem.easy)
        prob = factory(seed=42)
        adapter = _GraphColoringAdapter(prob)
        visualize_graph_coloring_optimization(
            prob, adapter, GeneticAlgorithm, algo_name='GA',
            params={'pop_size': 40}, iterations=iterations, interval=speed,
            save_path=save_path, show=True)

    elif category == 'all':
        factory = getattr(ShortestPathProblem, difficulty, ShortestPathProblem.easy)
        grid_prob = factory(seed=42)
        prob_map = {
            'Sphere': Sphere, 'Rastrigin': Rastrigin, 'Ackley': Ackley,
            'Rosenbrock': Rosenbrock, 'Griewank': Griewank,
        }
        cont_prob = prob_map.get(problem, Sphere)(dim=2)
        uninf = {'BFS': BreadthFirstSearch, 'DFS': DepthFirstSearch}
        inf = {'A*': AStar}
        loc = {
            'PSO': (ParticleSwarmOptimization, {'pop_size': 30}),
            'GA': (GeneticAlgorithm, {'pop_size': 30}),
            'DE': (DifferentialEvolution, {'pop_size': 30}),
            'SA': (SimulatedAnnealing, {}),
            'HC': (HillClimbing, {}),
            'ABC': (ArtificialBeeColony, {'pop_size': 30}),
            'CS': (CuckooSearch, {'n': 30}),
            'FA': (FireflyAlgorithm, {'pop_size': 30}),
            'TLBO': (TLBO, {'pop_size': 30}),
        }
        visualize_all_categories(
            grid_problem=grid_prob, continuous_problem=cont_prob,
            uninformed_algos=uninf, informed_algos=inf,
            local_search_algos=loc, iterations_local=iterations,
            interval=speed, save_path=save_path, show=True,
        )
    else:
        raise ValueError(f'Unknown category: {category}')
