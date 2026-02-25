import copy
import numpy as np

class Optimizer:
    def __init__(self, problem, params=None):
        self.problem = problem
        self.params = params or {}
        self.history = []  # Stores snapshots for visualization
        self.best_solution = None
        # Initialize based on optimization type (min or max)
        if hasattr(problem, 'opt_type') and problem.opt_type == "max":
            self.best_fitness = float('-inf')
        else:
            self.best_fitness = float('inf')

    def solve(self, iterations):
        """Main loop, must be implemented by subclasses"""
        raise NotImplementedError

    def save_history(self, population, fitness, global_best_sol, global_best_fit):
        """
        Saves the current state of the optimization.
        Uses deepcopy to ensure data is not overwritten in next iterations.
        """
        snapshot = {
            'positions': copy.deepcopy(population),
            'fitness': copy.deepcopy(fitness),
            'global_best_sol': copy.deepcopy(global_best_sol),
            'global_best_fit': global_best_fit
        }
        self.history.append(snapshot)