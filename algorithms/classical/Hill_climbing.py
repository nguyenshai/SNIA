import numpy as np
from ..base import Optimizer

class HillClimbing(Optimizer):
    """
    Random-restart Hill Climbing implementation.
    """

    def solve(self, iterations=1000):
        # Parameters
        restarts = self.params.get('restarts', 5)
        step_scale = self.params.get('step_scale', 0.1)
        
        bounds = np.asarray(self.problem.bounds)
        dim = self.problem.dim
        rng = np.random.default_rng()

        if bounds.ndim == 1 and bounds.shape[0] == 2:
            lb = np.full(dim, float(bounds[0]))
            ub = np.full(dim, float(bounds[1]))
        else:
            lb = bounds[:, 0].astype(float)
            ub = bounds[:, 1].astype(float)

        self.best_solution = None
        self.best_fitness = float('inf')

        iters_per_restart = iterations // restarts

        for r in range(restarts):
            # Random initialization
            current_x = rng.random(dim) * (ub - lb) + lb
            current_f = self.problem.evaluate(current_x)
            
            if current_f < self.best_fitness:
                self.best_fitness = float(current_f)
                self.best_solution = current_x.copy()

            for _ in range(iters_per_restart):
                # Create candidate neighbor
                candidate = current_x + rng.normal(scale=step_scale * (ub - lb))
                candidate = np.clip(candidate, lb, ub)
                
                cand_f = self.problem.evaluate(candidate)
                
                # Greedy acceptance
                if cand_f < current_f:
                    current_x = candidate
                    current_f = cand_f
                
                # Update Global Best
                if current_f < self.best_fitness:
                    self.best_fitness = float(current_f)
                    self.best_solution = current_x.copy()

                # --- VISUALIZATION HOOK ---
                # Save current climber position
                self.save_history(
                    population=np.array([current_x]), 
                    fitness=np.array([current_f]), 
                    global_best_sol=self.best_solution, 
                    global_best_fit=self.best_fitness
                )

        return self.best_solution, self.best_fitness