import numpy as np
from ..base import Optimizer

class SimulatedAnnealing(Optimizer):
    """
    Simulated Annealing (SA) implementation.
    Single-solution algorithm.
    """

    def solve(self, iterations=1000):
        # Parameters
        T0 = self.params.get('T0', 1.0)
        alpha = self.params.get('alpha', 0.99)
        
        bounds = np.asarray(self.problem.bounds)
        dim = self.problem.dim
        rng = np.random.default_rng()

        if bounds.ndim == 1 and bounds.shape[0] == 2:
            lb = np.full(dim, float(bounds[0]))
            ub = np.full(dim, float(bounds[1]))
        else:
            lb = bounds[:, 0].astype(float)
            ub = bounds[:, 1].astype(float)

        # Initialize single solution
        current_sol = rng.random(dim) * (ub - lb) + lb
        current_fit = self.problem.evaluate(current_sol)
        
        self.best_solution = current_sol.copy()
        self.best_fitness = float(current_fit)
        
        T = T0

        for _ in range(iterations):
            # Generate neighbor
            candidate = current_sol + rng.normal(scale=0.1 * (ub - lb))
            candidate = np.clip(candidate, lb, ub)
            
            cand_fit = self.problem.evaluate(candidate)
            
            # Acceptance probability
            # If better, accept. If worse, accept with prob exp(-delta/T)
            delta = cand_fit - current_fit
            if cand_fit < current_fit or rng.random() < np.exp(-delta / max(T, 1e-12)):
                current_sol = candidate
                current_fit = cand_fit
                
                # Update global best
                if current_fit < self.best_fitness:
                    self.best_fitness = float(current_fit)
                    self.best_solution = current_sol.copy()
            
            # Cool down
            T *= alpha

            # --- VISUALIZATION HOOK ---
            # Wrap single solution in list/array to mimic population
            self.save_history(
                population=np.array([current_sol]), 
                fitness=np.array([current_fit]), 
                global_best_sol=self.best_solution, 
                global_best_fit=self.best_fitness
            )

        return self.best_solution, self.best_fitness