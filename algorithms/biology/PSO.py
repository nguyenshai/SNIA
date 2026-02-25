import numpy as np
from ..base import Optimizer
class ParticleSwarmOptimization(Optimizer):
    """
    Particle Swarm Optimization (PSO) implementation inheriting from Optimizer.
    """

    def solve(self, iterations=200):
        # Extract parameters from self.params or use defaults
        pop_size = self.params.get('pop_size', 50)
        w = self.params.get('w', 0.7)
        c1 = self.params.get('c1', 1.5)
        c2 = self.params.get('c2', 1.5)
        
        # Retrieve bounds and dimension from the problem instance
        # Assuming self.problem.bounds is [min_val, max_val] or similar
        bounds = np.asarray(self.problem.bounds)
        dim = self.problem.dim  # Assumes problem has .dim
        
        rng = np.random.default_rng()
        
        # Handle bounds (simplified from original code logic)
        if bounds.ndim == 1 and bounds.shape[0] == 2:
             lb = np.full(dim, float(bounds[0]))
             ub = np.full(dim, float(bounds[1]))
        else:
             # Fallback or specific logic if bounds are (dim, 2)
             lb = bounds[:, 0].astype(float)
             ub = bounds[:, 1].astype(float)

        # Initialize population (positions and velocities)
        x = rng.random((pop_size, dim)) * (ub - lb) + lb
        v = rng.standard_normal((pop_size, dim)) * (ub - lb) * 0.1
        
        # Evaluate initial fitness using problem.evaluate
        fitness = np.array([self.problem.evaluate(xx) for xx in x])
        
        # Initialize Personal Best (pbest) and Global Best (gbest)
        pbest = x.copy()
        pbest_f = fitness.copy()
        
        gidx = np.argmin(pbest_f)
        self.best_solution = pbest[gidx].copy()
        self.best_fitness = float(pbest_f[gidx])

        # Main Loop
        for _ in range(iterations):
            r1 = rng.random((pop_size, dim))
            r2 = rng.random((pop_size, dim))
            
            # Update velocity and position
            v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (self.best_solution - x)
            x = x + v
            x = np.clip(x, lb, ub)
            
            # Evaluate new positions
            # Note: Optimized to pass batch if evaluate supports it, otherwise loop
            try:
                current_fitness = self.problem.evaluate(x)
            except:
                current_fitness = np.array([self.problem.evaluate(ind) for ind in x])

            # Update pbest and gbest
            improved_mask = current_fitness < pbest_f
            pbest[improved_mask] = x[improved_mask]
            pbest_f[improved_mask] = current_fitness[improved_mask]
            
            current_best_idx = np.argmin(pbest_f)
            if pbest_f[current_best_idx] < self.best_fitness:
                self.best_fitness = float(pbest_f[current_best_idx])
                self.best_solution = pbest[current_best_idx].copy()

            # --- VISUALIZATION HOOK ---
            self.save_history(x, current_fitness, self.best_solution, self.best_fitness)

        return self.best_solution, self.best_fitness
	