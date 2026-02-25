import numpy as np
from ..base import Optimizer

class FireflyAlgorithm(Optimizer):
    """
    Firefly Algorithm (FA) implementation.
    """

    def solve(self, iterations=200):
        # 1. Parameters
        pop_size = self.params.get('pop_size', 50)
        alpha = self.params.get('alpha', 0.5)   # Randomness strength
        beta0 = self.params.get('beta0', 1.0)   # Attractiveness at r=0
        gamma = self.params.get('gamma', 1.0)   # Absorption coefficient
        
        bounds = np.asarray(self.problem.bounds)
        dim = self.problem.dim
        rng = np.random.default_rng()

        # 2. Bounds
        if bounds.ndim == 1 and bounds.shape[0] == 2:
            lb = np.full(dim, float(bounds[0]))
            ub = np.full(dim, float(bounds[1]))
        else:
            lb = bounds[:, 0].astype(float)
            ub = bounds[:, 1].astype(float)

        # 3. Initialization
        pop = rng.random((pop_size, dim)) * (ub - lb) + lb
        fitness = np.array([self.problem.evaluate(x) for x in pop])
        
        best_idx = np.argmin(fitness)
        self.best_solution = pop[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])

        # 4. Main Loop
        for t in range(iterations):
            # Firefly interaction: O(N^2)
            for i in range(pop_size):
                for j in range(pop_size):
                    # Move firefly i towards j if j is brighter (lower fitness)
                    if fitness[j] < fitness[i]:
                        # Calculate Euclidean distance
                        r2 = np.sum((pop[i] - pop[j]) ** 2)
                        
                        # Attractiveness
                        beta = beta0 * np.exp(-gamma * r2)
                        
                        # Random movement dampening over time
                        alpha_t = alpha * (1.0 - t / iterations)
                        step = alpha_t * (rng.random(dim) - 0.5)
                        
                        # Update position
                        pop[i] = pop[i] + beta * (pop[j] - pop[i]) + step
                        pop[i] = np.clip(pop[i], lb, ub)
                        
                        # Evaluate new position
                        fitness[i] = self.problem.evaluate(pop[i])
                        
                        # Update global best
                        if fitness[i] < self.best_fitness:
                            self.best_fitness = float(fitness[i])
                            self.best_solution = pop[i].copy()

            # --- VISUALIZATION HOOK ---
            self.save_history(pop, fitness, self.best_solution, self.best_fitness)

        return self.best_solution, self.best_fitness