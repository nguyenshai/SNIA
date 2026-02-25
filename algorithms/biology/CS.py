import numpy as np
from ..base import Optimizer

class CuckooSearch(Optimizer):
    """
    Cuckoo Search (CS) implementation.
    """

    def _levy_flight(self, dim, rng):
        """
        Helper: Generates Levy flight steps using Cauchy distribution.
        """
        # Simple levy-like step using standard Cauchy
        return rng.standard_cauchy(size=dim)

    def solve(self, iterations=200):
        # 1. Extract parameters
        pop_size = self.params.get('n', 50)  # Population size
        pa = self.params.get('pa', 0.25)     # Discovery rate (probability of abandoning nest)
        
        bounds = np.asarray(self.problem.bounds)
        dim = self.problem.dim
        rng = np.random.default_rng()

        # 2. Handle bounds
        if bounds.ndim == 1 and bounds.shape[0] == 2:
            lb = np.full(dim, float(bounds[0]))
            ub = np.full(dim, float(bounds[1]))
        else:
            lb = bounds[:, 0].astype(float)
            ub = bounds[:, 1].astype(float)

        # 3. Initialize Nests
        nests = rng.random((pop_size, dim)) * (ub - lb) + lb
        fitness = np.array([self.problem.evaluate(x) for x in nests])
        
        # Find initial best
        best_idx = np.argmin(fitness)
        self.best_solution = nests[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])

        # 4. Main Loop
        for _ in range(iterations):
            # --- Levy Flight Phase ---
            for i in range(pop_size):
                # Calculate step size
                step = self._levy_flight(dim, rng) * 0.01 * (ub - lb)
                
                # Update position based on best solution
                new_sol = nests[i] + step * (nests[i] - self.best_solution)
                new_sol = np.clip(new_sol, lb, ub)
                
                fnew = self.problem.evaluate(new_sol)
                
                # Greedy selection
                if fnew < fitness[i]:
                    nests[i] = new_sol
                    fitness[i] = fnew
                    if fnew < self.best_fitness:
                        self.best_fitness = float(fnew)
                        self.best_solution = new_sol.copy()

            # --- Abandonment Phase (Discovery) ---
            # Randomly abandon worse nests with probability 'pa'
            abandon_mask = rng.random(pop_size) < pa
            
            # For vectorized replacement (or loop for clarity)
            for i in range(pop_size):
                if abandon_mask[i]:
                    # New random solution
                    nests[i] = rng.random(dim) * (ub - lb) + lb
                    fitness[i] = self.problem.evaluate(nests[i])
                    
                    if fitness[i] < self.best_fitness:
                        self.best_fitness = float(fitness[i])
                        self.best_solution = nests[i].copy()

            # --- VISUALIZATION HOOK ---
            self.save_history(nests, fitness, self.best_solution, self.best_fitness)

        return self.best_solution, self.best_fitness