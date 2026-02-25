import numpy as np
from ..base import Optimizer

class DifferentialEvolution(Optimizer):
    """
    Differential Evolution (DE) implementation.
    """

    def solve(self, iterations=200):
        # 1. Parameters
        pop_size = self.params.get('pop_size', 60)
        F = self.params.get('F', 0.8)   # Mutation factor
        CR = self.params.get('CR', 0.9) # Crossover probability
        
        bounds = np.asarray(self.problem.bounds)
        dim = self.problem.dim
        rng = np.random.default_rng()

        if bounds.ndim == 1 and bounds.shape[0] == 2:
            lb = np.full(dim, float(bounds[0]))
            ub = np.full(dim, float(bounds[1]))
        else:
            lb = bounds[:, 0].astype(float)
            ub = bounds[:, 1].astype(float)

        # 2. Initialization
        pop = rng.random((pop_size, dim)) * (ub - lb) + lb
        fitness = np.array([self.problem.evaluate(ind) for ind in pop])
        
        # Track best
        best_idx = np.argmin(fitness)
        self.best_solution = pop[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])

        # 3. Main Loop
        for _ in range(iterations):
            for i in range(pop_size):
                # Select 3 distinct random indices distinct from i
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = rng.choice(idxs, size=3, replace=False)
                
                # Mutation: DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover (Binomial)
                cross_points = rng.random(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.integers(dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                ftrial = self.problem.evaluate(trial)
                if ftrial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = ftrial
                    
                    if ftrial < self.best_fitness:
                        self.best_fitness = float(ftrial)
                        self.best_solution = trial.copy()

            # --- VISUALIZATION HOOK ---
            self.save_history(pop, fitness, self.best_solution, self.best_fitness)

        return self.best_solution, self.best_fitness