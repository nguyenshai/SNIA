import numpy as np
from ..base import Optimizer

class TLBO(Optimizer):
    """
    Teaching-Learning Based Optimization (TLBO).
    """

    def solve(self, iterations=200):
        pop_size = self.params.get('pop_size', 50)
        
        bounds = np.asarray(self.problem.bounds)
        dim = self.problem.dim
        rng = np.random.default_rng()

        if bounds.ndim == 1 and bounds.shape[0] == 2:
            lb = np.full(dim, float(bounds[0]))
            ub = np.full(dim, float(bounds[1]))
        else:
            lb = bounds[:, 0].astype(float)
            ub = bounds[:, 1].astype(float)

        # Initialize
        pop = rng.random((pop_size, dim)) * (ub - lb) + lb
        fitness = np.array([self.problem.evaluate(x) for x in pop])
        
        best_idx = np.argmin(fitness)
        self.best_solution = pop[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])

        for _ in range(iterations):
            # --- Teacher Phase ---
            # Calculate mean of the class
            mean_pop = pop.mean(axis=0)
            
            # Identify the teacher (best solution)
            teacher_idx = np.argmin(fitness)
            teacher = pop[teacher_idx]
            
            # Teaching factor (1 or 2)
            tf = rng.integers(1, 3)
            
            # Create new population candidates
            new_pop = pop.copy()
            for i in range(pop_size):
                # Move learner towards teacher
                new_sol = pop[i] + rng.random(dim) * (teacher - tf * mean_pop)
                new_sol = np.clip(new_sol, lb, ub)
                
                fnew = self.problem.evaluate(new_sol)
                if fnew < fitness[i]:
                    new_pop[i] = new_sol
                    fitness[i] = fnew
                    if fnew < self.best_fitness:
                        self.best_fitness = float(fnew)
                        self.best_solution = new_sol.copy()
            
            pop = new_pop

            # --- Learner Phase ---
            for i in range(pop_size):
                # Select a random partner to learn from
                j = rng.integers(pop_size)
                while j == i:
                    j = rng.integers(pop_size)
                
                if fitness[i] < fitness[j]:
                    step = pop[i] - pop[j]
                else:
                    step = pop[j] - pop[i]
                
                new_sol = pop[i] + rng.random(dim) * step
                new_sol = np.clip(new_sol, lb, ub)
                
                fnew = self.problem.evaluate(new_sol)
                if fnew < fitness[i]:
                    pop[i] = new_sol
                    fitness[i] = fnew
                    if fnew < self.best_fitness:
                        self.best_fitness = float(fnew)
                        self.best_solution = new_sol.copy()

            # --- VISUALIZATION HOOK ---
            self.save_history(pop, fitness, self.best_solution, self.best_fitness)

        return self.best_solution, self.best_fitness