import numpy as np
from ..base import Optimizer

class ArtificialBeeColony(Optimizer):
    """
    Artificial Bee Colony (ABC) implementation inheriting from Optimizer.
    """

    def solve(self, iterations=200):
        # Extract parameters
        pop_size = self.params.get('pop_size', 50)
        
        # Setup bounds and dimensions
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
        fitness = np.array([self.problem.evaluate(ind) for ind in pop])
        
        best_idx = np.argmin(fitness)
        self.best_solution = pop[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])
        
        trial = np.zeros(pop_size)

        for _ in range(iterations):
            # 1. Employed bees phase
            for i in range(pop_size):
                k = rng.integers(pop_size)
                while k == i:
                    k = rng.integers(pop_size)
                
                phi = rng.uniform(-1, 1, size=dim)
                new_sol = pop[i] + phi * (pop[i] - pop[k])
                new_sol = np.clip(new_sol, lb, ub)
                
                fnew = self.problem.evaluate(new_sol)
                
                if fnew < fitness[i]:
                    pop[i] = new_sol
                    fitness[i] = fnew
                    trial[i] = 0
                    if fnew < self.best_fitness:
                        self.best_fitness = float(fnew)
                        self.best_solution = new_sol.copy()
                else:
                    trial[i] += 1

            # 2. Onlooker bees phase
            # Avoid division by zero in probability calculation
            fit_shifted = fitness - fitness.min() + 1e-10 
            prob = (1.0 / (1.0 + fit_shifted)) # Simple fitness transform
            prob = prob / prob.sum()
            
            for _o in range(pop_size):
                i = rng.choice(pop_size, p=prob)
                k = rng.integers(pop_size)
                while k == i:
                    k = rng.integers(pop_size)
                
                phi = rng.uniform(-1, 1, size=dim)
                new_sol = pop[i] + phi * (pop[i] - pop[k])
                new_sol = np.clip(new_sol, lb, ub)
                
                fnew = self.problem.evaluate(new_sol)
                
                if fnew < fitness[i]:
                    pop[i] = new_sol
                    fitness[i] = fnew
                    trial[i] = 0
                    if fnew < self.best_fitness:
                        self.best_fitness = float(fnew)
                        self.best_solution = new_sol.copy()
                else:
                    trial[i] += 1

            # 3. Scout bees phase
            limit = self.params.get('limit', 20)  # Abandonment threshold
            for i in range(pop_size):
                if trial[i] > limit:
                    pop[i] = rng.random(dim) * (ub - lb) + lb
                    fitness[i] = self.problem.evaluate(pop[i])
                    trial[i] = 0
                    if fitness[i] < self.best_fitness:
                        self.best_fitness = float(fitness[i])
                        self.best_solution = pop[i].copy()

            # --- VISUALIZATION HOOK ---
            # Save the current population state
            self.save_history(pop, fitness, self.best_solution, self.best_fitness)

        return self.best_solution, self.best_fitness