import numpy as np
from ..base import Optimizer

class GeneticAlgorithm(Optimizer):
    """
    Genetic Algorithm (GA) implementation inheriting from Optimizer.
    """

    def solve(self, iterations=200):
        # Parameters
        pop_size = self.params.get('pop_size', 60)
        cx_rate = self.params.get('cx_rate', 0.7)
        mut_rate = self.params.get('mut_rate', 0.1)
        
        bounds = np.asarray(self.problem.bounds)
        dim = self.problem.dim
        rng = np.random.default_rng()

        # Bounds handling
        if bounds.ndim == 1 and bounds.shape[0] == 2:
            lb = np.full(dim, float(bounds[0]))
            ub = np.full(dim, float(bounds[1]))
        else:
            lb = bounds[:, 0].astype(float)
            ub = bounds[:, 1].astype(float)

        # Initialize Population
        pop = rng.random((pop_size, dim)) * (ub - lb) + lb
        fitness = np.array([self.problem.evaluate(ind) for ind in pop])
        
        best_idx = np.argmin(fitness)
        self.best_solution = pop[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])

        def tournament_select(k=3):
            idx = rng.integers(pop_size, size=k)
            best_in_tourn = idx[np.argmin(fitness[idx])]
            return best_in_tourn

        for _ in range(iterations):
            newpop = []
            # Elitism: optional, but often good to keep the best
            # newpop.append(self.best_solution.copy()) 
            
            # Generate new population
            while len(newpop) < pop_size:
                # Selection
                a_idx = tournament_select()
                b_idx = tournament_select()
                pa = pop[a_idx].copy()
                pb = pop[b_idx].copy()
                
                # Crossover
                if rng.random() < cx_rate:
                    alpha = rng.random(dim)
                    child1 = alpha * pa + (1 - alpha) * pb
                    child2 = alpha * pb + (1 - alpha) * pa
                else:
                    child1 = pa.copy()
                    child2 = pb.copy()
                
                # Mutation and Add
                for child in (child1, child2):
                    if len(newpop) >= pop_size: break
                    
                    if rng.random() < mut_rate:
                        i = rng.integers(dim)
                        child[i] += rng.normal(scale=0.1 * (ub[i] - lb[i]))
                    
                    child[:] = np.clip(child, lb, ub)
                    newpop.append(child)
            
            pop = np.array(newpop)
            fitness = np.array([self.problem.evaluate(ind) for ind in pop])

            # Update Global Best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = float(fitness[current_best_idx])
                self.best_solution = pop[current_best_idx].copy()

            # --- VISUALIZATION HOOK ---
            self.save_history(pop, fitness, self.best_solution, self.best_fitness)

        return self.best_solution, self.best_fitness