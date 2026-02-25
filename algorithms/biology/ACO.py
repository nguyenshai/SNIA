import numpy as np
from ..base import Optimizer

class AntColonyOptimization(Optimizer):
    """
    Ant Colony Optimization (ACO) specifically adapted for TSP Problems.
    """

    def solve(self, iterations=200):
        # Parameters
        n_ants = self.params.get('n_ants', 50)
        alpha = self.params.get('alpha', 1.0)
        beta = self.params.get('beta', 5.0)
        rho = self.params.get('rho', 0.5)
        q = self.params.get('q', 1.0)

        # Check if problem is indeed a TSP type (has distance_matrix)
        if not hasattr(self.problem, 'distance_matrix'):
            raise ValueError("ACO currently only supports TSP-like problems with 'distance_matrix'.")

        dist = self.problem.distance_matrix
        n = dist.shape[0]
        
        # Pheromone initialization
        pher = np.ones((n, n))
        heuristic = 1.0 / (dist + 1e-12) # Visibility
        rng = np.random.default_rng()

        # Helper to calculate route length
        def tour_length(tour):
            # Calculate distance: sum of edges + return to start
            # Assuming TSPProblem might store logic, but doing it here for speed
            length = 0
            for i in range(len(tour)):
                length += dist[tour[i-1], tour[i]]
            return length

        self.best_fitness = np.inf # Minimize distance
        self.best_solution = None

        for _ in range(iterations):
            all_tours = []
            all_lengths = np.zeros(n_ants)
            
            # 1. Ant construction phase
            for k in range(n_ants):
                start_node = rng.integers(n)
                tour = [start_node]
                visited = set(tour)
                
                for _ in range(n - 1):
                    i = tour[-1]
                    
                    # Calculate probabilities
                    unvisited = [j for j in range(n) if j not in visited]
                    
                    # Vectorized lookup for speed
                    tau = pher[i, unvisited] ** alpha
                    eta = heuristic[i, unvisited] ** beta
                    probs = tau * eta
                    
                    if probs.sum() <= 0:
                        chosen = rng.choice(unvisited)
                    else:
                        probs = probs / probs.sum()
                        chosen = rng.choice(unvisited, p=probs)
                    
                    tour.append(chosen)
                    visited.add(chosen)
                
                all_tours.append(tour)
                L = tour_length(tour)
                all_lengths[k] = L
                
                if L < self.best_fitness:
                    self.best_fitness = L
                    self.best_solution = tour

            # 2. Pheromone update phase
            pher *= (1 - rho) # Evaporation
            for tour, L in zip(all_tours, all_lengths):
                deposit = q / (L + 1e-12)
                for i in range(n):
                    a = tour[i - 1]
                    b = tour[i]
                    pher[a, b] += deposit
                    pher[b, a] += deposit

            # --- VISUALIZATION HOOK ---
            # For discrete problems like TSP, 'positions' might be the pheromone matrix 
            # or the collection of paths. Here we save the best tour of this iter.
            # Using custom structure for saving history
            iter_best_idx = np.argmin(all_lengths)
            
            self.save_history(
                population=all_tours,  # Save all ant paths
                fitness=all_lengths,
                global_best_sol=self.best_solution,
                global_best_fit=self.best_fitness
            )

        return self.best_solution, self.best_fitness