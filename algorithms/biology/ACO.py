import numpy as np


def solve_tsp(dist, n_ants=50, iters=200, alpha=1.0, beta=5.0, rho=0.5, q=1.0):
	"""Solve TSP with Ant Colony Optimization.

	Parameters
	- dist: (N,N) numpy array of distances
	- n_ants: number of ants
	- iters: iterations
	- alpha: pheromone importance
	- beta: heuristic importance
	- rho: pheromone evaporation rate
	- q: pheromone deposit factor

	Returns best_route (list of node indices), best_length
	"""
	n = dist.shape[0]
	pher = np.ones((n, n))
	heuristic = 1.0 / (dist + 1e-12)
	rng = np.random.default_rng()

	def tour_length(tour):
		return np.sum([dist[tour[i - 1], tour[i]] for i in range(len(tour))])

	best_len = np.inf
	best_tour = None

	for _ in range(iters):
		all_tours = []
		all_lengths = np.zeros(n_ants)
		for k in range(n_ants):
			start = rng.integers(n)
			tour = [start]
			visited = set(tour)
			for _ in range(n - 1):
				i = tour[-1]
				probs = np.zeros(n)
				unvisited = [j for j in range(n) if j not in visited]
				tau = pher[i, unvisited] ** alpha
				eta = heuristic[i, unvisited] ** beta
				probs_un = tau * eta
				if probs_un.sum() <= 0:
					chosen = rng.choice(unvisited)
				else:
					probs_un = probs_un / probs_un.sum()
					chosen = rng.choice(unvisited, p=probs_un)
				tour.append(chosen)
				visited.add(chosen)
			all_tours.append(tour)
			L = tour_length(tour)
			all_lengths[k] = L
			if L < best_len:
				best_len = L
				best_tour = tour

		# pheromone update
		pher *= (1 - rho)
		for tour, L in zip(all_tours, all_lengths):
			deposit = q / (L + 1e-12)
			for i in range(n):
				a = tour[i - 1]
				b = tour[i]
				pher[a, b] += deposit
				pher[b, a] += deposit

	return best_tour, float(best_len)

