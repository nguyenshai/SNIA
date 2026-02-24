import numpy as np

def optimize(func, bounds, pop_size=50, iters=200, alpha=0.5, beta0=1.0, gamma=1.0, dim=None):
	"""Firefly Algorithm for continuous optimization."""
	rng = np.random.default_rng()
	b = np.asarray(bounds)
	if b.ndim == 1 and b.shape[0] == 2:
		if dim is None:
			raise ValueError('When bounds is (low, high) pair, please pass dim=<int>')
		lb = np.full(dim, float(b[0]))
		ub = np.full(dim, float(b[1]))
	elif b.ndim == 2 and b.shape[1] == 2:
		lb = b[:, 0].astype(float)
		ub = b[:, 1].astype(float)
		if dim is not None and len(lb) != dim:
			raise ValueError('Provided dim does not match bounds shape')
		dim = lb.shape[0]
	else:
		raise ValueError('bounds must be (low,high) or array of shape (dim,2)')

	pop = rng.random((pop_size, dim)) * (ub - lb) + lb
	fitness = np.array([func(x) for x in pop])
	best_idx = np.argmin(fitness)
	best = pop[best_idx].copy()
	best_f = float(fitness[best_idx])

	for t in range(iters):
		for i in range(pop_size):
			for j in range(pop_size):
				if fitness[j] < fitness[i]:
					r2 = np.sum((pop[i] - pop[j]) ** 2)
					beta = beta0 * np.exp(-gamma * r2)
					step = alpha * (rng.random(dim) - 0.5) * (1.0 - t / iters)
					pop[i] = pop[i] + beta * (pop[j] - pop[i]) + step
					pop[i] = np.clip(pop[i], lb, ub)
					fitness[i] = func(pop[i])
					if fitness[i] < best_f:
						best_f = float(fitness[i])
						best = pop[i].copy()

	return best, best_f

