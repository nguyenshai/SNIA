import numpy as np


def optimize(func, bounds, pop_size=60, iters=200, F=0.8, CR=0.9, dim=None):
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
	fitness = np.array([func(ind) for ind in pop])

	for _ in range(iters):
		for i in range(pop_size):
			idxs = [idx for idx in range(pop_size) if idx != i]
			a, b, c = rng.choice(idxs, size=3, replace=False)
			mutant = pop[a] + F * (pop[b] - pop[c])
			mutant = np.clip(mutant, lb, ub)
			cross = rng.random(dim) < CR
			if not np.any(cross):
				cross[rng.integers(dim)] = True
			trial = np.where(cross, mutant, pop[i])
			ftrial = func(trial)
			if ftrial < fitness[i]:
				pop[i] = trial
				fitness[i] = ftrial

	best_idx = np.argmin(fitness)
	return pop[best_idx].copy(), float(fitness[best_idx])

