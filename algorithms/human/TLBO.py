import numpy as np

def optimize(func, bounds, pop_size=50, iters=200, dim=None):
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

	for _ in range(iters):
		# Teacher phase
		mean = pop.mean(axis=0)
		teacher = pop[np.argmin(fitness)]
		T = rng.integers(1, 3)
		newpop = pop.copy()
		for i in range(pop_size):
			new = pop[i] + rng.random(dim) * (teacher - T * mean)
			new = np.clip(new, lb, ub)
			fnew = func(new)
			if fnew < fitness[i]:
				newpop[i] = new
				fitness[i] = fnew
				if fnew < best_f:
					best_f = float(fnew)
					best = new.copy()
		pop = newpop

		# Learner phase
		for i in range(pop_size):
			j = rng.integers(pop_size)
			while j == i:
				j = rng.integers(pop_size)
			if fitness[i] < fitness[j]:
				step = pop[i] - pop[j]
			else:
				step = pop[j] - pop[i]
			new = pop[i] + rng.random(dim) * step
			new = np.clip(new, lb, ub)
			fnew = func(new)
			if fnew < fitness[i]:
				pop[i] = new
				fitness[i] = fnew
				if fnew < best_f:
					best_f = float(fnew)
					best = new.copy()

	return best, best_f

