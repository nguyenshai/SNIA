import numpy as np

def optimize(func, bounds, pop_size=60, iters=200, cx_rate=0.7, mut_rate=0.1, dim=None):
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

	def tournament_select(k=3):
		idx = rng.integers(pop_size, size=k)
		return idx[np.argmin(fitness[idx])]

	for _ in range(iters):
		newpop = []
		for _ in range(pop_size // 2):
			a = tournament_select()
			b = tournament_select()
			pa = pop[a].copy()
			pb = pop[b].copy()
			if rng.random() < cx_rate:
				# blend crossover
				alpha = rng.random(dim)
				child1 = alpha * pa + (1 - alpha) * pb
				child2 = alpha * pb + (1 - alpha) * pa
			else:
				child1 = pa.copy()
				child2 = pb.copy()
			# mutation
			for child in (child1, child2):
				if rng.random() < mut_rate:
					i = rng.integers(dim)
					child[i] += rng.normal(scale=0.1 * (ub[i] - lb[i]))
				child[:] = np.clip(child, lb, ub)
				newpop.append(child)
		pop = np.array(newpop)[:pop_size]
		fitness = np.array([func(ind) for ind in pop])

	best_idx = np.argmin(fitness)
	return pop[best_idx].copy(), float(fitness[best_idx])

