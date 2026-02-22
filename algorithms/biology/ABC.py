import numpy as np


def optimize(func, bounds, pop_size=50, iters=200, dim=None):
	"""Artificial Bee Colony for continuous optimization.

	bounds: array-like (dim,2)
	func: callable mapping x -> scalar
	returns best_x, best_f
	"""
	rng = np.random.default_rng()
	# normalize bounds: accept either (low, high) pair or (dim,2) array
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

	# initialize
	pop = rng.random((pop_size, dim)) * (ub - lb) + lb
	fitness = np.array([func(ind) for ind in pop])
	best_idx = np.argmin(fitness)
	best = pop[best_idx].copy()
	best_f = float(fitness[best_idx])
	trial = np.zeros(pop_size)

	for _ in range(iters):
		# Employed bees
		for i in range(pop_size):
			k = rng.integers(pop_size)
			while k == i:
				k = rng.integers(pop_size)
			phi = rng.uniform(-1, 1, size=dim)
			new = pop[i] + phi * (pop[i] - pop[k])
			new = np.clip(new, lb, ub)
			fnew = func(new)
			if fnew < fitness[i]:
				pop[i] = new
				fitness[i] = fnew
				trial[i] = 0
				if fnew < best_f:
					best_f = float(fnew)
					best = new.copy()
			else:
				trial[i] += 1

		# Onlooker bees
		prob = (1.0 / (1.0 + fitness))
		prob = prob / prob.sum()
		for _o in range(pop_size):
			i = rng.choice(pop_size, p=prob)
			k = rng.integers(pop_size)
			while k == i:
				k = rng.integers(pop_size)
			phi = rng.uniform(-1, 1, size=dim)
			new = pop[i] + phi * (pop[i] - pop[k])
			new = np.clip(new, lb, ub)
			fnew = func(new)
			if fnew < fitness[i]:
				pop[i] = new
				fitness[i] = fnew
				trial[i] = 0
				if fnew < best_f:
					best_f = float(fnew)
					best = new.copy()
			else:
				trial[i] += 1

		# Scout bees
		limit = 20
		for i in range(pop_size):
			if trial[i] > limit:
				pop[i] = rng.random(dim) * (ub - lb) + lb
				fitness[i] = func(pop[i])
				trial[i] = 0
				if fitness[i] < best_f:
					best_f = float(fitness[i])
					best = pop[i].copy()

	return best, best_f

