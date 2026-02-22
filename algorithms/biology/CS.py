import numpy as np


def _levy(dim, rng):
	# simple levy-like step using Cauchy
	return rng.standard_cauchy(size=dim)


def optimize(func, bounds, n=50, iters=200, pa=0.25, dim=None):
	"""Cuckoo Search for continuous optimization.

	bounds: (dim,2)
	"""
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

	nests = rng.random((n, dim)) * (ub - lb) + lb
	fitness = np.array([func(x) for x in nests])
	best_idx = np.argmin(fitness)
	best = nests[best_idx].copy()
	best_f = float(fitness[best_idx])

	for _ in range(iters):
		# generate new solutions by levy flights
		for i in range(n):
			step = _levy(dim, rng) * 0.01 * (ub - lb)
			new = nests[i] + step * (nests[i] - best)
			new = np.clip(new, lb, ub)
			fnew = func(new)
			if fnew < fitness[i]:
				nests[i] = new
				fitness[i] = fnew
				if fnew < best_f:
					best_f = float(fnew)
					best = new.copy()

		# abandon some nests
		K = rng.random(n) < pa
		for i in range(n):
			if K[i]:
				nests[i] = rng.random(dim) * (ub - lb) + lb
				fitness[i] = func(nests[i])
				if fitness[i] < best_f:
					best_f = float(fitness[i])
					best = nests[i].copy()

	return best, best_f

