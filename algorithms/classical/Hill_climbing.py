import numpy as np

def optimize(func, bounds, iters=1000, restarts=5, step_scale=0.1, dim=None):
	"""Random-restart hill climbing for continuous problems."""
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

	best_x = None
	best_f = float('inf')

	for _ in range(restarts):
		x = rng.random(dim) * (ub - lb) + lb
		fx = func(x)
		for _ in range(iters // restarts):
			cand = x + rng.normal(scale=step_scale * (ub - lb))
			cand = np.clip(cand, lb, ub)
			fc = func(cand)
			if fc < fx:
				x = cand
				fx = fc
		if fx < best_f:
			best_f = float(fx)
			best_x = x.copy()

	return best_x, best_f
