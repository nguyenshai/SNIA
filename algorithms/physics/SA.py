import numpy as np

def optimize(func, bounds, iters=1000, T0=1.0, alpha=0.99, dim=None):
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

	x = rng.random(dim) * (ub - lb) + lb
	fx = func(x)
	best_x = x.copy()
	best_f = float(fx)
	T = T0

	for _ in range(iters):
		cand = x + rng.normal(scale=0.1 * (ub - lb))
		cand = np.clip(cand, lb, ub)
		fc = func(cand)
		if fc < fx or rng.random() < np.exp((fx - fc) / max(T, 1e-12)):
			x = cand
			fx = fc
			if fc < best_f:
				best_f = float(fc)
				best_x = cand.copy()
		T *= alpha

	return best_x, best_f
