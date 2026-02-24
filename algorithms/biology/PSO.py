import numpy as np


def optimize(func, bounds, pop_size=50, iters=200, w=0.7, c1=1.5, c2=1.5, dim=None):
	"""Particle Swarm Optimization (real-valued)."""
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

	x = rng.random((pop_size, dim)) * (ub - lb) + lb
	v = rng.standard_normal((pop_size, dim)) * (ub - lb) * 0.1
	fitness = np.array([func(xx) for xx in x])
	pbest = x.copy()
	pbest_f = fitness.copy()
	gidx = np.argmin(pbest_f)
	gbest = pbest[gidx].copy()
	gbest_f = float(pbest_f[gidx])

	for _ in range(iters):
		r1 = rng.random((pop_size, dim))
		r2 = rng.random((pop_size, dim))
		v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
		x = x + v
		x = np.clip(x, lb, ub)
		for i in range(pop_size):
			f = func(x[i])
			if f < pbest_f[i]:
				pbest[i] = x[i].copy()
				pbest_f[i] = f
				if f < gbest_f:
					gbest_f = float(f)
					gbest = x[i].copy()

	return gbest, gbest_f

