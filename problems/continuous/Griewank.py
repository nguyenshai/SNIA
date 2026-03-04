import numpy as np
from ..base import Problem
"""
    Griewank Function.
    - Multimodal with regularly distributed minima.
    - Global Minimum: f(x) = 0 at x = [0, 0, ..., 0]
    - Algorithm: ACO vs SA vs FA
    """
class Griewank(Problem):
    def __init__(self, dim=10):
        super().__init__("Griewank Function", bounds=[-600.0, 600.0], min_val=0.0, opt_type="min")
        self.dim = dim

    def evaluate(self, x):
        x = np.asarray(x)
        # Helper to calculate single vector
        def _calc(vec):
            d = len(vec)
            sum_sq = np.sum(vec**2) / 4000.0
            prod_cos = np.prod(np.cos(vec / np.sqrt(np.arange(1, d + 1))))
            return 1.0 + sum_sq - prod_cos

        if x.ndim == 1:
            return _calc(x)
        else:
            # Apply along axis 1 for population
            return np.apply_along_axis(_calc, 1, x)

    def get_plotting_data(self, points=100):
        if self.dim != 2: return None
        lb, ub = self.bounds
        x = np.linspace(lb, ub, points)
        y = np.linspace(lb, ub, points)
        X, Y = np.meshgrid(x, y)
        
        # Manual grid calculation
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.evaluate(np.array([X[i, j], Y[i, j]]))
        return X, Y, Z