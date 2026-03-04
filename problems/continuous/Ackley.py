import numpy as np
from ..base import Problem

"""
    Ackley Function.
    - Almost flat outer region, deep hole at center.
    - Global Minimum: f(x) = 0 at x = [0, 0, ..., 0]
    - Algorithm: CS vs SA
"""
class Ackley(Problem):
    def __init__(self, dim=10):
        super().__init__("Ackley Function", bounds=[-32.0, 32.0], min_val=0.0, opt_type="min")
        self.dim = dim

    def evaluate(self, x):
        x = np.asarray(x)
        # Constants from your original code
        a, b, c = 20, 0.2, 2 * np.pi
        
        # Helper for vectorized calculation
        if x.ndim == 1:
            d = x.shape[0]
            sum_sq = np.sum(x**2)
            sum_cos = np.sum(np.cos(c * x))
            return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.exp(1)
        else:
            # Population mode: x shape (pop_size, dim)
            d = x.shape[1]
            sum_sq = np.sum(x**2, axis=1)
            sum_cos = np.sum(np.cos(c * x), axis=1)
            return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.exp(1)

    def get_plotting_data(self, points=100):
        if self.dim != 2: return None
        lb, ub = self.bounds
        x = np.linspace(lb, ub, points)
        y = np.linspace(lb, ub, points)
        X, Y = np.meshgrid(x, y)
        
        # Calculate Z for meshgrid
        # Manually computing for meshgrid format to avoid shape errors
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                vec = np.array([X[i,j], Y[i,j]])
                Z[i,j] = self.evaluate(vec)
        return X, Y, Z