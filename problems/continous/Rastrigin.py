import numpy as np
from ..base import Problem

"""
    Rastrigin Function.
    - Highly Multimodal (many local optima).
    - Global Minimum: f(x) = 0 at x = [0, 0, ..., 0]
    - Algorithm: GA vs SA
"""
class Rastrigin(Problem):
    def __init__(self, dim=10):
        super().__init__("Rastrigin Function", bounds=[-5.12, 5.12], min_val=0.0, opt_type="min")
        self.dim = dim

    def evaluate(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = x.shape[0]
            return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        else:
            d = x.shape[1]
            return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

    def get_plotting_data(self, points=100):
        if self.dim != 2: return None
        lb, ub = self.bounds
        x = np.linspace(lb, ub, points)
        y = np.linspace(lb, ub, points)
        X, Y = np.meshgrid(x, y)
        
        # Standard Rastrigin calculation for Meshgrid
        Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))
        return X, Y, Z