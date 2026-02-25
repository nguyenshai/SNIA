import numpy as np
from ..base import Problem

"""
    Sphere Function: f(x) = sum(x_i^2)
    - Convex, Unimodal.
    - Global Minimum: f(x) = 0 at x = [0, 0, ..., 0]
    - Algorithm: Hill climbing vs PSO
"""
class Sphere(Problem):
    def __init__(self, dim=10):
        super().__init__("Sphere Function", bounds=[-5.12, 5.12], min_val=0.0, opt_type="min")
        self.dim = dim

    def evaluate(self, x):
        x = np.asarray(x)
        # Vectorized for population (N, dim) or single (dim,)
        if x.ndim == 1:
            return np.sum(x**2)
        return np.sum(x**2, axis=1)

    def get_plotting_data(self, points=100):
        # Only support visual for 2D
        if self.dim != 2: return None
        lb, ub = self.bounds
        x = np.linspace(lb, ub, points)
        y = np.linspace(lb, ub, points)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2
        return X, Y, Z
