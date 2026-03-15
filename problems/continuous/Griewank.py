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
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            d = x.shape[0]
            idx = np.arange(1, d + 1, dtype=float)
            return 1.0 + np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(idx)))
        else:
            # Fully vectorized for population (N, dim)
            d = x.shape[1]
            idx = np.arange(1, d + 1, dtype=float)          # (dim,)
            sum_sq = np.sum(x ** 2, axis=1) / 4000.0         # (N,)
            prod_cos = np.prod(np.cos(x / np.sqrt(idx)), axis=1)  # (N,)
            return 1.0 + sum_sq - prod_cos

    def get_plotting_data(self, points=100):
        if self.dim != 2: return None
        lb, ub = self.bounds
        x = np.linspace(lb, ub, points)
        y = np.linspace(lb, ub, points)
        X, Y = np.meshgrid(x, y)
        # Vectorized: treat meshgrid as (N, 2) population
        pop = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = self.evaluate(pop).reshape(X.shape)
        return X, Y, Z

    def get_description(self) -> str:
        return (
            '========================================================' + '\n'
            f'  {self.name.upper()} FUNCTION (Continuous Optimization)\n'
            '========================================================' + '\n'
            'SCENARIO\n'
            '  Optimize math benchmark function in continuous search space.\n'
            f'  Formula: {self.__doc__.splitlines()[0] if self.__doc__ else "N/A"}\n'
            f'  Bounds: {self.bounds}\n'
            f'  Global Min: {self.min_val}\n'
            'DETAILS\n'
            '  Classic surface optimization benchmarking for algorithms.\n'
        )
