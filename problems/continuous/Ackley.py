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
        # Vectorized: treat as (N, 2) population
        pop = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = self.evaluate(pop).reshape(X.shape)
        return X, Y, Z

    def get_description(self) -> str:
        return (
            '=' * 56 + ''
            f'  {self.name.upper()} FUNCTION (Continuous Optimization)'
            '=' * 56 + ''
            'SCENARIO'
            '  Optimize math benchmark function in continuous search space.'
            f'  Formula: {self.__doc__.splitlines()[0] if self.__doc__ else "N/A"}'
            f'  Bounds: {self.bounds}'
            f'  Global Min: {self.min_val}'
            'DETAILS'
            '  Classic surface optimization benchmarking for algorithms.'
        )
