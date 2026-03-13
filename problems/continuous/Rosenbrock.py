import numpy as np
from ..base import Problem

"""
    Rosenbrock Function (Banana Valley).
    - Unimodal in low dims, but hard to converge due to narrow valley.
    - Global Minimum: f(x) = 0 at x = [1, 1, ..., 1]
    - Algorithm: DE vs Hill climbing
"""
class Rosenbrock(Problem):
    def __init__(self, dim=10):
        super().__init__("Rosenbrock Function", bounds=[-5.0, 10.0], min_val=0.0, opt_type="min")
        self.dim = dim

    def evaluate(self, x):
        x = np.asarray(x)
        # Logic: Sum of 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2
        if x.ndim == 1:
            return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
        else:
            # Vectorized for population (N, dim)
            return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2.0)**2.0 + (1 - x[:, :-1])**2.0, axis=1)

    def get_plotting_data(self, points=100):
        if self.dim != 2: return None
        lb, ub = self.bounds
        x = np.linspace(lb, ub, points)
        y = np.linspace(lb, ub, points)
        X, Y = np.meshgrid(x, y)
        
        # Calculate Z for 2D plot
        Z = 100.0 * (Y - X**2.0)**2.0 + (1 - X)**2.0
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
