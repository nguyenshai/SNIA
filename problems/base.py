import numpy as np

class Problem:
    def __init__(self, name, bounds=None, min_val=None, opt_type="min"):
        self.name = name
        self.bounds = bounds  # Tuple (lb, ub) for continuous
        self.min_val = min_val
        self.opt_type = opt_type

    def evaluate(self, solution):
        """Calculate fitness. Must handle 1D (single) or 2D (population) inputs."""
        raise NotImplementedError

    def get_plotting_data(self, points=100):
        """Returns meshgrid X, Y, Z for 2D contour plotting (Continuous only)."""
        return None