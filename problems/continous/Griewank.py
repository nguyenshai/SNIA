import numpy as np

"""
    Griewank Function.
    - Multimodal with regularly distributed minima.
    - Global Minimum: f(x) = 0 at x = [0, 0, ..., 0]
    - Algorithm: ACO vs SA vs FA
    """
def griewank_function(x):
    x = np.asarray(x)
    dim = len(x)
    sum_sq = np.sum(x**2) / 4000.0
    # Product of cos(x_i / sqrt(i+1))
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1))))
    return 1.0 + sum_sq - prod_cos

def generate_griewank_test(dim=10):
    # Standard domain for Griewank is quite large
    lower_bound, upper_bound = -600, 600
    # Generate random starting point within bounds
    start_node = np.random.uniform(lower_bound, upper_bound, dim)
    
    return {
        "bounds": (lower_bound, upper_bound),
        "dim": dim,
        "start_node": start_node,
        "expected_value": 0.0,
        "expected_coords": np.zeros(dim)
    }