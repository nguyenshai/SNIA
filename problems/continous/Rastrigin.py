import numpy as np

"""
    Rastrigin Function.
    - Highly Multimodal (many local optima).
    - Global Minimum: f(x) = 0 at x = [0, 0, ..., 0]
    - Algorithm: GA vs SA
"""
def rastrigin_function(x):
    x = np.asarray(x)
    dim = len(x)
    # 10*d + sum(x^2 - 10*cos(2*pi*x))
    return 10 * dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def generate_rastrigin_test(dim=10):
    # Standard domain for Rastrigin
    lower_bound, upper_bound = -5.12, 5.12
    # Generate random starting point within bounds
    start_node = np.random.uniform(lower_bound, upper_bound, dim)

    return {
        "bounds": (lower_bound, upper_bound),
        "dim": dim,
        "start_node": start_node,
        "expected_value": 0.0,
        "expected_coords": np.zeros(dim)
    }
