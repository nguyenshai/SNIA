import numpy as np

"""
    Ackley Function.
    - Almost flat outer region, deep hole at center.
    - Global Minimum: f(x) = 0 at x = [0, 0, ..., 0]
    - Algorithm: CS vs SA
"""
def ackley_function(x):

    x = np.asarray(x)
    dim = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / dim))
    term2 = -np.exp(sum_cos / dim)
    
    return term1 + term2 + a + np.exp(1)

def generate_ackley_test(dim=10):
    # Standard domain for Ackley
    lower_bound, upper_bound = -32, 32
    # Generate random starting point within bounds
    start_node = np.random.uniform(lower_bound, upper_bound, dim)
    
    return {
        "bounds": (lower_bound, upper_bound),
        "dim": dim,
        "start_node": start_node,
        "expected_value": 0.0,
        "expected_coords": np.zeros(dim)
    }