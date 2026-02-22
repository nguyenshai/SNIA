import numpy as np

"""
    Rosenbrock Function (Banana Valley).
    - Unimodal in low dims, but hard to converge due to narrow valley.
    - Global Minimum: f(x) = 0 at x = [1, 1, ..., 1]
    - Algorithm: DE vs Hill climbing
"""
def rosenbrock_function(x):
    x = np.asarray(x)
    # Sum of 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def generate_rosenbrock_test(dim=10):
    # Standard domain for Rosenbrock
    lower_bound, upper_bound = -5, 10
    # Generate random starting point within bounds
    start_node = np.random.uniform(lower_bound, upper_bound, dim)
    
    return {
        "bounds": (lower_bound, upper_bound),
        "dim": dim,
        "start_node": start_node,
        "expected_value": 0.0,
        "expected_coords": np.ones(dim) 
    }