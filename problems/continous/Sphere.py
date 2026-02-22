import numpy as np

"""
    Sphere Function: f(x) = sum(x_i^2)
    - Convex, Unimodal.
    - Global Minimum: f(x) = 0 at x = [0, 0, ..., 0]
    - Algorithm: Hill climbing vs PSO
"""
def sphere_function(x):
    x = np.asarray(x)
    return np.sum(x**2)

def generate_sphere_test(dim=10):
    # Standard domain for Sphere
    lower_bound, upper_bound = -5.12, 5.12
    # Generate random starting point within bounds
    start_node = np.random.uniform(lower_bound, upper_bound, dim)

    return {"bounds": (lower_bound, upper_bound),
        "dim": dim,
        "start_node": start_node,   
        "expected_value": 0.0,       
        "expected_coords": np.zeros(dim)
    }


