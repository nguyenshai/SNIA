# Base class
from .base import Problem

# --- Continuous Problems ---
# Note: Folder name is 'continous' based on your structure
from .continous.Ackley import Ackley
from .continous.Griewank import Griewank
from .continous.Rastrigin import Rastrigin
from .continous.Rosenbrock import Rosenbrock
from .continous.Sphere import Sphere

# --- Discrete Problems ---
from .discrete.GraphColoring import GraphColoringProblem
from .discrete.Knapsack import KnapsackProblem
from .discrete.ShortestPath import ShortestPathProblem
from .discrete.TSP import TSPProblem

# Define what gets imported when using "from problems import *"
__all__ = [
    # Base
    'Problem',
    
    # Continuous
    'Ackley',
    'Griewank',
    'Rastrigin',
    'Rosenbrock',
    'Sphere',
    
    # Discrete
    'GraphColoringProblem',
    'KnapsackProblem',
    'ShortestPathProblem',
    'TSPProblem'
]