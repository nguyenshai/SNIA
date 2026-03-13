# Base class
from .base import Problem

# --- Continuous Problems ---
# Note: Folder name is 'continuous'
from .continuous.Ackley import Ackley
from .continuous.Griewank import Griewank
from .continuous.Rastrigin import Rastrigin
from .continuous.Rosenbrock import Rosenbrock
from .continuous.Sphere import Sphere

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