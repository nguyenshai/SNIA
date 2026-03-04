"""Visualization utilities exported for the SNIA project.

Provides simple plotting and animation helpers that consume the
`Optimizer.history` snapshots produced by algorithms in `algorithms/`.
"""

from .plotter import plot_convergence, plot_population_2d
from .animator import animate_history
from .graph import plot_graph, draw_path

__all__ = [
	'plot_convergence', 'plot_population_2d',
	'animate_history', 'plot_graph', 'draw_path'
]
