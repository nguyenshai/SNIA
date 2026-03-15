"""Visualization utilities exported for the SNIA project.

Provides simple plotting and animation helpers that consume the
`Optimizer.history` snapshots produced by algorithms in `algorithms/`.
"""

from .visualize import (
	visualize_uninformed_search,
	visualize_informed_search,
	visualize_local_search,
	animate_grid_search,
	animate_local_search,
	visualize_all_categories,
	visualize_tsp_optimization,
	visualize_knapsack_optimization,
	visualize_graph_coloring_optimization,
)

__all__ = [
	'visualize_uninformed_search',
	'visualize_informed_search',
	'visualize_local_search',
	'animate_grid_search',
	'animate_local_search',
	'visualize_all_categories',
	'visualize_tsp_optimization',
	'visualize_knapsack_optimization',
	'visualize_graph_coloring_optimization',
]
