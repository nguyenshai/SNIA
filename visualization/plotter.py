"""Simple plotting utilities for optimization history and populations.

The functions here are intentionally lightweight: they accept the optimizer
`history` (list of snapshots saved by `Optimizer.save_history`) and produce
common diagnostic plots such as convergence curves and 2D population
scatter plots.
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(history: List[dict], figsize=(8, 4), save_path: Optional[str] = None):
	"""Plot global best fitness over iterations.

	Each snapshot in `history` is expected to contain the key
	`'global_best_fit'` (numeric).
	"""
	if not history:
		raise ValueError("history is empty")

	vals = [h.get('global_best_fit', None) for h in history]
	iters = list(range(1, len(vals) + 1))

	plt.figure(figsize=figsize)
	plt.grid(alpha=0.25)
	plt.plot(iters, vals, marker='o', linewidth=1.5)
	plt.xlabel('Iteration')
	plt.ylabel('Global Best Fitness')
	plt.title('Convergence')
	if save_path:
		plt.savefig(save_path, bbox_inches='tight', dpi=150)
	else:
		plt.show()


def plot_population_2d(history: List[dict], iteration: int = -1, ax=None, show_best=True):
	"""Scatter plot of the population for a 2D search space.

	- `history` element must have `positions` (ndarray-like, shape (n,2)).
	- `iteration` selects which snapshot to plot; -1 means last.
	"""
	if not history:
		raise ValueError("history is empty")

	snap = history[iteration]
	positions = np.asarray(snap.get('positions'))
	if positions.ndim != 2 or positions.shape[1] != 2:
		raise ValueError('positions must be shape (n,2) for 2D plotting')

	own_ax = False
	if ax is None:
		fig, ax = plt.subplots(figsize=(6, 6))
		own_ax = True

	ax.scatter(positions[:, 0], positions[:, 1], s=40, c='#58a6ff', edgecolors='k', alpha=0.8)
	if show_best and 'global_best_sol' in snap and snap['global_best_sol'] is not None:
		best = np.asarray(snap['global_best_sol'])
		if best.size == 2:
			ax.scatter([best[0]], [best[1]], s=140, c='#00d4aa', marker='*', edgecolors='w')

	ax.set_title(f'Population (iter {iteration if iteration>=0 else len(history)-1})')
	if own_ax:
		plt.show()
