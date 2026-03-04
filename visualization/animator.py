"""Animation helpers for visualizing optimization history.

This module provides a small wrapper around ``matplotlib.animation.FuncAnimation``
to animate 2D populations recorded in ``Optimizer.history`` snapshots.

The function is defensive about input shapes and offers convenient options
for saving or showing the animation.
"""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def animate_history(
	history: List[dict],
	interval: int = 200,
	xlim: Optional[Tuple[float, float]] = None,
	ylim: Optional[Tuple[float, float]] = None,
	save_path: Optional[str] = None,
	show: bool = False,
	marker_size: int = 40,
	best_marker_size: int = 140,
	color: str = '#58a6ff',
	best_color: str = '#00d4aa'
) -> animation.FuncAnimation:
	"""Animate a 2D population history.

	Parameters
	- history: list of snapshots produced by `Optimizer.save_history`.
	  Each snapshot should contain a `positions` key (array-like Nx2) and may
	  contain `global_best_sol` (length-2 array-like).
	- interval: milliseconds between frames.
	- xlim/ylim: optional axis limits. If omitted they are inferred.
	- save_path: if provided, attempt to save the animation to this path.
	- show: if True call ``plt.show()`` before returning.
	- marker_size/best_marker_size/color: visual options.

	Returns the created `FuncAnimation` object. The animation's ``fig`` is
	accessible via ``ani._fig`` (or ``ani._fig`` depending on matplotlib
	version); calling code can use it for further customization.
	"""

	if not history:
		raise ValueError('history is empty')

	# Gather all valid 2D positions to infer limits
	valid_positions = []
	for snap in history:
		pos = snap.get('positions')
		if pos is None:
			continue
		arr = np.asarray(pos)
		if arr.ndim == 1 and arr.size == 2:
			arr = arr.reshape((1, 2))
		if arr.ndim == 2 and arr.shape[1] == 2:
			valid_positions.append(arr)

	if not valid_positions:
		raise ValueError('no valid 2D positions found in history')

	concat = np.vstack(valid_positions)
	xmin, ymin = np.min(concat[:, 0]), np.min(concat[:, 1])
	xmax, ymax = np.max(concat[:, 0]), np.max(concat[:, 1])
	pad_x = max(1e-6, (xmax - xmin) * 0.06)
	pad_y = max(1e-6, (ymax - ymin) * 0.06)

	fig, ax = plt.subplots(figsize=(6, 6))
	ax.set_aspect('equal')
	scat = ax.scatter([], [], s=marker_size, c=color, edgecolors='k')
	best_sc = ax.scatter([], [], s=best_marker_size, c=best_color, marker='*', edgecolors='w')

	if xlim is None:
		ax.set_xlim(xmin - pad_x, xmax + pad_x)
	else:
		ax.set_xlim(*xlim)
	if ylim is None:
		ax.set_ylim(ymin - pad_y, ymax + pad_y)
	else:
		ax.set_ylim(*ylim)

	def init():
		scat.set_offsets(np.empty((0, 2)))
		best_sc.set_offsets(np.empty((0, 2)))
		return scat, best_sc

	def update(frame_index: int):
		snap = history[frame_index]
		pos = snap.get('positions') if snap.get('positions') is not None else []
		arr = np.asarray(pos)
		if arr.ndim == 1 and arr.size == 2:
			arr = arr.reshape((1, 2))
		if arr.ndim != 2 or arr.shape[1] != 2:
			arr = np.empty((0, 2))
		scat.set_offsets(arr)

		bpos = snap.get('global_best_sol')
		if bpos is not None:
			b = np.asarray(bpos)
			if b.ndim == 1 and b.size == 2:
				best_sc.set_offsets([b])
			else:
				best_sc.set_offsets(np.empty((0, 2)))
		else:
			best_sc.set_offsets(np.empty((0, 2)))

		return scat, best_sc

	ani = animation.FuncAnimation(fig, update, frames=len(history), init_func=init,
								  interval=interval, blit=False)

	if save_path:
		try:
			ani.save(save_path, dpi=150)
		except Exception as exc:  # pragma: no cover - IO/runtime dependent
			print(f'warning: failed to save animation to {save_path}: {exc}')

	if show:
		try:
			plt.show()
		except Exception:
			# In headless environments show may fail; ignore and return animation
			pass

	return ani
