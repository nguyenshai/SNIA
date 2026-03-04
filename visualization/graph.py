"""Graph drawing helpers used by discrete problem visualizers.

These helpers are lightweight wrappers around matplotlib to draw nodes,
edges and highlight solution paths for TSP / shortest-path / graph coloring
visualizations found in `problems/`.
"""

from typing import Iterable, Tuple, Optional
import matplotlib.pyplot as plt


def _get_coord(node):
	"""Extract (x,y) coordinate from `node` which may be a tuple or object."""
	if isinstance(node, tuple) and len(node) >= 2:
		return node[0], node[1]
	# object with x,y attributes (like City)
	x = getattr(node, 'x', None)
	y = getattr(node, 'y', None)
	if x is not None and y is not None:
		return x, y
	raise ValueError('node must be (x,y) or have x and y attributes')


def plot_graph(nodes: Iterable, edges: Iterable[Tuple[int, int, float]] = (), coords: Optional[dict] = None,
			   path: Optional[Iterable[int]] = None, ax=None, node_colors: Optional[dict] = None):
	"""Plot nodes and edges.

	- `nodes`: iterable of node identifiers or objects.
	- `edges`: iterable of (u, v, weight) or (u, v).
	- `coords`: optional dict mapping node -> (x,y) coordinates.
	- `path`: optional sequence of node ids to highlight as a route.
	"""
	own_ax = False
	if ax is None:
		fig, ax = plt.subplots(figsize=(8, 6))
		own_ax = True

	# Build coords map
	coord_map = {}
	if coords:
		coord_map.update(coords)
	else:
		# attempt to extract coordinates from nodes if they are objects or tuples
		for n in nodes:
			try:
				coord_map[n] = _get_coord(n)
			except Exception:
				# maybe node is an id and coords provided separately
				pass

	# Draw edges
	for e in edges:
		if len(e) >= 2:
			u, v = e[0], e[1]
		else:
			continue
		if u in coord_map and v in coord_map:
			ux, uy = coord_map[u]
			vx, vy = coord_map[v]
			ax.plot([ux, vx], [uy, vy], color='#c9d1d9', alpha=0.6, zorder=1)

	# Draw nodes
	xs = []
	ys = []
	labels = []
	colors = []
	for n in nodes:
		if n in coord_map:
			x, y = coord_map[n]
		else:
			try:
				x, y = _get_coord(n)
			except Exception:
				continue
		xs.append(x); ys.append(y); labels.append(str(getattr(n, 'id', n)))
		if node_colors and n in node_colors:
			colors.append(node_colors[n])
		else:
			colors.append('#58a6ff')

	ax.scatter(xs, ys, s=120, c=colors, edgecolors='k', zorder=2)
	for x, y, lab in zip(xs, ys, labels):
		ax.annotate(lab, (x, y), textcoords='offset points', xytext=(6, 6), fontsize=8, color='#c9d1d9')

	# Highlight path if provided
	if path:
		path_coords = []
		for p in path:
			if p in coord_map:
				path_coords.append(coord_map[p])
			else:
				try:
					path_coords.append(_get_coord(p))
				except Exception:
					pass
		if len(path_coords) >= 2:
			xs_p, ys_p = zip(*path_coords)
			ax.plot(xs_p, ys_p, color='#00d4aa', linewidth=2.5, zorder=3)

	if own_ax:
		plt.show()


def draw_path(coords: dict, path: Iterable[int], ax=None):
	"""Convenience to draw a path over coordinates dict mapping node->(x,y)."""
	return plot_graph(nodes=coords.keys(), coords=coords, path=path, ax=ax)


def draw_sample_graph(n_nodes: int = 8, edge_prob: float = 0.3, seed: Optional[int] = None,
					  palette: Optional[list] = None, layout: str = 'circle', ax=None, show: bool = True):
	"""Generate and draw a sample undirected graph with optional coloring.

	- `n_nodes`: number of nodes (5-10 recommended)
	- `edge_prob`: probability of edge between any pair
	- `seed`: RNG seed
	- `palette`: list of colors to use for node coloring (will assign randomly)
	- `layout`: 'circle' or 'random' for node placement
	- `ax`: optional matplotlib axis to draw into
	- `show`: whether to call `plt.show()` before returning

	Returns (coords, edges, color_map)
	"""
	import random
	import math
	import matplotlib.pyplot as plt

	rng = random.Random(seed)

	# Coordinates
	coords = {}
	if layout == 'circle':
		for i in range(n_nodes):
			theta = 2 * math.pi * i / n_nodes
			coords[i] = (math.cos(theta), math.sin(theta))
	else:
		for i in range(n_nodes):
			coords[i] = (rng.uniform(-1, 1), rng.uniform(-1, 1))

	# Edges (undirected)
	edges = []
	for i in range(n_nodes):
		for j in range(i + 1, n_nodes):
			if rng.random() < edge_prob:
				edges.append((i, j))

	# Coloring: either use provided palette or random distinct colors
	if palette is None:
		base_colors = ['#e11d48', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#06b6d4']
	else:
		base_colors = list(palette)

	color_map = {}
	for i in range(n_nodes):
		color_map[i] = rng.choice(base_colors)

	# Draw
	own_ax = False
	if ax is None:
		fig, ax = plt.subplots(figsize=(6, 6))
		own_ax = True

	plot_graph(nodes=list(range(n_nodes)), edges=edges, coords=coords, node_colors=color_map, ax=ax)

	if show and own_ax:
		plt.show()

	return coords, edges, color_map
