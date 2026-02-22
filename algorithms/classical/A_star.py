import heapq

def astar(adj, start, goal, h):
	"""A* search.

	adj: mapping node -> iterable of (neighbor, cost) pairs
	h: heuristic function h(node) -> estimate to goal
	returns path list and cost
	"""
	openq = [(h(start), 0.0, start, None)]  # (f, g, node, parent)
	parents = {}
	gscore = {start: 0.0}
	closed = set()

	while openq:
		f, g, node, parent = heapq.heappop(openq)
		if node in closed:
			continue
		parents[node] = parent
		if node == goal:
			# reconstruct path
			path = []
			cur = node
			while cur is not None:
				path.append(cur)
				cur = parents.get(cur)
			return path[::-1], g
		closed.add(node)
		for nb, cost in adj.get(node, []):
			ng = g + cost
			if nb in closed and ng >= gscore.get(nb, float('inf')):
				continue
			if ng < gscore.get(nb, float('inf')):
				gscore[nb] = ng
				heapq.heappush(openq, (ng + h(nb), ng, nb, node))

	return None, float('inf')

