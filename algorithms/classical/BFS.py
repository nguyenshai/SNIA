from collections import deque

def bfs(adj, start):
	"""Breadth-first search on graph given by adjacency dict/list.

	adj: mapping node -> iterable of neighbors (unweighted)
	start: start node
	Returns (order list, parent dict)
	"""
	q = deque([start])
	visited = {start}
	parent = {start: None}
	order = []
	while q:
		u = q.popleft()
		order.append(u)
		for v in adj.get(u, []):
			if v not in visited:
				visited.add(v)
				parent[v] = u
				q.append(v)
	return order, parent
