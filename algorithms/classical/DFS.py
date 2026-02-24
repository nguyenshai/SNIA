def dfs(adj, start):
	"""Depth-first search (iterative) on adjacency mapping.

	Returns (order list, parent dict)
	"""
	stack = [start]
	visited = set()
	parent = {start: None}
	order = []
	while stack:
		u = stack.pop()
		if u in visited:
			continue
		visited.add(u)
		order.append(u)
		# push neighbors (reverse for typical recursive order)
		for v in reversed(list(adj.get(u, []))):
			if v not in visited:
				parent[v] = u
				stack.append(v)
	return order, parent
