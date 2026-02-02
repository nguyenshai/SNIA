import numpy as np
def dfs(graph, start, end):
    stack = [np.array([start])]
    visited = np.zeros(len(graph), dtype=bool)
    visited[start] = True
    cost = 0
    while stack:
        path = stack.pop()
        node = path[-1]
        cost += 1
        if node == end:
            return path.tolist(), cost
        if not visited[node]:
            visited[node] = True
            for neighbor in graph[node]:
                new_path = np.append(path, neighbor)
                stack.append(new_path)
    return None, cost