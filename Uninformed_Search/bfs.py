import numpy as np
def bfs(graph, start, end):
    queue = [np.array([start])]
    visited = np.zeros(len(graph), dtype=bool)
    visited[start] = True
    cost = 0
    while queue:
        path = queue.pop(0)
        node = path[-1]
        cost += 1
        if node == end:
            return path.tolist(), cost
        for neighbor in range(len(graph)):
            if not visited[neighbor]:
                visited[neighbor] = True
                new_path = np.append(path, neighbor)
                queue.append(new_path)
    return None, cost