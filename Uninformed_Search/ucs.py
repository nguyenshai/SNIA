import numpy as np

def ucs(graph, start, goal):
    frontier = np.array([[0, start]], dtype=float)
    paths = {start: np.array([start])}
    visited_cost = {}
    expanded_nodes = 0
    while len(frontier) > 0:
        idx = np.argmin(frontier[:, 0])
        cost, node = frontier[idx]
        frontier = np.delete(frontier, idx, axis=0)
        expanded_nodes += 1
        if node == goal:
            return paths[node].tolist(), cost, expanded_nodes
        if node not in visited_cost or cost < visited_cost[node]:
            visited_cost[node] = cost

            for neighbor, weight in graph[node]:
                new_cost = cost + weight
                frontier = np.vstack((frontier, [new_cost, neighbor]))
                paths[neighbor] = np.append(paths[node], neighbor)
    return None, float('inf'), expanded_nodes
