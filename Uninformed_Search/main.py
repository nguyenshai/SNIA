import numpy as np
from bfs import bfs
from dfs import dfs
from ucs import ucs
def main():
    # ----- Đồ thị không trọng số -----
    graph_unweighted = {
        0: np.array([1, 2]),
        1: np.array([3]),
        2: np.array([3]),
        3: np.array([])
    }

    # ----- Đồ thị có trọng số -----
    graph_weighted = {
        0: np.array([[1, 1], [2, 5]]),
        1: np.array([[3, 1]]),
        2: np.array([[3, 1]]),
        3: np.array([])
    }

    start, goal = 0, 3

    print("===== BFS =====")
    path, expanded = bfs(graph_unweighted, start, goal)
    print("Path:", path)
    print("Expanded nodes:", expanded)

    print("\n===== DFS =====")
    path, expanded = dfs(graph_unweighted, start, goal)
    print("Path:", path)
    print("Expanded nodes:", expanded)

    print("\n===== UCS =====")
    path, cost, expanded = ucs(graph_weighted, start, goal)
    print("Path:", path)
    print("Cost:", cost)
    print("Expanded nodes:", expanded)
if __name__ == "__main__":
    main()