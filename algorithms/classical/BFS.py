from collections import deque
from ..base import Optimizer

class BreadthFirstSearch(Optimizer):
    """
    BFS for Graph/Grid problems.
    Visualize: Shows the expansion of the 'visited' set.
    """
    def solve(self, iterations=None):
        # Expects problem to have adjacency list or grid logic
        # For ShortestPathProblem, we use its structure
        start = self.problem.start
        goal = self.problem.goal
        
        q = deque([start])
        visited = {start}
        parent = {start: None}
        order = []
        
        found = False
        
        # We can treat 'iterations' as a safety break, 
        # or run until queue empty if iterations is None.
        steps = 0
        
        while q:
            if iterations and steps > iterations: break
            steps += 1
            
            u = q.popleft()
            order.append(u)
            
            if u == goal:
                found = True
                break
            
            # Get neighbors from problem definition
            # Assuming problem.neighbors(r, c) or similar interface
            if hasattr(self.problem, 'neighbors'):
                # For Grid problems (ShortestPath)
                neighbors = self.problem.neighbors(*u)
            else:
                # For Graph problems (Adjacency dict)
                neighbors = self.problem.adjacency.get(u, [])

            for v in neighbors:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
            
            # --- VISUALIZATION HOOK ---
            # Save the currently visited nodes as 'population' for visualizer
            self.history.append({
                'visited_nodes': list(visited),
                'current_node': u,
                'found': found
            })

        # Reconstruct path
        path = []
        if found:
            curr = goal
            while curr is not None:
                path.append(curr)
                curr = parent.get(curr)
            path = path[::-1]
            self.best_solution = path
            # Evaluate using problem's evaluate function if available
            res = self.problem.evaluate(path)
            self.best_fitness = res.get('total_cost', 0) if isinstance(res, dict) else 0
        
        return self.best_solution, self.best_fitness