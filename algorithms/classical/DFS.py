from ..base import Optimizer

class DepthFirstSearch(Optimizer):
    """
    DFS for Graph/Grid problems.
    """
    def solve(self, iterations=None):
        start = self.problem.start
        goal = self.problem.goal
        
        stack = [start]
        visited = set()
        parent = {start: None}
        
        found = False
        steps = 0
        
        while stack:
            if iterations and steps > iterations: break
            steps += 1
            
            u = stack.pop()
            if u in visited:
                continue
            
            visited.add(u)
            
            if u == goal:
                found = True
                break
                
            # Get neighbors
            if hasattr(self.problem, 'neighbors'):
                neighbors = self.problem.neighbors(*u)
            else:
                neighbors = self.problem.adjacency.get(u, [])
            
            # Reverse for typical recursive order simulation in stack
            for v in reversed(list(neighbors)):
                if v not in visited:
                    parent[v] = u
                    stack.append(v)

            # --- VISUALIZATION HOOK ---
            self.history.append({
                'visited_nodes': list(visited),
                'current_node': u,
                'found': found
            })
            
        path = []
        if found:
            curr = goal
            while curr is not None:
                path.append(curr)
                curr = parent.get(curr)
            path = path[::-1]
            self.best_solution = path
            res = self.problem.evaluate(path)
            self.best_fitness = res.get('total_cost', 0) if isinstance(res, dict) else 0

        return self.best_solution, self.best_fitness