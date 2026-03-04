import heapq
from ..base import Optimizer
import math

class AStar(Optimizer):
    """
    A* Search Algorithm.
    """
    def _heuristic(self, node, goal):
        # Euclidean distance for grid
        return math.sqrt((node[0]-goal[0])**2 + (node[1]-goal[1])**2)

    def solve(self, iterations=None):
        start = self.problem.start
        goal = self.problem.goal
        
        # Priority Queue: (f_score, g_score, node, parent)
        # We need a tie-breaker for heap to avoid comparing nodes directly if f is equal
        count = 0 
        openq = [(0, 0, count, start, None)] 
        
        parents = {}
        gscore = {start: 0.0}
        closed = set()
        
        found = False
        steps = 0
        
        while openq:
            if iterations and steps > iterations: break
            steps += 1
            
            f, g, _, node, parent = heapq.heappop(openq)
            
            if node in closed:
                continue
            
            parents[node] = parent
            closed.add(node)
            
            if node == goal:
                found = True
                break
                
            if hasattr(self.problem, 'neighbors'):
                neighbors = self.problem.neighbors(*node)
            else:
                neighbors = self.problem.adjacency.get(node, [])
                
            for nb in neighbors:
                if nb in closed:
                    continue
                
                # Calculate cost (problem specific)
                # Assuming grid problem has edge_cost method
                if hasattr(self.problem, 'edge_cost'):
                    cost = self.problem.edge_cost(node[0], node[1], nb[0], nb[1])
                else:
                    cost = 1.0 # Default unweighted
                
                ng = g + cost
                
                if ng < gscore.get(nb, float('inf')):
                    gscore[nb] = ng
                    h = self._heuristic(nb, goal)
                    count += 1
                    heapq.heappush(openq, (ng + h, ng, count, nb, node))

            # --- VISUALIZATION HOOK ---
            # Visualize closed set (visited)
            self.history.append({
                'visited_nodes': list(closed),
                'current_node': node,
                'found': found
            })

        path = []
        if found:
            curr = goal
            while curr is not None:
                path.append(curr)
                curr = parents.get(curr)
            path = path[::-1]
            self.best_solution = path
            res = self.problem.evaluate(path)
            self.best_fitness = res.get('total_cost', 0) if isinstance(res, dict) else 0

        return self.best_solution, self.best_fitness