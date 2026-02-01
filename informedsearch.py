import math
import heapq

class GraphSearchLib:
    def __init__(self, adjacency_list, node_coords):
        self.adj = adjacency_list
        self.coords = node_coords

    def heuristic(self, node_a, node_b):
        (x1, y1) = self.coords[node_a]
        (x2, y2) = self.coords[node_b]
        return round(math.sqrt((x1 - x2)**2 + (y1 - y2)**2), 2)

    def reconstruct_path(self, came_from, current):
        if current not in came_from and len(came_from) > 0: return [] # Bảo vệ lỗi
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def greedy_best_first_search(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        visited = set()
        history = []

        while open_set:
            h_val, current = heapq.heappop(open_set)
            
            # Lưu snapshot came_from để vẽ đường đi tạm thời
            history.append({
                'node': current,
                'h': self.heuristic(current, goal),
                'g': 0, 
                'f': self.heuristic(current, goal),
                'parent_map': came_from.copy() 
            })

            if current == goal:
                return self.reconstruct_path(came_from, current), history

            visited.add(current)

            for neighbor in self.adj.get(current, {}):
                if neighbor not in visited and neighbor not in [i[1] for i in open_set]:
                    priority = self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current
        
        return None, history

    def a_star_search(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        
        g_score = {node: float('inf') for node in self.adj}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.adj}
        f_score[start] = self.heuristic(start, goal)
        
        history = []

        while open_set:
            f_val, current = heapq.heappop(open_set)
            
            # Lưu snapshot came_from
            history.append({
                'node': current,
                'h': self.heuristic(current, goal),
                'g': g_score[current],
                'f': f_score[current],
                'parent_map': came_from.copy()
            })

            if current == goal:
                return self.reconstruct_path(came_from, current), history

            for neighbor, weight in self.adj.get(current, {}).items():
                tentative_g = g_score[current] + weight
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None, history