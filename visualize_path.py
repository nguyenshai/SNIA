import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button
from informedsearch import GraphSearchLib

# ==========================================
# 1. INPUT DỮ LIỆU
# ==========================================
node_coords = {
    'S': (0, 0),    
    'G': (12, 0),   
    'M1': (3, 0), 'M2': (6, 0), 'M3': (9, 0), 
    'H1': (2, 4), 'H2': (6, 5), 'H3': (10, 4), 
    'D1': (3, -2), 'D2': (6, -3), 'D3': (9, -2), 
    'X1': (1, 2), 'X2': (11, 2) 
}

adjacency_list = {
    'S':  {'X1': 2.0, 'M1': 5.0, 'D1': 4.0},
    'M1': {'S': 5.0,  'M2': 30.0, 'X1': 10.0, 'D1': 5.0},
    'M2': {'M1': 30.0, 'M3': 30.0, 'H2': 15.0, 'D2': 5.0},
    'M3': {'M2': 30.0, 'G': 30.0,  'H3': 10.0, 'D3': 5.0},
    'X1': {'S': 2.0,  'H1': 3.0, 'M1': 10.0},
    'H1': {'X1': 3.0, 'H2': 4.0},
    'H2': {'H1': 4.0, 'H3': 4.0, 'M2': 15.0}, 
    'H3': {'H2': 4.0, 'X2': 3.0, 'M3': 10.0},
    'X2': {'H3': 3.0, 'G': 2.0},
    'D1': {'S': 4.0,  'M1': 5.0, 'D2': 10.0},
    'D2': {'D1': 10.0,'M2': 5.0, 'D3': 10.0},
    'D3': {'D2': 10.0,'M3': 5.0, 'G': 8.0},
    'G':  {'M3': 30.0, 'X2': 2.0, 'D3': 8.0}
}

start_node = 'S'
goal_node = 'G'

# ==========================================
# 2. CHẠY THUẬT TOÁN & TÍNH CHI PHÍ
# ==========================================
solver = GraphSearchLib(adjacency_list, node_coords)
path_greedy, hist_greedy = solver.greedy_best_first_search(start_node, goal_node)
path_astar, hist_astar = solver.a_star_search(start_node, goal_node)

max_steps = max(len(hist_greedy), len(hist_astar))

# Hàm tính tổng chi phí đường đi
def calculate_total_cost(path):
    if not path: return 0
    total = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        # Lấy trọng số từ adjacency_list
        weight = adjacency_list[u][v]
        total += weight
    return total

cost_greedy = calculate_total_cost(path_greedy)
cost_astar = calculate_total_cost(path_astar)

# ==========================================
# 3. CLASS VISUALIZE
# ==========================================
class PathVisualizer:
    def __init__(self):
        self.step = 0
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(bottom=0.3, top=0.95, wspace=0.1) 
        
        self.G = nx.Graph()
        for node, pos in node_coords.items(): self.G.add_node(node, pos=pos)
        for u, neighbors in adjacency_list.items():
            for v, w in neighbors.items(): self.G.add_edge(u, v, weight=w)
        self.pos = nx.get_node_attributes(self.G, 'pos')
        
        self.draw_current_step()

    def get_path_segments(self, history_item):
        current = history_item['node']
        parent_map = history_item['parent_map']
        path_segments = []
        curr_trace = current
        while curr_trace in parent_map:
            prev = parent_map[curr_trace]
            path_segments.append((prev, curr_trace))
            curr_trace = prev
        return path_segments

    def draw_single_graph(self, ax, algo_name, history, final_path, final_cost, color_theme):
        ax.clear()
        
        curr_idx = min(self.step, len(history) - 1)
        current_data = history[curr_idx]
        current_node = current_data['node']
        
        # 1. NỀN
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, edge_color='lightgray', style='dashed', width=1.5)
        
        # 2. TRACE PATH 
        active_path_edges = self.get_path_segments(current_data)
        if active_path_edges:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=active_path_edges, 
                                 edge_color=color_theme, width=3, ax=ax, alpha=0.6)

        # 3. FINAL PATH 
        is_finished = (self.step >= len(history) - 1)
        if is_finished and final_path:
            final_edges = list(zip(final_path, final_path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, edgelist=final_edges, edge_color=color_theme, width=5, ax=ax)

        # 4. NODES
        visited_nodes = [item['node'] for item in history[:curr_idx+1]]
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=visited_nodes, node_color=color_theme, node_size=600, ax=ax, alpha=0.2)
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_color='white', edgecolors='black', node_size=800)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[current_node], node_color=color_theme, node_size=900, ax=ax)
        nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_weight='bold')

        # 5. INFO BOX
        info_text = ""
        if "Greedy" in algo_name:
            info_text = f"h={current_data['h']}"
        else:
            info_text = f"g={current_data['g']}\n+ h={current_data['h']}\n= f={current_data['f']}"
        
        x, y = self.pos[current_node]
        ax.text(x, y + 0.6, info_text, fontsize=9, bbox=dict(facecolor='lightyellow', alpha=0.9), 
                horizontalalignment='center', verticalalignment='bottom', fontweight='bold')

        # 6. TRỌNG SỐ
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(
            self.G, self.pos, edge_labels=edge_labels, ax=ax, font_size=8, font_color='#8B0000',
            bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=1.5)
        )
        
        # LABEL DƯỚI ĐÁY 
        if is_finished:
            status = f"DONE! Total Cost: {final_cost}"
            status_color = 'red' if final_cost > 50 else 'green' # Highlight màu nếu cost quá cao/thấp
        else:
            status = f"Thinking... (Step {self.step + 1})"
            status_color = '#333333'

        label_text = f"{algo_name}\n{status}\nNode: {current_node}"
        
        ax.text(0.5, -0.12, label_text, transform=ax.transAxes, 
                horizontalalignment='center', verticalalignment='top', 
                fontsize=11, fontweight='bold', color=status_color)
        ax.axis('off')

    def draw_current_step(self):
        # Truyền thêm tham số Cost vào hàm vẽ
        self.draw_single_graph(self.ax1, "Greedy BFS", hist_greedy, path_greedy, cost_greedy, 'orange')
        self.draw_single_graph(self.ax2, "A* Search (Optimal)", hist_astar, path_astar, cost_astar, 'green')
        self.fig.canvas.draw()

    def next(self, event):
        if self.step < max_steps - 1:
            self.step += 1
            self.draw_current_step()
    
    def prev(self, event):
        if self.step > 0:
            self.step -= 1
            self.draw_current_step()

    def restart(self, event):
        self.step = 0
        self.draw_current_step()

vis = PathVisualizer()

ax_prev = plt.axes([0.3, 0.05, 0.1, 0.05])
ax_restart = plt.axes([0.45, 0.05, 0.1, 0.05])
ax_next = plt.axes([0.6, 0.05, 0.1, 0.05])

btn_prev = Button(ax_prev, '<< Prev')
btn_restart = Button(ax_restart, 'Restart')
btn_next = Button(ax_next, 'Next >>')

btn_prev.on_clicked(vis.prev)
btn_restart.on_clicked(vis.restart)
btn_next.on_clicked(vis.next)

print(f"Greedy Cost: {cost_greedy}")
print(f"A* Cost: {cost_astar}")
plt.show()