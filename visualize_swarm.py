import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

# Import các file thư viện cùng thư mục
from benchmark_lib import BenchmarkFunc
from FA import FireflyAlgorithm
from CS import CuckooSearch

# ==========================================
# 1. CẤU HÌNH DỮ LIỆU & TỐI ƯU HÓA
# ==========================================
BOUNDS = [(-5.12, 5.12), (-5.12, 5.12)]
FUNC = BenchmarkFunc.rastrigin

# [TỐI ƯU 1] Giảm Resolution xuống 30 (Thay vì 50 hoặc 100)
# Giúp vẽ bề mặt nhẹ hơn, xoay chuột mượt hơn
RESOLUTION = 30 
x_val = np.linspace(BOUNDS[0][0], BOUNDS[0][1], RESOLUTION)
y_val = np.linspace(BOUNDS[1][0], BOUNDS[1][1], RESOLUTION)
X, Y = np.meshgrid(x_val, y_val)

# Tính toán trước bề mặt Z
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = FUNC([X[i, j], Y[i, j]])

# ==========================================
# 2. CLASS QUẢN LÝ VISUALIZE 3D
# ==========================================
class SwarmVisualizer3D:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 8))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        
        plt.subplots_adjust(bottom=0.15)
        
        self.reset_algorithms()
        self.ani = None 

    def reset_algorithms(self):
        # [MẸO] Giảm pop_size nếu máy vẫn yếu
        self.fa = FireflyAlgorithm(FUNC, BOUNDS, pop_size=30)
        self.cs = CuckooSearch(FUNC, BOUNDS, pop_size=25)
        print(">>> Đã reset dữ liệu thuật toán.")

    def plot_surface_background(self, ax, title):
        # Vẽ bề mặt
        # rstride, cstride: Độ thưa của lưới vẽ, càng cao càng mượt nhưng càng lag
        ax.plot_surface(X, Y, Z, cmap='viridis_r', edgecolor='none', alpha=0.4, rstride=1, cstride=1)
        
        # Đánh dấu đích
        ax.scatter([0], [0], [0], color='red', s=100, marker='*', label='Global Min')
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Cost')
        ax.view_init(elev=45, azim=120)
        ax.set_zlim(0, 100)

    def update(self, frame):
        # --- 1. FIREFLY ALGORITHM ---
        fa_pos, fa_best = self.fa.step()
        
        self.ax1.clear()
        self.plot_surface_background(self.ax1, f"Firefly Algorithm (FA)\nBest: {fa_best:.4f}")
        
        xs = [p[0] for p in fa_pos]
        ys = [p[1] for p in fa_pos]
        # Tính độ cao Z cho từng con đom đóm
        zs = [FUNC(p) for p in fa_pos]
        self.ax1.scatter(xs, ys, zs, c='#FFD700', s=40, edgecolors='black', depthshade=False)

        # --- 2. CUCKOO SEARCH ---
        cs_pos, cs_best = self.cs.step()
        
        self.ax2.clear()
        self.plot_surface_background(self.ax2, f"Cuckoo Search (CS)\nBest: {cs_best:.4f}")
        
        xs = [p[0] for p in cs_pos]
        ys = [p[1] for p in cs_pos]
        zs = [FUNC(p) for p in cs_pos]
        self.ax2.scatter(xs, ys, zs, c='#00FFFF', s=40, edgecolors='black', depthshade=False)

    def run_animation(self):
        if self.ani is not None:
            self.ani.event_source.stop()
            
        # [TỐI ƯU 2] Tăng interval lên 200ms
        # Giảm tần suất vẽ lại giúp CPU đỡ quá tải khi bạn xoay hình
        self.ani = FuncAnimation(self.fig, self.update, frames=100, interval=200, repeat=False)
        plt.draw()

    def start(self):
        self.run_animation()
        plt.show()

    def restart_callback(self, event):
        self.reset_algorithms()
        self.run_animation()

# ==========================================
# 3. CHẠY CHƯƠNG TRÌNH
# ==========================================
vis = SwarmVisualizer3D()

ax_restart = plt.axes([0.45, 0.05, 0.1, 0.05])
btn_restart = Button(ax_restart, 'Restart 3D')
btn_restart.on_clicked(vis.restart_callback)

print("Đang khởi tạo môi trường 3D (Đã tối ưu hóa)...")
vis.start()