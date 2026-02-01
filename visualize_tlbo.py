import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Import thư viện
from benchmark_lib import BenchmarkFunc
from tlbo import TLBO

# ==========================================
# 1. CẤU HÌNH (Dùng hàm Sphere cho dễ nhìn sự hội tụ về tâm)
# ==========================================
# Ta dùng hàm Sphere để dễ thấy thầy kéo trò về đáy bát
BOUNDS = [(-5.12, 5.12), (-5.12, 5.12)]
FUNC = BenchmarkFunc.rastrigin # Hoặc Sphere nếu muốn đơn giản hơn

# Tạo lưới 3D (Resolution 30 cho mượt)
x_val = np.linspace(BOUNDS[0][0], BOUNDS[0][1], 30)
y_val = np.linspace(BOUNDS[1][0], BOUNDS[1][1], 30)
X, Y = np.meshgrid(x_val, y_val)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = FUNC([X[i, j], Y[i, j]])

# ==========================================
# 2. CLASS VISUALIZE TLBO
# ==========================================
class TLBOVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.15)
        
        self.reset_algo()
        self.ani = None

    def reset_algo(self):
        self.tlbo = TLBO(FUNC, BOUNDS, pop_size=40)
        
    def update(self, frame):
        # Chạy thuật toán 1 bước
        pop_pos, teacher_pos, mean_pos, best_cost = self.tlbo.step()
        
        self.ax.clear()
        
        # 1. Vẽ địa hình 
        self.ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.15, rstride=2, cstride=2)
        
        # 2. Vẽ Học Sinh - Màu xanh nhạt
        xs = pop_pos[:, 0]
        ys = pop_pos[:, 1]
        zs = [FUNC(p) for p in pop_pos]
        self.ax.scatter(xs, ys, zs, c='cyan', s=30, alpha=0.6, label='Learners (Học sinh)')
        
        # 3. Vẽ TRUNG BÌNH LỚP - Màu Xanh Dương Đậm
        # Đây là khái niệm quan trọng của TLBO
        z_mean = FUNC(mean_pos)
        self.ax.scatter([mean_pos[0]], [mean_pos[1]], [z_mean], 
                        c='blue', s=100, marker='o', label='Class Mean (TB Lớp)')
        
        # 4. Vẽ GIÁO VIÊN (Teacher) - Màu Đỏ, Hình Sao Lớn
        z_teacher = FUNC(teacher_pos)
        self.ax.scatter([teacher_pos[0]], [teacher_pos[1]], [z_teacher], 
                        c='red', s=200, marker='*', edgecolors='black', label='Teacher (Giáo viên)')
        
        # 5. VẼ VECTOR GIẢNG DẠY 
        # Vẽ một đường thẳng nối từ Mean -> Teacher
        # Để thể hiện lực kéo của thầy đối với trình độ trung bình lớp
        self.ax.plot([mean_pos[0], teacher_pos[0]], 
                     [mean_pos[1], teacher_pos[1]], 
                     [z_mean, z_teacher], 
                     color='magenta', linewidth=3, linestyle='--', label='Teaching Force')
        
        # Setup trục
        self.ax.set_title(f"TLBO Simulation \nBest Grade (Cost): {best_cost:.4f}", fontsize=14)
        self.ax.set_zlim(0, 100)
        self.ax.legend(loc='upper right')
        self.ax.view_init(elev=50, azim=130)

    def start(self):
        self.ani = FuncAnimation(self.fig, self.update, frames=100, interval=200, repeat=False)
        plt.show()

    def restart_callback(self, event):
        self.reset_algo()
        self.ani = FuncAnimation(self.fig, self.update, frames=100, interval=200, repeat=False)
        plt.draw()

# ==========================================
# 3. CHẠY
# ==========================================
vis = TLBOVisualizer()

ax_restart = plt.axes([0.45, 0.05, 0.1, 0.05])
btn_restart = Button(ax_restart, 'Restart Class')
btn_restart.on_clicked(vis.restart_callback)

print("Đang mở lớp học TLBO...")
print("- Điểm Sao Đỏ: Giáo viên")
print("- Điểm Tròn Xanh Đậm: Trình độ trung bình của cả lớp")
print("- Dây màu Hồng: Lực hút của thầy kéo cả lớp tiến bộ")
vis.start()