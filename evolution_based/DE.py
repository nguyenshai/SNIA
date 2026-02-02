# DE.py
import numpy as np

class DE:
    def __init__(self, func, bounds, pop_size=50,
                 F=0.8, CR=0.9, iters=200):
        # Hàm mục tiêu
        self.func = func
        # Giới hạn không gian tìm kiếm
        self.bounds = bounds
        # Kích thước quần thể
        self.pop_size = pop_size
        # Hệ số khuếch đại vi phân
        self.F = F
        # Xác suất lai ghép
        self.CR = CR
        # Số vòng lặp
        self.iters = iters
        # Số chiều bài toán
        self.dim = len(bounds)

    def optimize(self):
        # Khởi tạo quần thể ban đầu
        pop = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            (self.pop_size, self.dim)
        )

        # Lưu lịch sử nghiệm tốt nhất
        best_history = []

        for _ in range(self.iters):
            for i in range(self.pop_size):
                # Chọn ngẫu nhiên 3 cá thể khác nhau
                a, b, c = pop[np.random.choice(self.pop_size, 3, replace=False)]

                # Tạo vector đột biến
                mutant = a + self.F * (b - c)

                # Lai ghép giữa mutant và cá thể hiện tại
                cross = np.random.rand(self.dim) < self.CR
                trial = np.where(cross, mutant, pop[i])

                # Chọn lọc: giữ cá thể tốt hơn
                if self.func(trial) < self.func(pop[i]):
                    pop[i] = trial

            # Lưu giá trị fitness tốt nhất mỗi vòng lặp
            fitness = np.array([self.func(ind) for ind in pop])
            best_history.append(np.min(fitness))

        # Trả về nghiệm tốt nhất
        best = pop[np.argmin([self.func(ind) for ind in pop])]
        return best, best_history
