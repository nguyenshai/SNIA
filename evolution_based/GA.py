# GA.py
import numpy as np

class GA:
    def __init__(self, func, bounds, pop_size=50, generations=200,
                 pc=0.9, pm=0.1):
        # Hàm mục tiêu
        self.func = func
        # Giới hạn không gian tìm kiếm
        self.bounds = bounds
        # Kích thước quần thể
        self.pop_size = pop_size
        # Số thế hệ
        self.generations = generations
        # Xác suất lai ghép
        self.pc = pc
        # Xác suất đột biến
        self.pm = pm
        # Số chiều bài toán
        self.dim = len(bounds)

    def optimize(self):
        # Khởi tạo quần thể ban đầu ngẫu nhiên
        pop = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            (self.pop_size, self.dim)
        )

        # Lưu lịch sử giá trị fitness tốt nhất
        best_history = []

        for _ in range(self.generations):
            # Tính fitness cho từng cá thể
            fitness = np.array([self.func(ind) for ind in pop])

            # Chọn lọc (random selection đơn giản)
            idx = np.random.choice(self.pop_size, self.pop_size)
            parents = pop[idx]

            # Lai ghép
            for i in range(0, self.pop_size, 2):
                if np.random.rand() < self.pc:
                    alpha = np.random.rand()
                    parents[i] = alpha * parents[i] + (1 - alpha) * parents[i + 1]

            # Đột biến
            for i in range(self.pop_size):
                if np.random.rand() < self.pm:
                    j = np.random.randint(self.dim)
                    parents[i, j] += np.random.normal(0, 0.1)

            # Cập nhật quần thể mới
            pop = parents

            # Lưu nghiệm tốt nhất của thế hệ hiện tại
            best_history.append(np.min(fitness))

        # Trả về nghiệm tốt nhất cuối cùng
        best = pop[np.argmin([self.func(ind) for ind in pop])]
        return best, best_history
