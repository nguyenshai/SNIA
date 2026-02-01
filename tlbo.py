import numpy as np
import random

class TLBO:
    def __init__(self, func, bounds, pop_size=30):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.dim = len(bounds)
        
        # Khởi tạo quần thể (Học sinh)
        # Sử dụng numpy để tính toán vector nhanh hơn (Modern approach)
        self.population = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            self.population[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], pop_size)
            
        # Tính điểm số (Cost) ban đầu
        self.costs = np.array([self.func(ind) for ind in self.population])
        
        # Tìm Giáo viên (Người giỏi nhất)
        best_idx = np.argmin(self.costs)
        self.teacher = self.population[best_idx].copy()
        self.best_cost = self.costs[best_idx]

    def step(self):
        # --- GIAI ĐOẠN 1: TEACHER PHASE (Giáo viên giảng bài) ---
        # Mục tiêu: Kéo trung bình lớp (Mean) về phía Giáo viên
        
        # 1. Tính trung bình trình độ cả lớp hiện tại
        class_mean = np.mean(self.population, axis=0)
        
        # 2. Cập nhật kiến thức cho từng học sinh dựa trên Thầy
        for i in range(self.pop_size):
            # Teaching Factor (TF): Thầy giảng 1 hiểu 1 hoặc hiểu 2 (Random 1 hoặc 2)
            tf = random.randint(1, 2) 
            r = np.random.random(self.dim)
            
            # Công thức cốt lõi: Difference Mean
            difference = r * (self.teacher - tf * class_mean)
            
            # Học sinh cố gắng tiếp thu
            new_pos = self.population[i] + difference
            
            # Kiểm tra biên (Clip bounds)
            for d in range(self.dim):
                new_pos[d] = np.clip(new_pos[d], self.bounds[d][0], self.bounds[d][1])
            
            # Nếu hiểu bài (Cost giảm) thì chấp nhận kiến thức mới
            new_cost = self.func(new_pos)
            if new_cost < self.costs[i]:
                self.population[i] = new_pos
                self.costs[i] = new_cost

        # --- GIAI ĐOẠN 2: LEARNER PHASE (Học nhóm) ---
        # Mục tiêu: Học sinh trao đổi ngẫu nhiên với nhau
        for i in range(self.pop_size):
            # Chọn ngẫu nhiên một bạn học khác (j) khác mình (i)
            idxs = list(range(self.pop_size))
            idxs.remove(i)
            j = random.choice(idxs)
            
            r = np.random.random(self.dim)
            
            # Nếu bạn j giỏi hơn mình -> Học theo bạn j
            if self.costs[j] < self.costs[i]:
                step = self.population[i] + r * (self.population[j] - self.population[i])
            # Nếu mình giỏi hơn bạn j -> Tránh xa cái sai của bạn (hoặc dạy bạn)
            else:
                step = self.population[i] + r * (self.population[i] - self.population[j])
                
            # Kiểm tra biên
            for d in range(self.dim):
                step[d] = np.clip(step[d], self.bounds[d][0], self.bounds[d][1])
                
            new_cost = self.func(step)
            if new_cost < self.costs[i]:
                self.population[i] = step
                self.costs[i] = new_cost

        # Cập nhật lại Giáo viên (Nếu có trò nào xuất sắc vượt thầy)
        current_best_idx = np.argmin(self.costs)
        if self.costs[current_best_idx] < self.best_cost:
            self.best_cost = self.costs[current_best_idx]
            self.teacher = self.population[current_best_idx].copy()
            
        # Trả về dữ liệu để Visualize: 
        # (Vị trí lớp, Vị trí thầy, Vị trí trung bình lớp, Điểm tốt nhất)
        return self.population.copy(), self.teacher.copy(), class_mean, self.best_cost