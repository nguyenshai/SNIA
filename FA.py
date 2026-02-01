import math
import random

class FireflyAlgorithm:
    def __init__(self, func, bounds, pop_size=25, alpha=0.5, beta0=1.0, gamma=1.0):
        """
        alpha: Độ ngẫu nhiên - Giảm dần theo thời gian
        beta0: Độ hấp dẫn cơ bản
        gamma: Hệ số hấp thụ ánh sáng - Môi trường mờ hay rõ
        """
        self.func = func
        self.bounds = bounds
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        
        # Khởi tạo quần thể
        self.population = []
        for _ in range(pop_size):
            pos = [random.uniform(b[0], b[1]) for b in bounds]
            # Intensity tỉ lệ nghịch với giá trị hàm (Càng nhỏ càng sáng)
            val = self.func(pos)
            self.population.append({'pos': pos, 'val': val})
            
        self.best_val = min(p['val'] for p in self.population)

    def distance_sq(self, pos_a, pos_b):
        # Tính khoảng cách bình phương (Euclidean squared)
        return sum((a - b)**2 for a, b in zip(pos_a, pos_b))

    def step(self):
        pop = self.population
        n = len(pop)
        
        # Vòng lặp so sánh từng cặp
        for i in range(n):
            for j in range(n):
                # Nếu con j tốt hơn (sáng hơn) con i -> i bay về phía j
                if pop[j]['val'] < pop[i]['val']:
                    r_sq = self.distance_sq(pop[i]['pos'], pop[j]['pos'])
                    
                    # Độ hấp dẫn giảm dần theo khoảng cách
                    beta = self.beta0 * math.exp(-self.gamma * r_sq)
                    
                    new_pos = []
                    for k in range(len(pop[i]['pos'])):
                        # Công thức di chuyển cốt lõi của FA
                        # Vị trí mới = Cũ + Lực hút (Beta) + Ngẫu nhiên (Alpha)
                        move = beta * (pop[j]['pos'][k] - pop[i]['pos'][k])
                        rand_step = self.alpha * (random.random() - 0.5)
                        
                        val = pop[i]['pos'][k] + move + rand_step
                        # Giữ trong biên (Clip bounds)
                        val = max(self.bounds[k][0], min(self.bounds[k][1], val))
                        new_pos.append(val)
                    
                    # Cập nhật vị trí và giá trị mới
                    new_val = self.func(new_pos)
                    pop[i]['pos'] = new_pos
                    pop[i]['val'] = new_val

        # Giảm độ ngẫu nhiên để thuật toán hội tụ dần
        self.alpha *= 0.97 
        
        # Tìm con tốt nhất hiện tại
        current_best = min(p['val'] for p in pop)
        if current_best < self.best_val:
            self.best_val = current_best
            
        return [p['pos'] for p in pop], self.best_val