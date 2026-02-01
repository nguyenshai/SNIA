import math
import random

class CuckooSearch:
    def __init__(self, func, bounds, pop_size=25, pa=0.25):
        """
        pa: Probability of Abandonment 
        """
        self.func = func
        self.bounds = bounds
        self.pa = pa 
        
        # Khởi tạo các tổ chim
        self.nests = []
        for _ in range(pop_size):
            pos = [random.uniform(b[0], b[1]) for b in bounds]
            self.nests.append({'pos': pos, 'val': self.func(pos)})
            
        # Tìm tổ tốt nhất ban đầu
        self.best_nest = min(self.nests, key=lambda x: x['val'])

    def levy_flight_step(self, current_pos):
        """
        Mô phỏng chuyến bay Levy: Thường đi bước nhỏ, nhưng thỉnh thoảng nhảy rất xa.
        Dùng thuật toán đơn giản hóa để không cần thư viện Scipy.
        """
        new_pos = []
        for i, x in enumerate(current_pos):
            # Tạo bước nhảy ngẫu nhiên
            # 90% là đi bình thường (Gaussian), 10% là nhảy vọt 
            step = random.gauss(0, 1)
            if random.random() < 0.1: 
                step *= 10.0 # Cú nhảy vọt 
            
            # Scale bước nhảy (0.1 là step size scale)
            val = x + 0.1 * step 
            val = max(self.bounds[i][0], min(self.bounds[i][1], val))
            new_pos.append(val)
        return new_pos

    def step(self):
        # BƯỚC 1: CUCKOO ĐẺ TRỨNG 
        # Mỗi con cuckoo chọn ngẫu nhiên một tổ để đẻ nhờ
        cuckoo_idx = random.randint(0, len(self.nests)-1)
        current_nest = self.nests[cuckoo_idx]
        
        # Tạo trứng mới bằng cách bay Levy từ vị trí cũ
        new_cuckoo_pos = self.levy_flight_step(current_nest['pos'])
        new_cuckoo_val = self.func(new_cuckoo_pos)
        
        # Chọn ngẫu nhiên một tổ nạn nhân (j) để tráo trứng
        j = random.randint(0, len(self.nests)-1)
        
        # Nếu trứng mới tốt hơn trứng trong tổ nạn nhân -> Thay thế
        if new_cuckoo_val < self.nests[j]['val']:
            self.nests[j] = {'pos': new_cuckoo_pos, 'val': new_cuckoo_val}

        # BƯỚC 2: CHỦ TỔ PHÁT HIỆN VÀ XÂY TỔ MỚI 
        # Sắp xếp các tổ từ tốt đến xấu
        self.nests.sort(key=lambda x: x['val'])
        
        # Giữ lại các tổ tốt, bỏ đi một phần tổ xấu nhất 
        num_abandon = int(len(self.nests) * self.pa)
        worst_nests_start_index = len(self.nests) - num_abandon
        
        # Thay thế các tổ xấu bằng tổ ngẫu nhiên mới hoàn toàn 
        for k in range(worst_nests_start_index, len(self.nests)):
             pos = [random.uniform(b[0], b[1]) for b in self.bounds]
             self.nests[k] = {'pos': pos, 'val': self.func(pos)}
             
        # Cập nhật Best Global 
        if self.nests[0]['val'] < self.best_nest['val']:
            self.best_nest = self.nests[0].copy()
            
        return [p['pos'] for p in self.nests], self.best_nest['val']