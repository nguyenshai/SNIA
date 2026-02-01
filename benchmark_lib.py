import math

class BenchmarkFunc:
    @staticmethod
    def rastrigin(position):
        # Hàm Rastrigin: Cực tiểu toàn cục tại (0,0) = 0
        # Rất nhiều cực tiểu địa phương (hố bẫy)
        n = len(position)
        total = 10 * n
        for x in position:
            total += x**2 - 10 * math.cos(2 * math.pi * x)
        return total