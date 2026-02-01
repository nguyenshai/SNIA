import math

class BenchmarkFunc:
    @staticmethod
    def rastrigin(position):
        # Hàm mục tiêu kinh điển để test độ thoát bẫy
        # Global Min = 0 tại (0, 0, ...)
        # f(x) = 10n + sum(x^2 - 10cos(2pi*x))
        n = len(position)
        total = 10 * n
        for x in position:
            total += x**2 - 10 * math.cos(2 * math.pi * x)
        return total