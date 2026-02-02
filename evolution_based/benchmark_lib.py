# benchmark_lib.py
import numpy as np

class BenchmarkFunc:
    # Hàm Sphere: f(x) = tổng bình phương các phần tử
    def sphere(self, x):
        return np.sum(x ** 2)

    # Hàm Rastrigin: nhiều cực trị cục bộ, khó tối ưu
    def rastrigin(self, x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    # Hàm Ackley: kiểm tra khả năng thoát local optimum
    def ackley(self, x):
        return (-20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
                - np.exp(np.mean(np.cos(2 * np.pi * x)))
                + 20 + np.e)
