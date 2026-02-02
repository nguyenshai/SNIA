import numpy as np

def simulated_annealing(
    f,
    dim,
    lb,
    ub,
    T0,
    T_min,
    alpha,
    max_iter
):
    # Khởi tạo nghiệm ban đầu
    x = lb + (ub - lb) * np.random.rand(dim)
    fx = f(x)

    best_x = x.copy()
    best_fx = fx
    history = []

    T = T0
    while T > T_min:
        for _ in range(max_iter):
            # Sinh nghiệm lân cận
            x_new = x + np.random.randn(dim)
            x_new = np.clip(x_new, lb, ub)
            fx_new = f(x_new)

            # Quy tắc chấp nhận
            if fx_new < fx or np.random.rand() < np.exp(-(fx_new - fx) / T):
                x, fx = x_new, fx_new
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx

        history.append(best_fx)
        T *= alpha

    return best_x, best_fx, history
