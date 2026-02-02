# visualize_swarm.py
import matplotlib.pyplot as plt

def plot_convergence(histories, labels, title):
    for hist, label in zip(histories, labels):
        plt.plot(hist, label=label)

    plt.xlabel("Số vòng lặp")
    plt.ylabel("Giá trị fitness tốt nhất")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
