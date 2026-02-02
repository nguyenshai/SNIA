import matplotlib.pyplot as plt

def plot_convergence(histories, labels, title):
    plt.figure()
    for h, label in zip(histories, labels):
        plt.plot(h, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
