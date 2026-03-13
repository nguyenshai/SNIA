# SNIA - Swarm and Nature-Inspired Algorithms

This is a comprehensive optimization framework built in Python that implements various traditional and nature-inspired optimization algorithms (Biological, Physical, and Evolutionary) to solve both Continuous and Discrete benchmark problems.

## 📂 Project Structure

The project is logically divided into self-contained modules:

- **`algorithms/`**: Contains the implementations of the optimization algorithms.
  - `biology/`: Swarm intelligence and biological algorithms (ACO, ABC, CS, PSO, FA).
  - `evolution/`: Evolutionary algorithms (GA, DE).
  - `physics/`: Physics-inspired algorithms (SA).
  - `human/`: Human-inspired algorithms (TLBO).
  - `classical/`: Traditional derivative-free search algorithms (BFS, DFS, A*, Hill Climbing).
- **`problems/`**: Defines the optimization problems.
  - `continuous/`: Classical continuous benchmarks (Sphere, Rastrigin, Ackley, Rosenbrock, Griewank).
  - `discrete/`: Discrete and combinatorial benchmarks (Graph Coloring, Knapsack, Shortest Path, TSP).
- **`visualization/`**: Advanced calculation and plotting utilities tailored for different problem types.
  - Supports Real-time Matplotlib Animations showing the algorithm's state over time.
  - Provides a *Scrollable Matplotlib Window* module (`scrollable.py`) to prevent aspect ratio distortion when viewing large comparison grids.
- **`scripts/`**: Advanced analytical scripts for calculating Execution Time tracking, Robustness checking, Parameter Sensitivity, All-in-One Comprehensive Benchmarking, and GIF generation.
- **`results/`**: The automated output directory where all generated plots, boxplots, heatmaps, and results from test scripts are saved.
- **`main.py`**: 🔥 **The Flet GUI Dashboard** application that acts as the primary user interface. It connects algorithms to problems and integrates interactive controls through a multi-threaded architecture.

---

## 🚀 1. The Dashboard Console (`main.py` Flet UI)

Unlike the bare-bones command-line interface in early versions, the project now features a fully-fledged **Flet UI Dashboard** inside `main.py` representing a beautiful Desktop User Interface!

**Key Features:**
1. Point-and-click GUI to intuitively configure algorithm selections, hyperparameters, population sizes, and problem map difficulties right from the application window.
2. A built-in Sidebar displaying a highly visual **Problem Description** outlining the mathematical intricacies and constraints of the selected problem.
3. Live Log/Terminal tracing in the background. It utilizes **Multiprocessing queues** to bridge data between the Flet UI and Matplotlib logic, preventing UI freezes while computing complex real-time plots.

**How to Launch:**
```bash
python main.py
```
*(If the Flet namespace or Tkinter environment for Matplotlib is missing, refer to the installation instructions below).*

---

## 📊 2. Advanced Analytical Scripts (`scripts/`)

Apart from running single standalone visualizers through `main.py`, the project maintains a suite of comprehensive automation scripts meant for generating data reports, dashboards, or scientific evaluations.

### A. All-in-One Comparison (Compare All) - MOST RECOMMENDED
A comprehensive test bridging both Discrete and Continuous complexities. It generates a 100% complete set of comparison charts involving Speed, Convergence Curves, Final Solution Quality, Median Execution Time, and Robustness Boxplots.
- **Run:** `python scripts/compare_all.py`
- *Output automatically generates into a 3-column grid structure under:* `results/compare/continuous/` & `results/compare/discrete/`

### B. Parameter Sensitivity Analysis
Investigates how changing specific hyperparameters configuration (`w` in PSO, or `F` in DE) affects the performance of an algorithm over target problems, displayed on radar webs and line plots.
- **Discrete Problems:** `python scripts/param_sensitivity.py`
- **Continuous Problems:** `python scripts/param_sensitivity_continuous.py`

### C. Robustness & Final Consistency Evaluation
Analyzes the stochastic (randomness) reliance of Swarm algorithms by running 30 independent loops under the same deterministic map seed and generating distribution **Boxplots**.
- **Discrete Problems:** `python scripts/robustness_discrete.py`
- **Continuous Problems:** `python scripts/robustness_continuous.py`

### D. Execution Time Profiling (Median)
Visualizes the median computational time taken to resolve problems using Bar charts alongside highly detailed Heatmaps.
- **Discrete Problems:** `python scripts/exec_time_discrete.py`
- **Continuous Problems:** `python scripts/exec_time_continuous.py`

### E. GIF Generator 
Records and compiles the entire evolutionary progression of a population (ants, birds, fireflies) converging towards the optimal space per iteration into GIF format.
- **Run:** `python scripts/generate_problem_gifs.py`

---

## 🛠 Dependencies & Requirements

Ensure you have **Python 3.10** (or 3.8+) installed, and set up the environment:

```bash
pip install -r requirements.txt
```

Alternatively, install the core packages manually:
```bash
pip install numpy matplotlib flet Pillow
```

**⚠️ OS Specific Notice for Linux (Ubuntu/Pop_OS!):**
To successfully render Real-time Animations, the native Python GUI toolkit `Tkinter` bindings must be installed on your operating system. (Example below for Python 3.10):
```bash
sudo apt update
sudo apt install python3.10-tk
```
