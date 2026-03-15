# SNIA - Swarm and Nature-Inspired Algorithms

This is a comprehensive optimization framework built in Python that implements various traditional and nature-inspired optimization algorithms (Biological, Physical, and Evolutionary) to solve both Continuous and Discrete benchmark problems.

## 📂 Project Structure

The project is logically divided into self-contained modules:

```text
SNIA/
├── algorithms/                # Implementations of optimization algorithms
│   ├── biology/               # Swarm & biological (ACO, ABC, CS, PSO, FA)
│   ├── classical/             # Traditional search (BFS, DFS, A*, Hill Climbing)
│   ├── evolution/             # Evolutionary (GA, DE)
│   ├── human/                 # Human-inspired (TLBO)
│   └── physics/               # Physics-inspired (SA)
├── problems/                  # Optimization problem definitions
│   ├── continuous/            # Continuous benchmarks (Sphere, Rastrigin, Ackley...)
│   └── discrete/              # Discrete benchmarks (Graph Coloring, TSP, Knapsack...)
├── results/                   # Auto-generated outputs (plots, heatmaps, tables, GIFs)
├── scripts/                   # Advanced analytical and benchmarking scripts
├── visualization/             # Advanced calculation and plotting utilities
│   └── scrollable.py          # Scrollable Matplotlib UI context to prevent ratio distortion
├── main.py                    # 🔥 Flet GUI Dashboard (Primary User Interface)
├── README.md                  # Project documentation
└── report.pdf                 # Report of the project
└── requirements.txt           # Project dependencies
```

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

### F. Per-Problem Detailed Analysis
Provides granular insights for each specific benchmark problem (both continuous and discrete). It evaluates the entire supported algorithm pool against a single problem and calculates Mean Best Fitness and Median Execution Time. Outcomes are presented using a dual-axis Bar/Line Chart, highlighting precision down to the millisecond in edge-cases.
- **Run:** `python scripts/per_problem_benchmark.py`
- *Output:* `results/visualize/per_problem/`

### G. Algorithm Performance Tables
Generates quantitative structured tables evaluating Mean Best Fitness (± Std) alongside Execution Time metrics for each specific algorithm across all compatible problems.
- **Run:** `python scripts/compare_all_problems.py`
- *Output:* `results/tables/per_algo/`

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

**⚠️ OS Specific Notice for Linux (Ubuntu/Linux Distro):**
Flet for Windows is different from Linux one. Some syntax may heavily go wrong!
