# SNIA - Search and Nature-Inspired Algorithms

## 📋 Project Description

SNIA is a comprehensive Python library for implementing, analyzing, and comparing classical graph search algorithms and nature-inspired optimization algorithms. The project is built entirely with NumPy and basic Python, implementing data structures and optimization processes from scratch without relying on external optimization libraries.

## 🎯 Objectives

- Provide clean and easy-to-understand implementations of optimization algorithms
- Compare performance of different algorithms on various optimization problems
- Enable learning and research of optimization techniques
- Offer visualization tools for analyzing and interpreting results
- Depend on common scientific libraries (NumPy, Matplotlib) listed in `requirements.txt`

---

## 📁 Project Structure

```
SNIA/
├── algorithms/              # Optimization algorithms
│   ├── biology/             # Nature‑inspired biology algorithms
│   │   ├── ABC.py           # Artificial Bee Colony
│   │   ├── ACO.py           # Ant Colony Optimization
│   │   ├── CS.py            # Cuckoo Search
│   │   ├── FA.py            # Firefly Algorithm
│   │   └── PSO.py           # Particle Swarm Optimization
│   ├── classical/          # Classical graph search algorithms
│   │   ├── A_star.py        # A* Search
│   │   ├── BFS.py           # Breadth‑First Search
│   │   ├── DFS.py           # Depth‑First Search
│   │   └── Hill_climbing.py # Hill Climbing
│   ├── evolution/          # Evolutionary algorithms
│   │   ├── DE.py            # Differential Evolution
│   │   └── GA.py            # Genetic Algorithm
│   ├── human/              # Human‑inspired algorithms
│   │   └── TLBO.py          # Teaching‑Learning Based Optimization
│   ├── physics/            # Physics‑inspired algorithms
│   │   └── SA.py            # Simulated Annealing
│   ├── base.py             # Father class
│   └── __init__.py
│
├── problems/               # Optimization problems
│   ├── base.py
│   ├── continous/          # Continuous optimization problems (note the spelling)
│   │   ├── Ackley.py        # Ackley Function
│   │   ├── Griewank.py      # Griewank Function
│   │   ├── Rastrigin.py     # Rastrigin Function
│   │   ├── Rosenbrock.py    # Rosenbrock Function
│   │   ├── Sphere.py        # Sphere Function
│   │   └── __init__.py
│   └── discrete/           # Discrete optimization problems
│       ├── GraphColoring.py # Graph Coloring Problem
│       ├── Knapsack.py      # Knapsack Problem
│       ├── ShortestPath.py  # Shortest Path Problem
│       ├── TSP.py           # Traveling Salesman Problem
│       └── __init__.py
│
├── visualization/          # Visualization utilities
│   ├── animator.py            # Code to create GIF/Video (Contour, Particles)
│   ├── graph_viz.py           # Graph drawing (Nodes, Edges, Path)
│   ├── plotter.py             # Statistical plots (Convergence, Boxplot)
│   └── __init__.py
│
├── main.py                 # Entry point for command‑line use
├── runner.py               # High‑level experiment runner
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
└── LICENSE                 # Project license

```
---

## 🔧 Implemented Algorithms

### 1️⃣ Biology Algorithms
- **ABC (Artificial Bee Colony)**: Based on foraging behavior of honey bees
- **ACO (Ant Colony Optimization)**: Based on pheromone trail behavior of ants
- **CS (Cuckoo Search)**: Based on breeding behavior of cuckoo birds
- **FA (Firefly Algorithm)**: Based on attraction behavior of fireflies
- **PSO (Particle Swarm Optimization)**: Based on movement behavior of bird flocking

### 2️⃣ Classical Algorithms
- **A\* Search**: Heuristic-based shortest path search
- **BFS (Breadth-First Search)**: Level-by-level graph traversal
- **DFS (Depth-First Search)**: Deep-first graph traversal
- **Hill Climbing**: Local greedy optimization

### 3️⃣ Evolutionary Algorithms
- **GA (Genetic Algorithm)**: Based on natural selection process
- **DE (Differential Evolution)**: Uses vector differences for optimization

### 4️⃣ Human-Inspired Algorithms
- **TLBO (Teaching-Learning Based Optimization)**: Based on teaching and learning process

### 5️⃣ Physics Algorithms
- **SA (Simulated Annealing)**: Based on metal cooling process

---

## 📊 Optimization Problems

### Continuous Optimization
- **Ackley**: Common high‑dimensional multimodal benchmark
- **Rastrigin**: Challenging multimodal landscape with many local minima
- **Sphere**: Simple convex quadratic benchmark
- **Griewank**: Function with many widespread local minima
- **Rosenbrock**: Valley‑shaped non‑convex problem used for testing convergence

### Discrete Optimization
These problems include additional real‑world constraints and rich evaluation routines.

- **TSP ("Storm Chaser Delivery Network")** – multi‑constrained traveling salesman on a 2D map with time windows, terrain cost multipliers (mountain/highway), storm‑zone penalties, and fuel limits; evaluation returns cost, time/fuel violations, and feasibility.
- **Knapsack ("Space Cargo Loading")** – 0/1 knapsack with three resource dimensions (weight, volume, power), item synergies and conflicts, fragility penalties reducing effective volume, and minimum critical‑item requirements.
- **GraphColoring ("Festival Scheduling")** – weighted graph coloring where edges represent event conflicts (hard/soft), forbidden adjacent slots, pre‑assigned nodes, and node popularity weights; penalty‑based evaluation and rich visualization.
- **ShortestPath ("Mars Rover Navigation")** – grid‑based pathfinding over an elevation/terrain map with sand/rock/smooth/lava, energy cost from elevation changes, hazard zones with risk penalties, mandatory waypoints that make the problem NP‑hard.

**Last Updated**: 2026-02-24
