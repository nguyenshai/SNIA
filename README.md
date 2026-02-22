# SNIA - Search and Nature-Inspired Algorithms

## 📋 Project Description

SNIA is a comprehensive Python library for implementing, analyzing, and comparing classical graph search algorithms and nature-inspired optimization algorithms. The project is built entirely with NumPy and basic Python, implementing data structures and optimization processes from scratch without relying on external optimization libraries.

## 🎯 Objectives

- Provide clean and easy-to-understand implementations of optimization algorithms
- Compare performance of different algorithms on various optimization problems
- Enable learning and research of optimization techniques
- Offer visualization tools for analyzing and interpreting results

---

## 📁 Project Structure

```
SNIA/
├── algorithms/              # Optimization algorithms
│   ├── biology/            # Nature-inspired biology algorithms
│   │   ├── ABC.py          # Artificial Bee Colony
│   │   ├── ACO.py          # Ant Colony Optimization
│   │   ├── CS.py           # Cuckoo Search
│   │   ├── FA.py           # Firefly Algorithm
│   │   └── PSO.py          # Particle Swarm Optimization
│   ├── classical/          # Classical graph search algorithms
│   │   ├── A_star.py       # A* Search
│   │   ├── BFS.py          # Breadth-First Search
│   │   ├── DFS.py          # Depth-First Search
│   │   └── Hill_climbing.py # Hill Climbing
│   ├── evolution/          # Evolutionary algorithms
│   │   ├── DE.py           # Differential Evolution
│   │   └── GA.py           # Genetic Algorithm
│   ├── human/              # Human-inspired algorithms
│   │   └── TLBO.py         # Teaching-Learning Based Optimization
│   ├── physics/            # Physics-inspired algorithms
│   │   └── SA.py           # Simulated Annealing
│   └── __init__.py
│
├── problems/               # Optimization problems
│   ├── continous/          # Continuous optimization problems
│   │   ├── Ackley.py       # Ackley Function
│   │   ├── Rastrigin.py    # Rastrigin Function
│   │   └── Sphere.py       # Sphere Function
│   └── discrete/           # Discrete optimization problems
│       ├── GraphColoring.py # Graph Coloring Problem
│       ├── Knapsack.py     # Knapsack Problem
│       ├── ShortestPath.py # Shortest Path Problem
│       └── TSP.py          # Traveling Salesman Problem
│
├── utils/                  # Utility modules
│   └── visualization.py    # Result visualization tools
│
├── main.py                 # Main entry point
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
- **Ackley**: Ackley function - common benchmark for optimization
- **Rastrigin**: Rastrigin function - challenging multimodal problem
- **Sphere**: Sphere function - basic optimization benchmark

### Discrete Optimization
- **TSP (Traveling Salesman Problem)**: Find shortest route visiting all cities
- **Knapsack**: Maximize value with weight constraints
- **GraphColoring**: Color graph with minimum colors
- **ShortestPath**: Find shortest path between graph nodes

**Last Updated**: 2026-02-04
