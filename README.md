# Multi-UAV Cooperative Spraying Optimization via C-GWO+

This repository contains the official implementation of the **C-GWO+ (Cooperative Grey Wolf Optimization with OBL)** algorithm for multi-UAV path planning in agricultural spraying tasks.

The project simulates a complex urban/agricultural environment considering wind drift, obstacles, and no-spray zones. It compares the proposed **C-GWO+** against traditional engineering baselines (Lawnmower pattern) and other State-of-the-Art (SOTA) meta-heuristic algorithms (PSO, SSA, GWO).

## 📖 Project Overview

Efficient path planning for multi-UAV systems is critical in precision agriculture. This project addresses the challenge of optimizing **Coverage**, **Uniformity**, and **Energy Efficiency** while minimizing **Overspray** (drift into prohibited zones) under wind disturbance.

### Key Features

* **C-GWO+ Algorithm**: An improved Grey Wolf Optimizer featuring:
* **Opposition-Based Learning (OBL)** for better initialization.
* **Cooperative Sub-swarm Mechanism** for multi-drone coordination.
* **Stagnation Reset** to escape local optima.


* **Physics-based Simulation**:
* Gaussian Plume Model for pesticide deposition.
* Wind drift simulation ( scaling based on wind speed/altitude).
* Deposition decay model ().


* **Comprehensive Metrics**: Evaluates Coverage (Cov), Uniformity (Uni), Overspray (Over), and Energy Consumption.
* **Baselines Included**: Lawnmower Pattern (Engineering baseline), Vanilla GWO, PSO, SSA, SGA.

## 📂 File Structure

```text
.
├── OptimizedCode_withDrawing.py   # [MAIN] Simulation of C-GWO+ vs GWO vs Lawnmower
                                   # Generates heatmaps, trajectory plots, and metrics logs.
├── SOTA_Experiment.py             # [BENCHMARK] Comparison against PSO, SSA, SGA
                                   # Runs statistical analysis (30 runs) and plots boxplots/convergence.
├── Algorithm_Comparison.py        # Alternative version of the main simulation logic.
├── requirements.txt               # List of dependencies.
└── runs/                          # Directory where results, logs, and figures are saved.

```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

```


2. Install the required dependencies:
```bash
pip install numpy matplotlib pandas seaborn scipy

```



## 🚀 Usage

### 1. Run the Main Simulation (C-GWO+ vs Baselines)

This script simulates the physical environment, runs the Lawnmower pattern, Vanilla GWO, and C-GWO+. It generates visualization of trajectories and deposition heatmaps.

```bash
python OptimizedCode_withDrawing.py
```

* **Output**: Results are saved in the `runs/cgwo_demo/` directory.
* `figs/`: Contains heatmaps (`fig3*.png`), convergence curves (`fig4*.png`), and radar charts.
* `metrics.csv`: Detailed performance metrics.
* `params.json`: Simulation parameters.



### 2. Run SOTA Comparison

This script runs a statistical comparison (30 independent runs) between C-GWO+, PSO, SSA, and SGA to validate robustness.

```bash
python SOTA_Experiment.py
```

* **Output**: Displays and saves:
* (A) Convergence Analysis curves.
* (B) Final Fitness Distribution boxplots.
* Prints statistical tables (Mean ± Std) to the console.



## 📊 Visualizations

The code automatically generates high-quality figures for academic papers:

1. **Deposition Heatmaps**: Visualizes spray distribution under wind effects.
2. **Trajectory Overlays**: Shows flight paths avoiding obstacles/no-spray zones.
3. **Convergence Curves**: Log-scale comparison of optimization speed.
4. **Radar Charts**: Multi-objective performance balance (Coverage vs Energy vs Safety).

## ⚙️ Configuration

You can modify simulation parameters directly in the `__main__` section of `OptimizedCode_withDrawing.py`:

```python
# Domain settings
nx, ny = 180, 120    # Grid size
dx = 5.0             # Resolution (meters)

# Environment settings
wind_speed = 2.0     # m/s
wind_dir_deg = 45.0  # Degrees

```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
