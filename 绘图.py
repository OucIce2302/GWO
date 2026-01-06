import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


# Create a 3D plot for the objective function (example: f(x, y) = x^2 + y^2)
def create_3d_surface_plot():
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    x, y = np.meshgrid(x, y)
    z = x ** 2 + y ** 2  # Objective function: f(x, y) = x^2 + y^2

    fig = plt.figure(figsize=(10, 8))

    # 3D surface plot
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F1')
    ax.set_title('Objective Function f(x, y)')

    return fig, ax, x, y, z


# Convergence curve
def plot_convergence_curve(gwo_scores, cgwo_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(gwo_scores, label='GWO', linestyle='--', color='r')
    plt.plot(cgwo_scores, label='C-GWO', linestyle='--', color='b')

    plt.yscale('log')  # Log scale for better visualization
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value (log scale)')
    plt.title('Convergence Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# Run the visualization for 3D surface plot and convergence curve
fig, ax, x, y, z = create_3d_surface_plot()

# Generate example convergence data (use actual algorithm results here)
gwo_scores = np.random.rand(500) * 1e5  # Example random data for GWO
cgwo_scores = np.random.rand(500) * 1e5  # Example random data for C-GWO

# Plot convergence curve
plot_convergence_curve(gwo_scores, cgwo_scores)
