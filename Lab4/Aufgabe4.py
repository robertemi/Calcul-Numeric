import numpy as np
import matplotlib.pyplot as plt

def runge_function(x):
    return 1 / (1 + 25 * x**2)

def get_chebyshev_nodes(n):
    """Calculates Chebyshev nodes of the 1st kind for degree n."""
    i = np.arange(n + 1)
    return np.cos((2 * i + 1) * np.pi / (2 * n + 2))

def divided_differences(x, y):
    """Calculates the Newton divided differences tableau."""
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x[i] - x[i-j])
    return coef

def newton_interpolation(x_nodes, coef, x_eval):
    """Evaluates the Newton polynomial at x_eval points."""
    n = len(x_nodes) - 1
    p = coef[n]
    for i in range(n - 1, -1, -1):
        p = coef[i] + (x_eval - x_nodes[i]) * p
    return p

degrees = [68, 70]

for n in degrees:
    x_nodes = get_chebyshev_nodes(n)
    y_nodes = runge_function(x_nodes)
    
    coef = divided_differences(x_nodes, y_nodes)
    
    x_range = np.linspace(-1, 1, 1000)
    y_runge = runge_function(x_range)
    y_interp = newton_interpolation(x_nodes, coef, x_range)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_range, y_runge, 'r--', label='Runge Function')
    plt.plot(x_range, y_interp, 'b-', label=f'Newton Interp (n={n})')
    plt.scatter(x_nodes, y_nodes, color='black', s=10, label='Nodes')
    plt.title(f'Numerical Instability of Newton Form (n={n})')
    plt.ylim(-0.2, 1.2)
    plt.legend()
    plt.grid(True)
    plt.show()