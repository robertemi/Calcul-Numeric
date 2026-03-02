import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



# Rosenbrock funktion 
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2


# Gradient und Hesse Matrix (fur die Kovexitat)
def grad_rosenbrock(x, y):
    dfdx = 2*(x - 1) - 400*x*(y - x**2)
    dfdy = 200*(y - x**2)
    return np.array([dfdx, dfdy])


def hessian_rosenbrock(x, y):
    f_xx = 2 - 400*y + 1200*x**2
    f_xy = -400*x
    f_yy = 200
    return np.array([[f_xx, f_xy],
                     [f_xy, f_yy]])


# plots: 3D plot und contour plot
# Domane fur den plot 
x_min, x_max = -2.0, 2.0
y_min, y_max = -1.0, 3.0

N = 400
x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# 3D surface plot
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
ax1.set_title("Rosenbrock funktion (3D surface)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x,y)")
fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1)

# Contour plot
ax2 = fig.add_subplot(1, 2, 2)
# Use log-spaced-ish levels to see the valley better (avoid 0 in log)
levels = np.geomspace(1e-3, Z.max(), 30)
cs = ax2.contour(X, Y, Z, levels=levels)
ax2.clabel(cs, inline=True, fontsize=8)
ax2.set_title("Rosenbrock function (contour plot)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# markieren des bekannten globalen Minimum
ax2.plot(1, 1, 'r*', markersize=12, label="global minimum (1,1)")
ax2.legend()

plt.tight_layout()
plt.savefig('rosenbrock_plots.png', dpi=300)
plt.close()


# Minimum der Funktion
print("Analytical global minimum: f(1,1) =", rosenbrock(1, 1), "at (x,y)=(1,1)")



# 5) Uberprufen der Konvexitat unter Benutzung der Hesse Matrix
# Eine 2-Mal ableitbare Funktion ist konvex auf einem Intervall wenn und nur wenn ihre Hesse Matrix positiv semidefinit ist auf dem gegebenen Intervall

grid_n = 120
xs = np.linspace(x_min, x_max, grid_n)
ys = np.linspace(y_min, y_max, grid_n)

min_eig = np.empty((grid_n, grid_n))
for i, xi in enumerate(xs):
    for j, yj in enumerate(ys):
        H = hessian_rosenbrock(xi, yj)
        # symmetric -> use eigvalsh
        eigs = np.linalg.eigvalsh(H)
        min_eig[j, i] = eigs.min()

print("\nHessian eigenvalue sampling on the plotting domain:")
print("Minimum sampled smallest-eigenvalue =", min_eig.min())
print("If this value is negative -> Hessian not PSD everywhere -> not convex globally.")

# Visualisierung der positiv semidefinierten Hesse Matrix
plt.figure(figsize=(7, 5))
plt.contourf(xs, ys, min_eig, levels=40)
plt.colorbar(label="smallest eigenvalue of Hessian")
plt.contour(xs, ys, min_eig, levels=[0], colors='red', linewidths=2)
plt.plot(1, 1, 'w*', markersize=12)
plt.title("Convexity indicator: smallest Hessian eigenvalue\n(red curve = boundary where min eigenvalue = 0)")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('rosenbrock_convexity.png', dpi=300)
plt.close()

print("\nConclusion about convexity:")
if min_eig.min() < 0:
    print("- NOT convex on the whole domain (and in fact not globally convex).")
else:
    print("- Appears convex on the sampled domain (but this does NOT prove global convexity).")