import numpy as np
import matplotlib
matplotlib.use("Agg")   # wichtig fuer WSL / ohne GUI
import matplotlib.pyplot as plt

# --------------------------------------------------
# Daten
# --------------------------------------------------
A = np.array([[3.0, 2.0],
              [2.0, 6.0]])

# --------------------------------------------------
# Energie, Gradient, Hesse-Matrix
# --------------------------------------------------
def energy(x):
    return 0.5 * x.T @ A @ x

def grad(x):
    return A @ x

def hessian(x):
    return A


# --------------------------------------------------
# Gradient Descent
# x_{n+1} = x_n - h * grad(x_n)
# --------------------------------------------------
def gradient_descent(x0, h, n_steps):
    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for _ in range(n_steps):
        x = x - h * grad(x)
        path.append(x.copy())

    return np.array(path)


# --------------------------------------------------
# Newton
# x_{n+1} = x_n - H^{-1} grad(x_n)
# --------------------------------------------------
def newton_method(x0, n_steps):
    x = np.array(x0, dtype=float)
    H_inv = np.linalg.inv(A)
    path = [x.copy()]

    for _ in range(n_steps):
        x = x - H_inv @ grad(x)
        path.append(x.copy())

    return np.array(path)


# --------------------------------------------------
# Ausgabe der mathematischen Groessen
# --------------------------------------------------
print("Matrix A:")
print(A)

print("\nGradient:")
print("grad E(x) = A x = [3*x1 + 2*x2, 2*x1 + 6*x2]^T")

print("\nHesse-Matrix:")
print(hessian(np.array([0.0, 0.0])))

eigvals = np.linalg.eigvalsh(A)
lam_min, lam_max = eigvals[0], eigvals[1]

print("\nEigenwerte von A:")
print(eigvals)

print("\nKonvergenzbereich fuer GD:")
print(f"0 < h < 2/lambda_max = 2/{lam_max:.1f} = {2/lam_max:.6f}")

h_opt = 2 / (lam_min + lam_max)
print(f"Optimaler konstanter Schritt: h* = 2/(lambda_min + lambda_max) = {h_opt:.6f}")


# --------------------------------------------------
# Konturdiagramm der Energie
# --------------------------------------------------
x1 = np.linspace(-4, 4, 400)
x2 = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(x1, x2)

Z = 0.5 * (3*X1**2 + 4*X1*X2 + 6*X2**2)

plt.figure(figsize=(8, 6))
levels = np.geomspace(0.01, np.max(Z), 25)
cp = plt.contour(X1, X2, Z, levels=levels)
plt.clabel(cp, inline=True, fontsize=8)
plt.plot(0, 0, "ro", label="Minimum")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Konturdiagramm von E_A(x) = 1/2 x^T A x")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ea_contour.png", dpi=300)
plt.close()


# --------------------------------------------------
# Verschiedene Schrittweiten fuer GD
# --------------------------------------------------
x0 = np.array([3.0, 2.0])
step_sizes = [0.05, 0.15, 2/9, 0.27, 0.30]
n_steps = 20

# Zoomed-in grid around the minimum
x1_zoom = np.linspace(-4, 4, 400)
x2_zoom = np.linspace(-4, 4, 400)
X1z, X2z = np.meshgrid(x1_zoom, x2_zoom)

Zz = 0.5 * (3*X1z**2 + 4*X1z*X2z + 6*X2z**2)

plt.figure(figsize=(14, 10))
levels_zoom = np.geomspace(0.001, np.max(Zz), 25)
cp = plt.contour(X1z, X2z, Zz, levels=levels_zoom)

for h in step_sizes:
    path = gradient_descent(x0, h, n_steps)
    plt.plot(path[:, 0], path[:, 1], marker="o", markersize=3, label=f"h={h:.3f}")

plt.plot(0, 0, "ro", label="Minimum")

plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Gradient Descent (zoomed)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gd_step_sizes_zoomed.png", dpi=300)
plt.close()

for h in step_sizes:
    path = gradient_descent(x0, h, n_steps)
    plt.plot(path[:, 0], path[:, 1], marker="o", markersize=3, label=f"GD, h={h:.3f}")

plt.plot(0, 0, "ro", label="Minimum")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Gradient Descent fuer verschiedene Schrittweiten")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gd_step_sizes.png", dpi=300)
plt.close()


# --------------------------------------------------
# Vergleich GD vs Newton
# --------------------------------------------------
h_compare = 2/9
gd_path = gradient_descent(x0, h_compare, 10)
newton_path = newton_method(x0, 2)

plt.figure(figsize=(8, 6))
cp = plt.contour(X1, X2, Z, levels=levels)

plt.plot(gd_path[:, 0], gd_path[:, 1], marker="o", linewidth=2, label=f"Gradient Descent, h={h_compare:.3f}")
plt.plot(newton_path[:, 0], newton_path[:, 1], marker="s", linewidth=2, label="Newton")
plt.plot(0, 0, "ro", label="Minimum")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Vergleich: Gradient Descent vs Newton")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gd_vs_newton.png", dpi=300)
plt.close()


# --------------------------------------------------
# Numerische Ausgabe einiger Bahnen
# --------------------------------------------------
print("\nStartpunkt:")
print(x0)

for h in step_sizes:
    path = gradient_descent(x0, h, 8)
    print(f"\nGD mit h = {h:.6f}")
    for k, x in enumerate(path[:5]):
        print(f"  Iteration {k}: x = {x}, E(x) = {energy(x):.6f}")

newton_path = newton_method(x0, 2)
print("\nNewton:")
for k, x in enumerate(newton_path):
    print(f"  Iteration {k}: x = {x}, E(x) = {energy(x):.6f}")

print("\nPlots gespeichert als:")
print("  - ea_contour.png")
print("  - gd_step_sizes.png")
print("  - gd_vs_newton.png")