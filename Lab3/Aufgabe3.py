import numpy as np
import matplotlib
matplotlib.use("Agg")   # fuer WSL / ohne GUI
import matplotlib.pyplot as plt


# =========================================================
# 1) Energien, Gradienten, Hesse-Matrizen
# =========================================================

def E_L(x):
    x1, x2 = x
    u = (x1 + x2) / np.sqrt(2.0)
    return (x1**2) / 1.5 + (x2**2) / 1.5 + 3.0 * np.sin(u)**2


def grad_E_L(x):
    x1, x2 = x
    u = (x1 + x2) / np.sqrt(2.0)
    s = (3.0 / np.sqrt(2.0)) * np.sin(2.0 * u)
    return np.array([
        (4.0 / 3.0) * x1 + s,
        (4.0 / 3.0) * x2 + s
    ])


def hess_E_L(x):
    x1, x2 = x
    u = (x1 + x2) / np.sqrt(2.0)
    c = 3.0 * np.cos(2.0 * u)
    return np.array([
        [4.0 / 3.0 + c, c],
        [c, 4.0 / 3.0 + c]
    ])


def E_R(x):
    x1, x2 = x
    return (1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2


def grad_E_R(x):
    x1, x2 = x
    return np.array([
        2.0 * (x1 - 1.0) - 400.0 * x1 * (x2 - x1**2),
        200.0 * (x2 - x1**2)
    ])


def hess_E_R(x):
    x1, x2 = x
    return np.array([
        [2.0 - 400.0 * x2 + 1200.0 * x1**2, -400.0 * x1],
        [-400.0 * x1, 200.0]
    ])


# =========================================================
# 2) Allgemeines Gradient Descent
# =========================================================

def gradient_descent(grad, energy, x0, h, tol=1e-4, max_iter=5000):
    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break

        x = x - h * g
        path.append(x.copy())

        if np.linalg.norm(x) > 1e8 or not np.isfinite(energy(x)):
            break

    return np.array(path), x, k + 1


# =========================================================
# 3) Allgemeines Newton-Verfahren
# =========================================================

def newton_method(grad, hess, x0, tol=1e-4, max_iter=100):
    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break

        H = hess(x)

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print("Newton abgebrochen: Hesse-Matrix singulaer.")
            break

        x = x - step
        path.append(x.copy())

        if np.linalg.norm(x) > 1e8 or not np.all(np.isfinite(x)):
            break

    return np.array(path), x, k + 1


# =========================================================
# 4) Konturplot + Bahnen
# =========================================================

def plot_contours_and_paths(energy, paths, labels, filename, xlim, ylim, title):
    x1 = np.linspace(xlim[0], xlim[1], 500)
    x2 = np.linspace(ylim[0], ylim[1], 500)
    X1, X2 = np.meshgrid(x1, x2)

    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = energy(np.array([X1[i, j], X2[i, j]]))

    plt.figure(figsize=(8, 6))

    positive = Z[Z > 0]
    if len(positive) > 0:
        levels = np.geomspace(max(1e-6, np.min(positive)), np.max(Z), 30)
    else:
        levels = 20

    plt.contour(X1, X2, Z, levels=levels)

    for path, label in zip(paths, labels):
        plt.plot(path[:, 0], path[:, 1], marker="o", markersize=3, linewidth=1.5, label=label)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().set_aspect("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# =========================================================
# 5) Schrittweiten-Test
# =========================================================

def test_step_sizes(name, energy, grad, x0, step_sizes, tol=1e-4):
    print(f"\n===== Schrittweiten-Test fuer {name} =====")
    print(f"Startpunkt: {x0}")

    for h in step_sizes:
        path, x_end, iters = gradient_descent(grad, energy, x0, h, tol=tol, max_iter=5000)
        gnorm = np.linalg.norm(grad(x_end))
        val = energy(x_end)
        converged = gnorm < tol and np.all(np.isfinite(x_end))

        print(f"h = {h:8.5f} | konvergiert = {converged!s:5} | Iterationen = {iters:4d} | "
              f"x_end = {x_end} | E = {val:.8f} | ||grad|| = {gnorm:.3e}")


def main():
    x0_L = np.array([2.0, -1.5])
    x0_R = np.array([-1.2, 1.0])

    # Schrittweiten untersuchen
    step_sizes_L = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    step_sizes_R = [0.0005, 0.001, 0.002, 0.005, 0.01]

    test_step_sizes("E_L", E_L, grad_E_L, x0_L, step_sizes_L)
    test_step_sizes("E_R", E_R, grad_E_R, x0_R, step_sizes_R)

    # Sinnvolle Vergleichswerte
    h_L = 0.10
    h_R = 0.001

    # Vergleich fuer E_L
    gd_path_L, gd_end_L, gd_iter_L = gradient_descent(grad_E_L, E_L, x0_L, h_L)
    newton_path_L, newton_end_L, newton_iter_L = newton_method(grad_E_L, hess_E_L, x0_L)

    print("\n===== Vergleich fuer E_L =====")
    print(f"GD:     Iterationen = {gd_iter_L}, Endpunkt = {gd_end_L}, E = {E_L(gd_end_L):.8f}")
    print(f"Newton: Iterationen = {newton_iter_L}, Endpunkt = {newton_end_L}, E = {E_L(newton_end_L):.8f}")

    plot_contours_and_paths(
        E_L,
        [gd_path_L, newton_path_L],
        [f"GD, h={h_L}", "Newton"],
        "vergleich_EL.png",
        xlim=(-3, 3),
        ylim=(-3, 3),
        title="Vergleich GD vs Newton fuer E_L"
    )

    # Vergleich fuer E_R
    gd_path_R, gd_end_R, gd_iter_R = gradient_descent(grad_E_R, E_R, x0_R, h_R, max_iter=20000)
    newton_path_R, newton_end_R, newton_iter_R = newton_method(grad_E_R, hess_E_R, x0_R)

    print("\n===== Vergleich fuer E_R =====")
    print(f"GD:     Iterationen = {gd_iter_R}, Endpunkt = {gd_end_R}, E = {E_R(gd_end_R):.8f}")
    print(f"Newton: Iterationen = {newton_iter_R}, Endpunkt = {newton_end_R}, E = {E_R(newton_end_R):.8f}")

    plot_contours_and_paths(
        E_R,
        [gd_path_R, newton_path_R],
        [f"GD, h={h_R}", "Newton"],
        "vergleich_ER.png",
        xlim=(-2, 2),
        ylim=(-1, 3),
        title="Vergleich GD vs Newton fuer E_R"
    )

    print("\nPlots gespeichert als:")
    print("  - vergleich_EL.png")
    print("  - vergleich_ER.png")


if __name__ == "__main__":
    main()