import numpy as np
import matplotlib.pyplot as plt


class NewtonOptimization:
    def __init__(self, tol=1e-4, max_iter=100, seed=42):
        self.tol = tol
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

    # =========================
    # Functions
    # =========================
    def EL(self, x1, x2):
        return (x1**2) / 1.5 + (x2**2) / 1.5 + 3 * np.sin((x1 + x2) / np.sqrt(2))**2

    def ER(self, x1, x2):
        return (1 - x1)**2 + 100 * (x2 - x1**2)**2

    # =========================
    # Part a) Gradient + Hessian
    # =========================
    def grad_EL(self, x):
        x1, x2 = x
        u = (x1 + x2) / np.sqrt(2)

        d_common = 3 * np.sin(2 * u) / np.sqrt(2)

        dE_dx1 = (2 / 1.5) * x1 + d_common
        dE_dx2 = (2 / 1.5) * x2 + d_common

        return np.array([dE_dx1, dE_dx2], dtype=float)

    def hess_EL(self, x):
        x1, x2 = x
        u = (x1 + x2) / np.sqrt(2)

        c = 3 * np.cos(2 * u)

        d2_x1x1 = 2 / 1.5 + c
        d2_x2x2 = 2 / 1.5 + c
        d2_x1x2 = c

        return np.array([
            [d2_x1x1, d2_x1x2],
            [d2_x1x2, d2_x2x2]
        ], dtype=float)

    def grad_ER(self, x):
        x1, x2 = x

        dE_dx1 = 2 * (x1 - 1) - 400 * x1 * (x2 - x1**2)
        dE_dx2 = 200 * (x2 - x1**2)

        return np.array([dE_dx1, dE_dx2], dtype=float)

    def hess_ER(self, x):
        x1, x2 = x

        d2_x1x1 = 2 - 400 * x2 + 1200 * x1**2
        d2_x1x2 = -400 * x1
        d2_x2x2 = 200

        return np.array([
            [d2_x1x1, d2_x1x2],
            [d2_x1x2, d2_x2x2]
        ], dtype=float)

    # =========================
    # Helpers
    # =========================
    def _get_components(self, func_name):
        if func_name == "EL":
            return self.EL, self.grad_EL, self.hess_EL, np.array([0.0, 0.0])
        elif func_name == "ER":
            return self.ER, self.grad_ER, self.hess_ER, np.array([1.0, 1.0])
        else:
            raise ValueError("func_name must be 'EL' or 'ER'")

    def _safe_eval_grid(self, func, X, Y):
        Z = np.zeros_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(X[i, j], Y[i, j])
        return Z

    # =========================
    # Part b) Contour plots
    # =========================
    def contour_plot(self, func_name, x1_range, x2_range, levels=30, show=True):
        func, _, _, _ = self._get_components(func_name)

        x1_vals = np.linspace(*x1_range)
        x2_vals = np.linspace(*x2_range)
        X, Y = np.meshgrid(x1_vals, x2_vals)
        Z = self._safe_eval_grid(func, X, Y)

        plt.figure(figsize=(7, 6))
        cp = plt.contour(X, Y, Z, levels=levels)
        plt.clabel(cp, inline=True, fontsize=8)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Contour plot of {func_name}")
        plt.grid(True, alpha=0.3)

        if show:
            plt.show()

    def plot_path_on_contour(self, func_name, path, x1_range, x2_range, levels=30, show=True):
        func, _, _, target = self._get_components(func_name)

        x1_vals = np.linspace(*x1_range)
        x2_vals = np.linspace(*x2_range)
        X, Y = np.meshgrid(x1_vals, x2_vals)
        Z = self._safe_eval_grid(func, X, Y)

        path = np.asarray(path)

        plt.figure(figsize=(7, 6))
        cp = plt.contour(X, Y, Z, levels=levels)
        plt.clabel(cp, inline=True, fontsize=8)

        plt.plot(path[:, 0], path[:, 1], marker="o", linewidth=2, label="Newton path")
        plt.plot(path[0, 0], path[0, 1], "s", markersize=8, label="Start")
        plt.plot(path[-1, 0], path[-1, 1], "*", markersize=12, label="End")
        plt.plot(target[0], target[1], "kx", markersize=10, label="Expected minimum")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Newton path on {func_name} contour plot")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if show:
            plt.show()

    # =========================
    # Part c) General Newton method
    # =========================
    def newton_step(self, grad, hess, x):
        g = grad(x)
        H = hess(x)
        delta = np.linalg.solve(H, g)
        return x - delta

    def newton_method(self, func_name, x0, tol=None, max_iter=None):
        func, grad, hess, _ = self._get_components(func_name)

        if tol is None:
            tol = self.tol
        if max_iter is None:
            max_iter = self.max_iter

        x = np.array(x0, dtype=float)
        path = [x.copy()]

        for k in range(max_iter):
            g = grad(x)

            if np.linalg.norm(g) < tol:
                return {
                    "x_min": x,
                    "path": np.array(path),
                    "iterations": k,
                    "converged": True,
                    "grad_norm": np.linalg.norm(g),
                    "f_value": func(x[0], x[1]),
                }

            H = hess(x)

            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return {
                    "x_min": x,
                    "path": np.array(path),
                    "iterations": k,
                    "converged": False,
                    "grad_norm": np.linalg.norm(g),
                    "f_value": func(x[0], x[1]),
                    "message": "Hessian is singular."
                }

            x_new = x - delta
            path.append(x_new.copy())

            if np.linalg.norm(x_new - x) < tol:
                return {
                    "x_min": x_new,
                    "path": np.array(path),
                    "iterations": k + 1,
                    "converged": True,
                    "grad_norm": np.linalg.norm(grad(x_new)),
                    "f_value": func(x_new[0], x_new[1]),
                }

            x = x_new

        return {
            "x_min": x,
            "path": np.array(path),
            "iterations": max_iter,
            "converged": False,
            "grad_norm": np.linalg.norm(grad(x)),
            "f_value": func(x[0], x[1]),
            "message": "Maximum iterations reached."
        }

    # =========================
    # Part d) Random starts + experiment
    # =========================
    def generate_random_start_points(self, func_name, n=4):
        if func_name == "EL":
            return self.rng.uniform(-3.0, 3.0, size=(n, 2))
        elif func_name == "ER":
            return self.rng.uniform(-2.0, 2.0, size=(n, 2))
        else:
            raise ValueError("func_name must be 'EL' or 'ER'")

    def run_experiment(self, func_name, start_points=None, tol=None, max_iter=None):
        if start_points is None:
            start_points = self.generate_random_start_points(func_name, n=4)

        results = []
        for x0 in start_points:
            result = self.newton_method(func_name, x0, tol=tol, max_iter=max_iter)
            result["start_point"] = np.array(x0, dtype=float)
            results.append(result)

        return results

    def plot_all_paths(self, func_name, results, x1_range, x2_range, levels=30, show=True):
        func, _, _, target = self._get_components(func_name)

        x1_vals = np.linspace(*x1_range)
        x2_vals = np.linspace(*x2_range)
        X, Y = np.meshgrid(x1_vals, x2_vals)
        Z = self._safe_eval_grid(func, X, Y)

        plt.figure(figsize=(8, 7))
        cp = plt.contour(X, Y, Z, levels=levels)
        plt.clabel(cp, inline=True, fontsize=8)

        for i, result in enumerate(results, start=1):
            path = result["path"]
            plt.plot(path[:, 0], path[:, 1], marker="o", linewidth=1.8, label=f"Path {i}")
            plt.plot(path[0, 0], path[0, 1], "s", markersize=7)
            plt.plot(path[-1, 0], path[-1, 1], "*", markersize=11)

        plt.plot(target[0], target[1], "kx", markersize=10, label="Expected minimum")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Newton paths on contour plot of {func_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if show:
            plt.show()

    # =========================
    # Output for part a)
    # =========================
    def print_part_a(self):
        print("Part a) Analytical derivatives\n")

        print("EL(x1, x2) = x1^2/1.5 + x2^2/1.5 + 3*sin^2((x1+x2)/sqrt(2))")
        print("grad EL = [ (2/1.5)x1 + 3*sin(2u)/sqrt(2), (2/1.5)x2 + 3*sin(2u)/sqrt(2) ]")
        print("with u = (x1+x2)/sqrt(2)")
        print("Hess EL = [[ 2/1.5 + 3*cos(2u), 3*cos(2u) ],")
        print("           [ 3*cos(2u),        2/1.5 + 3*cos(2u) ]]\n")

        print("ER(x1, x2) = (1-x1)^2 + 100(x2-x1^2)^2")
        print("grad ER = [ 2(x1-1) - 400x1(x2-x1^2), 200(x2-x1^2) ]")
        print("Hess ER = [[ 2 - 400x2 + 1200x1^2, -400x1 ],")
        print("           [ -400x1,               200     ]]")

    def print_results(self, func_name, results):
        print(f"\nResults for {func_name}:")
        for i, r in enumerate(results, start=1):
            print(f"\nStart point {i}: {r['start_point']}")
            print(f"Approx. minimum: {r['x_min']}")
            print(f"Function value:  {r['f_value']:.8f}")
            print(f"Iterations:      {r['iterations']}")
            print(f"Gradient norm:   {r['grad_norm']:.8e}")
            print(f"Converged:       {r['converged']}")
            if "message" in r:
                print(f"Message:         {r['message']}")


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    solver = NewtonOptimization(tol=1e-4, max_iter=100, seed=7)

    solver.print_part_a()

    solver.contour_plot("EL", x1_range=(-3.0, 3.0, 400), x2_range=(-3.0, 3.0, 400), levels=35)
    solver.contour_plot("ER", x1_range=(-2.0, 2.0, 400), x2_range=(-1.0, 3.0, 400), levels=40)

    start_EL = solver.generate_random_start_points("EL", n=4)
    start_ER = solver.generate_random_start_points("ER", n=4)

    results_EL = solver.run_experiment("EL", start_points=start_EL)
    results_ER = solver.run_experiment("ER", start_points=start_ER)

    solver.print_results("EL", results_EL)
    solver.print_results("ER", results_ER)

    solver.plot_all_paths("EL", results_EL, x1_range=(-3.0, 3.0, 400), x2_range=(-3.0, 3.0, 400), levels=35)
    solver.plot_all_paths("ER", results_ER, x1_range=(-2.0, 2.0, 400), x2_range=(-1.0, 3.0, 400), levels=40)