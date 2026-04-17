import numpy as np
import matplotlib
matplotlib.use("Agg")   # for WSL
import matplotlib.pyplot as plt

# RUNGE Funktion
def f(x):
    return 1.0 / (1.0 + 25.0 * x**2)

# Dividierte Differenzen
def divided_differences(x_nodes, y_nodes):
    n = len(x_nodes)
    coeff = np.array(y_nodes, dtype=float)

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coeff[i] = (coeff[i] - coeff[i - 1]) / (x_nodes[i] - x_nodes[i - j])

    return coeff

# Newton-Polynom an einer Stelle x auswerten
def newton_evaluate(x, x_nodes, coeff):
    n = len(coeff)
    p = coeff[n - 1]

    for k in range(n - 2, -1, -1):
        p = coeff[k] + (x - x_nodes[k]) * p

    return p

# Newton-Polynom fur ein Array auswerten
def newton_evaluate_array(x_values, x_nodes, coeff):
    return np.array([newton_evaluate(x, x_nodes, coeff) for x in x_values])

# Plot fur gegebenes n
# n = Grad des Polynoms
# => n+1 Stuetzstellen
def make_plot(n):
    num_nodes = n + 1
    x_nodes = np.linspace(-1.0, 1.0, num_nodes)
    y_nodes = f(x_nodes)

    coeff = divided_differences(x_nodes, y_nodes)

    x_plot = np.linspace(-1.0, 1.0, 1000)
    y_true = f(x_plot)
    y_interp = newton_evaluate_array(x_plot, x_nodes, coeff)

    max_error = np.max(np.abs(y_true - y_interp))

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_true, label=r"$f(x)=\frac{1}{1+25x^2}$")
    plt.plot(x_plot, y_interp, "--", label=fr"Newton-Interpolation, n={n}")
    plt.plot(x_nodes, y_nodes, "o", label="Stuetzstellen")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(fr"Newton-Interpolation der Runge-Funktion auf [-1,1] mit n={n}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # WICHTIG: Y-Achse begrenzen, um Runges Phaenomen gut zu sehen
    plt.ylim(-0.5, 2.0)

    plt.tight_layout()

    filename = f"runge_interpolation_n{n}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Bild gespeichert: {filename}")
    print(f"Maximaler Fehler fuer n={n}: {max_error:.6e}")

    return coeff, max_error

def main():
    for n in [10, 20, 30]:
        print("\n" + "=" * 50)
        print(f"Interpolation fuer n = {n}")
        coeff, err = make_plot(n)

        print("Newton-Koeffizienten (dividierte Differenzen):")
        print(coeff)

if __name__ == "__main__":
    main()