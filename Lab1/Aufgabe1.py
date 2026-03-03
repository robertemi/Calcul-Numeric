import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# A) f(x) >= 0 fur alle x in R
#  fur x -> inf => f(x) -> 1
# also ist 0<= f(x) < 1 fur alle x in R
def f(x):
    return x**2 / (x**2 + 1)

def plot_func(x_min, x_max, y_min, y_max, num_points=2000):
    x = np.linspace(x_min, x_max, num_points)
    y = f(x)
    plt.plot(x, y, label='f(x) = x^2 / (x^2 + 1)')
    plt.title('Plot of the function f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

# # x ∈ [-1,1], y ∈ [-1,1]
# plot_func(-1, 1, -1, 1)

# # x ∈ [-100,100], y ∈ [-1,1]
# plot_func(-100, 100, -1, 1)

# # x ∈ [-50,50], y ∈ [-1,1]
# plot_func(-50, 50, -1, 1)

# # x ∈ [-25,25], y ∈ [-1,1]
# plot_func(-25, 25, -1, 1)

# Abschluss: Die Funktion f(x) ist immer zwischen 0 und 1,
# und nähert sich 1 an, wenn x gegen unendlich geht.
# Also sind werte kleiner als 0 fur y nicht moglich, helfen aber um die Bild besser zu sehen.
# Fur x Werte größer als 25 nähert sich die Funktion schon sehr nah an 1 an,
# daher ist es nicht mehr nötig,
# größere x Werte zu plotten, um das Verhalten der Funktion zu verstehen.
# Hier ist unser Vorschlag fur die finale Intervale: x ∈ [-10,10], y ∈ [0,1]
plot_func(-10, 10, 0, 1)


# B) limes x->inf f(x) = 1 ==> f(x) hat eine horizontale Asymptote bei y=1

def plot_func_with_asymptote(x_min, x_max, y_min, y_max, num_points=2000):
    x = np.linspace(x_min, x_max, num_points)
    y = f(x)
    plt.plot(x, y, label='f(x) = x^2 / (x^2 + 1)')
    plt.axhline(y=1, color='r', linestyle='--', label='Asymptote y=1')
    plt.title('Plot of the function f(x) with Asymptote')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

# Den Interval fur x soll breiter sein um die Asymptote besser zu sehen.
plot_func_with_asymptote(-25, 25, 0, 2)

# C)  Taylor Polinom 2. Ordnung um x=0 ist f(0) + f'(0)*x + (f''(0)/2)*x^2
# f(0) = 0, f'(0) = 0, f''(0) = 2
# Also ist das Taylor-Polynom 2. Ordnung um x=0: 0 + 0*x + (2/2)*x^2 = x^2

def taylor_poly_2nd_order(x):
    return x**2

def plot_func_and_taylor(x_min, x_max, y_min, y_max, num_points=2000):
    x = np.linspace(x_min, x_max, num_points)
    y_original = f(x)
    y_taylor = taylor_poly_2nd_order(x)
    
    plt.plot(x, y_original, label='f(x) = x^2 / (x^2 + 1)')
    plt.plot(x, y_taylor, label='Taylor-Polynom 2. Ordnung')
    plt.axhline(y=1, color='r', linestyle='--', label='Asymptote y=1')
    plt.title('Original Function and its Taylor-Polynom of Order 2')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

plot_func_and_taylor(-25, 25, 0, 2)
# Und noch ein angeren Interval
plot_func_and_taylor(-10, 10, 0, 2)
    