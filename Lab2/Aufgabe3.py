import numpy as np
import time

# Matrix A and Vector b from Task 1
A = np.array([[10, 7, 8, 7],
              [7, 5, 6, 5],
              [8, 6, 10, 9],
              [7, 5, 9, 10]], dtype=float)

b = np.array([32, 23, 33, 31], dtype=float)
epsilon = 0.0001

def jacobi(A, b, eps, max_iterations=10000):
    n = len(b)
    x = np.zeros_like(b)
    D = np.diag(A)
    R = A - np.diagflat(D)
    
    for i in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, i + 1
        x = x_new
    return x, max_iterations

def gauss_seidel(A, b, eps, max_iterations=10000):
    n = len(b)
    x = np.zeros_like(b)
    
    for k in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            sum_j = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sum_j) / A[i, i]
            
        if np.linalg.norm(x - x_old, ord=np.inf) < eps:
            return x, k + 1
    return x, max_iterations

def cholesky_solve(A, b):
    # Task 2 Implementation for comparison
    L = np.linalg.cholesky(A)
    # Forward substitution Ly = b
    y = np.linalg.solve(L, b)
    # Backward substitution L^T x = y
    x = np.linalg.solve(L.T, y)
    return x

# Execution and Comparison (Task 3c)
# Jacobi
start = time.perf_counter()
sol_j, iter_j = jacobi(A, b, epsilon)
time_j = time.perf_counter() - start

# Gauss-Seidel
start = time.perf_counter()
sol_gs, iter_gs = gauss_seidel(A, b, epsilon)
time_gs = time.perf_counter() - start

# Cholesky
start = time.perf_counter()
sol_cho = cholesky_solve(A, b)
time_cho = time.perf_counter() - start

print(f"Jacobi: {iter_j} iterations, Time: {time_j:.6f}s")
print(f"Gauss-Seidel: {iter_gs} iterations, Time: {time_gs:.6f}s")
print(f"Cholesky Time: {time_cho:.6f}s")