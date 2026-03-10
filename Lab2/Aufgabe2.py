import numpy as np


A = np.array([
    [10, 7, 8, 7],
    [7, 5, 6, 5],
    [8, 6, 10, 9],
    [7, 5, 9, 10]
], dtype=float)

b = np.array([32, 23, 33, 31], dtype=float)


# -----------------------------
# a) prufen, ob a symmetrisch und positiv semi-definit ist
# -----------------------------
def is_symmetric(M, tol=1e-12):
    return np.allclose(M, M.T, atol=tol)

def is_positive_definite(M):
    # A symmetric matrix is positive definite iff all eigenvalues > 0
    eigvals = np.linalg.eigvalsh(M)
    return np.all(eigvals > 0), eigvals


# -----------------------------
# b) cholesky decomposition
#    A = L L^T
# -----------------------------
def cholesky_decomposition(M):
    n = M.shape[0]
    L = np.zeros_like(M, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]

            if i == j:
                value = M[i, i] - s
                if value <= 0:
                    raise ValueError("Matrix is not positive definite.")
                L[i, j] = np.sqrt(value)
            else:
                L[i, j] = (M[i, j] - s) / L[j, j]

    return L


# -----------------------------
# forward substitution:
# solve L y = b
# -----------------------------
def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n, dtype=float)

    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]

    return y


# -----------------------------
# backward substitution:
# solve L^T x = y
# -----------------------------
def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]

    return x


def main():
    print("Matrix A:")
    print(A)
    print("\nVector b:")
    print(b)

    # a) symmetrisch?
    sym = is_symmetric(A)
    print("\na) Ist A symmetrisch?")
    print(sym)

    # a) positive definit?
    pd, eigvals = is_positive_definite(A)
    print("\nEigenwerte von  A:")
    print(eigvals)
    print("\nIst A positiv definit?")
    print(pd)

    # b) cholesky decomposition
    if sym and pd:
        L = cholesky_decomposition(A)
        print("\nb) Cholesky factor L:")
        print(L)

        print("\nCheck: L @ L.T =")
        print(L @ L.T)

        # c) solve Ax = b using Cholesky
        y = forward_substitution(L, b)
        x = backward_substitution(L.T, y)

        print("\nc) Solution of Ax = b:")
        print(x)

        # verification
        print("\nCheck A @ x =")
        print(A @ x)
    else:
        print("\nCholesky Zerlegung ist nicht moglich, da A nicht symmetrisch und positiv definit")


if __name__ == '__main__':
    main()