import numpy as np

class ErrorAnaliysis():
    def __init__(self, function, matrix, vektor):
        self.function = function
        self.matrix = matrix
        self.vektor = vektor

    def det_matrix(self):
        return np.linalg.det(self.matrix)
    
    def inv_matrix(self):
        return np.linalg.inv(self.matrix)
    
    
    def swap_rows(self, row1, row2):
        # =============================================================================
        #     A is a NumPy array.  RowSwap will return duplicate array with rows
        #     1 and 2 swapped.
        # =============================================================================
        m = self.matrix.shape[0] # number of rows
        n = self.matrix.shape[1] # number of columns

        C = np.copy(self.matrix).astype(float) # copy of A
        for j in range(n):
            temp = C[row1, j]
            C[row1, j] = C[row2, j]
            C[row2, j] = temp
        return C
    
    def scale_row(self,row, scalar):
        # =============================================================================
        #     A is a NumPy array.  RowScale will return duplicate array with row
        #     multiplied by scalar.
        # =============================================================================
        m = self.matrix.shape[0] # number of rows
        n = self.matrix.shape[1] # number of columns

        C = np.copy(self.matrix).astype(float) # copy of A
        for j in range(n):
            C[row, j] *= scalar
        return C
    
    def add_multiple_of_row_to_row(self, row_to_be_added_to, row_to_be_added, scalar):
        # =============================================================================
        #     A is a NumPy array.  RowAdd will return duplicate array with row
        #     multiplied by scalar and added to row.
        # =============================================================================
        m = self.matrix.shape[0] # number of rows
        n = self.matrix.shape[1] # number of columns

        C = np.copy(self.matrix).astype(float) # copy of A
        for j in range(n):
            C[row_to_be_added_to, j] += scalar * C[row_to_be_added, j]
        return C



    def gaus_elimination(self):
        """
        Performs Gaussian elimination with partial pivoting to transform
        self.matrix into an upper triangular form.
        It uses the class's helper methods for row operations on the matrix
        and manually applies the same transformations to self.vektor.
        This method modifies the instance attributes in-place.
        """
        m, n = self.matrix.shape
        if m != n:
            raise ValueError("Matrix must be square for this implementation.")

        # Ensure we are working with floating point numbers for division.
        self.matrix = self.matrix.astype(float)
        self.vektor = self.vektor.astype(float)

        # Forward elimination loop
        for k in range(n - 1):
            # --- Partial Pivoting ---
            # Find the row with the largest pivot element in the current column k.
            # We search from the current row 'k' downwards.
            pivot_row_index = k + np.argmax(np.abs(self.matrix[k:, k]))

            # If the pivot row is not the current row, swap them.
            if pivot_row_index != k:
                # Use the helper method to swap rows in the matrix.
                self.matrix = self.swap_rows(k, pivot_row_index)
                
                # Manually swap corresponding elements in the vector.
                self.vektor[k], self.vektor[pivot_row_index] = self.vektor[pivot_row_index], self.vektor[k]

            # --- Elimination ---
            # For all rows below the pivot row.
            for i in range(k + 1, n):
                if self.matrix[k, k] == 0:
                    continue # Matrix is singular, or column is already zeroed.
                
                factor = self.matrix[i, k] / self.matrix[k, k]
                self.matrix = self.add_multiple_of_row_to_row(i, k, -factor)
                self.vektor[i] -= factor * self.vektor[k]
                
        return self.matrix, self.vektor
    
    def solve_system(self):
        """
        Solves the linear system Ax = b using Gaussian elimination followed by back substitution.
        This method modifies the instance attributes in-place and returns the solution vector.
        """
        # First, perform Gaussian elimination to get an upper triangular matrix.
        U, c = self.gaus_elimination()

        n = U.shape[0]
        x = np.zeros(n)

        # Back substitution to solve Ux = c
        for i in range(n - 1, -1, -1):
            if U[i, i] == 0:
                raise ValueError("Matrix is singular or nearly singular.")
            x[i] = (c[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

        return x

if __name__ == '__main__':

    A = np.array([
        [10., 7., 8., 7.],
        [7., 5., 6., 5.],
        [8., 6., 10., 9.],
        [7., 5., 9., 10.],
    ])
    b = np.array([32., 23., 33., 31.])

    print("Original Matrix (A):")
    print(A)
    print("\nOriginal Vector (b):")
    print(b)

    # Create an instance of the class
    analyzer = ErrorAnaliysis(function=None, matrix=A, vektor=b)

    # Run the Gaussian elimination
    U, c = analyzer.gaus_elimination()

    print("\n--- Running gaus_elimination ---")
    print("\nResulting Upper Triangular Matrix (U):")
    print(np.round(U, 4))
    print("\nResulting Vector (c):")
    print(np.round(c, 4))

    # Solve the system    solution = analyzer.solve_system()
    print("\n--- Solving the system ---")
    solution = analyzer.solve_system()
    print("\nSolution:")
    print(solution)

    A_prim = np.array([
        [10., 7., 8.1, 7.2],
        [7.08, 5.04, 6., 5.],
        [8., 6., 9.98, 9.],
        [6.99, 4.99, 9., 9.98],
    ])
    b_prim = np.array([32.1, 22.9, 33.1, 30.9])

    print("\n--- Testing with a modified matrix ---")
    print("Modified Matrix (A'):")
    print(A_prim)
    print("\nModified Vector (b'):")
    print(b_prim)

    # Create an instance of the class with the modified matrix
    analyzer_prim = ErrorAnaliysis(function=None, matrix=A_prim, vektor=b_prim)

    # Run the Gaussian elimination
    U_prim, c_prim = analyzer_prim.gaus_elimination()

    print("\nResulting Upper Triangular Matrix (U'):")
    print(np.round(U_prim, 4))
    print("\nResulting Vector (c'):")
    print(np.round(c_prim, 4))

    # Solve the system
    solution_prim = analyzer_prim.solve_system()
    print("\nSolution (x'):")
    print(solution_prim)