import numpy as np
import warnings

def swapRows(A, i, j):
    A[[i, j]] = A[[j, i]]

def relError(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.abs(a - b) / np.max(np.abs([a, b]))
        except:
            return 0.0

def rowReduce(A, i, j, pivot):
    factor = A[j][pivot] / A[i][pivot]
    A[j] = A[j] - factor * A[i]

def forwardElimination(B):
    A = B.copy().astype(float)
    m, n = A.shape
    for i in range(m - 1):
        leftmostRow = m
        leftmostCol = n
        for h in range(i, m):
            for k in range(i, n):
                if A[h][k] != 0.0 and k < leftmostCol:
                    leftmostRow = h
                    leftmostCol = k
                    break
        if leftmostRow == m:
            break
        if leftmostRow > i:
            swapRows(A, leftmostRow, i)
        for h in range(i + 1, m):
            rowReduce(A, i, h, leftmostCol)
    return A

def inconsistentSystem(A):
    m, n = A.shape
    for i in range(m):
        nonZeros = np.nonzero(A[i])[0]
        if len(nonZeros) == 0:
            continue
        if nonZeros[0] == n - 1 and len(nonZeros) == 1:
            return True
    return False

def backsubstitution(A):
    A = A.copy().astype(float)
    m, n = A.shape
    for i in range(m - 1, -1, -1):
        nonzero_idx = np.nonzero(A[i])[0]
        if len(nonzero_idx) == 0:
            continue
        pivot = nonzero_idx[0]
        if pivot == n - 1:
            continue
        A[i] = A[i] / A[i, pivot]
        for j in range(i):
            A[j] -= A[j, pivot] * A[i]
    return A

# Masukkan sistem persamaan:
A = np.array([
    [2, -3, 2, 3],
    [1, -1, -2, -1],
    [-1, 2, -3, -4]
])

A_1 = forwardElimination(A)
A_2 = backsubstitution(A_1)

print("Row Echelon Form (A_1):\n", A_1)
print("\nReduced Row Echelon Form (A_2):\n", A_2)
print("\nInconsistent?:", inconsistentSystem(A_2))

solution = A_2[:, -1]
print("\nSolution (x, y, z):", solution)
print("\nPretty Solution:")
print(f"x = {solution[0]:.0f}, y = {solution[1]:.0f}, z = {solution[2]:.0f}")
