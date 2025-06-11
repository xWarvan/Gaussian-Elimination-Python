import numpy as np
import warnings

def swapRows(A, i, j):
    tmp = A[i].copy()
    A[i] = A[j]
    A[j] = tmp

def relError(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.abs(a-b)/np.max(np.abs(np.array([a, b])))
        except:
            return 0.0

def rowReduce(A, i, j, pivot):
    factor = A[j][pivot] / A[i][pivot]
    for k in range(len(A[j])):
        if np.isclose(A[j][k], factor * A[i][k]):
            A[j][k] = 0.0
        else:
            A[j][k] = A[j][k] - factor * A[i][k]

def forwardElimination(B):
    A = B.copy().astype(float)
    m, n = np.shape(A)
    for i in range(m-1):
        leftmostNonZeroRow = m
        leftmostNonZeroCol = n
        for h in range(i, m):
            for k in range(i, n):
                if (A[h][k] != 0.0) and (k < leftmostNonZeroCol):
                    leftmostNonZeroRow = h
                    leftmostNonZeroCol = k
                    break
        if leftmostNonZeroRow == m:
            break
        if leftmostNonZeroRow > i:
            swapRows(A, leftmostNonZeroRow, i)
        for h in range(i+1, m):
            rowReduce(A, i, h, leftmostNonZeroCol)
    return A

def inconsistentSystem(A):
    m, n = np.shape(A)
    for i in range(m):
        nonZeros = np.nonzero(A[i])[0]
        if len(nonZeros) == 0:
            continue
        if nonZeros[0] == n - 1 and len(nonZeros) == 1:
            return True
    return False

def backsubstitution(A):
    m, n = np.shape(A)
    A = A.copy().astype(float)
    for i in range(m-1, -1, -1):
        nonzeroindex = np.nonzero(A[i])[0]
        if len(nonzeroindex) == 0:
            continue
        pivot = nonzeroindex[0]
        if pivot == n - 1:
            continue
        A[i] = A[i] / A[i, pivot]
        for j in range(i):
            A[j] -= A[j, pivot] * A[i]
    return A

# Matriks augmented untuk sistem:
# 2x – 3y + 2z = 3
# x  - y  -2z = -1
# -x + 2y -3z = -4

A = np.array([
    [ 2, -3,  2,  3],
    [ 1, -1, -2, -1],
    [-1,  2, -3, -4]
])

A_1 = forwardElimination(A)
A_2 = backsubstitution(A_1)

print("Row Echelon Form (A_1):\n", A_1)
print("\nReduced Row Echelon Form (A_2):\n", A_2)
print("\nInconsistent?:", inconsistentSystem(A_2))

solution = A_2[:, -1]
print("\nSolution (x, y, z):", solution)
