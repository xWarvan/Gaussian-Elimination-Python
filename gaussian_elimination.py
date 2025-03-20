import numpy as np
import warnings

def swapRows(A, i, j):
    """
    interchange two rows of A
    operates on A in place
    """
    tmp = A[i].copy()
    A[i] = A[j]
    A[j] = tmp

def relError(a, b):
    """
    compute the relative error of a and b
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.abs(a-b)/np.max(np.abs(np.array([a, b])))
        except:
            return 0.0

def rowReduce(A, i, j, pivot):
    """
    reduce row j using row i with pivot pivot, in matrix A
    operates on A in place
    """
    factor = A[j][pivot] / A[i][pivot]
    for k in range(len(A[j])):
        if np.isclose(A[j][k], factor * A[i][k]):
            A[j][k] = 0.0
        else:
            A[j][k] = A[j][k] - factor * A[i][k]


# stage 1 (forward elimination)
def forwardElimination(B):
    """
    Return the row echelon form of B
    """
    A = B.copy().astype(float)
    m, n = np.shape(A)
    for i in range(m-1):
        # Let lefmostNonZeroCol be the position of the leftmost nonzero value 
        # in row i or any row below it 
        leftmostNonZeroRow = m
        leftmostNonZeroCol = n
        ## for each row below row i (including row i)
        for h in range(i,m):
            ## search, starting from the left, for the first nonzero
            for k in range(i,n):
                if (A[h][k] != 0.0) and (k < leftmostNonZeroCol):
                    leftmostNonZeroRow = h
                    leftmostNonZeroCol = k
                    break
        # if there is no such position, stop
        if leftmostNonZeroRow == m:
            break
        # If the leftmostNonZeroCol in row i is zero, swap this row 
        # with a row below it
        # to make that position nonzero. This creates a pivot in that position.
        if (leftmostNonZeroRow > i):
            swapRows(A, leftmostNonZeroRow, i)
        # Use row reduction operations to create zeros in all positions 
        # below the pivot.
        for h in range(i+1,m):
            rowReduce(A, i, h, leftmostNonZeroCol)
    return A

#################### 

# If any operation creates a row that is all zeros except the last element,
# the system is inconsistent; stop.
def inconsistentSystem(A):
    """
    B is assumed to be in echelon form; return True if it represents
    an inconsistent system, and False otherwise
    """
    # remove the next line
    #pass
    m, n = np.shape(A)
    for i in range(m):
        nonZeros = np.nonzero(A[i])[0] #tracks indices of nonzero elements in array
        if len(nonZeros) == 0:
            continue  #returns row if it is all zeroes

        if nonZeros[0]== n- 1 and len(nonZeros)==1:
            return True #returns true if system is inconsistent
    
    return False #else returns false




def backsubstitution(A):
    """
    return the reduced row echelon form matrix of B
    """
    # remove the next line
    #pass
    m, n = np.shape(A)
    A = A.copy().astype(float)  #copies orginal matrix float
    
    for i in range(m-1, -1, -1): #starts loop from bottom row
        nonzeroindex = np.nonzero(A[i])[0]
        if len(nonzeroindex) == 0:
            continue  #continues row if all elements are zeroes
        pivot = nonzeroindex[0]
                #pivot rows
        if pivot == n - 1:
            continue #continues if pivot is in last col
        A[i] = A[i] / A[i, pivot] 
        # makes sure each element above pivot is zero
        for j in range(i):
            A[j] -= A[j, pivot] * A[i]
    
    return A

# A = np.array([[1,-3,2,-3],
#               [5,4,6,41],
#               [10,-9,8,-6]])

# A = np.loadtxt("h2m6.txt")
              
# A_1 = forwardElimination(A)
# A_2 = backsubstitution(A_1)

# print(A_2)
# print("inconsistent?:",inconsistentSystem(A_2))



# print("Solution:", A_2[:, -1])

#####################

