import numpy as np
import cmath

def input_matrix(file):
    matrix = []
    row = list(map(float, file.readline().split()))
    dim = len(row)
    
    matrix.append(row)
    for i in range(1, dim):
        row = list(map(float, file.readline().split()))
        matrix.append(row)
    
    return matrix

def QRDecomposition(matrix):
    dim = len(matrix)
    R = np.array(matrix)
    Q = np.identity(dim)
    
    for i in range(dim - 1):
        v = [R[j][i] if j > i else 0 for j in range(dim)]
        v[i] = R[i][i] + np.sign(R[i][i]) * np.linalg.norm(R[i:, i])
        
        H = np.identity(dim) - 2 * np.dot(np.transpose([v]), [v]) / np.linalg.norm(v) ** 2
        
        Q = np.dot(Q, H)
        R = np.dot(H, R)
    
    return Q, R

def lowerNorm(A):
    return max([np.linalg.norm(A[i + 2:,i]) for i in range(len(A))])

def getEigenvalues(matrix, precision):
    dim = len(matrix)
    A = np.array(matrix)
    eigenValues = []
    
    while lowerNorm(A) >= precision:
        Q, R = QRDecomposition(A)
        A = np.dot(R, Q)
    
    for i in range(5):
        Q, R = QRDecomposition(A)
        A = np.dot(R, Q)

    skip = False
    for i in range(dim):
        if skip:
            skip = False
            continue
        
        if np.linalg.norm(A[i + 1:, i]) < precision:
            eigenValues.append(A[i][i])
        else:
            b = -(A[i][i] + A[i + 1][i + 1])
            c = A[i][i] * A[i + 1][i + 1] - A[i + 1][i] * A[i][i + 1]
            D = b ** 2 - 4 * c
            eigenValues.append((-b + cmath.sqrt(D)) / 2)
            eigenValues.append((-b - cmath.sqrt(D)) / 2)
            skip = True
    
    return eigenValues

if __name__ == '__main__':
    file = open('D:/myprog/numeric_methods/lab1/5/test.txt', 'r')
    
    matrix = input_matrix(file)
    precision = float(file.readline())
    
    print('My Eigenvalues:')
    print(getEigenvalues(matrix, precision))
    print()
    print('NumPy\'s Eigenvalues:')
    print(np.linalg.eig(matrix)[0])