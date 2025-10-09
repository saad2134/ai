import numpy as np

def find_eigen(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

A = np.array([[4, -2], [1, 1]])
eigenvals, eigenvecs = find_eigen(A)
print("Eigenvalues:", eigenvals)
print("Eigenvectors:\n", eigenvecs)