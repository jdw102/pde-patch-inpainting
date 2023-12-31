'''
The findBoundaryHelper2.m MATLAB function is designed to compute a boundary given a path matrix and an index (ind). The function creates a matrix named boundary and updates it based on the provided path and index. This operation is typically used in image processing algorithms to delineate boundaries based on computed paths.

Key operations in this function include:

Matrix Initialization: Initializes the boundary matrix with zeros, having the same size as the path matrix.

Boundary Calculation: Iterates through the rows of the path matrix, updating the boundary matrix based on the path values and the initial index.

Updating Boundary Matrix: The boundary matrix is updated in a way that seems to trace a path or boundary within the matrix.
'''
import numpy as np
# Translating MATLAB's findBoundaryHelper2 function to Python
def find_boundary_helper2(path, ind):
    m, _ = path.shape
    boundary = np.zeros_like(path)
    boundary[0, :ind + 1] = 1
    prev = ind
    for i in range(1, m):
        prev += int(path[i - 1, prev])
        boundary[i, 0:prev+1] = 1
    return boundary

# Displaying the Python function for verification
