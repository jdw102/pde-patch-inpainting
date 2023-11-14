import numpy as np

'''
The findBoundaryHelper1.m MATLAB function appears to be part of the texture transfer algorithm, specifically used for finding a boundary in the context of image processing. It takes an error matrix as input and computes the cost and path matrices, which are likely used to determine the optimal boundary in the texture synthesis.

Here's a summary of the key operations in this function:

Matrix Initialization: Initializes path and cost matrices of the same size as the input error matrix.

Looping Over Matrix Rows: Iterates over the rows of the error matrix, performing calculations to update the cost and path matrices.

Calculating Minimum Tree: Involves creating a 'mintree' matrix and calculating minimum values to update the cost and path.

Adjusting Path Values: Modifies the path matrix based on certain conditions.
'''


# Translating MATLAB's findBoundaryHelper1 function to Python
def find_boundary_helper1(error):
    x, y = error.shape
    path = np.zeros_like(error)
    cost = np.zeros_like(error)
    cost[-1, :] = error[-1, :]

    for i in range(x - 2, -1, -1):
        mintree = np.vstack([
            np.pad(cost[i + 1, :-1], (1, 0), 'constant', constant_values=np.inf),
            cost[i + 1, :],
            np.pad(cost[i + 1, 1:], (0, 1), 'constant', constant_values=np.inf)
        ]) + error[i, :]
        cost[i, :], path[i, :] = np.min(mintree, axis=0), np.argmin(mintree, axis=0)

    path[path == 2] = 0
    path[path == 1] = -1
    path[path == 3] = 1
    return cost, path

# Displaying the Python function for verification
find_boundary_helper1
