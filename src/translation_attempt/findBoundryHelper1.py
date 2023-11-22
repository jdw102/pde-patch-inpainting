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
    path = np.zeros(error.shape)
    cost = np.zeros(error.shape)
    cost[x - 1, :] = error[x - 1, :]
    for i in range(x - 2, -1, -1):
        a = cost[i+1, 0:y-1]
        a = np.insert(a, 0, np.inf)
        b = cost[i+1,:]
        c = cost[i+1, 1:y]
        if len(c) == 0:
            c = np.insert(c, 0, np.inf)
        else:
            c = np.insert(c, -1, np.inf)
        mintree = np.array([a,
                            b,
                            c]) + error[i, :]
        cost[i, :], path[i, :] = np.min(mintree, axis=0), np.argmin(mintree) + 1
    path[path == 2] = 0
    path[path == 1] = -1
    path[path == 3] = 1
    return cost, path

