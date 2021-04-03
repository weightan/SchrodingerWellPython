
import numpy as np
import math

from scipy.sparse.linalg import eigs 
from scipy.sparse import diags, dia_matrix



def makeSphereWellMatrix (n, inW, outW):
    well_matrix = np.empty((n, n, n))
    
    #00100
    #01110
    #00100
    
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                if math.dist([i, j, k], [(n-1)/2, (n-1)/2, (n-1)/2]) <= (n-1)/2:
                    well_matrix[i][j] = inW
                else:
                    well_matrix[i][j] = outW

    return well_matrix


if __name__ == '__main__':
    N = 10
    print (makeSphereWellMatrix(N, 1, 0)) 