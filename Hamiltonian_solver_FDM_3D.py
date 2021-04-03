
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

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
                    well_matrix[i][j][k] = inW
                else:
                    well_matrix[i][j][k] = outW

    return well_matrix



def general_potential_3d(matrixWell3D, N, Elevels):

    position_mesh = np.matrix.flatten( matrixWell3D )

    No_points = N**3

    x_intervals = np.linspace(0, 1, N)

    increment = pow(x_intervals[1], 2)

    incrementValue = -1/increment

    zeroV = 6 / increment

    diagmNN = [incrementValue * position_mesh[i] for i in range(0, No_points - N**2 )]
    diagmN =  [incrementValue * position_mesh[i] for i in range(0, No_points - N )]
    diagm1 =  [incrementValue * position_mesh[i] for i in range(0, No_points - 1 )]
    diag0  =  [              zeroV               for i in range(0, No_points     )]
    diagp1 =  [incrementValue * position_mesh[i] for i in range(0, No_points - 1 )]
    diagpN =  [incrementValue * position_mesh[i] for i in range(0, No_points - N )]
    diagpNN = [incrementValue * position_mesh[i] for i in range(0, No_points - N**2 )]

    diagsK = [-N*N, -N, -1, 0, 1, N, N*N]
    diagsV = [diagmNN, diagmN, diagm1, diag0, diagp1, diagpN, diagpNN]

    Hamiltonian = diags(diagsV, diagsK, format = 'dia')
    
    print('Hamiltonian values done')


    ################################################################################
    
    #Hamiltonian.tocsr()
    e_values, e_vec = eigs(Hamiltonian, k = Elevels )

    print('All Hamiltonian done')

    ################################################################################

    return [e_values, e_vec]



def displayVec (vectorToImage):
    plot = plt.imshow( vectorToImage, cmap='nipy_spectral') 
    #plot = plt.imshow( vectorToImage, cmap='nipy_spectral', interpolation='gaussian') 
    plt.show()
    plt.close()



if __name__ == '__main__':

    N = 40

    mesh = makeSphereWellMatrix(N, 1, 0)
    e_values, e_vec = general_potential_3d(mesh, N, 16)

    Elevel = pow(np.absolute( e_vec[:, 4].reshape(N, N, N) ), 2) 

    '''
    for i in range(N):
        ar = mesh[:,:, i]
        #print(ar)
        displayVec(ar)
    '''

    for i in range(0, N):
        displayVec(Elevel[:,:, i])
