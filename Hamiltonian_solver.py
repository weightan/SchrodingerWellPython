


import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.sparse.linalg import eigs 
from scipy.sparse import diags
from scipy.sparse import dia_matrix

Elevels = 100

images = [20, 30]

N = 100


inWell = 1
outWell = 0




def makeCircleWellMatrix (n, inW, outW):
    well_matrix = np.empty((n, n))
    
    #00100
    #01110
    #00100
    
    for i in range(0, n):
        for j in range(0, n):
            if math.dist([i, j], [(n-1)/2, (n-1)/2]) <= (n-1)/2 :
                well_matrix[i][j] = inW
            else:
                well_matrix[i][j] = outW
                
    return well_matrix





def general_potential(matrixWell2D):
    
    ################################################################################
     

    position_mesh = np.matrix.flatten( matrixWell2D )



    No_points = N*N
    x_intervals = np.linspace(0, 1, N)
    increment = np.absolute(x_intervals[0] - x_intervals[1])
    print ('increment ' + str(increment))
    
    ################################################################################
    
    
    
    
    increment = pow(increment, 2)
    incrementValue = -1/increment
    zeroV = 4 / increment
    
    diagmN =  [incrementValue * position_mesh[i] for i in range(0, No_points - N )]
    diagm1 =  [incrementValue * position_mesh[i] for i in range(0, No_points     )]
    diag0  =  [              zeroV               for i in range(0, No_points+1   )]
    diagp1 =  [incrementValue * position_mesh[i] for i in range(0, No_points     )]
    diagpN =  [incrementValue * position_mesh[i] for i in range(0, No_points - N )]

    diagK = [-N, -1, 0, 1, N]

    Hamiltonian = diags([diagmN, diagm1, diag0, diagp1, diagpN],  diagK, format = 'dia')
    
    print('Hamiltonian values done')


    ################################################################################
    
    #Hamiltonian.tocsr()
    e_values, e_vec = eigs(Hamiltonian, k = Elevels )


    print('All Hamiltonian done')
    ################################################################################
    
    return [e_values, e_vec]




def displayAndSaveIm (vectors):
    for i in range(int(images[0]), int(images[1])):
        figi, axi = plt.subplots(1, 1)
           
        plot = plt.imshow( pow( np.absolute( vectors[:,i].reshape(N,N) ) ,2), cmap='nipy_spectral', interpolation='gaussian') 
        
        plt.setp(axi, xticks=[], yticks=[])
        
        plt.savefig( str(i) + 'c100.png', bbox_inches = 'tight') 
           
        plt.show()




if __name__ == '__main__':


    mesh = makeCircleWellMatrix(N, inWell, outWell)
    
    e_values, e_vec = general_potential(mesh)   


    idx = e_values.argsort()[::-1]   
    e_values = e_values[idx]
    e_vec = e_vec[:,idx]

    print('vectors done')

    if 1:
        np.save('data_E_vectors_circle' + str(N) +'x'+ str(N) + 'e' + str(Elevels) , e_vec)
        print('save e_vec done')

    displayAndSaveIm(e_vec)
    
    print('****************************** all done *******************************')



    



  

