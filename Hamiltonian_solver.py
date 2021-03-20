


#import sys
import numpy as np
import matplotlib.pyplot as plt
import math
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from numpy import array, empty
import numpy.version

from scipy.sparse.linalg import eigs
from scipy import sparse


from scipy.sparse import diags
from scipy.sparse import dia_matrix

Elevels = 500

images = [0, 5]

N = 100



def general_potential():
    
    ################################################################################
     
    

    
    binary_well_matrix = empty((N, N), None)    
    
    
    for i in range(0, N):
        for j in range(0, N):
            if math.dist([i, j], [(N-1)/2, (N-1)/2]) <= (N-1)/2:
                binary_well_matrix[i][j] = 1
            else:
                binary_well_matrix[i][j] = 0
    
               
    #global position_mesh
    position_mesh = binary_well_matrix
    #print(position_mesh)
    position_mesh = np.matrix.flatten(position_mesh)

    #for debag
    #position_mesh = [1 for i in position_mesh ]
    #
    
    ################################################################################

    No_points = N*N
    x_intervals = np.linspace(0, 1, N)
    increment = np.absolute(x_intervals[0] - x_intervals[1])
    print (increment)
    
    ################################################################################
    
    
    
    
    increment = pow(increment, 2)
    incrementValue = -1/increment
    zeroV = 4 / increment
    
    diagmN =  [incrementValue * position_mesh[i] for i in range(0, No_points - N )]
    diagm1 =  [incrementValue * position_mesh[i] for i in range(0, No_points )]
    diag0  =  [zeroV for i in range(0, No_points+1)]
    diagp1 =  [incrementValue * position_mesh[i] for i in range(0, No_points)]
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



 
e_values, e_vec = general_potential()   


idx = e_values.argsort()[::-1]   
e_values = e_values[idx]
e_vec = e_vec[:,idx]

print('vectors done')

if 1:
    np.save('data_E_vectors_circle_100x100', e_vec)


print('save e_vec done')


def displayAndSaveIm ():
    for i in range(int(images[0]), int(images[1])):
        figi, axi = plt.subplots(1, 1)
           
        plot = plt.imshow( pow( np.absolute( e_vec[:,i].reshape(N,N) ) ,2), cmap='nipy_spectral', interpolation='gaussian') 
        
        
        plt.setp(axi, xticks=[], yticks=[])
        
        plt.savefig( str(i) + 'c100.png', bbox_inches = 'tight') 
           
        plt.show()

displayAndSaveIm()

       
print('****************************** all done *******************************')



    



  

