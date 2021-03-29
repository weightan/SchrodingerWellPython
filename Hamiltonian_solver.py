

import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.sparse.linalg import eigs 
from scipy.sparse import diags, dia_matrix

#calculate this number of eigenvectors = number of energy levels
Elevels = 120

#image this eigenvectors
images = [80, 100]

# size of matrix N x N for well
# set N = 30 to reduce the calculation time. 
# You shouldn't set more than 500 otherwise the computation time will increase to hours
N = 200 

inWell = 1

#set 0.1 or so to reduce the calculation time. 
outWell = 0


#for polygon well matrix
def polygon(sides, radius = 1, rotation = 0, translation = None):
    one_segment = math.pi * 2 / sides

    points = [
        (math.sin(one_segment * i + rotation) * radius,
         math.cos(one_segment * i + rotation) * radius)
        for i in range(sides)]

    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]

    return points


#for polygon well matrix
def inPolygon(x, y, xp, yp):
    
    c = 0
    
    for i in range(len(xp)):
        if (((yp[i]<=y and y<yp[i-1]) or (yp[i-1]<=y and y<yp[i])) and 
            (x > (xp[i-1] - xp[i]) * (y - yp[i]) / (yp[i-1] - yp[i]) + xp[i])): c = 1 - c    
    return c

#for polygon well matrix
def makePolygonWellMatrix (n, inW, outW):
    well_matrix = np.empty((n, n))

    sides = 6
    
    pol = polygon(sides, 100, 0, (100, 100))
    
    xp = tuple([ pol[i][0] for i in range(0, sides) ])
    yp = tuple([ pol[i][1] for i in range(0, sides) ])
    
    print(xp)
    print(yp)
    
    for i in range(0, n):
        for j in range(0, n):
            
            if inPolygon(i, j, xp, yp):
                
                well_matrix[i][j] = inW
                
            else:
                
                well_matrix[i][j] = outW
                
    return well_matrix


#to make the matrix in the form of a circle
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

    position_mesh = np.matrix.flatten( matrixWell2D )

    No_points = N*N

    x_intervals = np.linspace(0, 1, N)

    increment = pow(x_intervals[1], 2)

    incrementValue = -1/increment

    zeroV = 4 / increment
    
    diagmN =  [incrementValue * position_mesh[i] for i in range(0, No_points - N )]
    diagm1 =  [incrementValue * position_mesh[i] for i in range(0, No_points - 1 )]
    diag0  =  [              zeroV               for i in range(0, No_points     )]
    diagp1 =  [incrementValue * position_mesh[i] for i in range(0, No_points - 1 )]
    diagpN =  [incrementValue * position_mesh[i] for i in range(0, No_points - N )]

    diagsK = [-N, -1, 0, 1, N]
    diagsV = [diagmN, diagm1, diag0, diagp1, diagpN]

    Hamiltonian = diags(diagsV, diagsK, format = 'dia')
    
    print('Hamiltonian values done')


    ################################################################################
    
    #Hamiltonian.tocsr()
    e_values, e_vec = eigs(Hamiltonian, k = Elevels )

    print('All Hamiltonian done')

    ################################################################################

    return [e_values, e_vec]




def displayAndSaveIm (vectorsToImage):
    for i in range(int(images[0]), int(images[1])):
        figi, axi = plt.subplots(1, 1)
           
        plot = plt.imshow( pow( np.absolute( vectorsToImage[:,i].reshape(N,N) ) ,2), cmap='nipy_spectral', interpolation='gaussian') 
        
        plt.setp(axi, xticks=[], yticks=[])
        
        plt.savefig( str(i) + 'c.png', bbox_inches = 'tight') 
           
        plt.show()




if __name__ == '__main__':


    mesh = makeCircleWellMatrix (N, inWell, outWell)
    
    #mesh = makePolygonWellMatrix (N, inWell, outWell)
    
    e_values, e_vec = general_potential(mesh)   


    idx = e_values.argsort()[::-1]   
    e_values = e_values[idx]
    e_vec = e_vec[:,idx]

    #array for picture of energy level i:  Elevel = pow(np.absolute( e_vec[:,i].reshape(N,N) ), 2)
    #pictures are e_vec, not e_values

    print('vectors done')

    if 1:
        np.save('data_E_vectors_circle' + str(N) +'x'+ str(N) + 'e' + str(Elevels) , e_vec)
        print ('save e_vec done')

    displayAndSaveIm(e_vec)
    
    print('****************************** all done *******************************')
