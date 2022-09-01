
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse.linalg import eigs 
from scipy.sparse import diags, dia_matrix

import open3d as o3d


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



def tetrahedron (radius = 1, translation = 0):

    v1 = [0, 0, 0]
    v2 = [1, 1, 0]
    v3 = [0, 1, 1]
    v4 = [1, 0, 1]

    points = np.array([v1, v2, v3, v4])

    points[:, 0] = radius * points[:,0] + translation[0]
    points[:, 1] = radius * points[:,1] + translation[1]
    points[:, 2] = radius * points[:,2] + translation[2]

    return points

def PointInTetrahedron(arr, p):
    a =   (SameSide(arr[0], arr[1], arr[2], arr[3], p) and
           SameSide(arr[1], arr[2], arr[3], arr[0], p) and
           SameSide(arr[2], arr[3], arr[0], arr[1], p) and
           SameSide(arr[3], arr[0], arr[1], arr[2], p) )
    return a

def SameSide(v1, v2, v3, v4, p):

    normal = np.cross(v2 - v1, v3 - v1)

    dotV4 = np.dot(normal, v4 - v1)
    dotP = np.dot(normal, p - v1)

    s1 = math.copysign(1, dotV4)
    s2 = math.copysign(1, dotP)

    return s1 == s2

def makeTetrahedronWellMatrix (n, inW, outW):
    well_matrix = np.empty((n, n, n))

    tetr = tetrahedron(n - 20, (0, 0, 0))


    #00100
    #01110
    #00100
    
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                if PointInTetrahedron(tetr, (i, j, k)):
                    well_matrix[i][j][k] = inW
                else:
                    well_matrix[i][j][k] = outW

    #well_matrix[0][0][0] = 1

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
    
    print('Hamiltonian done')


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


def toList(arr, n):

    temp = []

    for i in range(0, n):

        for j in range(0, n):

            for k in range(0, n):

                if arr[i][j][k] >= 0.00001:

                    color = 1 - arr[i][j][k]
                    temp.append([i, j, k])

    return np.array(temp)


def run():

    print('start')

    level_to_show = 11
    
    Elevels = 25
    N = 50

    #mesh = makeSphereWellMatrix(N, 1, 0)
    mesh = makeTetrahedronWellMatrix(N, 1, 0)
    
    e_values, e_vec = general_potential_3d(mesh, N, Elevels)

    if 1:
        np.save('data_E_vectors_Tetrahedron' + str(N) +'x'+ str(N) +'x'+ str(N) + 'e' + str(Elevels) , e_vec)

    Elevel = pow(np.absolute( e_vec[:, level_to_show].reshape(N, N, N) ), 2) 


    '''
    for i in range(N):
        ar = mesh[:,:, i]
        #print(ar)
        displayVec(ar)
    '''

    xyz = toList(Elevel, N)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.visualization.draw_geometries([pcd])


    '''
    for i in range(0, N):
        displayVec(Elevel[:,:, i])
    '''

def test():
    N = 70
    k = makeTetrahedronWellMatrix(N, 1, 0)
    xyz = toList(k, N)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    run()
    
