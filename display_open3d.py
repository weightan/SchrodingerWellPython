import numpy as np
import open3d as o3d



def toList(arr, n):

    temp = []

    for i in range(0, n):

        for j in range(0, n):

            for k in range(0, n):

                if arr[i][j][k] >= 0.00001:
                    color = 1 - arr[i][j][k]
                    temp.append([i, j, k, color, color, color])

    return np.array(temp)


if __name__ == '__main__':

    e_vec = np.load('data_E_vectors_sphere100x100e16.npy')

    N = 100

    print('done load')

    Elevel = pow(np.absolute( e_vec[:, 2].reshape(N, N, N) ), 2) 
    xyz = toList(Elevel, N)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.visualization.draw_geometries([pcd])