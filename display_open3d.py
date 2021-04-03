import numpy as np
import open3d as o3d



def toList(arr, n):

    temp = []
    cmap = []
    for i in range(0, n):

        for j in range(0, n):

            for k in range(0, n):

                if arr[i][j][k] >= 0.000001:

                    color = 1 - 20000*arr[i][j][k]

                    temp.append([i, j, k])
                    cmap.append([color, color, color])

    return [np.array(temp), np.array(cmap)]

'''
def toTxt(arr):
    f = open("data_d.txt", "w")
    for i in range(len(arr)):
        f.write()
    f.close()
'''





if __name__ == '__main__':

    e_vec = np.load('data_E_vectors_sphere70x70x70e50.npy')

    N = 70

    print('done load')

    for i in range(5, 20):
        Elevel = pow(np.absolute( e_vec[:, i].reshape(N, N, N) ), 2) 
        xyz, cmap= toList(Elevel, N)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        #pcd.colors = o3d.utility.Vector3dVector(cmap)

        o3d.visualization.draw_geometries([pcd], window_name=str(i), width=1080, height=1080)