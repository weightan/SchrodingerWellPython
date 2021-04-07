import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os



def toList1(arr, n):

    temp = []
    cmap = []
    for i in range(0, n):

        for j in range(0, n):

            for k in range(0, n):

                if arr[i][j][k] >= 0.00000001 and arr[i][j][k] < 0.000005:

                    color = 0.95

                    temp.append([i - 100, j, k])
                    cmap.append([color, color, color])

                if arr[i][j][k] >= 0.000005 and arr[i][j][k] < 0.00001:

                    color = 0.7

                    temp.append([i, j, k])
                    cmap.append([color, color, color])

                if arr[i][j][k] >= 0.00001 and arr[i][j][k] < 0.00005:

                    color = 0.4

                    temp.append([i + 100, j, k])
                    cmap.append([color, color, color])

                if arr[i][j][k] >=  0.00005 and arr[i][j][k] < 0.0001:

                    color = 0.3

                    temp.append([i + 200, j, k])
                    cmap.append([color, color, color])

                if arr[i][j][k] >=  0.0001 and arr[i][j][k] < 0.001:

                    color = 0.2

                    temp.append([i + 300, j, k])
                    cmap.append([color, color, color])

                if arr[i][j][k] >=  0.001 :

                    color = 0.1

                    temp.append([i + 400, j, k])
                    cmap.append([color, color, color])

    return [np.array(temp), np.array(cmap)]



def toList2(arr, n):

    temp = []
    cmap = []
    for i in range(0, n):

        for j in range(0, n):

            for k in range(0, n):

                if arr[i][j][k] >= 0.000005 and arr[i][j][k] < 0.00005:

                    color = 0.5

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


def custom_draw_geometry_with_rotation1(pcd):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([230/255, 225/255, 227/255])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../test_data/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    

    def rotate_view(vis):
        ctr = vis.get_view_control()
        for i in range(180):
            ctr.rotate(10.0, 0.0)
            image = vis.capture_screen_float_buffer(True)
            plt.imsave("{:05d}.png".format(i), np.asarray(image), dpi = 1)
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = rotate_view
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image

    o3d.visualization.draw_geometries_with_key_callbacks([pcd], 
                                                        key_to_callback,
                                                        window_name=str(i),
                                                        width=1080,
                                                        height=1080)

def custom_draw_geometry_with_custom_fov(pcd):

    for i in range(90):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        vis.run()
        image = vis.capture_screen_float_buffer(True)
        plt.imsave("{:05d}.png".format(i), np.asarray(image), dpi = 1)
        vis.destroy_window()

def custom_draw_geometry_with_rotation2(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        image = vis.capture_screen_float_buffer(True)
        plt.imsave("{:05d}.png".format(i), np.asarray(image), dpi = 1)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)

def custom_draw_geometry_with_camera_trajectory(pcd):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
            o3d.io.read_pinhole_camera_trajectory(
                    "../../test_data/camera_trajectory.json")
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    if not os.path.exists("../../test_data/image/"):
        os.makedirs("../../test_data/image/")
    if not os.path.exists("../../test_data/depth/"):
        os.makedirs("../../test_data/depth/")

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave("../../test_data/depth/{:05d}.png".format(glb.index),\
                    np.asarray(depth), dpi = 1)
            plt.imsave("../../test_data/image/{:05d}.png".format(glb.index),\
                    np.asarray(image), dpi = 1)
            #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                    register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("../../test_data/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':


    e_vec = np.load('data_E_vectors_Tetrahedron70x70x70e50.npy')

    N = 70

    print('done load')

    for i in range(35, 40):
        Elevel = pow(np.absolute( e_vec[:, i].reshape(N, N, N) ), 2) 
        xyz, cmap= toList1(Elevel, N)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(cmap)

        custom_draw_geometry_with_rotation1(pcd)