import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

img = o3d.io.read_image('../../images/test_3d/10_1.jpg')
depth = o3d.io.read_image('../../images/test_3d/10_1_depth.png')
rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)
print(rgbd_img)

plt.subplot(1, 2, 1)
plt.title('GrayScale image')
plt.imshow(rgbd_img.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd_img.depth)
plt.show()

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd], height=640)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
pcd = pcd.select_by_index(ind)

# estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()

# surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

# rotate the mesh
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=(0, 0, 0))

# save the mesh
o3d.io.write_triangle_mesh(f'./mesh.obj', mesh)

