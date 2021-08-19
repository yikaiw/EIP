import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
# create unit mesh from point cloud, for rendering


path_name = 'can_voxels.ply'
for path_name in tqdm(os.listdir('.')):
    if '.ply' not in path_name:
        continue
    pcd = o3d.io.read_point_cloud(path_name)
    # pcd = o3d.io.read_triangle_mesh('voxels_cylinder_vertical_gear.obj')
    point_num = len(np.array(pcd.points))
    max_num = 70000

    mesh = o3d.geometry.TriangleMesh()
    for pos in np.array(pcd.points):
        if point_num > max_num and np.random.rand() > max_num / point_num:
            continue
        cube = o3d.geometry.TriangleMesh.create_box()
        cube.scale(0.001, center=cube.get_center())
        cube.translate((pos[0], pos[1], pos[2]), relative=False)
        mesh += cube

    o3d.io.write_triangle_mesh('%s_voxel.obj' % path_name.replace('.ply', ''), mesh)
