import os
import numpy as np
import open3d as o3d
import argparse


def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)


def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = model.get_center()
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model, center, max(max_bound) - min(min_bound)


def voxel_carving(obj_file, camera_path, cubic_size, voxel_resolution, 
                  w=300, h=300, use_depth=True, surface_method='pointcloud', visualization=False):

    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh.compute_vertex_normals()
    camera_sphere = o3d.io.read_triangle_mesh(camera_path)

    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
        color=[1.0, 0.7, 0.0])

    # rescale geometry
    camera_sphere, _, _ = preprocess(camera_sphere)
    mesh, center, scale = preprocess(mesh)
    o3d.io.write_triangle_mesh('mesh.obj', mesh)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True

    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    pcd_agg = o3d.geometry.PointCloud()
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    i = 0
    for cid, xyz in enumerate(camera_sphere.vertices):
        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        centers_pts[cid, :] = c[:3]
        ctr.convert_from_pinhole_camera_parameters(param)

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)
        pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth),
            param.intrinsic,
            param.extrinsic,
            depth_scale=1)

        # depth map carving method
        if use_depth:
            voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
        else:
            voxel_carving.carve_silhouette(o3d.geometry.Image(depth), param)
        print('Carve view %03d/%03d' % (cid + 1, len(camera_sphere.vertices)))

    vis.destroy_window()

    if visualization:
        print('visualize camera center')
        centers = o3d.geometry.PointCloud()
        centers.points = o3d.utility.Vector3dVector(centers_pts)
        o3d.visualization.draw_geometries([centers, mesh, voxel_carving])

    save_voxel_grid(voxel_carving, obj_file, scale, center)


def save_voxel_grid(voxel_grid, obj_file, scale, center):
    save_name = obj_file.split('/')
    save_name[-1] = save_name[-1].replace('.obj', '_voxels.obj')
    save_name = os.path.join(*save_name)
    voxels = voxel_grid.get_voxels()
    voxels = np.asarray([voxels[i].grid_index for i in range(len(voxels))])
    voxels = (voxels - voxels.mean()) / (voxels.max() - voxels.min()) * scale + center
    print('%d voxels' % len(voxels))
    f = open(save_name, 'w')
    for voxel in voxels:
        f.write('v %.3f %.3f %.3f\n' % (voxel[0], voxel[1], voxel[2]))
    f.close()


def surface_voxelization(obj_file):
    voxel_size = 0.03
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    save_voxel_grid(voxel_grid, obj_file)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mesh2voxel')
    parser.add_argument('--path', help='mesh path', required=True, type=str)  # e.g., 'obj/gear.obj'
    args = parser.parse_args()

    obj_file = args.path
    camera_path = 'obj/sphere.ply'
    cubic_size = 2.0
    voxel_resolution = 256.0

    voxel_carving(obj_file, camera_path, cubic_size, voxel_resolution,
                    surface_method='mesh', visualization=False)
