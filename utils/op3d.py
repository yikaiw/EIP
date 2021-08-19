import open3d as o3d
import torch
from pytorch3d.io import load_obj
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures import Meshes
import numpy as np


def point2mesh(x, filename, save_file_only=False):
    pcd = o3d.geometry.PointCloud()
    points = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    for point in points:
        pcd.points.append(point)
    # pcd = o3d.io.read_point_cloud('x_target_boundary_2.ply')
    # pcd = o3d.io.read_triangle_mesh('x_target_boundary_2.obj')
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.asarray(pcd.points).mean(0))
    normals = np.asarray(pcd.normals)
    while len(np.asarray(pcd.normals)) > 0:
        pcd.normals.pop()
    pcd.normals.extend(-normals)
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, 
        scale=1.1, linear_fit=False)[0]
    o3d.io.write_triangle_mesh(filename, poisson_mesh, write_vertex_colors=False)
    if save_file_only:
        return
    # o3d.visualization.draw_geometries([pcd, poisson_mesh], point_show_normal=True)
    verts, faces, aux = load_obj(filename)
    faces_idx = faces.verts_idx
    mesh = Meshes(verts=[verts], faces=[faces_idx])
    return mesh, torch.Tensor(normals)

    # for filename in pcds:
    #     pcd = o3d.io.read_point_cloud(filename)
    #     pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    #     pcd.orient_normals_towards_camera_location(np.asarray(pcd.points).mean(0))
    #     normals = np.asarray(pcd.normals)
    #     while len(np.asarray(pcd.normals)) > 0:
    #         pcd.normals.pop()
    #     pcd.normals.extend(-normals)
    #     poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, 
    #         scale=1.1, linear_fit=False)[0]
    #     o3d.io.write_triangle_mesh(filename.replace('ply', 'obj'), poisson_mesh, write_vertex_colors=True)


def pcd2ply(filename, save_file=True):
    pcd = o3d.io.read_point_cloud(filename)
    ply = o3d.geometry.PointCloud(pcd)
    if save_file:
        o3d.io.write_point_cloud(filename.replace('pcd', 'ply'), ply)

    # pcds = os.listdir('.')
    # for pcd in pcds:
    #     if '.pcd' in pcd:
    #         pcd2ply(pcd)


def load_obj_file(filename):
    verts, faces, aux = load_obj(filename)
    faces_idx = faces.verts_idx
    mesh = Meshes(verts=[verts], faces=[faces_idx])
    return mesh


def get_mesh_chamfer_distance(src_mesh, trg_mesh):
    scale = 100
    src_sample = sample_points_from_meshes(src_mesh, 5000).detach() * scale  # torch.Size([1, 5000, 3])
    trg_sample = sample_points_from_meshes(trg_mesh, 5000).detach() * scale
    loss_chamfer, _ = chamfer_distance(src_sample, trg_sample)
    return loss_chamfer


def get_knn_vector(x_boundary, trg_mesh):
    trg_sample = sample_points_from_meshes(trg_mesh, x_boundary.shape[0]).detach()
    lengths = torch.tensor([x_boundary.shape[0]])
    x_nn = knn_points(x_boundary[None], trg_sample, lengths1=lengths, lengths2=lengths, K=1)
    x_knn_vector = (x_boundary[None] - trg_sample[:, x_nn.idx.squeeze()]).squeeze()
    x_knn_vector_norm = x_knn_vector.norm(dim=1)
    return x_knn_vector, x_knn_vector_norm  # torch.Size([n, 3]), torch.Size([n])


def check_intersection(x, v, x_region, z_region, y_plane=0):
    t = (y_plane - x[1]) / (v[1] + 1e-6)
    if t < 0:
        return False
    inter_x = x[0] + t * v[0]
    inter_z = x[2] + t * v[2]
    check_x = inter_x * 1.05 >= x_region[0] and inter_x * 0.95 <= x_region[1]
    check_z = inter_z * 1.05 >= z_region[0] and inter_z * 0.95 <= z_region[1]
    return check_x and check_z
