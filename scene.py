import os
import taichi as ti
from math import *
import random
import numpy as np
from config import *
import open3d as o3d
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.io import load_obj, save_obj


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.boundary_type = []
        self.pos_j = []
        self.tactile_type = []
        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0
        self.n_actuators = 0

    def new_actuator(self):
        self.n_actuators += 1
        return self.n_actuators - 1

    def offset(self, ptype, w, i, real_dx):
        if ptype == MATERIAL_WATER:
            return random.random() * w
        elif abs(ptype) == MATERIAL_ELASTIC or ptype == 2:
            return (i + 0.5) * real_dx

    def add_rect(self, x, y, z, w, h, d, actuation, ptype=MATERIAL_ELASTIC):
        global n_particles
        w_count = int(w / dx * density)
        h_count = int(h / dx * density)
        d_count = int(d / dx * density)
        real_dx = w / w_count
        real_dy = h / h_count
        real_dz = d / d_count
        center = [x + w / 2 + self.offset_x, y + h / 2 + self.offset_y, z + d / 2 + self.offset_z]
        theta = pi / 4
        
        for i in range(w_count):
            for j in range(h_count):
                for k in range(d_count):
                    pos = [x + self.offset(ptype, w, i, real_dx) + self.offset_x,
                           y + self.offset(ptype, h, j, real_dy) + self.offset_y,
                           z + self.offset(ptype, d, k, real_dz) + self.offset_z]
                    self.x.append(pos)
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.pos_j.append(j / h_count)
                    if abs(ptype) == MATERIAL_ELASTIC:
                        if j == 0:
                            self.tactile_type.append(1)
                        elif j == h_count - 1:
                            self.tactile_type.append(-1)
                        else:
                            self.tactile_type.append(0)
                    else:
                        self.tactile_type.append(0)
                    self.n_particles += 1
                    self.n_solid_particles += int(abs(ptype) == MATERIAL_ELASTIC)
                    if i == 0 or i == w_count - 1 or j == 0 or j == h_count - 1 or k == 0 or k == d_count - 1:
                        self.boundary_type.append(1)
                    else:
                        self.boundary_type.append(0)
        return center

    def add_from_file(self, filename, exp_name, scale, pos=0, actuation=-1, 
                      ptype=MATERIAL_ELASTIC, rotate=None, max_particles=None):
        verts, faces, aux = load_obj(filename)
        verts = (verts - verts.mean(dim=0)) / (verts.max() - verts.min())
        if rotate is not None:
            for (angle, axis) in rotate:
                verts = RotateAxisAngle(angle, axis=axis).transform_points(verts)
        verts = verts * scale + np.array(pos) + np.array([0.5, 0.5, 0.5]) + \
            np.array([self.offset_x, self.offset_y, self.offset_z])

        if ptype == MATERIAL_VONMISES and False:
            new_mesh_name = filename.split('/')[-1].replace('_voxels', '')
            new_mesh_name = os.path.join('outputs', exp_name, new_mesh_name)
            mesh = o3d.io.read_triangle_mesh(filename.replace('_voxels', ''))
            mesh_verts = np.array(mesh.vertices)
            mesh_verts = (mesh_verts - mesh_verts.mean(axis=0)) / (mesh_verts.max() - mesh_verts.min())
            mesh_verts = mesh_verts * scale + np.array(pos) + np.array([0.5, 0.5, 0.5]) + \
                np.array([self.offset_x, self.offset_y, self.offset_z])
            mesh.vertices.clear()
            mesh.vertices.extend(mesh_verts)
            o3d.io.write_triangle_mesh(new_mesh_name, mesh)

            new_points_name = filename.split('/')[-1].replace('obj', 'ply')
            new_points_name = os.path.join('outputs', exp_name, new_points_name)
            points = o3d.geometry.PointCloud()
            points.points.extend(verts)
            o3d.io.write_point_cloud(new_points_name, points)
            print('New mesh and points saved')

        if not max_particles:
            max_particles = 10000
        p = max_particles / len(verts) if len(verts) > max_particles else 1
        
        for i in range(0, len(verts), int(1 / p)):
            self.x.append(verts[i])
            self.actuator_id.append(actuation)
            self.particle_type.append(ptype)
            self.tactile_type.append(0)
            self.pos_j.append(0)
            self.boundary_type.append(0)
            self.n_particles += 1
        return verts.mean(axis=0)

    def set_offset(self, x, y, z):
        self.offset_x = x
        self.offset_y = y
        self.offset_z = z
