import taichi as ti
from mpl_toolkits.mplot3d import Axes3D
import os, time
import argparse
from random import random
import pickle
from math import *
import numpy as np
from tqdm import tqdm
from scene import Scene
from config import *

import torch
from pytorch3d.loss import chamfer_distance
from utils import *

gravity = 0
bound = 2

n_grid = 130
steps = max_steps = 256

E = 1
mu = E
la = E


@ti.layout
def place():
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type, boundary_type, pos_j, tactile_type)
    ti.root.dense(ti.l, max_steps).dense(ti.k, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ijk, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)
    ti.root.dense(ti.ij, (visualize_resolution, visualize_resolution)).place(screen)
    ti.root.lazy_grad()


def zero_vec():
    return [0.0, 0.0, 0.0]


def zero_matrix():
    return [zero_vec(), zero_vec(), zero_vec()]


@ti.kernel
def clear_grid():
    for i, j, k in grid_m_in:
        grid_v_in[i, j, k] = [0, 0, 0]
        grid_m_in[i, j, k] = 0


@ti.func
def vonmises_projection(sigma, p):
    exp_H = ti.Matrix.zero(ti.f32, dim, dim)
    epsilon = ti.Vector.zero(ti.f32, dim)
    for i in ti.static(range(dim)):
        epsilon[i] = ti.log(max(abs(sigma[i, i]), 1e-4))
    trace_epsilon = 0.0
    for i in ti.static(range(dim)):
        trace_epsilon += epsilon[i]
    epsilon_hat = ti.Matrix.zero(ti.f32, dim, dim)
    epsilon_hat_norm = 0.0
    for i in ti.static(range(dim)):
        epsilon_hat[i, i] = epsilon[i] - trace_epsilon / dim
        epsilon_hat_norm += epsilon_hat[i, i] ** 2
    epsilon_hat_norm = ti.sqrt(epsilon_hat_norm)
    delta_gamma = epsilon_hat_norm - yield_stress / (2.0 * mu_0)
    if delta_gamma > 0:
        for i in ti.static(range(dim)):
            new_sig = min(max(sigma[i, i], 1 - 1e-1), 1 + 4.5e-3)
            exp_H[i, i] = new_sig
    return delta_gamma, exp_H


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=dim, val=1) + dt * C[f, p]) @ F[f, p]
        # J = (new_F).determinant()
        U, sig, V = ti.svd(new_F)
        J = 1.0
        for d in ti.static(range(dim)):
            J *= sig[d, d]
        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)
        if particle_type[p] == MATERIAL_WATER:
            sqrtJ = ti.sqrt(J)  # TODO: need pow(x, 1/3)
            new_F = ti.Matrix([[sqrtJ, 0, 0], [0, sqrtJ, 0], [0, 0, 1]])
            F[f + 1, p] = new_F

        act_id = actuator_id[p]
        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) * act
        cauchy = ti.Matrix(zero_matrix())
        mass = 0.0
        ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if particle_type[p] == MATERIAL_WATER:
            mass = 4
            cauchy = ti.Matrix(ident) * (J - 1) * E
            cauchy += new_F @ A @ new_F.transpose()
        elif abs(particle_type[p]) == MATERIAL_ELASTIC:
            mass = 2
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + ti.Matrix.diag(3, la * (J - 1) * J)
        elif particle_type[p] == MATERIAL_VONMISES:
            mass = 10
            delta_gamma, exp_H = vonmises_projection(sig, p)
            if delta_gamma > 0:
                new_F = U @ exp_H @ V.transpose()
                F[f + 1, p] = new_F
            log_sig_sum = 0.0
            center = ti.Matrix(zero_matrix())
            for i in ti.static(range(dim)):
                log_sig_sum += ti.log(sig[i, i])
                center[i, i] = 2.0 * mu_0 * ti.log(sig[i, i]) * (1 / sig[i, i])
            for i in ti.static(range(dim)):
                center[i, i] += lambda_0 * log_sig_sum * (1 / sig[i, i])
            cauchy = U @ center @ V.transpose() @ new_F.transpose()

        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * dx
                    weight = w[i](0) * w[j](1) * w[k](2)
                    grid_v_in[base + offset].atomic_add(weight * (mass * v[f, p] + affine @ dpos))
                    grid_m_in[base + offset].atomic_add(weight * mass)


@ti.kernel
def grid_op():
    for i, j, k in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j, k] + 1e-10)
        v_out = inv_m * grid_v_in[i, j, k]
        v_out[1] -= dt * gravity
        half_grid = n_grid / 2

        if i < (half_grid + bound) and i > (half_grid - bound) and \
           k < (half_grid + bound) and k > (half_grid - bound) and \
           j < (half_grid + bound) and j > (half_grid - bound):
            ratio = max(((i - half_grid) ** 2 + (j - half_grid) ** 2) / bound ** 2, 0.1)
            if i < half_grid + bound and v_out[0] < 0:
                v_out[0], v_out[1], v_out[2] = v_out[0] * ratio, v_out[1] * ratio, v_out[2] * ratio
            if i > half_grid - bound and v_out[0] > 0:
                v_out[0], v_out[1], v_out[2] = v_out[0] * ratio, v_out[1] * ratio, v_out[2] * ratio
            if k < half_grid + bound and v_out[2] < 0:
                v_out[0], v_out[1], v_out[2] = v_out[0] * ratio, v_out[1] * ratio, v_out[2] * ratio
            if k > half_grid - bound and v_out[2] > 0:
                v_out[0], v_out[1], v_out[2] = v_out[0] * ratio, v_out[1] * ratio, v_out[2] * ratio
            if j < half_grid + bound and v_out[1] < 0:
                v_out[0], v_out[1], v_out[2] = v_out[0] * ratio, v_out[1] * ratio, v_out[2] * ratio
            if j > half_grid - bound and v_out[1] > 0:
                v_out[0], v_out[1], v_out[2] = v_out[0] * ratio, v_out[1] * ratio, v_out[2] * ratio

        grid_v_out[i, j, k] = v_out


@ti.kernel
def g2p(v_x: ti.f32, v_y: ti.f32, v_z: ti.f32, f: ti.i32):
    for p in range(n_particles):
        sign = 0
        if particle_type[p] == MATERIAL_ELASTIC:
            sign = 1
        elif particle_type[p] == -MATERIAL_ELASTIC:
            sign = -1
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector(zero_vec())
        new_C = ti.Matrix(zero_matrix())
        if abs(particle_type[p]) == MATERIAL_ELASTIC:
            neg_gravity_v = ti.Vector([0, dt * gravity, 0])
            apply_v = ti.Vector([v_x, v_y, v_z])
            new_v = neg_gravity_v + apply_v * sign

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                    g_v = grid_v_out[base(0) + i, base(1) + j, base(2) + k]
                    weight = w[i](0) * w[j](1) * w[k](2)
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        if particle_type[p] == MATERIAL_VONMISES:
            v[f + 1, p].fill(0)
        else:
            v[f + 1, p] = new_v
        if abs(particle_type[p]) == MATERIAL_ELASTIC:
            tmp = 1 - pos_j[p]
            v[f + 1, p] = tmp * new_v + (1 - tmp) * ti.Vector([v_x, v_y, v_z]) * 100 * sign
            
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


def forward(v, total_steps=steps):
    # simulation
    for s in tqdm(range(total_steps - 1)):
        clear_grid()
        p2g(s)
        grid_op()
        v_x, v_y, v_z = v 
        if get_tactile_value(s) * 1e5 > 18:
            v_x, v_y, v_z = 0, 0, 0
            break
        g2p(v_x, v_y, v_z, s)
    return s


def get_tactile_value(s):
    x_torch = x.to_torch()[s]
    index = (particle_type.to_torch() == MATERIAL_ELASTIC) & (tactile_type.to_torch() == 1)
    x_torch = x_torch[index].detach()
    x_torch = x_torch - x_torch.mean(dim=0)
    distance = chamfer_distance(x_init_torch[None], x_torch[None])[0]
    return distance


def visualize_(s, x_knn_vector_norm=None):
    particles = x.to_numpy()[s][:, :2]
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    idx = 0
    for i in range(n_particles):
        color = 0x111111
        if boundary_type[i] == 1 and x_knn_vector_norm is not None:
            act = x_knn_vector_norm[idx] * 10
            color = ti.rgb_to_hex((0.5 - act * 2, 0.5 + act, 0.5 + act * 5))
            idx += 1
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.023), (0.95, 0.023), radius=3, color=0x0)
    gui.show()


def visualize(s, tactile1, tactile2, folder):
    tactile1 = tactile1 * 0.15 + np.array([0.1, 0.05])
    tactile2 = tactile2 * 0.15 + np.array([0.3, 0.05])
    particles = x.to_numpy()[s][:, :2]
    colors = 0x111111
    gui.circles(pos=tactile1, color=colors, radius=1.5)
    gui.circles(pos=tactile2, color=colors, radius=1.5)
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.show(f'{folder}/{s:04d}.png')


def visualize_tactile(s, tactile1, tactile2, folder):
    tactile1 = tactile1 * 0.1 + np.array([0.1, 0.1])
    tactile2 = tactile2 * 0.1 + np.array([0.25, 0.1])
    particles = x.to_numpy()[s][:, :2]
    colors = 0x111111
    gui.circles(pos=tactile1, color=colors, radius=1.5)
    gui.circles(pos=tactile2, color=colors, radius=1.5)
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.show(f'{folder}/{s:04d}.png')


@ti.kernel
def clear_particle(f: ti.i32, ptype: ti.i32):
    for p in range(n_particles):
        if particle_type[p] == ptype:
            x[f, p].fill(-1)
            v[f, p].fill(0)


@ti.kernel
def snap_particle(stop_step: ti.i32):
    for p in range(n_particles):
        if particle_type[p] == MATERIAL_VONMISES:
            x[0, p] = x[stop_step, p]
            v[0, p].fill(0)# = v[stop_step, p]


@ti.kernel
def splat(t: ti.i32):
    for p in range(n_particles):
        pos = ti.cast(x[t, p] * visualize_resolution, ti.i32)
        screen[pos[0], pos[1]][0] += 0.1


def copy_back_and_clear(img: np.ndarray):
    for i in range(res[0]):
        for j in range(res[1]):
            coord = ((res[1] - 1 - j) * res[0] + i) * 3
            for c in ti.static(range(3)):
                img[coord + c] = screen[i, j][2 - c]
                screen[i, j][2 - c] = 0


def robot(scene, exp_name):
    block_size = 0.1
    global center_von, center_ela
    center_ela = {1: None, -1: None}
    center_von = scene.add_from_file('obj/gear_voxels.obj', exp_name,
                                      0.16, 0, scene.new_actuator(), 2, 
                                      max_particles=100000)
    center_ela[1] = scene.add_rect(0., 0., 0., 0.24, 0.04, 0.24, -1, 1)
    center_ela[-1] = scene.add_rect(0., 0., 0., 0.24, 0.04, 0.24, -1, -1)


def set_xv(x_init, x_new, v_new):
    global x_init_torch
    new_center = {1: [0] * dim, -1: [0] * dim}
    a, b = 0, 0
    for i in range(dim):
        a += v_new[i] * (x_new[i] - center_von[i])
        b += v_new[i] ** 2
    t = -2 * a / b
    x_new = {1: [x_new[i] - center_ela[1][i] for i in range(dim)], 
            -1: [x_new[i] + t * v_new[i] - center_ela[-1][i] for i in range(dim)]}
    for i in range(dim):
        new_center[1][i] = center_ela[1][i] + x_new[1][i]
        new_center[-1][i] = center_ela[-1][i] + x_new[-1][i]
    theta = {1: atan(v_new[1] / (v_new[0] + 1e-6)) + pi / 2, 
            -1: atan(v_new[1] / (v_new[0] + 1e-6)) + pi / 2 + pi}
    for p in range(n_particles):
        if abs(particle_type[p]) != MATERIAL_ELASTIC:
            continue
        if particle_type[p] == MATERIAL_ELASTIC:
            sign = 1
        elif particle_type[p] == -MATERIAL_ELASTIC:
            sign = -1
        pos = [0] * dim
        for i in range(dim):
            x[0, p][i] = x_init[0, p][i] + x_new[sign][i]
            pos[i] = x[0, p][i] - new_center[sign][i]
        x[0, p][0] = pos[0] * cos(theta[sign]) - pos[1] * sin(theta[sign]) + new_center[sign][0]
        x[0, p][1] = pos[1] * cos(theta[sign]) + pos[0] * sin(theta[sign]) + new_center[sign][1]

    x_set_init = x.to_numpy()
    x_init_torch = torch.from_numpy(x_set_init[0])
    index = (particle_type.to_torch() == MATERIAL_ELASTIC) & (tactile_type.to_torch() == 1)
    x_init_torch = x_init_torch[index].detach()
    x_init_torch = x_init_torch - x_init_torch.mean(dim=0)


def get_x_torch(ptype, s=0, get_boundary=True):
    x_torch = x.to_torch()[s]
    particle_type_torch = particle_type.to_torch()
    boundary_type_torch = boundary_type.to_torch()
    index = particle_type_torch == ptype
    if get_boundary:
        index = index & (boundary_type_torch == 1)
    return x_torch[index].detach()


def save_preforward(x_init):
    x_new, v_new = [0.7, 0.23, 0.2], [0, 0, 0]
    set_xv(x_init, x_new, v_new)
    print("Pre-forward", flush=True)
    forward(v_new)
    for s in range(7, steps, 5):
        visualize(s, None)
    snap_particle()
    x_pre_body = get_x_torch(MATERIAL_VONMISES, get_boundary=False)
    x_pre_boundary = get_x_torch(MATERIAL_VONMISES, get_boundary=True)
    save_x(x_pre_body, 'assets/x-pre-body')
    save_x(x_pre_boundary, 'assets/x-pre-boundary')


def get_tactile(s, ptype):
    x_torch = x.to_torch()[s]
    index_1 = (particle_type.to_torch() == ptype) & (tactile_type.to_torch() == 1)
    index_0 = (particle_type.to_torch() == ptype) & (tactile_type.to_torch() == -1)
    sqrt_index = int(sqrt(x_torch[index_0].shape[0]))
    a, b, c = x_torch[index_0][0], x_torch[index_0][sqrt_index - 1], x_torch[index_0][-1]
    ab, bc = b - a, c - b
    x1_torch = x_torch[index_1].detach()
    tactile = []
    for idx, t in enumerate(x1_torch):
        ax, bx = t - a, t - b
        p0 = (ab[0] * ax[0] + ab[1] * ax[1] + ab[2] * ax[2]) / torch.norm(ab)
        p1 = (bc[0] * bx[0] + bc[1] * bx[1] + bc[2] * bx[2]) / torch.norm(bc)
        tactile.append([p0.cpu().numpy(), p1.cpu().numpy()])
    tactile = np.array(tactile)
    tactile = (tactile - tactile.min()) / (tactile.max() - tactile.min())
    return tactile


def main(args):
    folder = os.path.join('outputs', args.name)
    os.makedirs(folder, exist_ok=True)
    ti.set_gdb_trigger()
    scene = Scene()
    robot(scene, args.name)

    global n_particles, n_actuators
    n_particles = scene.n_particles
    n_actuators = scene.n_actuators
    print('n_particles', n_particles)

    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
        boundary_type[i] = scene.boundary_type[i]
        pos_j[i] = scene.pos_j[i]
        tactile_type[i] = scene.tactile_type[i]
    x_init = x.to_numpy()
    
    v_x, v_y, v_z = dt * 2, 0, 0
    x_new, v_new = [0.33, 0.5, 0.5], [v_x, v_y, v_z]
    set_xv(x_init, x_new, v_new)
    print("Forward", flush=True)
    start_step, inter_step = 7, 5
    stop_step = forward(v_new)

    tactiles1, tactiles2 = {}, {}
    print("Save tactile", flush=True)
    for s in tqdm(range(start_step, stop_step, 5)):
        tactile1 = get_tactile(s, MATERIAL_ELASTIC)
        tactile2 = 1 - get_tactile(s, -MATERIAL_ELASTIC)
        if s > start_step:
            tactile1 = (tactile1 - tactiles1[start_step]) * 4 + tactiles1[start_step]
            tactile2 = (tactile2 - tactiles2[start_step]) * 4 + tactiles2[start_step]
        tactiles1[s] = tactile1
        tactiles2[s] = tactile2
    with open(os.path.join(folder, 'tactile1.pkl'), 'wb') as f:
        pickle.dump(tactile1, f)
    with open(os.path.join(folder, 'tactile2.pkl'), 'wb') as f:
        pickle.dump(tactile2, f)

    print("Visualize", flush=True)
    os.makedirs(os.path.join(folder, 'visual'), exist_ok=True)
    for s in range(start_step, stop_step, inter_step):
        # visualize(s, x_knn_vector_norm)
        visualize(s, tactiles1[s], tactiles2[s], os.path.join(folder, 'visual'))
    # snap_particle(stop_step)
    
    print("Save elastic models", flush=True)
    os.makedirs(os.path.join(folder, 'models'), exist_ok=True)
    for s in tqdm(range(start_step, stop_step, 1)):
        x_boundary = get_x_torch(MATERIAL_ELASTIC, s, get_boundary=True) 
        pcd = o3d.geometry.PointCloud()
        points = x_boundary.detach().cpu().numpy()
        for point in points:
            pcd.points.append(point)
        o3d.io.write_point_cloud(os.path.join(folder, 'models', '%d_elastic_a.ply' % s), pcd)
        x_boundary = get_x_torch(-MATERIAL_ELASTIC, s, get_boundary=True) 
        pcd = o3d.geometry.PointCloud()
        points = x_boundary.detach().cpu().numpy()
        for point in points:
            pcd.points.append(point)
        o3d.io.write_point_cloud(os.path.join(folder, 'models', '%d_elastic_b.ply' % s), pcd)

    print("Save voxel particles", flush=True)
    os.makedirs(os.path.join(folder, 'voxels'), exist_ok=True)
    for s in tqdm(range(start_step, stop_step, 1)):
        x_voxel = get_x_torch(MATERIAL_ELASTIC, s, get_boundary=False) 
        pcd = o3d.geometry.PointCloud()
        points = x_voxel.detach().cpu().numpy()
        for point in points:
            pcd.points.append(point)
        o3d.io.write_point_cloud(os.path.join(folder, 'voxels', '%d_elastic_voxel_a.ply' % s), pcd)
        x_voxel = get_x_torch(-MATERIAL_ELASTIC, s, get_boundary=False) 
        pcd = o3d.geometry.PointCloud()
        points = x_voxel.detach().cpu().numpy()
        for point in points:
            pcd.points.append(point)
        o3d.io.write_point_cloud(os.path.join(folder, 'voxels', '%d_elastic_voxel_b.ply' % s), pcd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixel2Mesh Training Entrypoint')
    parser.add_argument('--name', help='experiment name', required=True, type=str)
    args = parser.parse_args()
    main(args)
