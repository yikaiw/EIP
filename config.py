import taichi as ti

real = ti.f32
gui = ti.GUI("Differentiable MPM", (1024, 1024), background_color=0xFFFFFF)
# ti.init(default_fp=real, arch=ti.cuda, flatten_if=True)

MATERIAL_WATER = 0
MATERIAL_ELASTIC = 1  # and -1
MATERIAL_VONMISES = 2

dim = 3
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
density = 2
dt = 2e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
steps = max_steps = 512
# steps = max_steps = 96
gravity = 5
target = [0.8, 0.2, 0.2]
use_apic = False
visualize_resolution = 512
res = [visualize_resolution, visualize_resolution]
bound = 3
coeff = 1.5

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

actuator_id = ti.var(ti.i32)
particle_type = ti.var(ti.i32)
boundary_type = ti.var(ti.i32)
pos_j = ti.var(ti.f32)
tactile_type = ti.var(ti.f32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

screen = ti.Vector(3, dt=real)

loss = scalar()
n_sin_waves = 4
weights = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 40
act_strength = 5

youngs_modulus = 10
poisson_ratio = 0.1
lambda_0 = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
mu_0 = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
yield_stress = 1.0
