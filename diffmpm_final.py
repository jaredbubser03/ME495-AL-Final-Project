import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import json

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
# x_avg = vec()
x_leftmost = vec()

actuation = scalar()
actuation_omega = 20
act_strength = 4

def allocate_fields():
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_leftmost)

    ti.root.lazy_grad()


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                          2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)


# @ti.kernel
# def compute_x_avg():
#     for i in range(n_particles):
#         contrib = 0.0
#         if particle_type[i] == 1:
#             contrib = 1.0 / n_solid_particles
#         ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])

@ti.kernel
def compute_x_leftmost():
    # # Initialize to a large value
    # x_leftmost[None] = [1.0, 0.0]  # Using 1.0 as it's larger than our domain (0-1)
    
    # Find the minimum x-coordinate among solid particles
    for i in range(n_particles):
        if particle_type[i] == 1:  # Only consider solid particles
            # Use atomic_min to find the minimum x-coordinate
            ti.atomic_min(x_leftmost[None][0], x[steps - 1, i][0])

@ti.kernel
def compute_loss():
    # Initialize loss to zero
    loss[None] = 0.0
    
    # Set loss to negative of leftmost x-coordinate (to maximize distance)
    loss[None] = -x_leftmost[None][0]


    # dist = x_avg[None][0]  # Distance traveled in x-direction
    # max_height = 0.0
    # for i in range(n_particles):
    #     if particle_type[i] == 1:  # Only consider solid particles
    #         max_height = ti.max(max_height, x[steps - 1, i][1])
    # loss[None] = -dist + 0.5 * max_height  # Reward forward distance, penalize high jumps


@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    # x_avg[None] = [0, 0]
    # compute_x_avg()
    x_leftmost[None] = [10.0,0]
    compute_x_leftmost()
    compute_loss()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def geno2pheno(self,geno): # convert genotype to phenotype
        repeats = geno[-1]
        for i in range(repeats): # for each desired number of genotypes
            for j in range(len(geno)-1): # for each rectangle in the genotype
                if geno[j][4] == -1: # if solid body
                    self.add_rect(geno[j][0] + i*geno[j][2],geno[j][1],geno[j][2],geno[j][3],geno[j][4],geno[j][5],geno[j][6])
                else: # else if actuator
                    self.add_rect(geno[j][0] + i*geno[0][2],geno[j][1],geno[j][2],geno[j][3],geno[j][4]+i,geno[j][5],geno[j][6])

    def add_rect(self, x, y, w, h, actuation, ptype=1, corner=0):
        # corner argument:
        # 0 = top-left (default)
        # 1 = top-right
        # 2 = bottom-left
        # 3 = bottom-right

        if ptype == 0:
            assert actuation == -1
        global n_particles

        # Adjust coordinates based on corner choice
        if corner == 1:  # top-right
            x1 = x - w
            y1 = y - h
        elif corner == 2:  # bottom-left
            x1 = x
            y1 = y
        elif corner == 3:  # bottom-right
            x1 = x - w
            y1 = y
        else:  # top-left (default)
            x1 = x
            y1 = y - h

        # Calculate the number of particles in the x and y directions
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count

        # Create the particles in the specified rectangle area
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x1 + (i + 0.5) * real_dx + self.offset_x,
                    y1 + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act

    def reset(self):
        clear_grid()
        clear_particle_grad()
        clear_actuation_grad()
        x.fill(0)
        v.fill(0)
        C.fill(0)
        F.fill(0)
        actuation.fill(0)
        weights.fill(0)
        bias.fill(0)
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []             # List of particle positions
        self.actuator_id = []   # List of actuator ids
        self.particle_type = [] # List indicating type (e.g., fluid vs. solid)
        # x_avg[None] = [0, 0]
        x_leftmost[None] = [0,0]
        loss[None] = 0


def fish(scene):
    scene.add_rect(0.025, 0.025, 0.95, 0.1, -1, ptype=0)
    scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
    scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
    scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
    scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
    scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
    scene.set_n_actuators(4)


def robot(scene):
    scene.set_offset(0.1, 0.03)
    scene.add_rect(0.0, 0.1, 0.3, 0.1, -1)
    scene.add_rect(0.0, 0.0, 0.05, 0.1, 0)
    scene.add_rect(0.05, 0.0, 0.05, 0.1, 1)
    scene.add_rect(0.2, 0.0, 0.05, 0.1, 2)
    scene.add_rect(0.25, 0.0, 0.05, 0.1, 3)
    scene.set_n_actuators(4)

def geno_robot(scene,geno): # new function, defines robot genotype
    scene.set_offset(0.1, 0.1)
    scene.geno2pheno(geno) # genotype->phenotype
    scene.set_n_actuators(geno[-1]*2)

def mutate_geno(geno, mutation_rate=0.6, size_variation=0.02):
    """Randomly mutates the sizes and positions of rectangles in geno_robot while ensuring connectivity."""
    new_geno = []
    repeats = geno[-1]
    geno.pop()
    for i, rect in enumerate(geno):
        x, y, w, h, actuation, ptype, corner = rect
        if random.random() < mutation_rate:
            w = round(max(0.02, w + random.uniform(-size_variation, size_variation)),2)
            h = round(max(0.02, h + random.uniform(-size_variation, size_variation)),2)
        
        if i == 2:
            x = new_geno[0][2]
        
        new_rect = (x, y, w, h, actuation, ptype, corner)
        new_geno.append(new_rect)

    if random.random() < mutation_rate/2:
        if random.random() < 0.5:
            repeats = min(repeats + 1,5)

        else:
            repeats = max(repeats - 1,1)
    new_geno.append(repeats)
    return new_geno


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

def run(geno):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--iters', type=int, default=100)
    # options = parser.parse_args()

    # geno = [(0.0, 0.00, 0.15, 0.08, -1, 1,2),(0.0, 0.0, 0.05, 0.05, 0, 1,0),(0.1, 0.0, 0.05, 0.05, 1, 1,1)] # list of tuples defining genotype rectangles
    # geno = mutate_geno(geno)
    # print(geno)

    scene = Scene()
    geno_robot(scene, geno)
    scene.finalize()
    allocate_fields()

    for i in range(n_actuators):
        for j in range(n_sin_waves):
            weights[i, j] = np.random.randn() * 0.01

    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

        if particle_type[i] == 0:  # Solid particles
            F[0, i] = [[1, 0], [0, 1]]  # Identity matrix
        else:
            F[0, i] = [[1, 0], [0, 1]]  # Ensuring actuators also get correct F

        # Debugging print
        # print(f"Particle {i}: Type={particle_type[i]}, F={F[0, i]}")

    # for i in range(scene.n_particles):    # debugging
    #     if particle_type[i] == 0:
    #         print(f"Solid Particle {i}: F={F[0, i]}")

    losses = []
    for iter in range(80):#range(options.iters):
        with ti.ad.Tape(loss):
            forward()
        l = loss[None]
        losses.append(l)
        
        if iter % 10 == 0:
            print('i=', iter, 'loss=', l)
        
        learning_rate = 0.1
        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights[i, j] -= learning_rate * weights.grad[i, j]
            bias[i] -= learning_rate * bias.grad[i]

        # if iter == 0:
        #     forward(1500)
        #     for s in range(15, 1500, 16):
        #         visualize(s, 'diffmpm/iter{:03d}/'.format(iter))


    # Run the visualization once at the end
    forward(1500)
    for s in range(15, 1500, 16):
        visualize(s, 'output_folder')

    # ti.reset()
    scene.reset()
    # allocate_fields()
    gui.clear()

    # ti.reset()
    # ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

    return geno, losses[-1]

    # plt.title("Optimization of Initial Velocity")
    # plt.ylabel("Loss")
    # plt.xlabel("Gradient Descent Iterations")
    # plt.plot(losses)
    # plt.show()

# if __name__ == '__main__':
#     main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotype", type=str, required=True)
    args = parser.parse_args()

    # Deserialize JSON and convert lists back to tuples
    genotype = [tuple(item) for item in json.loads(args.genotype)]

    run(genotype)