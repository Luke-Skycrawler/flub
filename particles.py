import warp as wp
import numpy as np
from const_param import n_particle_per_cell, h0, res, timer_active, gridx, gridy, gridz, MACGrid, Field, shapeu, shapev, shapew, shape, get_uf
from utils.util import sample_velocity_3d, divide, inv_bilerpu, inv_bilerpv, inv_bilerpw, sample_3d, trilerp
from fenwick import FenwickTree
from typing import Any
from redistance import redistance
int8 = int
max_num_particles_per_cell = n_particle_per_cell * 3
clip_max = n_particle_per_cell * 2
flip_alpha = 0.0
@wp.kernel
def min_kernel_offset(q1: Field, q2: Field, h: float):
    i, j, k = wp.tid()
    if q2.f[i, j, k] < q1.f[i, j, k] + h:
        q1.f[i, j, k] = q2.f[i, j, k]
    # else:
    #     q1[i, j, k] = q1[i, j, k] + h

@wp.kernel
def _compress(deleted: wp.array1d(dtype = int8), prefix_deleted: wp.array1d(dtype = int), a: wp.array1d(dtype = Any), b: wp.array1d(dtype = Any)):
    i = wp.tid()
    if not deleted[i]:
        b[i - prefix_deleted[i]] = a[i]
    
@wp.func
def get_cell(p: wp.vec3, res: int):
    # return int(wp.floor(p * float(res)))
    pf = p * float(res)
    return wp.vec3i(int(wp.floor(pf[0])), int(wp.floor(pf[1])), int(wp.floor(pf[2])))

@wp.kernel
def cnt_activate(activated: wp.array1d(dtype = int), prefix_sum: wp.array1d(dtype = int)):
    # n_particles = prefix_sum.shape[0]
    n_particles = activated[0]
    activated[0] = n_particles - prefix_sum[n_particles-1]



@wp.kernel
def _resample_1(phi: Field, 
              x: wp.array1d(dtype = wp.vec3), v: wp.array1d(dtype = wp.vec3),
              cnt_particles: wp.array3d(dtype = int), 
              activated: wp.array1d(dtype = int),
              deleted: wp.array1d(dtype = int8), h: float, r: float, R: float):
    shape = wp.vec3i(cnt_particles.shape[0], cnt_particles.shape[1], cnt_particles.shape[2])
    shape -= wp.vec3i(1, 1, 1)
    p = wp.tid()
    cell = get_cell(x[p], shape[0])
    i, j, k = cell[0], cell[1], cell[2]
    dele = int(0)

    offs = wp.atomic_add(cnt_particles, i, j, k, 1)
    cond1 = wp.min(cell) >= 0 and wp.max(cell, shape) == shape and p < activated[0]
    cond2 = offs < clip_max and not (phi.f[i, j, k] < -R or phi.f[i, j, k] > h)

    dele = wp.where(cond1 and cond2, 0, 1)

    deleted[p] = dele

    

@wp.func
def init_xv(
    uf: wp.array3d(dtype = float), vf: wp.array3d(dtype = float), wf: wp.array3d(dtype = float),
    i: int, j: int, k: int, res: int, _seed: int
    ):
    seed = wp.rand_init(_seed, i << 16 + j << 8 + k)
    x = wp.vec3(float(i) + wp.randf(seed), float(j) + wp.randf(seed), float(k) + wp.randf(seed)) / float(res)
    v = sample_velocity_3d(uf, vf, wf, x * float(res))

    return x, v
    
@wp.kernel
def _resample_2(phi: Field, 
              vel: MACGrid, 
              x: wp.array1d(dtype = wp.vec3), v: wp.array1d(dtype = wp.vec3), 
              cnt_particles: wp.array3d(dtype = int), 
              activated: wp.array1d(dtype = int),
              deleted: wp.array1d(dtype = int8), 
              h: float, r: float, R: float
              ):
    res = cnt_particles.shape[0]

    i, j, k = wp.tid()
    n_particles = cnt_particles[i, j, k]
    if -R <= phi.f[i, j, k] and phi.f[i, j, k] <= -h:
        # resample

        
        for kk in range(n_particles, n_particle_per_cell):
            p = wp.atomic_add(activated, 0, 1)
            
            # seed = wp.rand_init(i << 16 + j << 8 + k, 0)
            # x[p] = wp.vec3(float(i) + wp.randf(seed), float(j) + wp.randf(seed), float(k) + wp.randf(seed)) / float(res)
            # v[p] = sample_velocity_3d(uf, vf, wf, x[p] * float(res))

            xp, vp = init_xv(vel.u.f, vel.v.f, vel.w.f, i, j, k, res, kk)

            if p < x.shape[0] and p >= 0 :
                x[p] = xp
                v[p] = vp
                deleted[p] = 0

@wp.func 
def grad_at(
    pf: wp.array3d(dtype = float), pos: wp.vec3):
    iu = int(wp.floor(pos[0] - 0.5))
    iv = int(wp.floor(pos[1] - 0.5))
    iw = int(wp.floor(pos[2] - 0.5))

    fu = pos[0] - 0.5 - float(iu)
    fv = pos[1] - 0.5 - float(iv)
    fw = pos[2] - 0.5 - float(iw)

    phi000 = sample_3d(pf, iu, iv, iw)
    phi100 = sample_3d(pf, iu + 1, iv, iw)
    phi010 = sample_3d(pf, iu, iv + 1, iw)
    phi110 = sample_3d(pf, iu + 1, iv + 1, iw)

    phi001 = sample_3d(pf, iu, iv, iw + 1)
    phi101 = sample_3d(pf, iu + 1, iv, iw + 1)
    phi011 = sample_3d(pf, iu, iv + 1, iw + 1)
    phi111 = sample_3d(pf, iu + 1, iv + 1, iw + 1)

    nx_00 = phi100 - phi000 
    nx_01 = phi101 - phi001
    nx_10 = phi110 - phi010
    nx_11 = phi111 - phi011

    nx = wp.lerp(wp.lerp(nx_00, nx_01, fw), wp.lerp(nx_10, nx_11, fw), fv)

    ny0_0 = phi010 - phi000
    ny0_1 = phi011 - phi001
    ny1_0 = phi110 - phi100
    ny1_1 = phi111 - phi101

    ny = wp.lerp(wp.lerp(ny0_0, ny0_1, fw), wp.lerp(ny1_0, ny1_1, fw), fu)

    nz00_ = phi001 - phi000
    nz01_ = phi011 - phi010
    nz10_ = phi101 - phi100
    nz11_ = phi111 - phi110

    nz = wp.lerp(wp.lerp(nz00_, nz01_, fv), wp.lerp(nz10_, nz11_, fv), fu)
    return wp.vec3(nx, ny, nz)

@wp.func
def normal_at(
    phi: wp.array3d(dtype = float), pos: wp.vec3):
    return wp.normalize(grad_at(phi, pos))
    
@wp.kernel
def exile(
    x: wp.array1d(dtype = wp.vec3),
    v: wp.array1d(dtype = wp.vec3),
    phi_rigid: Field,
    vel: MACGrid):
    p = wp.tid()
    pos = x[p] / h0
    phi = trilerp(phi_rigid.f, pos)
    if phi < 0.0:
        normal = normal_at(phi_rigid.f, pos)
        x[p] = x[p] + normal * (-phi)
        # x[p] = wp.vec3(-1.0, -1.0, -1.0)
        v[p] = sample_velocity_3d(vel.u.f, vel.v.f, vel.w.f, pos)
    
@wp.kernel
def _nbcombine(
    vel_grid: MACGrid,
    vel_p: MACGrid, 
    phi: Field,
    r: float,
    d: int):
    
    i, j, k = wp.tid()
    phiu = 0.0
    uf = get_uf(vel_grid, d).f
    uu = get_uf(vel_p, d).f
    idx = wp.vec3i(i, j, k)
    ii = idx[d - 1]
    if 0 < ii and ii < uf.shape[d - 1] - 1:
        idx[d - 1] -= 1
        phil = phi.f[idx[0], idx[1], idx[2]]
        phir = phi.f[i, j, k]
        phiu = (phil + phir) * 0.5
    if phiu >= -r:
        uf[i, j, k] = uu[i, j, k]
    
        
class ParticleSystem:
    def __init__(self, shape = (256, 256, 256), sparsity = 1):

        self.n_particle_per_cell = n_particle_per_cell
        # self.n_particles = n_particles = shape[0] * shape[1] * shape[2] * n_particle_per_cell // sparsity
        self.n_particles = n_particles = shape[0] * shape[1] * shape[2] * n_particle_per_cell // sparsity
        self.cnt_particles = wp.empty(shape = shape, dtype = int)
        self.activated =wp.zeros((1), dtype = int)
        self.activated_int:int = 0
        self.deleted = wp.zeros(shape = (n_particles), dtype = int8)

        self.shape = shape

        digits = int(np.rint(np.log2(n_particles)))
        self.ft = FenwickTree(digits)
        
        self.tmp_x = wp.zeros(shape = (n_particles), dtype = wp.vec3)
        
        self.h = 1.0 / shape[0]
        self.r = 2 * self.h
        self.R = 3 * self.h

        self.clip_max = self.n_particle_per_cell * 2

        
        self.x = wp.zeros(shape = (n_particles), dtype = wp.vec3)
        self.v = wp.zeros(shape = (n_particles), dtype = wp.vec3)
        self.fields = [self.cnt_particles, self.activated, self.deleted]
        self.dofs = [self.x, self.v]
        # self.rd = ss.Redistance(3)
    
    def initialize_particles(self, phi: Field, vel: MACGrid):
        self.activated.zero_()
        self.deleted.fill_(1)
        self.cnt_particles.zero_()
        args = [phi, vel] + self.dofs + self.fields + [self.h, self.r, self.R]
        wp.launch(_initialize, dim = shape, inputs = args)

        
        
    def compress(self):
        # FIXME: prevent activated overflow
        self.activated_int = int(self.activated.numpy()[0])
        self.activated_int = min(self.n_particles, self.activated_int)
        self.ft.prefix_fast(self.deleted, self.activated_int)


        wp.launch(_compress, dim = (self.activated_int), inputs = [self.deleted, self.ft.sum, self.x, self.tmp_x])
        wp.copy(self.x, self.tmp_x)
        wp.launch(_compress, dim = (self.activated_int), inputs = [self.deleted, self.ft.sum, self.v, self.tmp_x])
        wp.copy(self.v, self.tmp_x)

        wp.launch(cnt_activate, dim = (1), inputs = [self.activated, self.ft.sum])
        self.activated_int= int(self.activated.numpy()[0])
        
    def nb_flip_step_phi(self, phi_prime: Field, phi_p: Field):
        '''
        phi_prime: phi after advection 
        phi_p: returns phi from particles 
        '''
        phi_p.f.fill_(1.0)
        self.compute_water_phi(phi_p, radius = h0)
        wp.launch(min_kernel_offset, dim = phi_prime.f.shape, inputs = [phi_prime, phi_p, h0])

        self.redistance(phi_prime, h0)

    def compute_water_phi(self, ret: Field, radius = h0):
        '''
        radius: particle radius for phi compute 
        '''
        activated = self.activated_int
        wp.launch(_compute_water_phi, dim=(
            activated), inputs=[self.x, ret, radius])
        for i in range(8):
            wp.launch(_expand, dim=shape, inputs=[ret, i % 2])
            
    def redistance(self, phi: Field, h: float, set_boundary = 0):

        with wp.ScopedTimer("redistance", active=timer_active, synchronize=True):
            redistance(phi.f, h, set_boundary = set_boundary)

    def out(self, phi_rigid: Field, vel: MACGrid):
        wp.launch(exile, dim = self.activated_int, inputs = [self.x, self.v, phi_rigid, vel])

    def resample(self, phi_prime: Field, vel: MACGrid):
        with wp.ScopedTimer("resample", active=timer_active, synchronize=True):
            self.cnt_particles.zero_()
            self.deleted.fill_(1)
            args = [phi_prime, vel] + self.dofs + self.fields + [self.h, self.r, self.R]
            args1 = [phi_prime] + self.dofs + self.fields + [self.h, self.r, self.R]
            n_particles = self.activated_int
            wp.launch(_resample_1, dim = n_particles, inputs = args1)
            wp.launch(_resample_2, dim = phi_prime.f.shape, inputs = args)
            
            self.compress()
    
    def nbcombine(self, vel_grid: MACGrid, vel_p: MACGrid, phi: Field):  
        wp.launch(_nbcombine, dim = shapeu, inputs = [vel_grid, vel_p, phi, self.r, 1])
        wp.launch(_nbcombine, dim = shapev, inputs = [vel_grid, vel_p, phi, self.r, 2])
        wp.launch(_nbcombine, dim = shapew, inputs = [vel_grid, vel_p, phi, self.r, 3])

    def step(self, vel0: MACGrid, dt, pf: Field = None, vel1: MACGrid = None):
        if pf is None:
            wp.launch(particle_step, dim = int(self.activated_int), inputs = [self.x, self.v, vel0, self.activated, dt])
        else :

            wp.launch(particle_step_flip, dim = int(self.activated_int), inputs = [self.x, self.v, vel0, vel1, pf, self.activated, dt])

        
    def p2g(self, vel: MACGrid, weight: MACGrid):

        '''
        uf, vf, wf: returns the grid velocity field
        u1, v1, w1: temporary fields
        '''
        # _p2g(particles, uf, vf, grid_mu, grid_mv)
        weight.u.f.zero_()
        weight.v.f.zero_()
        weight.w.f.zero_()

        vel.u.f.zero_()
        vel.v.f.zero_()
        vel.w.f.zero_()
        
        x = self.x
        v = self.v
        n_particles = self.activated_int
        wp.launch(p2g_kernel, dim = int(n_particles), inputs = [x, v, vel, weight])

        wp.launch(divide, dim = shapeu, inputs = [vel.u.f, weight.u.f])
        wp.launch(divide, dim = shapev, inputs = [vel.v.f, weight.v.f])
        wp.launch(divide, dim = shapew, inputs = [vel.w.f, weight.w.f])

        wp.launch(enforce_boundary, dim = (1), inputs = [vel])
        # enforce_boundary(uf, vf)



@wp.func
def _avg_neighbor(vf: wp.array3d(), i: int, j: int, k: int):
    sum = float(0.0)
    cnt = int(0)
    for ii in range(-1, 2):
        for jj in range(-1, 2):
            for kk in range(-1, 2):
                cond1 = 0 <= i + ii and i + ii < vf.shape[0]
                cond2 = 0 <= j + jj and j + jj < vf.shape[1]
                cond3 = 0 <= k + kk and k + kk < vf.shape[2]
                if cond1 and cond2 and cond3:
                    sum += vf[i + ii, j + jj, k + kk]
                    cnt += 1
    ret = float(0.0)
    if cnt > 0:
        ret = sum / float(cnt)
    return ret

@wp.kernel
def _expand(phi: Field, odd: int):
    i, j, k = wp.tid()
    if (i + j + k) % 2 == odd:
        phi.f[i, j, k] = min(phi.f[i, j, k], _avg_neighbor(phi.f, i, j, k))


@wp.func
def update_min(phi: wp.array3d(dtype=float), i: int, j: int, k: int, dis: float):
    if i >= 0 and i < phi.shape[0] and j >= 0 and j < phi.shape[1] and k >= 0 and k < phi.shape[2]:
        wp.atomic_min(phi, i, j, k, dis)


@wp.kernel
def _compute_water_phi(
    px: wp.array1d(dtype=wp.vec3),
    phi: Field, r: float):
    '''
    NOTE: particle size needs to be decided: do we use finer particles on finer patches? add field for particle radius? Or do we just stick to coarse manage all the particles (may limit the adaptivity)
    '''
    i = wp.tid()
    p = px[i]
    _p = px[i]

    ix = int(wp.floor(_p[0] / h0 - 0.5))
    iy = int(wp.floor(_p[1] / h0 - 0.5))
    iz = int(wp.floor(_p[2] / h0 - 0.5))

    ix = wp.max(ix, 0)
    iy = wp.max(iy, 0)
    iz = wp.max(iz, 0)

    ix = wp.min(ix, gridx - 1)
    iy = wp.min(iy, gridy - 1)
    iz = wp.min(iz, gridz - 1)

    pa = wp.vec3(float(ix) + 0.5, float(iy) + 0.5, float(iz) + 0.5) * h0
    pb = pa + wp.vec3(1.0, 0.0, 0.0) * h0
    pc = pa + wp.vec3(0.0, 1.0, 0.0) * h0
    pd = pa + wp.vec3(1.0, 1.0, 0.0) * h0

    pe = pa + wp.vec3(0.0, 0.0, 1.0) * h0
    pf = pa + wp.vec3(1.0, 0.0, 1.0) * h0
    pg = pa + wp.vec3(0.0, 1.0, 1.0) * h0
    ph = pa + wp.vec3(1.0, 1.0, 1.0) * h0

    da = wp.length(p - pa) - r * 1.0
    db = wp.length(p - pb) - r * 1.0
    dc = wp.length(p - pc) - r * 1.0
    dd = wp.length(p - pd) - r * 1.0


    de = wp.length(p - pe) - r * 1.0
    df = wp.length(p - pf) - r * 1.0
    dg = wp.length(p - pg) - r * 1.0
    dh = wp.length(p - ph) - r * 1.0


    update_min(phi.f, ix, iy, iz, da)
    update_min(phi.f, ix + 1, iy, iz, db)
    update_min(phi.f, ix, iy + 1, iz, dc)
    update_min(phi.f, ix + 1, iy + 1, iz, dd)

    update_min(phi.f, ix, iy, iz + 1, de)
    update_min(phi.f, ix + 1, iy, iz + 1, df)
    update_min(phi.f, ix, iy + 1, iz + 1, dg)
    update_min(phi.f, ix + 1, iy + 1, iz + 1, dh)
    


@wp.func 
def forward_rk3_3d(
    uf: wp.array3d(dtype = float), vf: wp.array3d(dtype = float), wf: wp.array3d(dtype = float),
    p: wp.vec3, dt: float):
    v1 = sample_velocity_3d(uf, vf, wf, p)
    p1 = p + 0.5 * dt * v1 / h0
    v2 = sample_velocity_3d(uf, vf, wf, p1)
    p2 = p + 0.75 * dt * v2 / h0
    v3 = sample_velocity_3d(uf, vf, wf, p2)
    p3 = p + dt * ((2.0 / 9.0) * v1 + (1.0 / 3.0) * v2 + (4.0 / 9.0) * v3) / h0
    return p3

@wp.kernel
def particle_step(
    # tracer_particles: Particles, 
    x: wp.array1d(dtype = wp.vec3), v: wp.array1d(dtype = wp.vec3),
    vel: MACGrid,
    activated: wp.array1d(dtype = int),
    dt: float):

    i = wp.tid()
    if i < activated[0]:
        p = x[i] / h0

        v[i] = sample_velocity_3d(vel.u.f, vel.v.f, vel.w.f, p)
        bound = wp.vec3(h0 * float(gridx), h0 * float(gridy), h0 * float(gridz))

        xi = forward_rk3_3d(vel.u.f, vel.v.f, vel.w.f, p, dt) * h0
        x[i] = wp.vec3(wp.clamp(xi[0], 0.0, bound[0]), wp.clamp(xi[1], 0.0, bound[1]), wp.clamp(xi[2], 0.0 , bound[2]))

@wp.kernel
def particle_step_flip(
    x: wp.array1d(dtype = wp.vec3), v: wp.array1d(dtype = wp.vec3),
    vel: MACGrid,
    vel_last: MACGrid,
    pf: Field,
    activated: wp.array1d(dtype = int),
    dt: float):
    i = wp.tid()            
    if i < activated[0]:
        p = x[i] / h0
        v1 = sample_velocity_3d(vel.u.f, vel.v.f, vel.w.f, p)
        v0 = sample_velocity_3d(vel_last.u.f, vel_last.v.f, vel_last.w.f, p)
        v[i] = ((v1 - v0) + v[i]) * flip_alpha + v1 * (1.0 - flip_alpha)

        bound = wp.vec3(h0 * float(gridx), h0 * float(gridy), h0 * float(gridz))

        xi = forward_rk3_3d(vel.u.f, vel.v.f, vel.w.f, p, dt) * h0
        x[i] = wp.vec3(wp.clamp(xi[0], 0.0, bound[0]), wp.clamp(xi[1], 0.0, bound[1]), wp.clamp(xi[2], 0.0 , bound[2]))



@wp.kernel
def p2g_kernel(
    x: wp.array1d(dtype = wp.vec3), v: wp.array1d(dtype = wp.vec3),
    vel: MACGrid, weight: MACGrid):
    i = wp.tid()
    p = x[i] / h0 
    inv_bilerpu(weight.u.f, p, 1.0)
    inv_bilerpv(weight.v.f, p, 1.0)
    inv_bilerpw(weight.w.f, p, 1.0)

    inv_bilerpu(vel.u.f, p, v[i][0])
    inv_bilerpv(vel.v.f, p, v[i][1])
    inv_bilerpw(vel.w.f, p, v[i][2])

@wp.kernel
def enforce_boundary(vel: MACGrid):
    # for j, k in range(gridy, gridz):
    for j in range(gridy):
        for k in range(gridz):
            vel.u.f[0, j, k] = 0.0
            vel.u.f[gridx, j, k] = 0.0 
    # for i, k in range(gridx, gridz):
    for i in range(gridx):
        for k in range(gridz):
            vel.v.f[i, 0, k] = 0.0
            vel.v.f[i, gridy, k] = 0.0
    # for i, j in range(gridx, gridy):
    for i in range(gridx):
        for j in range(gridy):
            vel.w.f[i, j, 0] = 0.0
            vel.w.f[i, j, gridz] = 0.0

@wp.kernel
def _initialize(phi: Field,
              vel: MACGrid,
              x: wp.array1d(dtype = wp.vec3), v: wp.array1d(dtype = wp.vec3), 
              cnt_particles: wp.array3d(dtype = int), 
              activated: wp.array1d(dtype = int),
              deleted: wp.array1d(dtype = int8), 
              h: float, r: float, R: float
              ):
    res = phi.f.shape[0]  
    i, j, k = wp.tid()
    phiijk = phi.f[i, j, k]
    if (phiijk < 0.0 and phiijk >= -R):
        p = wp.atomic_add(activated, 0, n_particle_per_cell)
        for ii in range(n_particle_per_cell):
            # FIXME: dangerous use of res
            xp, vp = init_xv(vel.u.f, vel.v.f, vel.w.f, i, j, k, res, ii)
            x[p + ii] = xp
            v[p + ii] = vp
            deleted[p + ii] = 0
        