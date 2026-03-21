from const_param import dt, h0, Field, MACGrid, field_like
import warp as wp
from utils.util import sample_velocity_3d, sample_3d, trilerp_stagger, trilerp

@wp.func
def in_bound(p: float, res: int):
    return 0.0 <= p and p < float(res)


@wp.func
def backtrace(
    uf: wp.array3d(dtype=float), 
    vf: wp.array3d(dtype=float), 
    wf: wp.array3d(dtype=float),
    p: wp.vec3, dt: float, h: float):
    v1 = sample_velocity_3d(uf, vf, wf, p)
    p1 = p - 0.5 * dt / h * v1
    v2 = sample_velocity_3d(uf, vf, wf, p1)
    p2 = p - 0.75 * dt / h * v2
    v3 = sample_velocity_3d(uf, vf, wf, p2)
    p -= dt / h * ((2.0 / 9.0) * v1 + (1.0 / 3.0) * v2 + (4.0 / 9.0) * v3)
    return p

@wp.kernel
def advect_simple_3d(
    uf: wp.array3d(dtype=float),
    vf: wp.array3d(dtype=float),
    wf: wp.array3d(dtype=float),
    qf: wp.array3d(dtype=float),
    new_qf: wp.array3d(dtype=float),
    dt: float,
    h: float
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    p = backtrace(uf, vf, wf, p, dt, h)
    new_qf[i, j, k] = trilerp(qf, p)

@wp.kernel
def advect_phi(
    uf: wp.array3d(dtype=float),
    vf: wp.array3d(dtype=float),
    wf: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    new_phi: wp.array3d(dtype=float),
    dt: float,
    h: float
):
    i, j, k = wp.tid()
    stagger = wp.vec3(0.5, 0.5, 0.5)
    p = wp.vec3(float(i), float(j), float(k)) + stagger
    p = backtrace(uf, vf, wf, p, dt, h)
    gx = phi.shape[0]
    gy = phi.shape[1]
    gz = phi.shape[2]
    if in_bound(p[0], gx) and in_bound(p[1], gy) and in_bound(p[2], gz):
        new_phi[i, j, k] = trilerp_stagger(phi, p, stagger)
    else: 
        new_phi[i, j, k] = h0


@wp.kernel
def _advect_mc_1(
    vel: MACGrid,
    q0: Field,
    q_tmp: Field,
    q1: Field
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + q0.stagger
    p1 = backtrace(vel.u.f, vel.v.f, vel.w.f, p, dt, h0)
    q1.f[i, j, k] = trilerp_stagger(q0.f, p1, q0.stagger)

@wp.kernel
def _advect_mc_2(
    vel: MACGrid,
    q0: Field,
    q_tmp: Field,
    q1: Field
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + q0.stagger
    
    p2 = backtrace(vel.u.f, vel.v.f, vel.w.f, p, -dt, h0)
    q_tmp.f[i, j, k] = trilerp_stagger(q1.f, p2, q0.stagger)
    

@wp.kernel
def _advect_mc_3(
    vel: MACGrid,
    q0: Field,
    q_tmp: Field,
    q1: Field
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + q0.stagger

    p3 = backtrace(vel.u.f, vel.v.f, vel.w.f, p, dt, h0)
    q_sl = trilerp_stagger(q0.f, p3, q0.stagger)
    
    q1.f[i, j, k] = q1.f[i, j, k] + 0.5 * (q0.f[i, j, k] - q_tmp.f[i, j, k])

    min_val, max_val = sample_minmax(q0.f, p3, h0, q0.stagger)
    cond = min_val < q1.f[i, j, k] and q1.f[i, j, k] < max_val
    if not cond:
        q1.f[i, j, k] = q_sl

@wp.func
def sample_minmax(q_in: wp.array3d(dtype = float), p: wp.vec3, h: float, stagger: wp.vec3):
    uvw = p / h
    rst = uvw - stagger 

    iuvw = wp.vec3i(int(wp.floor(rst[0])), int(wp.floor(rst[1])), int(wp.floor(rst[2])))

    a = sample_3d(q_in, iuvw[0], iuvw[1], iuvw[2])
    b = sample_3d(q_in, iuvw[0] + 1, iuvw[1], iuvw[2])
    c = sample_3d(q_in, iuvw[0], iuvw[1] + 1, iuvw[2])
    d = sample_3d(q_in, iuvw[0] + 1, iuvw[1] + 1, iuvw[2])
    e = sample_3d(q_in, iuvw[0], iuvw[1], iuvw[2] + 1)
    f = sample_3d(q_in, iuvw[0] + 1, iuvw[1], iuvw[2] + 1)
    g = sample_3d(q_in, iuvw[0], iuvw[1] + 1, iuvw[2] + 1)
    h = sample_3d(q_in, iuvw[0] + 1, iuvw[1] + 1, iuvw[2] + 1)

    min_val = wp.min(wp.min(wp.vec4(a, b, c, d)), wp.min(wp.vec4(e, f, g, h)))
    max_val = wp.max(wp.max(wp.vec4(a, b, c, d)), wp.max(wp.vec4(e, f, g, h)))

    return min_val, max_val

def advect_mc(vel0: MACGrid, q0: Field, q1: Field):
    q_tmp = field_like(q0)

    wp.launch(_advect_mc_1, dim = q0.f.shape, inputs = [vel0, q0, q_tmp, q1])
    wp.launch(_advect_mc_2, dim = q0.f.shape, inputs = [vel0, q0, q_tmp, q1])
    wp.launch(_advect_mc_3, dim = q0.f.shape, inputs = [vel0, q0, q_tmp, q1])

def advect_uv_mc(vel0: MACGrid, vel1: MACGrid):
    advect_mc(vel0, vel0.u, vel1.u)
    advect_mc(vel1, vel0.v, vel1.v)
    advect_mc(vel0, vel0.w, vel1.w)


def advect_3d(vel0, vel1, q0, q1):    
    advect_mc(vel0, q0, q1)
    advect_uv_mc(vel0, vel1)
