from const_param import dt, h0, shape, shapeu, shapev, shapew, MACGrid, Field
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
    vel: MACGrid,
    phi: Field, 
    new_phi: Field
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + phi.stagger
    p = backtrace(vel.u.f, vel.v.f, vel.w.f, p, dt, h0)
    gx = phi.f.shape[0]
    gy = phi.f.shape[1]
    gz = phi.f.shape[2]
    if in_bound(p[0], gx) and in_bound(p[1], gy) and in_bound(p[2], gz):
        new_phi.f[i, j, k] = trilerp_stagger(phi.f, p, phi.stagger)
    else: 
        new_phi.f[i, j, k] = h0


@wp.kernel
def _advect_mc_1(
    args_common: MACGrid,
    q_in: Field,
    q_out: Field,
    q_tmp: wp.array3d(dtype = float)
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + q_in.stagger
    p1 = backtrace(args_common.u.f, args_common.v.f, args_common.w.f, p, dt, h0)
    q_out.f[i, j, k] = trilerp_stagger(q_in.f, p1, q_in.stagger)

@wp.kernel
def _advect_mc_2(
    args_common: MACGrid,
    q_in: Field,
    q_out: Field,
    q_tmp: wp.array3d(dtype = float)
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + q_in.stagger
    
    p2 = backtrace(args_common.u.f, args_common.v.f, args_common.w.f, p, -dt, h0)
    q_tmp[i, j, k] = trilerp_stagger(q_in.f, p2, q_in.stagger)
    

@wp.kernel
def _advect_mc_3(
    args_common: MACGrid,
    q_in: Field,
    q_out: Field,
    q_tmp: wp.array3d(dtype = float),
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + q_in.stagger

    p3 = backtrace(args_common.u.f, args_common.v.f, args_common.w.f, p, dt, h0)
    q_sl = trilerp_stagger(q_in.f, p3, q_in.stagger)
    
    q_out.f[i, j, k] = q_out.f[i, j, k] + 0.5 * (q_in.f[i, j, k] - q_tmp[i, j, k])

    min_val, max_val = sample_minmax(q_in.f, p3, h0, q_in.stagger)
    cond = min_val < q_out.f[i, j, k] and q_out.f[i, j, k] < max_val
    if not cond:
        q_out.f[i, j, k] = q_sl

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

def advect_mc(args_common: MACGrid, q_in: Field, q_out: Field):
    q_tmp = wp.empty_like(q_in.f)
    wp.launch(_advect_mc_1, dim = q_in.f.shape, inputs = [args_common, q_in, q_out, q_tmp])
    wp.launch(_advect_mc_2, dim = q_in.f.shape, inputs = [args_common, q_in, q_out, q_tmp])
    wp.launch(_advect_mc_3, dim = q_in.f.shape, inputs = [args_common, q_in, q_out, q_tmp])

def advect_uv_mc(vel_in: MACGrid, vel_out: MACGrid):
    advect_mc(vel_in, vel_in.u, vel_out.u)
    advect_mc(vel_in, vel_in.v, vel_out.v)
    advect_mc(vel_in, vel_in.w, vel_out.w)


def advect_3d(vel: MACGrid, qf: Field, new_vel: MACGrid, new_qf: Field):    
    advect_mc(vel, qf, new_qf)
    advect_uv_mc(vel, new_vel)