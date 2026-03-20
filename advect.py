from const_param import dt, h0
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


def advect_3d(uf, vf, wf, qf, new_uf, new_vf, new_wf, new_qf, h = h0):    
    advect_mc(qf, new_qf, uf, vf, wf, dt, h)
    advect_uv_mc(uf, vf, wf, new_uf, new_vf, new_wf, dt, h)

@wp.kernel
def _advect_mc_1(
    uf: wp.array3d(dtype = float),
    vf: wp.array3d(dtype = float),
    wf: wp.array3d(dtype = float),
    q_in: wp.array3d(dtype = float),
    q_out: wp.array3d(dtype = float),
    q_tmp: wp.array3d(dtype = float),
    dt: float, h: float, stagger: wp.vec3
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + stagger
    p1 = backtrace(uf, vf, wf, p, dt, h)
    q_out[i, j, k] = trilerp_stagger(q_in, p1, stagger)

@wp.kernel
def _advect_mc_2(
    uf: wp.array3d(dtype = float),
    vf: wp.array3d(dtype = float),
    wf: wp.array3d(dtype = float),
    q_in: wp.array3d(dtype = float),
    q_out: wp.array3d(dtype = float),
    q_tmp: wp.array3d(dtype = float),
    dt: float, h: float, stagger: wp.vec3
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + stagger
    
    p2 = backtrace(uf, vf, wf, p, -dt, h)
    q_tmp[i, j, k] = trilerp_stagger(q_out, p2, stagger)
    

@wp.kernel
def _advect_mc_3(
    uf: wp.array3d(dtype = float),
    vf: wp.array3d(dtype = float),
    wf: wp.array3d(dtype = float),
    q_in: wp.array3d(dtype = float),
    q_out: wp.array3d(dtype = float),
    q_tmp: wp.array3d(dtype = float),
    dt: float, h: float, stagger: wp.vec3
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + stagger

    p3 = backtrace(uf, vf, wf, p, dt, h)
    q_sl = trilerp_stagger(q_in, p3, stagger)
    
    q_out[i, j, k] = q_out[i, j, k] + 0.5 * (q_in[i, j, k] - q_tmp[i, j, k])

    min_val, max_val = sample_minmax(q_in, p3, h, stagger)
    cond = min_val < q_out[i, j, k] and q_out[i, j, k] < max_val
    if not cond:
        q_out[i, j, k] = q_sl

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

    
def advect_mc(q_in, q_out, uf, vf, wf, dt = dt, h = h0):
    q_tmp = wp.empty_like(q_in)
    stagger = wp.vec3(0.5, 0.5, 0.5)
    wp.launch(_advect_mc_1, dim = q_in.shape, inputs = [uf, vf, wf, q_in, q_out, q_tmp, dt, h, stagger])
    wp.launch(_advect_mc_2, dim = q_in.shape, inputs = [uf, vf, wf, q_in, q_out, q_tmp, dt, h, stagger])
    wp.launch(_advect_mc_3, dim = q_in.shape, inputs = [uf, vf, wf, q_in, q_out, q_tmp, dt, h, stagger])

def advect_uv_mc(u_in, v_in, w_in, u_out, v_out, w_out, dt = dt, h = h0):
    u_tmp = wp.empty_like(u_in)
    v_tmp = wp.empty_like(v_in)
    w_tmp = wp.empty_like(w_in)

    ust = wp.vec3(0.0, 0.5, 0.5)
    vst = wp.vec3(0.5, 0.0, 0.5)
    wst = wp.vec3(0.5, 0.5, 0.0)

    wp.launch(_advect_mc_1, dim = u_in.shape, inputs = [u_in, v_in, w_in, u_in, u_out, u_tmp, dt, h, ust])
    wp.launch(_advect_mc_2, dim = u_in.shape, inputs = [u_in, v_in, w_in, u_in, u_out, u_tmp, dt, h, ust])
    wp.launch(_advect_mc_3, dim = u_in.shape, inputs = [u_in, v_in, w_in, u_in, u_out, u_tmp, dt, h, ust])


    wp.launch(_advect_mc_1, dim = v_in.shape, inputs = [u_in, v_in, w_in, v_in, v_out, v_tmp, dt, h, vst])
    wp.launch(_advect_mc_2, dim = v_in.shape, inputs = [u_in, v_in, w_in, v_in, v_out, v_tmp, dt, h, vst])
    wp.launch(_advect_mc_3, dim = v_in.shape, inputs = [u_in, v_in, w_in, v_in, v_out, v_tmp, dt, h, vst])

    wp.launch(_advect_mc_1, dim = w_in.shape, inputs = [u_in, v_in, w_in, w_in, w_out, w_tmp, dt, h, wst])
    wp.launch(_advect_mc_2, dim = w_in.shape, inputs = [u_in, v_in, w_in, w_in, w_out, w_tmp, dt, h, wst])
    wp.launch(_advect_mc_3, dim = w_in.shape, inputs = [u_in, v_in, w_in, w_in, w_out, w_tmp, dt, h, wst])

    
    

@wp.func
def is_interior(
    p: wp.vec3,
    q: wp.array3d(dtype = float)
):
    if p[0] < 0.5 or p[0] > float(q.shape[0]) - 0.5 or p[1] < 0.5 or p[1] > float(q.shape[1]) - 0.5 or p[2] < 0.5 or p[2] > float(q.shape[2]) - 0.5:
        return False
    else:
        return True

@wp.func
def is_interior_uvw(
    p: wp.vec3,
    q: int
):
    if p[0] < 1.0 or p[0] > float(q) - 1.0 or p[1] < 1.0 or p[1] > float(q) - 1.0 or p[2] < 1.0 or p[2] > float(q) - 1.0:
        return False
    else:
        return True

