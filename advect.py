from const_param import *
from utils.util import backtrace, bilerp, bilerpu, bilerpv, trilerp, trilerpu, trilerpv, trilerpw, in_bound, sample_3d, trilerp_stagger

@wp.kernel
def advect_simple(
    uf: wp.array2d(dtype=float),
    vf: wp.array2d(dtype=float),
    qf: wp.array2d(dtype=float),
    new_qf: wp.array2d(dtype=float),
    dt: float, h: float
):
    i, j = wp.tid()
    p = wp.vec2(float(i) + 0.5, float(j) + 0.5)
    p = backtrace(uf, vf, p, dt, h)
    new_qf[i, j] = bilerp(qf, p)

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
        # new_phi[i, j, k] = trilerp(phi, p)
        new_phi[i, j, k] = trilerp_stagger(phi, p, stagger)
        # phi_ijk = trilerp(phi, p)
        # new_phi[i, j, k] = wp.select(phi_ijk > 0.0, - h0, h0)
    else: 
        new_phi[i, j, k] = h0

@wp.kernel
def advect_u(
    uf: wp.array2d(dtype=float),
    vf: wp.array2d(dtype=float),
    new_uf: wp.array2d(dtype=float),
    dt: float, h: float
):
    i, j = wp.tid()
    p = wp.vec2(float(i), float(j) + 0.5)
    p = backtrace(uf, vf, p, dt, h)
    if 0 < i < gridx:
        new_uf[i, j] = bilerpu(uf, p)
    else :
        new_uf[i, j] = 0.0
        
@wp.kernel
def advect_u_3d(
    uf: wp.array3d(dtype=float),
    vf: wp.array3d(dtype=float),
    wf: wp.array3d(dtype=float),
    new_uf: wp.array3d(dtype=float),
    dt: float, h: float
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j) + 0.5, float(k) + 0.5)
    p = backtrace(uf, vf, wf, p, dt, h)
    # if 0 < i and i < uf.shape[0] - 1:
    if True:
        new_uf[i, j, k] = trilerpu(uf, p)

    # FIXME: make it verseraltile for patch and global 
    elif h == h0 :
        new_uf[i, j, k] = 0.0

@wp.kernel
def advect_v(
    uf: wp.array2d(dtype=float),
    vf: wp.array2d(dtype=float),
    new_vf: wp.array2d(dtype=float),
    dt: float, h: float
):
    i, j = wp.tid()
    p = wp.vec2(float(i) + 0.5, float(j))
    p = backtrace(uf, vf, p, dt, h)
    if 0 < j < gridy:
        new_vf[i, j] = bilerpv(vf, p)
    else:
        new_vf[i, j] = 0.0

@wp.kernel
def advect_v_3d(
    uf: wp.array3d(dtype=float),
    vf: wp.array3d(dtype=float),
    wf: wp.array3d(dtype=float),
    new_vf: wp.array3d(dtype=float),
    dt: float, h: float
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i) + 0.5, float(j), float(k) + 0.5)
    p = backtrace(uf, vf, wf, p, dt, h)
    # if 0 < j and j < vf.shape[1] - 1:
    if True:
        new_vf[i, j, k] = trilerpv(vf, p)
    elif h == h0:
        new_vf[i, j, k] = 0.0

@wp.kernel
def advect_w_3d(
    uf: wp.array3d(dtype=float),
    vf: wp.array3d(dtype=float),
    wf: wp.array3d(dtype=float),
    new_wf: wp.array3d(dtype=float),
    dt: float, h: float
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k))
    p = backtrace(uf, vf, wf, p, dt, h)
    # if 0 < k and k < wf.shape[2] - 1:
    if True:
        new_wf[i, j, k] = trilerpw(wf, p)
    elif h == h0:
        new_wf[i, j, k] = 0.0


def advect_staggered(uf, vf, qf, new_uf, new_vf, new_qf, h = h0):

    wp.launch(advect_simple, dim = shape, inputs = [uf, vf, qf, new_qf, dt, h])
    wp.launch(advect_u, dim = shapeu, inputs = [uf, vf, new_uf, dt, h])
    wp.launch(advect_v, dim = shapev, inputs = [uf, vf, new_vf, dt, h])

def advect_3d(uf, vf, wf, qf, new_uf, new_vf, new_wf, new_qf, h = h0):
    wp.launch(set_boundary_u, dim = (shapeu[1], shapeu[2]), inputs = [uf, v_inflow])

    # wp.launch(advect_simple_3d, dim = shape, inputs = [uf, vf, wf, qf, new_qf, dt, h])
    # wp.launch(advect_u_3d, dim = shapeu, inputs = [uf, vf, wf, new_uf, dt, h])
    # wp.launch(advect_v_3d, dim = shapev, inputs = [uf, vf, wf, new_vf, dt, h])
    # wp.launch(advect_w_3d, dim = shapew, inputs = [uf, vf, wf, new_wf, dt, h])
    
    advect_mc(qf, new_qf, uf, vf, wf, dt, h)
    advect_uv_mc(uf, vf, wf, new_uf, new_vf, new_wf, dt, h)



    
@wp.kernel
def set_boundary_u(uf: wp.array3d(dtype=float), u_boundary: float):
    i, j = wp.tid()
    gx = uf.shape[0] - 1
    uf[0, i, j] = u_boundary
    uf[gx, i, j] = u_boundary
    

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

