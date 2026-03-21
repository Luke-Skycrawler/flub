from const_param import dt, h0, shape, shapeu, shapev, shapew
import warp as wp
from utils.util import sample_velocity_3d, sample_3d, trilerp_stagger, trilerp

@wp.struct 
class AdvectCommon:
    uf: wp.array3d(dtype=float)
    vf: wp.array3d(dtype=float)
    wf: wp.array3d(dtype=float)
    dt: float
    h: float

@wp.struct 
class AdvectFieldSpec:
    qf: wp.array3d(dtype=float)
    new_qf: wp.array3d(dtype=float)
    stagger: wp.vec3

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
    args_common: AdvectCommon,
    args_spec: AdvectFieldSpec,
    q_tmp: wp.array3d(dtype = float)
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + args_spec.stagger
    p1 = backtrace(args_common.uf, args_common.vf, args_common.wf, p, args_common.dt, args_common.h)
    args_spec.new_qf[i, j, k] = trilerp_stagger(args_spec.qf, p1, args_spec.stagger)

@wp.kernel
def _advect_mc_2(
    args_common: AdvectCommon,
    args_spec: AdvectFieldSpec,
    q_tmp: wp.array3d(dtype = float)
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + args_spec.stagger
    
    p2 = backtrace(args_common.uf, args_common.vf, args_common.wf, p, -args_common.dt, args_common.h)
    q_tmp[i, j, k] = trilerp_stagger(args_spec.new_qf, p2, args_spec.stagger)
    

@wp.kernel
def _advect_mc_3(
    args_common: AdvectCommon,
    args_spec: AdvectFieldSpec,
    q_tmp: wp.array3d(dtype = float),
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k)) + args_spec.stagger

    p3 = backtrace(args_common.uf, args_common.vf, args_common.wf, p, args_common.dt, args_common.h)
    q_sl = trilerp_stagger(args_spec.qf, p3, args_spec.stagger)
    
    args_spec.new_qf[i, j, k] = args_spec.new_qf[i, j, k] + 0.5 * (args_spec.qf[i, j, k] - q_tmp[i, j, k])

    min_val, max_val = sample_minmax(args_spec.qf, p3, args_common.h, args_spec.stagger)
    cond = min_val < args_spec.new_qf[i, j, k] and args_spec.new_qf[i, j, k] < max_val
    if not cond:
        args_spec.new_qf[i, j, k] = q_sl

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

def advect_mc(args_common: AdvectCommon, args_spec: AdvectFieldSpec):
    q_tmp = wp.empty_like(args_spec.qf)
    wp.launch(_advect_mc_1, dim = args_spec.qf.shape, inputs = [args_common, args_spec, q_tmp])
    wp.launch(_advect_mc_2, dim = args_spec.qf.shape, inputs = [args_common, args_spec, q_tmp])
    wp.launch(_advect_mc_3, dim = args_spec.qf.shape, inputs = [args_common, args_spec, q_tmp])

def advect_uv_mc(u_in, v_in, w_in, u_out, v_out, w_out, dt = dt, h = h0):
    common = AdvectCommon()
    common.uf = u_in
    common.vf = v_in
    common.wf = w_in
    common.dt = dt
    common.h = h
    

    ust = wp.vec3(0.0, 0.5, 0.5)
    vst = wp.vec3(0.5, 0.0, 0.5)
    wst = wp.vec3(0.5, 0.5, 0.0)

    specu = AdvectFieldSpec()
    specu.qf = u_in
    specu.new_qf = u_out
    specu.stagger = ust
    
    specv = AdvectFieldSpec()
    specv.qf = v_in
    specv.new_qf = v_out
    specv.stagger = vst

    specw = AdvectFieldSpec()
    specw.qf = w_in
    specw.new_qf = w_out
    specw.stagger = wst

    advect_mc(common, specu)
    advect_mc(common, specv)
    advect_mc(common, specw)


def advect_3d(uf, vf, wf, qf, new_uf, new_vf, new_wf, new_qf, h = h0):    
    midst = wp.vec3(0.5, 0.5, 0.5)
    common = AdvectCommon()
    common.uf = uf
    common.vf = vf
    common.wf = wf
    common.dt = dt
    common.h = h

    spec = AdvectFieldSpec()
    spec.qf = qf
    spec.new_qf = new_qf
    spec.stagger = midst

    advect_mc(common, spec)
    advect_uv_mc(uf, vf, wf, new_uf, new_vf, new_wf, dt, h)
