from ..const_param import *
import numpy as np
# enforcing no-through, free-slip boundary condition
class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

@wp.kernel
def mean(
    sum: wp.array(dtype = float),
    qf: wp.array2d(dtype = float)):

    i, j = wp.tid()
    wp.atomic_add(sum, 0, qf[i, j])

@wp.kernel
def min_kernel(
    q: wp.array3d(dtype = float),
    q1: wp.array3d(dtype = float)):
    i, j, k = wp.tid()
    q[i, j, k] = wp.min(q[i, j, k], q1[i, j, k])

@wp.kernel
def subtract_mean(
    qf: wp.array2d(dtype = float),
    sum: wp.array(dtype = float)
):
    n_avg = float(qf.shape[0] * qf.shape[1])
    
    i, j = wp.tid()
    qf[i, j] = qf[i, j] - sum[0] / n_avg
    

@wp.kernel
def divide_if_not_zero(
    qf: wp.array3d(dtype = float),
    lam: wp.array3d(dtype = float)
):
    i, j, k = wp.tid()
    if lam[i, j, k] != 0.0:
        qf[i, j, k] = qf[i, j, k] / lam[i, j, k]


# from warp.utils import array_sum

def zero_centered(div):
    # array_sum()
    
    # divnp = div.numpy()
    # divnp -= np.mean(divnp)
    # div = wp.from_numpy(divnp, dtype = float, shape = divnp.shape)
    pass

@wp.func
def sample_2d(qf: wp.array2d(dtype=float), u: int, v: int):
    u = wp.clamp(u, 0, gridx - 1)
    v = wp.clamp(v, 0, gridy - 1)
    return qf[u, v]

@wp.func 
def sample_3d(qf: wp.array3d(dtype = float), u: int, v: int, w: int):
    gx = qf.shape[0]
    gy = qf.shape[1]
    gz = qf.shape[2]
    u = wp.clamp(u, 0, gx - 1)
    v = wp.clamp(v, 0, gy - 1)
    w = wp.clamp(w, 0, gz - 1)
    return qf[u, v, w]

@wp.func 
def sample_3di(qf: wp.array3d(dtype = int), u: int, v: int, w: int):
    gx = qf.shape[0]
    gy = qf.shape[1]
    gz = qf.shape[2]
    u = wp.clamp(u, 0, gx - 1)
    v = wp.clamp(v, 0, gy - 1)
    w = wp.clamp(w, 0, gz - 1)
    return qf[u, v, w]

sample = sample_2d if dimension == 2 else sample_3d

@wp.func
def sample_u(uf: wp.array2d(dtype=float), u: int, v: int):
    u = wp.clamp(u, 0, gridx)
    v = wp.clamp(v, 0, gridy - 1)
    return uf[u, v]


@wp.func
def sample_v(vf: wp.array2d(dtype=float), u: int, v: int):
    u = wp.clamp(u, 0, gridx - 1)
    v = wp.clamp(v, 0, gridy)
    return vf[u, v]


@wp.func
def sample_velocity_2d(uf: wp.array2d(dtype=float), vf: wp.array2d(dtype=float), p: wp.vec2):
    # p is the unbiased position
    
    return wp.vec2(bilerpu(uf, p), bilerpv(vf, p))
    # (u, v) = p[0], p[1]
    # _iu, _iv = wp.floor(u), wp.floor(v)
    # fu, fv = u - _iu, v - _iv

    # s, t = u - 0.5, v - 0.5
    # _iu_shift, _iv_shift = wp.floor(s), wp.floor(t)
    # fs, ft = s - _iu_shift, t - _iv_shift

    # iu, iv, iu_shift, iv_shift = int(_iu), int(_iv), int(_iu_shift), int(_iv_shift)

    # a = sample_u(uf, iu, iv_shift)
    # b = sample_u(uf, iu + 1, iv_shift)
    # c = sample_u(uf, iu, iv_shift + 1)
    # d = sample_u(uf, iu + 1, iv_shift + 1)
    # vx = wp.lerp(wp.lerp(a, b, fu), wp.lerp(c, d, fu), ft)

    # a = sample_v(vf, iu_shift, iv)
    # b = sample_v(vf, iu_shift + 1, iv)
    # c = sample_v(vf, iu_shift, iv + 1)
    # d = sample_v(vf, iu_shift + 1, iv + 1)
    # vy = wp.lerp(wp.lerp(a, b, fs), wp.lerp(c, d, fs), fv)

    # return wp.vec2(vx, vy)

@wp.func
def sample_velocity_3d(
    u_staggered: wp.array3d(dtype = float), 
    v_staggered: wp.array3d(dtype = float), 
    w_staggered: wp.array3d(dtype = float), 
    p: wp.vec3):
    '''trilinear interpolation of velocity at unbiased position p; unit in [0, 256]'''
    (u, v, w) = (p[0], p[1], p[2])
    iu, iv, iw = int(wp.floor(u)), int(wp.floor(v)), int(wp.floor(w))
    fu, fv, fw = u - float(iu), v - float(iv), w - float(iw)

    s, t, r = u - 0.5, v - 0.5, w - 0.5
    iu_shift, iv_shift, iw_shift = int(wp.floor(s)), int(wp.floor(t)), int(wp.floor(r))
    fs, ft, fr = s - float(iu_shift), t - float(iv_shift), r - float(iw_shift)

    a = sample_3d(u_staggered, iu, iv_shift, iw_shift)
    b = sample_3d(u_staggered, iu + 1, iv_shift, iw_shift)
    c = sample_3d(u_staggered, iu, iv_shift + 1, iw_shift)
    d = sample_3d(u_staggered, iu + 1, iv_shift + 1, iw_shift)
    e = sample_3d(u_staggered, iu, iv_shift, iw_shift + 1)
    f = sample_3d(u_staggered, iu + 1, iv_shift, iw_shift + 1)
    g = sample_3d(u_staggered, iu, iv_shift + 1, iw_shift + 1)
    h = sample_3d(u_staggered, iu + 1, iv_shift + 1, iw_shift + 1)


    # vx = lerp(lerp(a, b, fu), lerp(c, d, fu), ft)
    vx = wp.lerp(wp.lerp(wp.lerp(a, b, fu), wp.lerp(c, d, fu), ft), wp.lerp(wp.lerp(e, f, fu), wp.lerp(g, h, fu), ft), fr)

    # a = sample_v(v_staggered, iu_shift, iv)
    # b = sample_v(v_staggered, iu_shift + 1, iv)
    # c = sample_v(v_staggered, iu_shift, iv + 1)
    # d = sample_v(v_staggered, iu_shift + 1, iv + 1)
    # vy = lerp(lerp(a, b, fs), lerp(c, d, fs), fv)

    a = sample_3d(v_staggered, iu_shift, iv, iw_shift)
    b = sample_3d(v_staggered, iu_shift + 1, iv, iw_shift)
    c = sample_3d(v_staggered, iu_shift, iv + 1, iw_shift)
    d = sample_3d(v_staggered, iu_shift + 1, iv + 1, iw_shift)
    e = sample_3d(v_staggered, iu_shift, iv, iw_shift + 1)
    f = sample_3d(v_staggered, iu_shift + 1, iv, iw_shift + 1)
    g = sample_3d(v_staggered, iu_shift, iv + 1, iw_shift + 1)
    h = sample_3d(v_staggered, iu_shift + 1, iv + 1, iw_shift + 1)
    vy = wp.wp.lerp(wp.wp.lerp(wp.wp.lerp(a, b, fs), wp.wp.lerp(c, d, fs), fv), wp.wp.lerp(wp.wp.lerp(e, f, fs), wp.wp.lerp(g, h, fs), fv), fr)

    a = sample_3d(w_staggered, iu_shift, iv_shift, iw)
    b = sample_3d(w_staggered, iu_shift + 1, iv_shift, iw)
    c = sample_3d(w_staggered, iu_shift, iv_shift + 1, iw)
    d = sample_3d(w_staggered, iu_shift + 1, iv_shift + 1, iw)
    e = sample_3d(w_staggered, iu_shift, iv_shift, iw + 1)
    f = sample_3d(w_staggered, iu_shift + 1, iv_shift, iw + 1)
    g = sample_3d(w_staggered, iu_shift, iv_shift + 1, iw + 1)
    h = sample_3d(w_staggered, iu_shift + 1, iv_shift + 1, iw + 1)

    vz = wp.lerp(wp.lerp(wp.lerp(a, b, fs), wp.lerp(c, d, fs), ft), wp.lerp(wp.lerp(e, f, fs), wp.lerp(g, h, fs), ft), fw)
    return wp.vec3(vx, vy, vz)


@wp.func
def backtrace(uf: wp.array2d(dtype=float), vf: wp.array2d(dtype=float), p: wp.vec2, dt: float, h: float):
    v1 = sample_velocity_2d(uf, vf, p)
    p1 = p - 0.5 * dt / h * v1
    v2 = sample_velocity_2d(uf, vf, p1)
    p2 = p - 0.75 * dt / h * v2
    v3 = sample_velocity_2d(uf, vf, p2)
    p -= dt / h * ((2.0 / 9.0) * v1 + (1.0 / 3.0) * v2 + (4.0 / 9.0) * v3)
    return p

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


@wp.func
def bilerp(qf: wp.array2d(dtype=float), p: wp.vec2):
    # takes unbiased position
    (u, v) = (p[0], p[1])
    s, t = u - 0.5, v - 0.5
    iu, iv = int(wp.floor(s)), int(wp.floor(t))
    fu, fv = s - float(iu), t - float(iv)
    a = sample_2d(qf, iu, iv)
    b = sample_2d(qf, iu + 1, iv)
    c = sample_2d(qf, iu, iv + 1)
    d = sample_2d(qf, iu + 1, iv + 1)
    return wp.lerp(wp.lerp(a, b, fu), wp.lerp(c, d, fu), fv)


@wp.func
def trilerp(qf: wp.array3d(dtype=float), p: wp.vec3):
    '''trilinear interpolation; takes unbiased position'''
    (u, v, w) = (p[0], p[1], p[2])
    s, t, r = u - 0.5, v - 0.5, w - 0.5
    return _trilerp(qf, s, t, r)

@wp.func
def trilerp_stagger(qf: wp.array3d(dtype=float), p: wp.vec3, stagger: wp.vec3):
    '''trilinear interpolation; takes unbiased position'''
    (u, v, w) = (p[0], p[1], p[2])
    s, t, r = u - stagger[0], v - stagger[1], w - stagger[2]
    # return _trilerp(qf, s, t, r)
    return _tricubic(qf, s, t, r)

@wp.func
def _trilerp(qf: wp.array3d(dtype = float), s: float, t: float, r: float):
    # floor
    iu, iv, iw = int(wp.floor(s)), int(wp.floor(t)), int(wp.floor(r))
    # fract
    fu, fv, fw = s - float(iu), t - float(iv), r - float(iw)
    a = sample_3d(qf, iu, iv, iw)
    b = sample_3d(qf, iu + 1, iv, iw)
    c = sample_3d(qf, iu, iv + 1, iw)
    d = sample_3d(qf, iu + 1, iv + 1, iw)
    e = sample_3d(qf, iu, iv, iw + 1)
    f = sample_3d(qf, iu + 1, iv, iw + 1)
    g = sample_3d(qf, iu, iv + 1, iw + 1)
    h = sample_3d(qf, iu + 1, iv + 1, iw + 1)
    return wp.lerp(wp.lerp(wp.lerp(a, b, fu), wp.lerp(c, d, fu), fv), wp.lerp(wp.lerp(e, f, fu), wp.lerp(g, h, fu), fv), fw)

@wp.func
def cubic(a: float, b: float, c: float, d: float, s: float):
    term0 = -1.0 / 3.0 * s + 0.5 * s * s - 1.0 / 6.0 * s * s * s
    term1 =  1.0 - s * s + 0.5 * (s * s * s - s)
    term2 = s + 0.5 * (s * s - s * s * s)
    term3 = 1.0 / 6.0 * (s * s * s - s)

    return wp.dot(wp.vec4(a, b, c, d), wp.vec4(term0, term1, term2, term3))

@wp.func
def _cubic(qf: wp.array3d(dtype = float), iu: int, iv: int, iw: int, fu: float):
    q0 = sample_3d(qf, iu - 1, iv, iw)
    q1 = sample_3d(qf, iu, iv, iw)
    q2 = sample_3d(qf, iu + 1, iv, iw)
    q3 = sample_3d(qf, iu + 2, iv, iw)
    return cubic(q0, q1, q2, q3, fu)

@wp.func
def _bicubic(qf: wp.array3d(dtype = float), iu: int, iv: int, iw: int, fu: float, fv: float):
    q_00 = _cubic(qf, iu, iv - 1, iw, fu)
    q_10 = _cubic(qf, iu, iv, iw, fu)
    q_20 = _cubic(qf, iu, iv + 1, iw, fu)
    q_30 = _cubic(qf, iu, iv + 2, iw, fu)

    return cubic(q_00, q_10, q_20, q_30, fv)
    
@wp.func
def _tricubic(qf: wp.array3d(dtype =float), s: float, t: float, r: float):
    '''tri-cubic interpolation'''
    iu, iv, iw = int(wp.floor(s)), int(wp.floor(t)), int(wp.floor(r))
    fu, fv, fw = s - float(iu), t - float(iv), r - float(iw)

    q__0 = _bicubic(qf, iu, iv, iw - 1, fu, fv)
    q__1 = _bicubic(qf, iu, iv, iw, fu, fv)
    q__2 = _bicubic(qf, iu, iv, iw + 1, fu, fv)
    q__3 = _bicubic(qf, iu, iv, iw + 2, fu, fv)

    return cubic(q__0, q__1, q__2, q__3, fw)
    
    
@wp.func
def bilerpu(vf: wp.array2d(dtype=float), p: wp.vec2):
    (u, v) = (p[0], p[1])
    s, t = u, v - 0.5
    # floor
    iu, iv = int(wp.floor(s)), int(wp.floor(t))
    # fract
    fu, fv = s - float(iu), t - float(iv)
    a = sample_u(vf, iu, iv)
    b = sample_u(vf, iu + 1, iv)
    c = sample_u(vf, iu, iv + 1)
    d = sample_u(vf, iu + 1, iv + 1)
    return wp.lerp(wp.lerp(a, b, fu), wp.lerp(c, d, fu), fv)

@wp.func
def trilerpu(
    uf: wp.array3d(dtype=float), 
    p: wp.vec3):
    (u, v, w) = (p[0], p[1], p[2])
    s, t, r = u, v - 0.5, w - 0.5
    return _trilerp(uf, s, t, r)

@wp.func
def trilerpv(
    vf: wp.array3d(dtype=float), 
    p: wp.vec3):
    (u, v, w) = (p[0], p[1], p[2])
    s, t, r = u - 0.5, v, w - 0.5
    return _trilerp(vf, s, t, r)

@wp.func
def trilerpw(
    wf: wp.array3d(dtype=float), 
    p: wp.vec3):
    (u, v, w) = (p[0], p[1], p[2])
    s, t, r = u - 0.5, v - 0.5, w
    return _trilerp(wf, s, t, r)
    
@wp.func
def bilerpv(vf: wp.array2d(dtype = float), p: wp.vec2):
    (u, v) = (p[0], p[1])
    s, t = u - 0.5, v
    # floor
    iu, iv = int(wp.floor(s)), int(wp.floor(t))
    # fract
    fu, fv = s - float(iu), t - float(iv)
    a = sample_v(vf, iu, iv)
    b = sample_v(vf, iu + 1, iv)
    c = sample_v(vf, iu, iv + 1)
    d = sample_v(vf, iu + 1, iv + 1)
    return wp.lerp(wp.lerp(a, b, fu), wp.lerp(c, d, fu), fv)

@wp.kernel
def scale_by(qf: wp.array2d(dtype=float), scale: float):
    i, j = wp.tid()
    qf[i, j] = qf[i, j] * scale

@wp.kernel
def scale_by_3d(qf: wp.array3d(dtype=float), scale: float):
    i, j, k = wp.tid()
    qf[i, j, k] = qf[i, j, k] * scale


@wp.func
def fraction_edge(yl: float, yr: float):
    """ 
    Fraction of the edge with sdf > 0
    """
    # tie breaker: if sdf(edge) = 0, categorize it to solid (F = 0)
    ret = 0.0
    if yl > 0 and yr > 0:
        ret = 1.0
    elif yl > 0:
        ret = yl / (yl - yr)
    elif yr > 0:
        ret = yr / (yr - yl)
    else:
        ret = 0.0
    if ret < 0.1:
        ret = 0.0
    elif ret > 1.0:
        ret = 1.0
    return ret

@wp.func
def fraction_triangle(ya: float, yb: float, yc: float):
    """ Fraction of the triangle with sdf > 0"""
    # tie breaker: if sdf(a, b, c) = 0, categorize it to solid (F = 0)
    ret = 0.0
    fab = fraction_edge(ya, yb)
    fbc = fraction_edge(yb, yc)
    fca = fraction_edge(yc, ya)

    if ya > 0 and yb > 0 and yc > 0:
        ret = 1.0
    elif ya > 0 and yb > 0:
        # yc <= 0
        ret = 1.0 - (1.0 - fca) * (1.0 - fbc)
    elif ya > 0 and yc > 0:
        # yb <= 0
        ret = 1.0 - (1.0 - fab) * (1.0 - fbc)
    elif yb > 0 and yc > 0:
        # ya <= 0
        ret = 1.0 - (1.0 - fab) * (1.0 - fca)
    elif ya > 0:
        ret = fca * fab
    elif yb > 0:
        ret = fab * fbc
    elif yc > 0:
        ret = fbc * fca
    else:
        ret = 0.0
    ret = wp.clamp(ret, 0.0, 1.0)
    return ret

@wp.func 
def fraction_quad(ya: float, yb: float, yc: float, yd: float, y_center: float):
    frac = fraction_triangle(ya, yb, y_center) + fraction_triangle(yb, yc, y_center) + fraction_triangle(yc, yd, y_center) + fraction_triangle(yd, ya, y_center)
    frac *= 0.25
    if frac < 0.1:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    return frac

@wp.func
def compute_v_solid_2d(uv: wp.vec2, x: wp.vec2, v: wp.vec2, omega: float):
    # takes in unbiased position
    p = uv - x
    ret = v + omega * wp.vec2(-p[1], p[0])
    return ret

@wp.func
def compute_v_solid_3d(uvw: wp.vec3, x: wp.vec3, v: wp.vec3, omega: wp.vec3):
    p = uvw - x
    ret = v + wp.cross(omega, p)
    return ret

compute_v_solid = compute_v_solid_2d if dimension == 2 else compute_v_solid_3d

@wp.func
def is_cutcell_2d(i: int, j: int, fu: wp.array2d(dtype = float), fv: wp.array2d(dtype = float)):
    l = sample_u(fu, i, j)
    r = sample_u(fu, i + 1, j)
    b = sample_v(fv, i, j)
    t = sample_v(fv, i, j + 1)
    ret = l + r + b + t < 4.0
    return ret

@wp.func
def is_cutcell_3d(
    i: int, j: int, k: int, 
    fu: wp.array3d(dtype = float),
    fv: wp.array3d(dtype = float),
    fw: wp.array3d(dtype = float)):
    l = fu[ i, j, k]
    r = fu[ i + 1, j, k]
    b = fv[ i, j, k]
    t = fv[ i, j + 1, k]
    f = fw[ i, j, k]
    n = fw[ i, j, k + 1]
    ret = l + r + b + t + f + n < 6.0
    return ret


# @wp.kernel    
# def set_boundary_fufv(
#     fu: wp.array3d(dtype = float), 
#     fv: wp.array3d(dtype = float), 
#     fw: wp.array3d(dtype = float),
#     val: float, 
#     l: float, b: float, f: float, h: float):
#     i, j = wp.tid()

#     gridx = fu.shape[0] - 1
#     gridy = fv.shape[1] - 1
#     gridz = fw.shape[2] - 1

#     if l < 0.01 * h0 and i < gridy and j < gridz:
#         fu[0, i, j] = val 
#     if l + float(gridx) * h > 1.0 - 0.01 * h0:
#         fu[gridx, i, j] = val


#     if b < 0.01 * h0 and i < gridx and j < gridz:
#         fv[i, 0, j] = val
#     if b + float(gridy) * h > 1.0 - 0.01 * h0:
#         fv[i, gridy, j] = val
    
#     if f < 0.01 * h0 and i < gridx and j < gridy:
#         fw[i, j, 0] = val
#     if f + float(gridz) * h > 1.0 - 0.01 * h0:
#         fw[i, j, gridz] = val

#     # fixme: upper bounday no longer 1.0
#     # buggy, use the one in geometry_derived




@wp.kernel
def divide(u1: wp.array3d(dtype = float), u2: wp.array3d(dtype = float)):
    i, j, k = wp.tid()
    if u2[i, j, k] != 0.0:
        u1[i, j, k] = u1[i, j, k] / u2[i, j, k]
    else :
        u1[i, j, k] = 0.0

@wp.func
def valid_uface(iu: int, iv: int):
    return 0 <= iu and iu <= res and 0 <= iv and iv < res

@wp.func
def valid_uface(iu: int, iv: int, iw: int):
    return 0 <= iu and iu <= res and 0 <= iv and iv < res and 0 <= iw and iw < res

@wp.func
def valid_vface(iu: int, iv: int):
    return 0 <= iu and iu < res and 0 <= iv and iv <= res

@wp.func
def valid_vface(iu: int, iv: int, iw: int):
    return 0 <= iu and iu < res and 0 <= iv and iv <= res and 0 <= iw and iw < res

@wp.func
def valid_wface(iu: int, iv: int, iw: int):
    return 0 <= iu and iu < res and 0 <= iv and iv < res and 0 <= iw and iw <= res


@wp.func
def inv_bilerpu(uf: wp.array2d(dtype = float), p: wp.vec2, q: float):
    u, v = p[0], p[1]
    s, t = u, v - 0.5
    # floor
    iu, iv = int(wp.floor(s)), int(wp.floor(t))
    # fract
    fu, fv = s - wp.floor(s), t - wp.floor(t)

    a_area = (1.0 - fu) * (1.0 - fv)
    b_area = fu * (1.0 - fv)
    c_area = (1.0 - fu) * fv
    d_area = fu * fv
    
    if valid_uface(iu, iv):
        wp.atomic_add(uf, iu, iv, a_area * q)
    if valid_uface(iu + 1, iv):
        wp.atomic_add(uf, iu + 1, iv, b_area * q)
        # uf[iu + 1, iv] += b_area * q
    if valid_uface(iu, iv + 1):
        wp.atomic_add(uf, iu, iv + 1, c_area * q)
        # uf[iu, iv + 1] += c_area * q
    if valid_uface(iu + 1, iv + 1):
        wp.atomic_add(uf, iu + 1, iv + 1, d_area * q)
        # uf[iu + 1, iv + 1] += d_area * q

@wp.func
def inv_bilerpu(uf: wp.array3d(dtype = float), p: wp.vec3, q: float):
    u, v, w = p[0], p[1], p[2]
    s, t, r = u, v - 0.5, w - 0.5
    # floor
    iu, iv, iw = int(wp.floor(s)), int(wp.floor(t)), int(wp.floor(r))
    # fract
    fu, fv, fw = s - wp.floor(s), t - wp.floor(t), r - wp.floor(r)

    a_area = (1.0 - fu) * (1.0 - fv) * (1.0 - fw)
    b_area = fu * (1.0 - fv) * (1.0 - fw)
    c_area = (1.0 - fu) * fv * (1.0 - fw)
    d_area = fu * fv * (1.0 - fw)
    
    e_area = (1.0 - fu) * (1.0 - fv) * fw
    f_area = fu * (1.0 - fv) * fw
    g_area = (1.0 - fu) * fv * fw
    h_area = fu * fv * fw

    if valid_uface(iu, iv, iw):
        wp.atomic_add(uf, iu, iv, iw, a_area * q)
    if valid_uface(iu + 1, iv, iw):
        wp.atomic_add(uf, iu + 1, iv, iw, b_area * q)
    if valid_uface(iu, iv + 1, iw):
        wp.atomic_add(uf, iu, iv + 1, iw, c_area * q)
    if valid_uface(iu + 1, iv + 1, iw):
        wp.atomic_add(uf, iu + 1, iv + 1, iw, d_area * q)

    if valid_uface(iu, iv, iw + 1):
        wp.atomic_add(uf, iu, iv, iw + 1, e_area * q)
    if valid_uface(iu + 1, iv, iw + 1):
        wp.atomic_add(uf, iu + 1, iv, iw + 1, f_area * q)
    if valid_uface(iu, iv + 1, iw + 1):
        wp.atomic_add(uf, iu, iv + 1, iw + 1, g_area * q)
    if valid_uface(iu + 1, iv + 1, iw + 1):
        wp.atomic_add(uf, iu + 1, iv + 1, iw + 1, h_area * q)

@wp.func
def inv_bilerpv(vf: wp.array2d(dtype = float), p: wp.vec2, q: float):
    u, v = p[0], p[1]
    s, t = u - 0.5, v
    # floor
    iu, iv = int(wp.floor(s)), int(wp.floor(t))
    # fract
    fu, fv = s - wp.floor(s), t - wp.floor(t)

    a_area = (1.0 - fu) * (1.0 - fv)
    b_area = fu * (1.0 - fv)
    c_area = (1.0 - fu) * fv
    d_area = fu * fv
    
    if valid_vface(iu, iv):
        wp.atomic_add(vf, iu, iv, a_area * q)
    if valid_vface(iu + 1, iv):
        wp.atomic_add(vf, iu + 1, iv, b_area * q)
        # vf[iu + 1, iv] += b_area * q
    if valid_vface(iu, iv + 1):
        wp.atomic_add(vf, iu, iv + 1, c_area * q)
        # vf[iu, iv + 1] += c_area * q
    if valid_vface(iu + 1, iv + 1):
        wp.atomic_add(vf, iu + 1, iv + 1, d_area * q)
        # vf[iu + 1, iv + 1] += d_area * q

@wp.func
def inv_bilerpv(vf: wp.array3d(dtype = float), p: wp.vec3, q: float):
    u, v, w = p[0], p[1], p[2]
    s, t, r = u - 0.5, v, w - 0.5
    # floor
    iu, iv, iw = int(wp.floor(s)), int(wp.floor(t)), int(wp.floor(w))
    # fract
    fu, fv, fw = s - wp.floor(s), t - wp.floor(t), r - wp.floor(r)

    a_area = (1.0 - fu) * (1.0 - fv) * (1.0 - fw)
    b_area = fu * (1.0 - fv) * (1.0 - fw)
    c_area = (1.0 - fu) * fv * (1.0 - fw)
    d_area = fu * fv * (1.0 - fw)

    e_area = (1.0 - fu) * (1.0 - fv) * fw
    f_area = fu * (1.0 - fv) * fw
    g_area = (1.0 - fu) * fv * fw
    h_area = fu * fv * fw
    
    if valid_vface(iu, iv, iw):
        wp.atomic_add(vf, iu, iv, iw, a_area * q)
    if valid_vface(iu + 1, iv, iw):
        wp.atomic_add(vf, iu + 1, iv, iw, b_area * q)
    if valid_vface(iu, iv + 1, iw):
        wp.atomic_add(vf, iu, iv + 1, iw, c_area * q)
    if valid_vface(iu + 1, iv + 1, iw):
        wp.atomic_add(vf, iu + 1, iv + 1, iw, d_area * q)

    if valid_vface(iu, iv, iw + 1):
        wp.atomic_add(vf, iu, iv, iw + 1, e_area * q)
    if valid_vface(iu + 1, iv, iw + 1):
        wp.atomic_add(vf, iu + 1, iv, iw + 1, f_area * q)
    if valid_vface(iu, iv + 1, iw + 1):
        wp.atomic_add(vf, iu, iv + 1, iw + 1, g_area * q)
    if valid_vface(iu + 1, iv + 1, iw + 1):
        wp.atomic_add(vf, iu + 1, iv + 1, iw + 1, h_area * q)

@wp.func
def inv_bilerpw(wf: wp.array3d(dtype = float), p: wp.vec3, q: float):
    u, v, w = p[0], p[1], p[2]
    s, t, r = u - 0.5, v - 0.5, w
    # floor
    iu, iv, iw = int(wp.floor(s)), int(wp.floor(t)), int(wp.floor(w))
    # fract
    fu, fv, fw = s - wp.floor(s), t - wp.floor(t), r - wp.floor(r)

    a_area = (1.0 - fu) * (1.0 - fv) * (1.0 - fw)
    b_area = fu * (1.0 - fv) * (1.0 - fw)
    c_area = (1.0 - fu) * fv * (1.0 - fw)
    d_area = fu * fv * (1.0 - fw)

    e_area = (1.0 - fu) * (1.0 - fv) * fw
    f_area = fu * (1.0 - fv) * fw
    g_area = (1.0 - fu) * fv * fw
    h_area = fu * fv * fw
    
    if valid_wface(iu, iv, iw):
        wp.atomic_add(wf, iu, iv, iw, a_area * q)
    if valid_wface(iu + 1, iv, iw):
        wp.atomic_add(wf, iu + 1, iv, iw, b_area * q)
    if valid_wface(iu, iv + 1, iw):
        wp.atomic_add(wf, iu, iv + 1, iw, c_area * q)
    if valid_wface(iu + 1, iv + 1, iw):
        wp.atomic_add(wf, iu + 1, iv + 1, iw, d_area * q)

    if valid_wface(iu, iv, iw + 1):
        wp.atomic_add(wf, iu, iv, iw + 1, e_area * q)
    if valid_wface(iu + 1, iv, iw + 1):
        wp.atomic_add(wf, iu + 1, iv, iw + 1, f_area * q)
    if valid_wface(iu, iv + 1, iw + 1):
        wp.atomic_add(wf, iu, iv + 1, iw + 1, g_area * q)
    if valid_wface(iu + 1, iv + 1, iw + 1):
        wp.atomic_add(wf, iu + 1, iv + 1, iw + 1, h_area * q)


@wp.func
def center_in_fluid(phi: wp.array2d(dtype = float), i: int, j: int) -> bool:
    return 0 <= i and i < phi.shape[0] and 0 <= j and j < phi.shape[1] and phi[i, j] < 0.0 

@wp.func
def center_in_fluid_3d(phi: wp.array3d(dtype = float), i: int, j: int, k: int) -> bool:
    return 0 <= i and i < phi.shape[0] and 0 <= j and j < phi.shape[1] and 0 <= k and k < phi.shape[2] and phi[i, j, k] < 0.0 

@wp.func
def is_interior_cell(tag: wp.array2d(dtype = int), i: int, j: int):
    return 0 <= i and i < tag.shape[0] and 0 <= j and j < tag.shape[1] and tag[i, j] != 0

@wp.func
def is_interior_cell_3d(tag: wp.array3d(dtype = int), i: int, j: int, k: int):
    return 0 <= i and i < tag.shape[0] and 0 <= j and j < tag.shape[1] and 0 <= k and k < tag.shape[2] and tag[i, j, k] != 0

@wp.func
def in_bound(p: float, res: int):
    return 0.0 <= p and p < float(res)
@wp.func 
def fraction_of_fluid(phi1: float, phi2: float):
    ret = 0.0
    if phi1 < 0.0 and phi2 < 0.0:
        ret = 1.0
    elif phi1 >= 0.0 and phi2 >= 0.0:
        ret = 0.0
    else:
        ret = 1.0 - wp.max(phi1, phi2) / wp.abs(phi1 - phi2)
    return ret


@wp.kernel
def integrate(u: wp.array2d(dtype=wp.vec2), rho: wp.array2d(dtype=float), dt: float):
    i, j = wp.tid()

    # gravity
    f_g = wp.vec2(-0.98, 0.0) * rho[i, j]

    # integrate
    u[i, j] = u[i, j] + dt * f_g

    # fade
    rho[i, j] = rho[i, j] * (1.0 - 0.1 * dt)

@wp.kernel
def integrate_staggered(u: wp.array2d(dtype=float), v: wp.array2d(dtype = float), rho: wp.array2d(dtype=float), dt: float):
    i, j = wp.tid()

    # gravity
    # f_g = wp.vec2(0.98, 0.0)# * rho[i, j]
    f_g = wp.vec2(0.0, -0.98) * 10.0 #* rho[i, j]
    # f_g = wp.vec2(0.0, 0.0) * rho[i, j]

    # integrate
    u[i, j] = u[i, j] + dt * f_g[0]
    v[i, j] = v[i, j] + dt * f_g[1]

    if i == gridx - 1:
        u[gridx - 1, j] = u[gridx - 1, j] + dt * f_g[0]
    if j == gridy - 1:
        v[i, gridy - 1] = v[i, gridy - 1] + dt * f_g[1]

    # fade
    # rho[i, j] = rho[i, j] * (1.0 - 0.1 * dt)
    rho[i, j] = rho[i, j] * (1.0 - dt)



@wp.kernel
def init(rho: wp.array2d(dtype=float), u: wp.array2d(dtype=wp.vec2), radius: int, dir: wp.vec2):
    i, j = wp.tid()

    d = wp.length(wp.vec2(float(i - gridx / 2), float(j - gridy / 2)))

    if d < radius:
        rho[i, j] = 1.0
        u[i, j] = dir

@wp.kernel
def init_staggered(rho: wp.array2d(dtype=float), u: wp.array2d(dtype=float), v: wp.array2d(dtype = float), radius: float, dir: wp.vec2):
    i, j = wp.tid()

    d = wp.length(wp.vec2(float(i - gridx / 4), float(j - gridy / 2))) * h0

    if d < radius:
        rho[i, j] = 1.0
        u[i, j] = dir[0]
        v[i, j] = dir[1]

@wp.kernel
def process_tag(
    tag: wp.array3d(dtype = int)
):
    i, j, k = wp.tid()
    tag[i, j, k] = wp.select(tag[i, j, k] == 0, 1, 0)


@wp.kernel
def exclude_rigid_phi(
    phi_rigid: wp.array3d(dtype=float),
    phi_water: wp.array3d(dtype=float)
):
    i, j, k = wp.tid()
    if phi_rigid[i, j, k] < 0.0 and phi_water[i, j, k] < 0.0:
        phi_water[i, j, k] = - phi_rigid[i, j, k]

@wp.kernel
def set_unknown_3d(
    phi: wp.array3d(dtype=float),
    u_tag: wp.array3d(dtype=int),
    v_tag: wp.array3d(dtype=int),
    w_tag: wp.array3d(dtype= int)
):
    i, j, k = wp.tid()
    pc = phi[i, j, k]
    if pc >= 0.0:
        u_tag[i, j, k] = 1
        v_tag[i, j, k] = 1
        w_tag[i, j, k] = 1
    else:
        u_tag[i, j, k] = 0
        v_tag[i, j, k] = 0
        w_tag[i, j, k] = 0

