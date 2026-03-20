from .const_param import *
from .geometry import _sdf_cube
@wp.kernel
def integrate_3d(
    v: wp.array3d(dtype=float), 
    rho: wp.array3d(dtype=float), 
    dt: float):
    i, j, k = wp.tid()

    # gravity
    # f_g = wp.vec2(-0.98, 0.0) * rho[i, j]
    f_g = wp.vec3(0.0, -gravity, 0.0) * 1.0# * rho[i, j, k]

    # # integrate
    if j != 0 and j != v.shape[1] - 1:
        v[i, j, k] = v[i, j, k] + dt * f_g[1]
    # if j == v.shape[1] - 2:
    #     v[i, j + 1, k] = v[i, j + 1, k] + dt * f_g[1]
    
    # fade
    # rho[i, j] = rho[i, j] * (1.0 - 0.1 * dt)
    rho[i, j, k] = rho[i, j, k] * (1.0 - 0.3 * dt)


@wp.kernel
def set_phi_3d(
    phi: wp.array3d(dtype=float),
    wl: float,
    l: float, b: float, f: float, h: float
):
    i, j, k = wp.tid()

    p = wp.vec3((float(i) + 0.5) * h, (float(j) + 0.5) * h, (float(k) + 0.5) * h) + wp.vec3(l, b, f)

    r_and_center = wp.vec3(0.5, wl / 2.0, 0.5)
    p -= r_and_center
    phi[i, j, k] = _sdf_cube(p, r_and_center, wp.quat_identity()) - 0.01

@wp.func
def should_emit(d: float) -> bool:
    return True
    d = d * 512.0
    ret = False
    if int(wp.rint(d)) % 5 < 3:
        ret = True
    return ret

@wp.kernel
def init_3d(
    rho: wp.array3d(dtype=float), 
    u: wp.array3d(dtype=float), 
    v: wp.array3d(dtype = float), 
    w: wp.array3d(dtype=float),
    radius: int, 
    dir: wp.vec3,
    src_pos: wp.vec3,
    h: float,
    l: float, b: float, f: float):
    i, j, k = wp.tid()

    # d = wp.length(wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * h + offset - src_pos)

    pos = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * h + wp.vec3(l, b, f)

    d = wp.length(pos - src_pos)

    if d < 0.05:
        rho[i, j, k] = 1.0
        u[i, j, k] = dir[0]
        v[i, j, k] = dir[1]
        w[i, j, k] = dir[2]

