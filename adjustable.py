import warp as wp
from const_param import gravity
from geometry import _sdf_cube
from const_param import Field, MACGrid, h0
@wp.kernel
def integrate_3d(
    vel: MACGrid,
    rho: Field,
    dt: float):
    i, j, k = wp.tid()

    # gravity
    # f_g = wp.vec2(-0.98, 0.0) * rho[i, j]
    f_g = wp.vec3(0.0, -gravity, 0.0) * 1.0# * rho[i, j, k]

    # # integrate
    if j != 0 and j != vel.v.f.shape[1] - 1:
        vel.v.f[i, j, k] = vel.v.f[i, j, k] + dt * f_g[1]
    # if j == v.shape[1] - 2:
    #     v[i, j + 1, k] = v[i, j + 1, k] + dt * f_g[1]
    
    # fade
    # rho[i, j] = rho[i, j] * (1.0 - 0.1 * dt)
    rho.f[i, j, k] *= (1.0 - 0.3 * dt)


@wp.kernel
def set_phi_3d(
    phi: Field,
    wl: float,
):
    i, j, k = wp.tid()

    p = wp.vec3((float(i) + 0.5) * h0, (float(j) + 0.5) * h0, (float(k) + 0.5) * h0)

    r_and_center = wp.vec3(0.25, wl / 2.0, 0.5)
    p -= r_and_center
    phi.f[i, j, k] = _sdf_cube(p, r_and_center, wp.quat_identity()) - 0.01


@wp.kernel
def init_3d(
    rho: Field,
    vel: MACGrid,
    dir: wp.vec3,
    src_pos: wp.vec3):

    i, j, k = wp.tid()
    pos = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * h0

    d = wp.length(pos - src_pos)

    if d < 0.05:
        rho.f[i, j, k] = 1.0
        vel.u.f[i, j, k] = dir[0]
        vel.v.f[i, j, k] = dir[1]
        vel.w.f[i, j, k] = dir[2]

