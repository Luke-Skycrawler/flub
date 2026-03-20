import warp as wp
import numpy as np
from const_param import *
from geometry import *
from utils.util import fraction_edge, fraction_quad, trilerp_stagger
from rigid_coupling import RigidCoupling

class PrecomputeFaceFraction:
    def __init__(self, use_mesh = False, vertices = [], edges = []):
        self.vertices = np.array(vertices, dtype = np.float32)
        self.edges = np.array(edges, dtype = int)
        self.use_mesh = use_mesh
    
    def tag_dirichlet(self, tag, phi):
        wp.launch(_tag_dirichlet, dim = phi.shape, inputs = [tag, phi])

    def tag_neumann(self, tag, phi):
        wp.launch(_tag_neumann, dim = phi.shape, inputs = [tag, phi])
    def precompute_tag(self, tag, x, theta, geometry, include_static = 1, lbf = np.zeros(3, np.float32), h = h0):
        wp.launch(_precompute_tag, dim = tag.shape, inputs = [tag, x, theta, geometry, include_static, lbf[0], lbf[1], lbf[2], h])

    def precompute_fij(self, fuf, fvf, x, theta, geometry, owneru, ownerv, x_offset, y_offset, h):
        if self.use_mesh:
            wp.launch(_mesh_fij_u, shape = fuf.shape, inputs = [self.vertices, self.edges, fuf, fvf, x, theta, geometry, owneru, ownerv, x_offset, y_offset, h])
            wp.launch(_mesh_fij_v, shape = fvf.shape, inputs = [self.vertices, self.edges, fuf, fvf, x, theta, geometry, owneru, ownerv, x_offset, y_offset, h])
        
        else:
            wp.launch(_precompute_fij_u, dim = fuf.shape, inputs = [fuf, fvf, x, theta, geometry, owneru, ownerv, x_offset, y_offset, h])
            wp.launch(_precompute_fij_v, dim = fvf.shape, inputs = [fuf, fvf, x, theta, geometry, owneru, ownerv, x_offset, y_offset, h])
        wp.launch(_set_fij_boundary, dim = (1), inputs = [fuf, fvf, x_offset, y_offset, h])
    
    def precompute_phi_with_cached_levelset(self, phi, rc: RigidCoupling, x_offset, y_offset, z_offset, h, owner = None, stagger = wp.vec3(0.0, 0.0, 0.0)):        
        if owner is None:
            owner = wp.zeros((1, 1, 1), dtype = int)

        # handle simple geometry first (boxes, spheres, etc.)
        wp.launch(precompute_grid_point_phi_prims, dim = phi.shape, inputs = [phi, owner, rc.x_pair.cur, rc.theta_pair.cur, rc.rigid_geometry, x_offset, y_offset, z_offset, h, stagger])
        # handle mesh
        xnp = rc.x_pair.cur.numpy()
        qnp = rc.theta_pair.cur.numpy()

        for i, lijk in rc.cached_levelset.items():
            levelset, ijk, h_levelset = lijk
            _x = xnp[i]
            _q = qnp[i]
            x = wp.vec3(_x[0], _x[1], _x[2]) 
            q = wp.quat(_q[0], _q[1], _q[2], _q[3])
            wp.launch(update_argmin, dim = phi.shape, inputs = [phi, owner, levelset, i, x, q, x_offset, y_offset, z_offset, h, h_levelset, stagger, ijk])
    
    def precompute_fij_with_cache(self, fuf, fvf, fwf, owneru, ownerv, ownerw, rc: RigidCoupling, x_offset = 0.0, y_offset = 0.0, z_offset = 0.0, h = h0): 
        ext_shape = (fuf.shape[0], fvf.shape[1], fwf.shape[2])
        phi_tmp = wp.empty(ext_shape, dtype = float)


        staggeru = wp.vec3(0.0, 0.5, 0.5)
        staggerv = wp.vec3(0.5, 0.0, 0.5)
        staggerw = wp.vec3(0.5, 0.5, 0.0)
        
        # grid points 
        self.precompute_phi_with_cached_levelset(phi_tmp, rc, x_offset, y_offset, z_offset, h)
        # u, v, w face center
        self.precompute_phi_with_cached_levelset(fuf, rc, x_offset, y_offset, z_offset, h, owneru, staggeru)
        self.precompute_phi_with_cached_levelset(fvf, rc, x_offset, y_offset, z_offset, h, ownerv, staggerv)
        self.precompute_phi_with_cached_levelset(fwf, rc, x_offset, y_offset, z_offset, h, ownerw, staggerw)


        
        wp.launch(phi_to_fraction, dim = fuf.shape, inputs = [phi_tmp, fuf, 0])
        wp.launch(phi_to_fraction, dim = fvf.shape, inputs = [phi_tmp, fvf, 1])
        wp.launch(phi_to_fraction, dim = fwf.shape, inputs = [phi_tmp, fwf, 2])

    def precompute_fij_3d_deprecated(self, fuf, fvf, fwf, x, q, geometry, owneru, ownerv, ownerw, x_offset, y_offset, z_offset, h, solid_wall_boundary = True, include_static = 1):
        '''
        include static: 0 to only include dynamic objects, 1 for both, 2 for static only
        if solid_wall_boundary is True, it will detect points at actual coarse grid boundary and set fraction to 0.0
            not necessary for a patch call, but won't do much damage if it's set True 
        '''
        ext_shape = (fuf.shape[0], fvf.shape[1], fwf.shape[2])
        
        phi_tmp = wp.empty(ext_shape, dtype = float)
        wp.launch(precompute_grid_point_phi, dim = ext_shape, inputs = [phi_tmp, x, q, geometry, x_offset, y_offset, z_offset, h, include_static])
        wp.launch(_precompute_fij_u_3d_cached, dim = fuf.shape, inputs = [phi_tmp, fuf, x, q, geometry, owneru, x_offset, y_offset, z_offset, h, include_static])
        wp.launch(_precompute_fij_v_3d_cached, dim = fvf.shape, inputs = [phi_tmp, fvf, x, q, geometry, ownerv, x_offset, y_offset, z_offset, h, include_static])
        wp.launch(_precompute_fij_w_3d_cached, dim = fwf.shape, inputs = [phi_tmp, fwf, x, q, geometry, ownerw, x_offset, y_offset, z_offset, h, include_static])
        # wp.launch(_precompute_fij_u_3d, dim = fuf.shape, inputs = [fuf, x, q, geometry, owneru, x_offset, y_offset, z_offset, h, include_static])
        # wp.launch(_precompute_fij_v_3d, dim = fvf.shape, inputs = [fvf, x, q, geometry, ownerv, x_offset, y_offset, z_offset, h, include_static])
        # wp.launch(_precompute_fij_w_3d, dim = fwf.shape, inputs = [fwf, x, q, geometry, ownerw, x_offset, y_offset, z_offset, h, include_static])
        if solid_wall_boundary:
            wp.launch(_set_fij_boundary_3d, dim = (1), inputs = [fuf, fvf, fwf, 0.0, x_offset, y_offset, z_offset, h])


@wp.kernel
def _precompute_fij_u(
    fuf: wp.array2d(dtype = float), fvf: wp.array2d(dtype = float), 
    x: wp.array1d(dtype = wp.vec2), theta: wp.array1d(dtype = float), 
    geometry: wp.array1d(dtype = GeometryType2D), 
    owneru: wp.array2d(dtype = int), ownerv: wp.array2d(dtype = int), 
    x_offset: float, y_offset: float, h: float
):
    '''2d face fraction kernel for u component'''
    # compute the weights on all edges
    i, j = wp.tid()
    (ut, vt) = to01(i, j + 1, x_offset, y_offset, h)
    (ub, vb) = to01(i, j, x_offset, y_offset, h)

    yt, tagt = sdf_argmin(ut, vt, x, theta, geometry)
    yb, tagb = sdf_argmin(ub, vb, x, theta, geometry)

    fuf[i, j] = fraction_edge(yb, yt)
    owneru[i, j] = tagb

@wp.kernel
def _precompute_fij_u_3d(
    fuf: wp.array3d(dtype = float),
    x: wp.array1d(dtype = wp.vec3), q: wp.array1d(dtype = wp.quat), 
    geometry: wp.array1d(dtype = GeometryType3D), 
    owneru: wp.array3d(dtype = int), 
    x_offset: float, y_offset: float, z_offset: float, h: float, include_static: int
):
    '''3d face fraction kernel for u component'''
    # compute the weights on all edges
    i, j, k = wp.tid()
    # wp.func can be overloaded but kernels cannot
    (ua, va, wa) = to01(i, j, k, x_offset, y_offset, z_offset, h)
    (ub, vb, wb) = to01(i, j + 1, k, x_offset, y_offset, z_offset, h)
    (uc, vc, wc) = to01(i, j + 1, k + 1, x_offset, y_offset, z_offset, h)
    (ud, vd, wd) = to01(i, j, k + 1, x_offset, y_offset, z_offset, h)

    (ya, taga) = sdf_argmin_3d(ua, va, wa, x, q, geometry, include_static)
    (yb, tagb) = sdf_argmin_3d(ub, vb, wb, x, q, geometry, include_static)
    (yc, tagc) = sdf_argmin_3d(uc, vc, wc, x, q, geometry, include_static)
    (yd, tagd) = sdf_argmin_3d(ud, vd, wd, x, q, geometry, include_static)
    (y_center, tag_center) = sdf_argmin_3d(ua, va + 0.5 * h, wa + 0.5 * h, x, q, geometry, include_static)
    fuf[i, j, k] = fraction_quad(ya, yb, yc, yd, y_center)
    owneru[i, j, k] = tag_center

@wp.kernel
def test_3d(
    fuf: wp.array3d(dtype = float), 
    x: wp.vec3, q: wp.quat, 
    x_offset: float, y_offset: float, z_offset: float, h: float
):
    '''3d face fraction kernel for u component'''
    # compute the weights on all edges
    i, j, k = wp.tid()
    # wp.func can be overloaded but kernels cannot
    (ua, va, wa) = to01(i, j, k, x_offset, y_offset, z_offset, h)
    (ub, vb, wb) = to01(i, j + 1, k, x_offset, y_offset, z_offset, h)
    (uc, vc, wc) = to01(i, j + 1, k + 1, x_offset, y_offset, z_offset, h)
    (ud, vd, wd) = to01(i, j, k + 1, x_offset, y_offset, z_offset, h)
    # include_static = True
    # (ya, taga) = sdf_argmin_3d(ua, va, wa, x, q, geometry, include_static)
    # (yb, tagb) = sdf_argmin_3d(ub, vb, wb, x, q, geometry, include_static)
    # (yc, tagc) = sdf_argmin_3d(uc, vc, wc, x, q, geometry, include_static)
    # (yd, tagd) = sdf_argmin_3d(ud, vd, wd, x, q, geometry, include_static)
    ya = sdf_cube(wp.vec3(ua, va, wa) - x, wp.vec3(0.2, 0.2, 0.2), q)
    yb = sdf_cube(wp.vec3(ub, vb, wb) - x, wp.vec3(0.2, 0.2, 0.2), q)
    yc = sdf_cube(wp.vec3(uc, vc, wc) - x, wp.vec3(0.2, 0.2, 0.2), q)
    yd = sdf_cube(wp.vec3(ud, vd, wd) - x, wp.vec3(0.2, 0.2, 0.2), q)
    # (y_center, tag_center) = sdf_argmin_3d(ua, va + 0.5 * h, wa + 0.5 * h, x, q, geometry, include_static)
    y_center = sdf_cube(wp.vec3(ua + 0.5 * h, va + 0.5 * h, wa + 0.5 * h) - x, wp.vec3(0.2, 0.2, 0.2), q)
    fuf[i, j, k] = fraction_quad(ya, yb, yc, yd, y_center)

@wp.kernel
def _precompute_fij_v(
    fuf: wp.array2d(dtype = float), fvf: wp.array2d(dtype = float), 
    x: wp.array1d(dtype = wp.vec2), theta: wp.array1d(dtype = float), 
    geometry: wp.array1d(dtype = GeometryType2D), 
    owneru: wp.array2d(dtype = int), ownerv: wp.array2d(dtype = int), 
    x_offset: float, y_offset: float, h: float, include_static: int
):
    # compute the weights on all edges
    i, j = wp.tid()
    ul, vl = to01(i, j, x_offset, y_offset, h)
    ur, vr = to01(i + 1, j, x_offset, y_offset, h)
    
    yl, tagl = sdf_argmin(ul, vl, x, theta, geometry)
    yr, tagr = sdf_argmin(ur, vr, x, theta, geometry)
    fvf[i, j] = fraction_edge(yl, yr)
    ownerv[i, j] = tagl

@wp.kernel
def _precompute_fij_v_3d(
    fvf: wp.array3d(dtype = float), 
    x: wp.array1d(dtype = wp.vec3), q: wp.array1d(dtype = wp.quat), 
    geometry: wp.array1d(dtype = GeometryType3D), 
    ownerv: wp.array3d(dtype = int), 
    x_offset: float, y_offset: float, z_offset: float, h: float, include_static: int
):
    '''3d face fraction kernel for v component'''
    i, j, k = wp.tid()
    (ua, va, wa) = to01(i, j, k, x_offset, y_offset, z_offset, h)
    (ub, vb, wb) = to01(i + 1, j, k, x_offset, y_offset, z_offset, h)
    (uc, vc, wc) = to01(i + 1, j, k + 1, x_offset, y_offset, z_offset, h)
    (ud, vd, wd) = to01(i, j, k + 1, x_offset, y_offset, z_offset, h)

    (ya, taga) = sdf_argmin_3d(ua, va, wa, x, q, geometry, include_static)
    (yb, tagb) = sdf_argmin_3d(ub, vb, wb, x, q, geometry, include_static)
    (yc, tagc) = sdf_argmin_3d(uc, vc, wc, x, q, geometry, include_static)
    (yd, tagd) = sdf_argmin_3d(ud, vd, wd, x, q, geometry, include_static)
    (y_center, tag_center) = sdf_argmin_3d(ua + 0.5 * h, va, wa + 0.5 * h, x, q, geometry, include_static)
    fvf[i, j, k] = fraction_quad(ya, yb, yc, yd, y_center)
    ownerv[i, j, k] = tag_center


@wp.kernel
def _precompute_fij_w_3d(
    fwf: wp.array3d(dtype = float),
    x: wp.array1d(dtype = wp.vec3), q: wp.array1d(dtype = wp.quat), 
    geometry: wp.array1d(dtype = GeometryType3D), 
    ownerw: wp.array3d(dtype = int),
    x_offset: float, y_offset: float, z_offset: float, h: float, include_static: int
):
    '''3d face fraction kernel for w component'''
    i, j, k = wp.tid()
    (ua, va, wa) = to01(i, j, k, x_offset, y_offset, z_offset, h)
    (ub, vb, wb) = to01(i + 1, j, k, x_offset, y_offset, z_offset, h)
    (uc, vc, wc) = to01(i + 1, j + 1, k, x_offset, y_offset, z_offset, h)
    (ud, vd, wd) = to01(i, j + 1, k, x_offset, y_offset, z_offset, h)

    (ya, taga) = sdf_argmin_3d(ua, va, wa, x, q, geometry, include_static)
    (yb, tagb) = sdf_argmin_3d(ub, vb, wb, x, q, geometry, include_static)
    (yc, tagc) = sdf_argmin_3d(uc, vc, wc, x, q, geometry, include_static)
    (yd, tagd) = sdf_argmin_3d(ud, vd, wd, x, q, geometry, include_static)
    (y_center, tag_center) = sdf_argmin_3d(ua + 0.5 * h, va + 0.5 * h, wa, x, q, geometry, include_static)
    fwf[i, j, k] = fraction_quad(ya, yb, yc, yd, y_center)
    ownerw[i, j, k] = tag_center



@wp.kernel
def _set_fij_boundary(
    fuf: wp.array2d(dtype = float), fvf: wp.array2d(dtype = float), 
    x_offset: float, y_offset: float, h: float
):
    if x_offset == 0.0:
        for j in range(res):
            fuf[0, j] = 0.0
    if x_offset + h * (float(gridx) + 0.01) >= 1.0:
        for j in range(res):
            fuf[res, j] = 0.0
    if y_offset == 0.0:
        for j in range(res):
            fvf[j, 0] = 0.0
    if y_offset + h * (float(gridy) + 0.01) >= float(gridy) * h0:
        for j in range(res):
            fvf[j, res] = 0.0

@wp.kernel
def _set_fij_boundary_3d(
    fuf: wp.array3d(dtype = float), fvf: wp.array3d(dtype = float), fwf: wp.array3d(dtype = float),
    value: float, x_offset: float, y_offset: float, z_offset: float, h: float
):
    gx = fvf.shape[0]
    gy = fuf.shape[1]
    gz = fuf.shape[2]
    if x_offset == 0.0:
        for i in range(fuf.shape[1]):
            for j in range(fuf.shape[2]):
                fuf[0, i, j] = value
    if x_offset + h * (float(gridx) + 0.01) >= 1.0:
        for i in range(fuf.shape[1]):
            for j in range(fuf.shape[2]):
                fuf[gx, i, j] = value
    if y_offset == 0.0:
        for i in range(fvf.shape[0]):
            for j in range(fvf.shape[2]):
                fvf[i, 0, j] = value
    if y_offset + h * (float(gridy) + 0.01) >= float(gridy) * h0:
        for i in range(fvf.shape[0]):
            for j in range(fvf.shape[2]):
                fvf[i, gy, j] = value
    if z_offset == 0.0:
        for i in range(fwf.shape[0]):
            for j in range(fwf.shape[1]):
                fwf[i, j, 0] = value
    if z_offset + h * (float(gridz) + 0.01) >= float(gridz) * h0:
        for i in range(fwf.shape[0]):
            for j in range(fwf.shape[1]):
                fwf[i, j, gz] = value
@wp.kernel
def _mesh_fij_u(
    vertices: wp.array1d(dtype = wp.vec2), 
    edges: wp.array1d(dtype = wp.vec2i), 
    fuf: wp.array2d(dtype = float), fvf: wp.array2d(dtype = float), 
    x: wp.array1d(dtype = wp.vec2), theta: wp.array1d(dtype = float), 
    geometry: wp.array1d(dtype = GeometryType2D), 
    owneru: wp.array2d(dtype = int), ownerv: wp.array2d(dtype = int), 
    x_offset: float, y_offset: float, h: float
):
    # compute the weights on all edges
    i, j = wp.tid()
    ut, vt = to01(i, j + 1, x_offset, y_offset, h)
    ub, vb = to01(i, j, x_offset, y_offset, h)
    pt = wp.vec2(ut, vt)
    pb = wp.vec2(ub, vb)
    yt = signed_dis_to_mesh(edges, vertices, pt, x[0], theta[0])
    yb = signed_dis_to_mesh(edges, vertices, pb, x[0], theta[0])
    
    fuf[i, j] = fraction_edge(yb, yt)
    owneru[i, j] = 0

@wp.kernel
def _mesh_fij_v(
    vertices: wp.array1d(dtype = wp.vec2), 
    edges: wp.array1d(dtype = wp.vec2i), 
    fuf: wp.array2d(dtype = float), fvf: wp.array2d(dtype = float), 
    x: wp.array1d(dtype = wp.vec2), theta: wp.array1d(dtype = float), 
    geometry: wp.array1d(dtype = GeometryType2D), 
    owneru: wp.array2d(dtype = int), ownerv: wp.array2d(dtype = int), 
    x_offset: float, y_offset: float, h: float
):
    i, j = wp.tid()
    ul, vl = to01(i, j, x_offset, y_offset, h)
    ur, vr = to01(i + 1, j, x_offset, y_offset, h)

    pl = wp.vec2(ul, vl)
    pr = wp.vec2(ur, vr)
    yl = signed_dis_to_mesh(edges, vertices, pl, x[0], theta[0])
    yr = signed_dis_to_mesh(edges, vertices, pr, x[0], theta[0])
    
    fvf[i, j] = fraction_edge(yl, yr)
    ownerv[i, j] = 0


@wp.kernel
def precompute_grid_point_phi(
    phi: wp.array3d(dtype = float),
    x: wp.array1d(dtype = wp.vec3), q: wp.array1d(dtype = wp.quat), 
    geometry: wp.array1d(dtype = GeometryType3D), 
    x_offset: float, y_offset: float, z_offset: float, h: float, include_static: int
):
    i, j, k = wp.tid()
    u, v, w = to01(i, j, k, x_offset, y_offset, z_offset, h)
    y, tag = sdf_argmin_3d(u, v, w, x, q, geometry, include_static)
    phi[i, j, k] = y

@wp.kernel
def precompute_grid_point_phi_prims(
    phi: wp.array3d(dtype = float), owner: wp.array3d(dtype = int),
    x: wp.array1d(dtype = wp.vec3), q: wp.array1d(dtype = wp.quat), 
    geometry: wp.array1d(dtype = GeometryType3D), 
    x_offset: float, y_offset: float, z_offset: float, h: float,
    stagger: wp.vec3
):
    i, j, k = wp.tid()
    u, v, w = to01(i, j, k, x_offset, y_offset, z_offset, h) 
    u += stagger[0] * h
    v += stagger[1] * h
    w += stagger[2] * h
    y, tag = sdf_argmin_prims(u, v, w, x, q, geometry)
    phi[i, j, k] = y
    if stagger[0] != 0.0 or stagger[1] != 0.0 or stagger[2] != 0.0:
        owner[i, j, k] = tag

@wp.kernel
def _precompute_fij_u_3d_cached(
    phi: wp.array3d(dtype = float),
    fuf: wp.array3d(dtype = float),
    x: wp.array1d(dtype = wp.vec3), q: wp.array1d(dtype = wp.quat), 
    geometry: wp.array1d(dtype = GeometryType3D), 
    owneru: wp.array3d(dtype = int), 
    x_offset: float, y_offset: float, z_offset: float, h: float, include_static: int
):
    '''3d face fraction kernel for u component'''
    # compute the weights on all edges
    i, j, k = wp.tid()

    ya = phi[i, j, k]
    yb = phi[i, j + 1, k]
    yc = phi[i, j + 1, k + 1]
    yd = phi[i, j, k + 1]

    (ua, va, wa) = to01(i, j, k, x_offset, y_offset, z_offset, h)
    (y_center, tag_center) = sdf_argmin_3d(ua, va + 0.5 * h, wa + 0.5 * h, x, q, geometry, include_static)
    fuf[i, j, k] = fraction_quad(ya, yb, yc, yd, y_center)
    owneru[i, j, k] = tag_center

@wp.kernel
def phi_to_fraction(
    phi: wp.array3d(dtype = float),
    fuf: wp.array3d(dtype = float),
    dim: int
):
    '''3d face fraction kernel for u component'''
    # compute the weights on all edges
    i, j, k = wp.tid()
    
    ya = phi[i, j, k]
    yb = phi[i, j + 1, k]
    yc = phi[i, j + 1, k + 1]
    yd = phi[i, j, k + 1]

    if dim == 1: 
        yb = phi[i + 1, j, k]
        yc = phi[i + 1, j, k + 1]
        yd = phi[i, j, k + 1]
    elif dim == 2:
        yb = phi[i + 1, j, k]
        yc = phi[i + 1, j + 1, k]
        yd = phi[i, j + 1, k]


    y_center = fuf[i, j, k]
    fuf[i, j, k] = fraction_quad(ya, yb, yc, yd, y_center)

@wp.kernel
def _precompute_fij_v_3d_cached(
    phi: wp.array3d(dtype = float),
    fvf: wp.array3d(dtype = float), 
    x: wp.array1d(dtype = wp.vec3), q: wp.array1d(dtype = wp.quat), 
    geometry: wp.array1d(dtype = GeometryType3D), 
    ownerv: wp.array3d(dtype = int), 
    x_offset: float, y_offset: float, z_offset: float, h: float, include_static: int
):
    '''3d face fraction kernel for v component'''
    i, j, k = wp.tid()
    (ua, va, wa) = to01(i, j, k, x_offset, y_offset, z_offset, h)

    ya = phi[i, j, k]
    yb = phi[i + 1, j, k]
    yc = phi[i + 1, j, k + 1]
    yd = phi[i, j, k + 1]
    (y_center, tag_center) = sdf_argmin_3d(ua + 0.5 * h, va, wa + 0.5 * h, x, q, geometry, include_static)
    fvf[i, j, k] = fraction_quad(ya, yb, yc, yd, y_center)
    ownerv[i, j, k] = tag_center


@wp.kernel
def _precompute_fij_w_3d_cached(
    phi: wp.array3d(dtype = float),
    fwf: wp.array3d(dtype = float),
    x: wp.array1d(dtype = wp.vec3), q: wp.array1d(dtype = wp.quat), 
    geometry: wp.array1d(dtype = GeometryType3D), 
    ownerw: wp.array3d(dtype = int),
    x_offset: float, y_offset: float, z_offset: float, h: float, include_static: int
):
    '''3d face fraction kernel for w component'''
    i, j, k = wp.tid()
    (ua, va, wa) = to01(i, j, k, x_offset, y_offset, z_offset, h)
    ya = phi[i, j, k]
    yb = phi[i + 1, j, k]
    yc = phi[i + 1, j + 1, k]
    yd = phi[i, j + 1, k]

    (y_center, tag_center) = sdf_argmin_3d(ua + 0.5 * h, va + 0.5 * h, wa, x, q, geometry, include_static)
    fwf[i, j, k] = fraction_quad(ya, yb, yc, yd, y_center)
    ownerw[i, j, k] = tag_center

@wp.kernel
def _tag_dirichlet(
    tag: wp.array3d(dtype = int),
    phi: wp.array3d(dtype = float)
):
    i, j, k = wp.tid()
    if phi[i, j, k] > 0.0:
        tag[i, j, k] = 0
    else:
        tag[i, j, k] = 1


@wp.kernel
def _precompute_tag(
    tag: wp.array3d(dtype = int),
    x: wp.array1d(dtype = wp.vec3), q: wp.array1d(dtype = wp.quat), 
    geometry: wp.array1d(dtype = GeometryType3D),
    include_static: int,
    l: float, b: float, f: float, h: float
):
    i, j, k = wp.tid()
    u, v, w = to01(i, j, k, l, b, f, h)
    u += 0.5 * h
    v += 0.5 * h
    w += 0.5 * h
    
    (y_center, tag_center) = sdf_argmin_3d(u, v, w, x, q, geometry, include_static)
    if y_center < 0.0:
        tag[i, j, k] = 2
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    res = 16
    wp.init()
    
    dt = np.pi / 30
    for k in range(6):
        fu = wp.zeros((res + 1, res, res), dtype = float)
        wp.launch(test_3d, dim = fu.shape, inputs =[fu, wp.vec3(0.5, 0.5, 0.5), wp.quat(0.0, 0.0, np.cos(dt * k), np.sin(dt * k)), 0.0, 0.0, 0.0, 1.0 / float(res)])
        fu_np = fu.numpy()
        img = plt.imshow(fu_np[:, :, 8], origin = "lower")
        plt.show()

@wp.kernel
def update_argmin(
    phi: wp.array3d(dtype = float),
    owner: wp.array3d(dtype = int),
    levelset: wp.array3d(dtype = float),
    id: int, x: wp.vec3, q: wp.quat,
    x_offset: float, y_offset: float, z_offset: float, h: float, h_levelset: float,
    stagger: wp.vec3,
    ijk: wp.vec3
):
    '''update phi with resampled levelset'''
    i, j, k = wp.tid()
    px, py, pz = to01(i, j, k, x_offset, y_offset, z_offset, h)
    p = wp.vec3(px, py, pz) + stagger * h


    p = wp.quat_rotate_inv(q, p - x) 
    p = p / h_levelset - ijk
    mesh_phi = trilerp_stagger(levelset, p, wp.vec3(0.0, 0.0, 0.0))
    if mesh_phi < phi[i, j, k]:
        phi[i, j, k] = mesh_phi
        if stagger[0] != 0.0 or stagger[1] != 0.0 or stagger[2] != 0.0:
            owner[i, j, k] = id
    

@wp.kernel
def _tag_neumann(
    tag: wp.array3d(dtype = int),
    phi: wp.array3d(dtype = float)
):
    i, j, k = wp.tid()
    if phi[i, j, k] < 0.0:
        tag[i, j, k] = 2