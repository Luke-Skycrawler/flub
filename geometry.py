from const_param import *
import numpy as np
from utils.mesh import Mesh
# from .utils.util import *
MAX_DIST = wp.constant(1e10)
@wp.struct
class GeometryType2D:
    geometry: int
    radius: wp.vec2
    rho: float

@wp.struct
class GeometryType3D:
    geometry: int
    radius: wp.vec3
    rho: float
    mesh_id: wp.uint64
    volume: float

_pi2 = wp.constant(np.pi / 2.0)
@wp.func
def _sdf_rect(
    p: wp.vec2, 
    radius: wp.vec2, 
    theta: float
) -> float:
    ret = 0.0
    p = wp.vec2(p[0] * wp.cos(theta) + p[1] * wp.sin(theta), p[0] * - wp.sin(theta) + p[1] * wp.cos(theta))

    margin = wp.max(p, -p) - radius # abs is not for vector
    if wp.max(margin) < 0.0:
        margin = - margin
        ret = -(wp.min(margin))
    else:
        margin = wp.max(margin, wp.vec2(0.0, 0.0))
        ret = wp.length(margin)
    return ret

@wp.func
def _sdf_cube(
    p: wp.vec3,
    radius: wp.vec3,
    q: wp.quat
) -> float:
    '''compute the sdf of a 3d cube'''
    p = wp.quat_rotate_inv(q, p)
    margin = wp.max(p, -p) - radius
    if wp.max(margin) < 0.0:
        margin = - margin
        ret = -(wp.min(margin))
    else:
        margin = wp.max(margin, wp.vec3(0.0, 0.0, 0.0))
        ret = wp.length(margin)
    return ret

sdf_cube = _sdf_cube
# shape: 0: none, 1: quad, 2: circle, 3: rect, 4: x
@wp.func
def compute_sdf_atom(
    u: float, v: float, 
    x: wp.array1d(dtype=wp.vec2), 
    _theta: wp.array1d(dtype=float), 
    _geometry: wp.array1d(dtype=GeometryType2D), 
    rigid_id: int
) -> float:
    '''u, v in [0, 1]'''
    ret = 0.0
    p = wp.vec2(u, v) - x[rigid_id]
    radius = _geometry[rigid_id].radius
    geometry = _geometry[rigid_id].geometry
    theta = _theta[rigid_id]
    if geometry == 0:
        '''none'''
        ret = 1.0
    elif geometry == 1 or geometry == 3:
        '''quad / rect'''
        ret = _sdf_rect(p, radius, theta)

    elif geometry == 2:
        '''circle'''
        ret = wp.length(p) - radius[0]
    elif geometry == 4:
        '''x'''
        ret = wp.min(_sdf_rect(p, radius, theta), _sdf_rect(p, radius, theta + _pi2))
        # ret = 1.0
    return ret

# shape: 0: none, 1: quad, 2: circle, 3: rect, 4: x
@wp.func
def compute_sdf_atom(
    u: float, v: float, w: float,
    x: wp.vec3, 
    q: wp.quat, 
    _geometry: GeometryType3D, 
) -> float:
    '''3d sdf function, returns the min sdf to the union of all rigid bodies; u, v, w in [0, 1]'''
    ret = 0.0
    p = wp.vec3(u, v, w) - x
    
    radius = _geometry.radius
    geometry = _geometry.geometry
    if geometry == 0:
        '''none'''
        ret = 1.0
    elif geometry == 1 or geometry == 3:
        '''quad / rect'''
        ret = _sdf_cube(p, radius, q)

    elif geometry == 2:
        '''circle'''
        ret = wp.length(p) - radius[0]
    elif geometry == 4:
        '''x'''
        z_pi_2 = wp.quat(0.0, 0.0, wp.cos(_pi2 / 2.0), wp.sin(_pi2 / 2.0))
        # rotate by the z axis by pi / 2        
        ret = wp.min(_sdf_cube(p, radius, q), _sdf_cube(p, radius, z_pi_2 * q * wp.quat_inverse(z_pi_2)))
    elif geometry == 5:
        '''mesh'''
        dis = wp.float32(0.0)
        face = wp.int32(0)
        bary_u = wp.float32(0.0)
        bary_v = wp.float32(0.0)
        id = _geometry.mesh_id
        # found = wp.mesh_query_point(id, p, 2.0, dis, face, bary_u, bary_v)
        _p = wp.quat_rotate_inv(q, p)
        found = wp.mesh_query_point_sign_normal(id, _p, 2.0, dis, face, bary_u, bary_v, 1e-5)
        if found:
            p_mesh = wp.mesh_eval_position(id, face, bary_u, bary_v)
            distance = wp.length(_p - p_mesh)
            ret = distance * wp.sign(dis)
        else :
            ret = 1.0
    return ret

@wp.func
def compute_sdf_atom_prims(
    u: float, v: float, w: float,
    x: wp.vec3, 
    q: wp.quat, 
    _geometry: GeometryType3D, 
) -> float:
    '''3d sdf function, returns the min sdf to the union of all rigid bodies; u, v, w in [0, 1]'''
    ret = 0.0
    p = wp.vec3(u, v, w) - x
    
    radius = _geometry.radius
    geometry = _geometry.geometry
    if geometry == 0:
        '''none'''
        ret = 1.0
    elif geometry == 1 or geometry == 3:
        '''quad / rect'''
        ret = _sdf_cube(p, radius, q)

    elif geometry == 2:
        '''circle'''
        ret = wp.length(p) - radius[0]
    elif geometry == 4:
        '''x'''
        z_pi_2 = wp.quat(0.0, 0.0, wp.cos(_pi2 / 2.0), wp.sin(_pi2 / 2.0))
        # rotate by the z axis by pi / 2        
        ret = wp.min(_sdf_cube(p, radius, q), _sdf_cube(p, radius, z_pi_2 * q * wp.quat_inverse(z_pi_2)))
    elif geometry == 5:
        # compute mesh phi later on
        ret = 2.0
    return ret

@wp.func
def sdf_argmin(
    u: float, 
    v: float, 
    x: wp.array1d(dtype=wp.vec2), 
    _theta: wp.array1d(dtype=float), 
    geometry: wp.array1d(dtype=GeometryType2D)
):
    interior = float(2.0)
    argmin = int(-1)
    n_rigids = x.shape[0]
    for ii in range(n_rigids):
        sdf = compute_sdf_atom(u, v, x, _theta, geometry, ii)
        if sdf < wp.min(h0, interior):
            interior = sdf
            argmin = ii
    
    return (interior, argmin)

@wp.func
def sdf_argmin_3d(
    u: float, 
    v: float,
    w: float, 
    x: wp.array1d(dtype=wp.vec3), 
    _q: wp.array1d(dtype=wp.quat), 
    geometry: wp.array1d(dtype=GeometryType3D),
    include_static: int
):
    '''include staic: 0 to only include dynamic bodies; 1 for both; 2 for static objects only'''

    interior = float(2.0)
    argmin = int(-1)
    n_rigids = x.shape[0]
    for ii in range(n_rigids):

        if geometry[ii].volume <= 0.0:
            if include_static == 0:
                continue
        else:
            if include_static == 2:
                continue
            
        sdf = compute_sdf_atom(u, v, w, x[ii], _q[ii], geometry[ii])
        # if sdf < interior: # useful for min sdf blending
        if sdf < wp.min(h0, interior):
            interior = sdf
            argmin = ii
    
    return (interior, argmin)

@wp.func
def sdf_argmin_prims(
    u: float, 
    v: float,
    w: float, 
    x: wp.array1d(dtype=wp.vec3), 
    _q: wp.array1d(dtype=wp.quat), 
    geometry: wp.array1d(dtype=GeometryType3D),
):
    '''include staic: 0 to only include dynamic bodies; 1 for both; 2 for static objects only'''

    interior = float(2.0)
    argmin = int(-1)
    n_rigids = x.shape[0]
    for ii in range(n_rigids):            
        sdf = compute_sdf_atom_prims(u, v, w, x[ii], _q[ii], geometry[ii])
        # if sdf < interior: # useful for min sdf blending
        if sdf < wp.min(h0, interior):
            interior = sdf
            argmin = ii
    
    return (interior, argmin)


@wp.func 
def dis_to_ab(
    a: wp.vec2, b: wp.vec2, p: wp.vec2):
    '''computes distance from p to nearest point on line segment ab'''
    ap = p - a
    ab = b - a
    t = wp.dot(ap, ab) / wp.length_sq(ab)
    t = wp.min(wp.max(t, 0.0), 1.0)
    ret = wp.length(ap - t * ab)
    return ret

@wp.func 
def dis_to_ab(
    a: wp.vec3, b: wp.vec3, p: wp.vec3):
    '''computes distance from p to nearest point on line segment ab'''
    ap = p - a
    ab = b - a
    t = wp.dot(ap, ab) / wp.length_sq(ab)
    t = wp.clamp(t, 0.0, 1.0)
    ret = wp.length(ap - t * ab)
    return ret

@wp.func
def project(p: wp.vec3, d: wp.vec3):
    '''return a vec3, the projection of p onto d'''
    d = wp.normalize(d)
    return wp.dot(p, d) * d

@wp.func
def inside_triangle(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, p_project: wp.vec3):
    normal = wp.cross(b - a, c - a)
    area1 = wp.dot(normal, wp.cross(p_project - a, p_project - b))
    area2 = wp.dot(normal, wp.cross(p_project - b, p_project - c))
    area3 = wp.dot(normal, wp.cross(p_project - c, p_project - a))

    if area1 * area2 >= 0.0 and area2 * area3 >= 0.0:
        return True

    return False
@wp.func 
def dis_to_abc(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    '''computes distance from p to nearest point on triangle abc'''
    ab = b - a
    ac = c - a
    pa = a - p
    normal = wp.cross(ab, ac)
    height_offplane = wp.dot(pa, wp.normalize(normal))
    
    dab = dis_to_ab(a, b, p)
    dac = dis_to_ab(a, c, p)
    dbc = dis_to_ab(b, c, p)
    
    p_project = pa - project(pa, normal)
    projection_inside = inside_triangle(a, b, c, p_project)


    if projection_inside:
        return height_offplane
    
    return wp.min(wp.min(dab, dac), dbc)

@wp.func
def v_transformed(
    pid: wp.int32, 
    vertices: wp.array1d(dtype = wp.vec2), 
    x: wp.vec2, theta: float
):
    _c = wp.cos(theta)
    _s = wp.sin(theta)
    p0 = vertices[pid]
    p0 = x + wp.vec2(p0[0] * _c - p0[1] * _s, p0[0] * _s + p0[1] * _c)
    return p0

@wp.func
def v_transformed(
    pid: wp.int32, 
    vertices: wp.array1d(dtype = wp.vec3), 
    x: wp.vec3, q: wp.quat
):
    p0 = vertices[pid]
    p0 = x + wp.quat_rotate(q, p0)
    return p0

@wp.func
def inside_polygon(
    sihoutte: wp.array1d(dtype = wp.vec2i), 
    vertices: wp.array1d(dtype = wp.vec2), 
    x: wp.vec2, theta: float,
    p: wp.vec2) -> wp.bool:
    '''decide whether p is inside polygon, only works when the polygon is convex'''
    ne = sihoutte.shape[0]
    sgn = 0
    ret = wp.bool(True)
    for i in range(ne):
        p0 = v_transformed(sihoutte[i][0], vertices, x, theta)
        p1 = v_transformed(sihoutte[i][1], vertices, x, theta)

        if (p0[1] > p[1]) != (p1[1] > p[1]) and (p[0] < (p1[0] - p0[0]) * (p[1] - p0[1]) / (p1[1] - p0[1]) + p0[0]):
            ret = wp.unot(ret)
    return ret

@wp.func
def inside_closed_mesh(
    indices: wp.array1d(dtype = wp.vec3i), 
    vertices: wp.array1d(dtype = wp.vec3), 
    x: wp.vec3, q: wp.quat,
    p: wp.vec3) -> wp.bool:
    '''decide whether p is inside polygon, only works when the polygon is convex'''
    ne = indices.shape[0]
	# Accumulate generalized winding number per triangle
    winding_number = float(0.0)
    for i in range(ne):
        p0 = v_transformed(indices[i][0], vertices, x, q) - p
        p1 = v_transformed(indices[i][1], vertices, x, q) - p
        p2 = v_transformed(indices[i][2], vertices, x, q) - p
        
        p0_norm = wp.length(p0)
        p1_norm = wp.length(p1)
        p2_norm = wp.length(p2)
        
        winding_number += wp.atan2(wp.determinant(wp.mat33(p0, p1, p2)), p0_norm * p1_norm * p2_norm + p2_norm * wp.dot(p0, p1) + p0_norm * wp.dot(p1, p2) + p1_norm * wp.dot(p2, p0))

    return winding_number >= 4.0 * _pi2

@wp.func
def to01(i: wp.int32, j: wp.int32, x_offset: float, y_offset: float, h: float):
    return (float(i) * h + x_offset, float(j) * h + y_offset)

@wp.func
def to01(
    i: int, j: int, k: int, 
    x_offset: float, y_offset: float, z_offset: float,
    h: float):
    return float(i) * h + x_offset, float(j) * h + y_offset, float(k) * h + z_offset
    # return (float(i) + x_offset) * h, (float(j) + y_offset) * h, (float(k) + z_offset) * h

@wp.func
def signed_dis_to_mesh(
    sihoutte: wp.array1d(dtype = wp.vec2i), 
    vertices: wp.array1d(dtype = wp.vec2), 
    p: wp.vec2, 
    x: wp.vec2, theta: float) -> float:
    '''sihoutte: (ne, 2), vertices(nv, 2), returns signed distance from p to the mesh'''
    
    ne = sihoutte.shape[0]
    ret = MAX_DIST

    inside = inside_polygon(sihoutte, vertices, x, theta, p)
    sgn = wp.select(inside, -1.0, 1.0)
        
    for i in range(ne):
        a = v_transformed(sihoutte[i][0], vertices, x, theta)
        b = v_transformed(sihoutte[i][1], vertices, x, theta)
        ret = wp.min(ret, dis_to_ab(a, b, p))
    return ret * sgn

@wp.func
def signed_dis_to_mesh_3d(
    indices: wp.array1d(dtype = wp.vec3i), 
    vertices: wp.array1d(dtype = wp.vec3), 
    p: wp.vec3, 
    x: wp.vec3, q: wp.quat) -> float:
    '''compute signed distance from p to a closed mesh'''
    
    ne = indices.shape[0]
    ret = MAX_DIST

    inside = inside_closed_mesh(indices, vertices, x, q, p)
    sgn = wp.select(inside, -1.0, 1.0)
        
    for i in range(ne):
        a = v_transformed(indices[i][0], vertices, x, q)
        b = v_transformed(indices[i][1], vertices, x, q)
        c = v_transformed(indices[i][2], vertices, x, q)
        ret = wp.min(ret, dis_to_abc(a, b, c, p))
    return ret * sgn

@wp.kernel
def test_cube_kernel(
    x: wp.array1d(dtype = wp.vec3),
    q: wp.quat,
    phi: wp.array1d(dtype = float)
):
    i = wp.tid()
    phi[i] = _sdf_cube(x[i] - wp.vec3(0.5, 0.5, 0.0), wp.vec3(0.2, 0.2, 0.2), q)

@wp.kernel
def init_samples(
    x: wp.array1d(dtype = wp.vec3),
):
    i = wp.tid()
    r = wp.rand_init(315, i)
    # x[i] = wp.vec3(wp.randf(r), wp.randf(r), 0.5)
    x[i] = wp.vec3(wp.randf(r), wp.randf(r), wp.randf(r))


def test_cube():
    import matplotlib.pyplot as plt
    sample_points = wp.zeros((n_sample), dtype = wp.vec3)
    phi = wp.zeros((n_sample), dtype = float)
    wp.launch(init_samples, dim = (n_sample), inputs = [sample_points])
    dt = np.pi / 30
    sample_points_np = sample_points.numpy()
    for k in range(6):
        wp.launch(test_cube_kernel, dim = (n_sample), inputs = [sample_points, wp.quat(0.0, 0.0, np.cos(dt * k), np.sin(dt * k)), phi])
        # print(phi.numpy())
        # print(sample_points.numpy())
        phi_np = phi.numpy()
        idx_neg = phi_np[:] < 0.0
        idx_pos = phi_np[:] > 0.0
        plt.plot(sample_points_np[idx_neg, 0], sample_points_np[idx_neg, 1], 'ro')
        plt.plot(sample_points_np[idx_pos, 0], sample_points_np[idx_pos, 1], 'bo')

        plt.show()


@wp.kernel
def mesh_phi(
    mesh_id: wp.uint64, 
    x: wp.array1d(dtype = wp.vec3),
    phi: wp.array1d(dtype = float)
):
    i = wp.tid()
    face = wp.int32(0)
    bary_u = wp.float32(0.0)
    bary_v = wp.float32(0.0)
    dis = wp.float32(0.0)
    ret = wp.mesh_query_point_sign_normal(mesh_id, x[i], 2.0, dis, face, bary_u, bary_v, 1e-4)
    phi[i] = dis

    