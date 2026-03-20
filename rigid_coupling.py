import warp as wp
from const_param import center_rigid, rotate, init_v0, h0, v_inflow, gridx, gridy, gridz, dt, gravity, config_file, use_scene, xtype, theta_type, omega_type, shape
from geometry import sdf_argmin_3d, GeometryType3D
from utils.util import TexPair, compute_v_solid_3d, is_cutcell_3d
import numpy as np
from utils.scene import WarpScene
# import warp.sim as sim
from typing import List
# import pyopenvdb as vdb
rho = 1.0

@wp.kernel
def compute_rigid_phi(
    x: wp.array1d(dtype = wp.vec3),
    q: wp.array1d(dtype = wp.quat),
    geometry: wp.array1d(dtype = GeometryType3D),
    l: float, b: float, f: float, h: float,
    phi: wp.array3d(dtype = float), include_static: int
):
    '''NOTE: also include ignore static object option'''
    i, j, k = wp.tid()
    u = (float(i) + 0.5) * h + l
    v = (float(j) + 0.5) * h + b
    w = (float(k) + 0.5) * h + f
    phiijk, tag = sdf_argmin_3d(u, v, w, x, q, geometry, include_static)
    phi[i, j, k] = phiijk

@wp.kernel
def reset_kernel(
    trans: wp.array1d(dtype = wp.transform), 
    transd: wp.array1d(dtype = wp.spatial_vector),
    x: wp.array1d(dtype = wp.vec3),
    q: wp.array1d(dtype = wp.quat),
    v: wp.array1d(dtype = wp.vec3),
    omega: wp.array1d(dtype = wp.vec3)
):
    i = wp.tid()
    trans[i] = wp.transform(x[i], q[i])
    transd[i] = wp.spatial_vector(omega[i], v[i]) 

@wp.kernel
def set_state(
    x: wp.array1d(dtype = wp.vec3),
    q: wp.array1d(dtype = wp.quat),
    v: wp.array1d(dtype = wp.vec3),
    omega: wp.array1d(dtype = wp.vec3),
    trans: wp.array1d(dtype = wp.transform),
    transd: wp.array1d(dtype = wp.spatial_vector)
):
    i = wp.tid()
    x[i] = wp.transform_get_translation(trans[i])
    q[i] = wp.transform_get_rotation(trans[i])
    omega[i] = wp.vec3(transd[i][0], transd[i][1], transd[i][2])
    v[i] = wp.vec3(transd[i][3], transd[i][4], transd[i][5])

@wp.kernel
def set_state_to_solver(
    x: wp.array1d(dtype = wp.vec3),
    q: wp.array1d(dtype = wp.quat),
    trans: wp.array1d(dtype = wp.transform)
):
    i = wp.tid()
    trans[i] = wp.transform(x[i], q[i])

@wp.kernel
def set_velocity(
    x: wp.array1d(dtype = wp.vec3), 
    q: wp.array1d(dtype = wp.quat),
    x_nxt: wp.array1d(dtype = wp.vec3),
    q_nxt: wp.array1d(dtype = wp.quat),
    v: wp.array1d(dtype = wp.vec3),
    omega: wp.array1d(dtype = wp.vec3),
    dt: float
):
    i =  wp.tid()
    v[i] = (x_nxt[i] - x[i]) / dt
    q1 = q[i]
    q2 = q_nxt[i]
    
    omega[i] = (2.0 / dt) * wp.vec3(
        q1[3] * q2[0] - q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[3] * q2[1] + q1[0] * q2[2] - q1[1] * q2[3] - q1[2] * q2[0],
        q1[3] * q2[2] - q1[0] * q2[1] + q1[1] * q2[0] - q1[2] * q2[3])

@wp.kernel
def set_v_to_solver(
    v: wp.array1d(dtype = wp.vec3),
    omega: wp.array1d(dtype = wp.vec3),
    transd: wp.array1d(dtype = wp.spatial_vector)
):
    i = wp.tid()
    transd[i] = wp.spatial_vector(
        omega[i][0], omega[i][1], omega[i][2],
        v[i][0], v[i][1], v[i][2]
    )
    
@wp.kernel
def get_xform(
    x: wp.array1d(dtype = wp.vec3),
    q: wp.array1d(dtype = wp.quat),
    m: wp.array1d(dtype = wp.mat44)
):
    i = wp.tid()
    # m33 = wp.mat33()
    # wp.quat_to_matrix(q[i], m33)
    # t = x[i]
    # xform = wp.transform(x[i], q[i])
    # m[i] = wp.
    # m[i] = wp.mat44.
    # m[i] = wp.mat44.from_trs(x[i], theta[i])
    xi = x[i]# - wp.vec3(0.5, 0.0, 0.5)

    m[i] = wp.transform_compose(xi, q[i], wp.vec3(1.0, 1.0, 1.0))


@wp.kernel
def initialize_solid_3d(
    x: wp.array1d(dtype=wp.vec3),
    q: wp.array1d(dtype=wp.quat),
    v: wp.array1d(dtype=wp.vec3),
    omega: wp.array1d(dtype=wp.vec3)
):
    n_rigids = x.shape[0]
    i = wp.tid()
    if center_rigid:
        # x[i] = wp.vec3(float(i + 1) / float(n_rigids + 1), 0.5, 0.5)
        x[i] = wp.vec3(0.0, -0.8, 0.0) + wp.vec3(0.5, 2.6 * float(i + 1) / float(n_rigids + 1), 0.5)
    else:
        x[i] = wp.vec3(float(i + 1) / float(n_rigids + 1), 0.0, 0.5)
    q[i] = wp.quat_identity()
    if rotate:
        omega[i] = wp.vec3(0.0, 0.0, 2.0)
    else:
        omega[i] = wp.vec3(0.0, 0.0, 0.0)
    if init_v0:
        v[i] = wp.vec3(0.01, 0.01, 0.01)
    else:
        v[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _precompute_solid_vel_u_3d(
        x: wp.array1d(dtype=wp.vec3),
        v: wp.array1d(dtype=wp.vec3),
        omega: wp.array1d(dtype=wp.vec3),
        u_rigid: wp.array3d(dtype=float),
        u_owner: wp.array3d(dtype=int),
        l: float, b: float, f: float, h: float):

    eps = h0 * 0.01
    i, j, k = wp.tid()
    owner = u_owner[i, j, k]
    tmp = float(i) * h + l
    if eps < tmp and tmp < 1.0 - eps and owner != -1:
        pos = wp.vec3(l + float(i) * h, b + (float(j) + 0.5) * h, f + (float(k) + 0.5) * h)
        u_rigid[i, j, k] = compute_v_solid_3d(
            pos, x[owner], v[owner], omega[owner])[0]
    else:
        u_rigid[i, j, k] = v_inflow
        # u_rigid[i, j, k] = 0.0


@wp.kernel
def _precompute_solid_vel_v_3d(
        x: wp.array1d(dtype=wp.vec3),
        v: wp.array1d(dtype=wp.vec3),
        omega: wp.array1d(dtype=wp.vec3),
        v_rigid: wp.array3d(dtype=float),
        v_owner: wp.array3d(dtype=int),
        l: float, b: float, f: float, h: float):
    eps = h0 * 0.01
    i, j, k = wp.tid()
    owner = v_owner[i, j, k]
    tmp = float(j) * h + b
    if eps < tmp and tmp < h0 * float(gridy) - eps and owner != -1:
        pos = wp.vec3(l + (float(i) + 0.5) * h, b + float(j) * h, f + (float(k) + 0.5) * h)
        v_rigid[i, j, k] = compute_v_solid_3d(
            pos, x[owner], v[owner], omega[owner])[1]
    else:
        v_rigid[i, j, k] = 0.0
@wp.kernel
def _precompute_solid_vel_w_3d(
        x: wp.array1d(dtype=wp.vec3),
        v: wp.array1d(dtype=wp.vec3),
        omega: wp.array1d(dtype=wp.vec3),
        w_rigid: wp.array3d(dtype=float),
        w_owner: wp.array3d(dtype=int),
        l: float, b: float, f: float, h: float):
    eps = h0 * 0.01
    i, j, k = wp.tid()
    owner = w_owner[i, j, k]
    tmp = float(k) * h + f
    if eps < tmp and tmp < h0 * float(gridz) - eps and owner != -1:
        pos = wp.vec3(l + (float(i) + 0.5) * h, b + (float(j) + 0.5) * h, f + float(k) * h)
        w_rigid[i, j, k] = compute_v_solid_3d(
            pos, x[owner], v[owner], omega[owner])[2]
    else:
        w_rigid[i, j, k] = 0.0

@wp.func
def cross(a: wp.vec2, b: wp.vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


@wp.kernel
def compute_forces_3d(
    pf: wp.array3d(dtype=float), 
    x: wp.array1d(dtype=wp.vec3), 
    fu: wp.array3d(dtype=float), 
    fv: wp.array3d(dtype=float), 
    fw: wp.array3d(dtype=float),
    tag: wp.array3d(dtype=int),

    force: wp.array1d(dtype = wp.vec3), 
    torque: wp.array1d(dtype = wp.vec3),
    l: float, b: float, f: float, h: float
):
    i, j, k = wp.tid()
    n_rigids = force.shape[0]
    for ii in range(n_rigids):
        if is_cutcell_3d(i, j, k, fu, fv, fw) and (ii == tag[i, j, k]):
            delta_f = pf[i, j, k] * \
                wp.vec3(
                    fu[i, j, k] - fu[i + 1, j, k],
                    fv[i, j, k] - fv[i, j + 1, k],
                    fw[i, j, k] - fw[i, j, k + 1]
                ) * (h * h)
            wp.atomic_add(force, ii, delta_f)
            wp.atomic_add(torque, ii, wp.cross(wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * h + wp.vec3(l, b, f) - x[ii], delta_f))

@wp.kernel
def integrate_pressure_forces(
    geometry: wp.array1d(dtype = GeometryType3D),
    v:  wp.array1d(dtype = wp.vec3),
    omega: wp.array1d(dtype = wp.vec3),
    force: wp.array1d(dtype = wp.vec3),
    torque: wp.array1d(dtype = wp.vec3),
    integrate: bool
):
    '''NOTE: no gravitational force in the integrand'''
    # _radius = rigid_geometry[ii].radius[0]
    # FIXME: fails if we use geometry
    ii = wp.tid()
    # radius = radius0
    radius = geometry[ii].radius[0]
    geo = geometry[ii].geometry
    df = force[ii]
    tau = torque[ii]
    # rho = geometry[ii].rho

    a = df / ((radius * 2.0) ** 3.0 * geometry[ii].rho)
    beta = tau / ((radius) ** 5.0 * 8.0 * geometry[ii].rho * 2.0 / 3.0)
    # 2 / 3 for a cube 

    # if False:
    if geo == 5 and geometry[ii].volume < 0.0:
        a = wp.vec3(0.0, 0.0, 0.0)
        beta = wp.vec3(0.0, 0.0, 0.0)
    elif geo == 5 and geometry[ii].volume > 0.0:
        a = df / (geometry[ii].volume * geometry[ii].rho)
        beta = tau / ((radius ** 2.0) * geometry[ii].volume * geometry[ii].rho * 2.0 / 3.0)

    # FIXME: rotate the I by q
    # diag = (radius) ** 4.0 * 4 * rho * 2.0 / 3.0
    # beta = tau * h ** 3.0 / wp.diag(wp.vec3(diag, diag, diag))
    if integrate:
        v[ii] = v[ii] + a
        omega[ii] = omega[ii] + beta

    force[ii] = a / dt
    torque[ii] = beta / dt

@wp.kernel
def integrate_single_pressure_force(
    geometry: wp.array1d(dtype = GeometryType3D),
    v:  wp.array1d(dtype = wp.vec3),
    omega: wp.array1d(dtype = wp.vec3),
    force: wp.array1d(dtype = wp.vec3),
    torque: wp.array1d(dtype = wp.vec3),
    ii: int
):
    '''NOTE: no gravitational force in the integrand'''
    # _radius = rigid_geometry[ii].radius[0]
    # FIXME: fails if we use geometry
    # radius = radius0
    radius = geometry[ii].radius[0]
    geo = geometry[ii].geometry
    df = force[ii]
    tau = torque[ii]
    rho = geometry[ii].rho

    a = df / ((radius * 2.0) ** 3.0 * rho)
    beta = tau / ((radius) ** 5.0 * 8.0 * rho * 2.0 / 3.0)
    # 2 / 3 for a cube 

    # FIXME: rotate the I by q
    # diag = (radius) ** 4.0 * 4 * rho * 2.0 / 3.0
    # beta = tau * h ** 3.0 / wp.diag(wp.vec3(diag, diag, diag))

    if geo == 5 and geometry[ii].volume > 0.0:
        a = df / (geometry[ii].volume * rho)
        beta = tau / ((radius ** 2.0) * geometry[ii].volume * rho * 2.0 / 3.0)
    elif geo == 5 and geometry[ii].volume < 0.0:
        a = wp.vec3(0.0, 0.0, 0.0)
        beta = wp.vec3(0.0, 0.0, 0.0)

    if geometry[ii].volume < 0.0:
        a = wp.vec3(0.0, 0.0, 0.0)
        beta = wp.vec3(0.0, 0.0, 0.0)

    v[ii] = v[ii] + a    
    omega[ii] = omega[ii] + beta

    force[ii] = a / dt
    torque[ii] = beta / dt


@wp.kernel
def _integrate_ext_forces(
    v: wp.array1d(dtype = wp.vec3),
    geometry: wp.array1d(dtype = GeometryType3D)
):
    f_g = wp.vec3(0.0, -gravity, 0.0) * 1.0# * rho[i, j, k]
    i = wp.tid()
    if geometry[i].volume > 0.0:
        v[i] = v[i] + f_g * dt

@wp.kernel
def override(
    v: wp.array1d(dtype = wp.vec3),
    omega: wp.array1d(dtype = wp.vec3),
    vel: float,
    i: int
): 
    v[i] = wp.vec3(0.0, vel, 0.0)
    omega[i] = wp.vec3(0.0, 0.0, 0.0)




class RigidCoupling:
    def __init__(self):
        # builder = sim.ModelBuilder(gravity = -gravity)
        self.ke = 1.0e5
        self.kd = 250
        self.kf = 0
        self.n_rigids = None
        self.scene = WarpScene(config_file=config_file)
        assert use_scene
        self._rigid_geometry: List[GeometryType3D] = self.geometry_from_scene(self.scene)
        self.rigid_geometry = wp.array(self._rigid_geometry, dtype = GeometryType3D, shape = (len(self.scene.scene_objects)))
        self.cached_levelset= {}
        
        self.n_rigids = self.rigid_geometry.shape[0]
        n_rigids = self.n_rigids
        x = wp.empty(shape=(n_rigids), dtype=xtype)
        x_nxt = wp.empty(shape=(n_rigids), dtype=xtype)
        theta = wp.empty(shape=(n_rigids), dtype=theta_type)
        theta_nxt = wp.empty(shape=(n_rigids), dtype=theta_type)

        self.v = wp.empty(shape=(n_rigids), dtype=xtype)
        self.force = wp.empty(shape=(n_rigids), dtype=xtype)
        self.torque = wp.empty(shape=(n_rigids), dtype=omega_type)
        self.omega = wp.empty(shape=(n_rigids), dtype=omega_type)

        self.v_stashed = np.zeros((n_rigids, 3), dtype = np.float32)
        self.omega_stashed = np.zeros((n_rigids, 3), dtype = np.float32)

        self.x_pair = TexPair(x, x_nxt)
        self.theta_pair = TexPair(theta, theta_nxt)
        self.model = None
        self.time = 0.0

        self.reset()
        
    def set_levelset_res(self):
        '''decide the resolution of the levelset according to the finest patch'''
        finest = 0.5
        for so in self.scene.scene_objects:
            finest = min(finest, so.patch_config.h_multiply)
        return finest * h0, np.rint(1.0 / finest).astype(int)

    @classmethod
    def geometry_from_mesh(cls, mesh):
        rg = GeometryType3D()
        rg.geometry = 5 if mesh.shape == "" else 2 if mesh.shape == "sphere" else 1
        rg.mesh_id = mesh.mesh.id if mesh.shape == "" else 0
        rg.radius = np.max(np.abs(mesh.V)) if mesh.shape == "" else mesh.radius
        rg.rho = mesh.density * 1.0 
        rg.volume = mesh.volume() if not mesh.static else -1.0 
        return rg

    @classmethod
    def geometry_from_scene(cls, scene):
        geometry_array: List[GeometryType3D] = []
        for so in scene.scene_objects:
            rg = cls.geometry_from_mesh(so)
            geometry_array.append(rg)
        return geometry_array


    @classmethod
    def pad(cls, x, n, default = np.zeros(3)):
        _x = x
        if x.shape[0] != n:
            _x = np.zeros((n, default.shape[0]))
            _x = np.tile(default, (n, 1))
            _x[:x.shape[0], :] = x
        return _x

    @classmethod
    def prescribed_motion(cls):
        ret = {
            "cos": lambda t: (np.array([0.0, np.cos(t * 10.0, 0.0)]), np.zeros(3)),
            "constx": lambda t: (np.array([0.0, 0.0, 1.0]), np.zeros(3)),
            "constz": lambda t: (np.array([1.0, 0.0, 0.0]), np.zeros(3)),
            "spiny": lambda t: (np.zeros(3), np.array([0.0, -20.0, 0.0])),
            "spinz": lambda t: (np.zeros(3), np.array([0.0, 0.0, 5.0])),
            "static": lambda t: (np.zeros(3), np.zeros(3))
        }
        return ret

    def hack_vel(self):
        w = self.omega.numpy()
        v = self.v.numpy()


        t = self.time
        sos = self.scene.scene_objects
        for i in range(self.n_rigids):
            orb = sos[i].orbit
            if orb != "":
                if orb in self.prescribed_motion().keys():
                    vel, ome = self.prescribed_motion()[orb](t)
                    v[i] = vel + np.array([0.0, gravity, 0.0]) * dt * 0.5
                    w[i] = ome
                    # print(f"modified vel {i} = {v[i]}")

        self.omega.assign(w)
        self.v.assign(v)

    def compute_forces_then_integrate(self, pf, fu, fv, fw, tag, lbf = np.zeros(3), h = h0, integrate = True):
        self.time += dt
        self.force.zero_()
        self.torque.zero_()
        l, b, f = lbf
        wp.launch(compute_forces_3d, dim = shape, inputs = [pf, self.x_pair.cur, fu, fv, fw, tag, self.force, self.torque, l, b, f, h])
        wp.launch(integrate_pressure_forces, dim = self.v.shape, inputs = [self.rigid_geometry, self.v, self.omega, self.force, self.torque, integrate])


        # override velocity script
        self.hack_vel()
        # wp.launch(override, dim = (1), inputs = [self.v, self.omega, velocity, 0])

    def reset(self):
        self.time = 0.0
        self.force.zero_()
        self.torque.zero_()

        wp.launch(initialize_solid_3d, dim = self.v.shape, inputs = [self.x_pair.cur, self.theta_pair.cur, self.v, self.omega])
        wp.launch(initialize_solid_3d, dim = self.v.shape, inputs = [self.x_pair.nxt, self.theta_pair.nxt, self.v, self.omega])

        x, q, v, omega = self.scene.get_init_state()
        # print(f"initialized: x = {x}, q = {q}, v = {v}, omega = {omega}")
        self.set_x0q0v0w0(x, q, v, omega)

        if self.model is None:
            pass


    def precompute_solid_vel_3d(self, x, theta, v, omega, geometry, u_rigid, v_rigid, w_rigid, u_owner, v_owner, w_owner, l = 0.0, b = 0.0, f = 0.0, h= h0):
        wp.launch(_precompute_solid_vel_u_3d, dim = u_rigid.shape, inputs = [x, v, omega, u_rigid, u_owner, l, b, f, h])
        wp.launch(_precompute_solid_vel_v_3d, dim = v_rigid.shape, inputs = [x, v, omega, v_rigid, v_owner, l, b, f, h])
        wp.launch(_precompute_solid_vel_w_3d, dim = w_rigid.shape, inputs = [x, v, omega, w_rigid, w_owner, l, b, f, h])


    def compute_rigid_phi(self, phi, lbf = np.zeros(3), h = h0, include_static = 1):
        x = self.x_pair.cur
        q =self.theta_pair.cur
        rigid_geometry = self.rigid_geometry

        wp.launch(compute_rigid_phi, dim = phi.shape, inputs = [x, q, rigid_geometry, lbf[0], lbf[1], lbf[2], h, phi, include_static])

    def copy_from(self, rc, dt):
        wp.copy(self.x_pair.cur, rc.x_pair.cur)
        wp.copy(self.theta_pair.cur, rc.theta_pair.cur)
        wp.copy(self.rigid_geometry, rc.rigid_geometry)
        wp.copy(self.v, rc.v)
        wp.copy(self.omega, rc.omega)
        self.rigid_step_paired(dt)

    def xform_matrices(self):
        m = wp.empty(shape = self.v.shape, dtype = wp.mat44)
        wp.launch(get_xform, dim = self.v.shape, inputs = [self.x_pair.cur, self.theta_pair.cur], outputs = [m])
        return m.numpy()

    def integrate_ext_forces(self):
        # assert dimension == 3
        return
        wp.launch(_integrate_ext_forces, dim = self.v.shape, inputs = [self.v, self.rigid_geometry])

    def set_x0q0v0w0(self, _x0, _q0, _v0, _w0):
        # NOTE: don't overwrite the array by allocating another buffer
        # otherwise omniverse will have trouble locating the data
        x = self.pad(_x0, self.n_rigids, np.array([0.5, 0.5, 0.5]))
        q = self.pad(_q0, self.n_rigids, np.array([0.0, 0.0, 0.0, 1.0]))
        v = self.pad(_v0, self.n_rigids, np.array([0.0, 0.0, 0.0]))
        omega = self.pad(_w0, self.n_rigids, np.array([0.0, 0.0, 0.0]))

        x0 = wp.array(x, dtype = wp.vec3, shape = self.v.shape)
        wp.copy(self.x_pair.cur, x0)
        q0 = wp.array(q, dtype = wp.quat, shape = self.v.shape)
        wp.copy(self.theta_pair.cur, q0)
        
        v0 = wp.array(v, dtype = wp.vec3, shape = self.v.shape)
        wp.copy(self.v, v0)

        omega0 = wp.array(omega, dtype = wp.vec3, shape = self.v.shape)
        wp.copy(self.omega, omega0)

        wp.copy(self.x_pair.nxt, x0)
        wp.copy(self.theta_pair.nxt, q0)

        print(f"set success, x0 = {self.x_pair.cur.numpy()}, q0 = {self.theta_pair.cur.numpy()}, v0 = {self.v.numpy()}, w0 = {self.omega.numpy()}")

    def integrate(self, dt):
        rc = self
        sos =  rc.scene.scene_objects
        forces = np.zeros((rc.n_rigids, 3))
        for i, so in enumerate(sos):
            forces[i * 3: i * 3 + 3] = so.force
        rc.force.assign(forces)
        rc.torque.zero_()
    
        wp.launch(integrate_pressure_forces, dim = rc.n_rigids, 
                inputs = [rc.rigid_geometry, rc.v, rc.omega, rc.force, rc.torque, True])
        
