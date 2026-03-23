import warp as wp
from const_param import center_rigid, rotate, init_v0, h0, v_inflow, gridx, gridy, gridz, dt, gravity, config_file, use_scene, xtype, theta_type, omega_type, shape, Field, MACGrid, IntField, MACGridInt, get_uf
from geometry import sdf_argmin_3d, GeometryType3D
from utils.util import TexPair, compute_v_solid_3d, is_cutcell_3d
import numpy as np
from utils.scene import WarpScene
# import warp.sim as sim
from typing import List
# import pyopenvdb as vdb
rho = 1.0

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
def precompute_solid_vel(
        x: wp.array1d(dtype=wp.vec3),
        v: wp.array1d(dtype=wp.vec3),
        omega: wp.array1d(dtype=wp.vec3),
        vel_rigid: MACGrid,
        vel_owner: MACGridInt,
        d: int):

    eps = h0 * 0.01
    i, j, k = wp.tid()
    u_rigid = get_uf(vel_rigid, d)
    owner = get_uf(vel_owner, d).f[i, j, k]
    tmp = float(i) * h0
    if eps < tmp and tmp < 1.0 - eps and owner != -1:
        # pos = wp.vec3(float(i) * h0, (float(j) + 0.5) * h0, (float(k) + 0.5) * h0)
        pos = (wp.vec3(float(i), float(j), float(k)) + u_rigid.stagger) * h0
        u_rigid.f[i, j, k] = compute_v_solid_3d(pos, x[owner], v[owner], omega[owner])[d - 1]
    else:
        u_rigid.f[i, j, k] = 0.0


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


    def precompute_solid_vel_3d(self, x, theta, v, omega, geometry, vel_rigid, vel_owner):
        wp.launch(precompute_solid_vel, dim = vel_rigid.u.f.shape, inputs = [x, v, omega, vel_rigid, vel_owner, 1])
        wp.launch(precompute_solid_vel, dim = vel_rigid.v.f.shape, inputs = [x, v, omega, vel_rigid, vel_owner, 2])
        wp.launch(precompute_solid_vel, dim = vel_rigid.w.f.shape, inputs = [x, v, omega, vel_rigid, vel_owner, 3])



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
        
