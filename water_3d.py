from .const_param import *
from .example_fluid import Example, flatten
from advect import advect_3d, advect_phi, advect_uv_mc
from .subtract_gradient import u_subtract_gradient_batty_3d, v_subtract_gradient_batty_3d, w_subtract_gradient_batty_3d, u_subtract_gradient_tag, v_subtract_gradient_tag, w_subtract_gradient_tag
from utils.util import center_in_fluid_3d, sample_3d, fraction_of_fluid, scale_by_3d, min_kernel, process_tag, exclude_rigid_phi, set_unknown_3d
from adjustable import integrate_3d, set_phi_3d
from extrapolate import extrapolate_by_1
from divergence import divergence_fractional_3d, invalidate, mask_with_tag, divergence_tag
from geometry_derived import _set_fij_boundary_3d
from particles import ParticleSystem
from .rigid_coupling import RigidCoupling
import numpy as np

@wp.kernel
def merge_water_solid_phi_3d(
    phi_water: wp.array3d(dtype=float),
    phi_solid: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    phi_solid[i, j, k] = wp.min(phi_water[i, j, k], phi_solid[i, j, k])

@wp.kernel
def tag_interior_cells_3d(
    phi: wp.array3d(dtype=float), 
    tag: wp.array3d(dtype=int)):
    i, j, k = wp.tid()
    tag[i, j, k] = wp.select(center_in_fluid_3d(phi, i, j, k), 0, 1)

@wp.kernel
def density_from_phi_3d(
    phi: wp.array3d(dtype=float),
    rho_u: wp.array3d(dtype=float),
    rho_v: wp.array3d(dtype=float),
    rho_w: wp.array3d(dtype=float)
):
    i, j, k = wp.tid()
    pc = phi[i, j, k]

    pl = sample_3d(phi, i - 1, j, k)
    pb = sample_3d(phi, i, j - 1, k)
    pf = sample_3d(phi, i, j, k - 1)

    rho_u[i, j, k] = fraction_of_fluid(pc, pl) 
    rho_v[i, j, k] = fraction_of_fluid(pc, pb)
    rho_w[i, j, k] = fraction_of_fluid(pc, pf)

    if i == gridx - 1:
        rho_u[i + 1,  j, k] = wp.select(pc < 0.0, 0.0, 1.0)    
    if j == gridy - 1:
        rho_v[i, j + 1, k] = wp.select(pc < 0.0, 0.0, 1.0)
    if k == gridz - 1:
        rho_w[i, j, k + 1] = wp.select(pc < 0.0, 0.0, 1.0)



@wp.kernel
def flatten_tag(
    u: wp.array3d(dtype = float),
    vis: wp.array2d(dtype = float),
    tag: wp.array3d(dtype = int)
):
    i, j = wp.tid()
    gridz = u.shape[2]

    if tag[i, j, gridz // 2] == 2:
        vis[i, j] = 0.5
    elif tag[i, j, gridz // 2] == 1:
        vis[i, j] = -0.5
    else: 
        vis[i, j] = 0.0
    # else: 
    #     vis[i, j] = u[i, j, gridz // 2]


@wp.kernel
def flatteni(
    u: wp.array3d(dtype = int),
    vis: wp.array2d(dtype = float),
    phi: wp.array3d(dtype = float)
):
    i, j = wp.tid()
    gridz = u.shape[2]
    # if phi[i, j, gridz * 2 // 4] <= 0.0:
    if False:
        vis[i, j] = 0.5
    else: 
        vis[i, j] = float(u[i, j, gridz * 2 // 4]) * 0.5


@wp.kernel
def expand(
    phi: wp.array3d(dtype = float),
    odd: int
):
    i, j, k = wp.tid()
    gridx = phi.shape[0]
    gridy = phi.shape[1]
    gridz = phi.shape[2]
    if (i  + j + k) %  2 == odd and i > 0 and j > 0 and k > 0 and i < gridx - 1 and j < gridy - 1 and k < gridz - 1:
        
        avg = (phi[i - 1, j, k ] + phi[i + 1, j, k] + phi[i, j - 1, k] + phi[i, j + 1, k] + phi[i, j, k - 1] + phi[i, j, k + 1]) / 6.0
        phi[i, j, k] = wp.min(phi[i, j, k], avg)


@wp.kernel
def include_water_phi(
    phi_water: wp.array3d(dtype=float),
    phi_rigid: wp.array3d(dtype=float)
):
    i, j, k = wp.tid()
    if phi_water[i, j, k] < 0.0:
        phi_rigid[i, j, k] = phi_water[i, j, k]


        
class WaterExample3d(Example):
    def __init__(self):
        # solid sdf

        # water sdf
        self.phi0 = wp.zeros(shape, dtype=float)
        self.phi1 = wp.zeros(shape, dtype=float)

        self.phi_static = wp.zeros(shape, dtype = float)
        self.p2g = True

        self.u2 = wp.zeros(shapeu, dtype = float)
        self.v2 = wp.zeros(shapev, dtype = float)
        self.w2 = wp.zeros(shapew, dtype = float)
        self.u3 = wp.zeros(shapeu, dtype = float)
        self.v3 = wp.zeros(shapev, dtype = float)
        self.w3 = wp.zeros(shapew, dtype = float)

        self.pm = ParticleSystem(shape, sparsity=4) if self.p2g else None

        super().__init__()

    def reset(self):
        super().reset()
        wl = self.rc.scene.water_level
        wp.launch(set_phi_3d, dim=shape, inputs=[self.phi0, wl, 0.0, 0.0, 0.0, h0])

        rc = self.rc
        rc.compute_rigid_phi(self.phi_static, include_static= 2)
        rc.compute_rigid_phi(self.phi_rigid)
        wp.launch(exclude_rigid_phi, dim = shape, inputs = [self.phi_rigid, self.phi0])
        if self.p2g:
            self.pm.initialize_particles(self.phi0, self.u0, self.v0, self.w0)

    def solve_multigrid_water_3d(self, mg_solver, p_pair = None):
        if p_pair is None:
            p_pair = self.p_pair
        
        mg_solver.set_fufvfw_rurvrw_tag_all(self.fu, self.fv, self.fw, self.rhou, self.rhov, self.rhow, self.tag)        
        mg_solver.solve(p_pair, self.div, smoother = "phases")

    def advect(self):     
        wp.launch(advect_phi, dim = shape, inputs = [self.u0, self.v0, self.w0, self.phi0, self.phi1, dt, h0])
        # advect_3d(self.u0, self.v0, self.w0, self.rho0, self.u1, self.v1, self.w1, self.rho1)
        advect_uv_mc(self.u0, self.v0, self.w0, self.u1, self.v1, self.w1, dt, h0)

        (self.u0, self.u1) = (self.u1, self.u0)
        (self.v0, self.v1) = (self.v1, self.v0)
        (self.w0, self.w1) = (self.w1, self.w0)
        # (self.rho0, self.rho1) = (self.rho1, self.rho0)
        (self.phi0, self.phi1) = (self.phi1, self.phi0)

        if self.p2g:
            with wp.ScopedTimer("p2g", active=timer_active, synchronize=True):
                self.pm.p2g(self.u2, self.v2, self.w2, self.u3, self.v3, self.w3)
            with wp.ScopedTimer("step phi", active=timer_active, synchronize=True):
                self.pm.nb_flip_step_phi(self.phi0, self.phi_rigid)
            with wp.ScopedTimer("nb combine", active=timer_active, synchronize=True):
                self.pm.nbcombine(self.u0, self.v0, self.w0, self.u2, self.v2, self.w2, self.phi0)

    def ext_forces(self):
        # not updating the emitters

        # force integrate
        wp.launch(integrate_3d, dim=shape, inputs=[self.v0, self.rho0, dt])
        if not static_objects:
            self.rc.integrate_ext_forces()

    def Dh(self, rc: RigidCoupling):
        pff = self.pff

        rc.x_pair.swap()
        rc.theta_pair.swap()
        with wp.ScopedTimer("precompute fij", active=timer_active, synchronize=True):

            # # compute fraction and ownership of faces
            # pff.precompute_fij_3d(self.fu, self.fv, self.fw, rc.x_pair.cur, rc.theta_pair.cur, rc.rigid_geometry, self.u_owner, self.v_owner, self.w_owner, 0.0, 0.0, 0.0, h0, include_static = 0)
            # wp.launch(min_kernel, dim = shapeu, inputs= [self.fu, self.fu_static])
            # wp.launch(min_kernel, dim = shapev, inputs= [self.fv, self.fv_static])
            # wp.launch(min_kernel, dim = shapew, inputs= [self.fw, self.fw_static])
            pff.precompute_fij_with_cache(self.fu, self.fv, self.fw, self.u_owner, self.v_owner, self.w_owner, rc, 0.0, 0.0, 0.0, h0)
        with wp.ScopedTimer("compute phi", active=timer_active, synchronize=True):
            self.phi_rigid.fill_(1.0)
            # rc.compute_rigid_phi(self.phi_rigid, include_static= 0)
            # wp.launch(min_kernel, dim = shape, inputs= [self.phi_rigid, self.phi_static])
            pff.precompute_phi_with_cached_levelset(self.phi_rigid, rc, 0.0, 0.0, 0.0, h0, stagger = wp.vec3(0.5, 0.5, 0.5))

        

        self.pff.tag_dirichlet(self.tag, self.phi0)
        self.pff.tag_neumann(self.tag, self.phi_rigid)
        # self.pff.precompute_tag(self.tag, rc.x_pair.cur, rc.theta_pair.cur, rc.rigid_geometry)

    def apply_coupling_force(self, rc: RigidCoupling = None):
        if rc is None:
            rc = self.rc

        self.divergence(rc)

        if use_mgpcg:
        # if False:
            wp.copy(self.tag_pc, self.tag)
            wp.launch(process_tag, dim = self.tag_pc.shape, inputs = [self.tag_pc])

            self.call_mgpcg(self.pc_pair.cur, self.tag_pc)
        else:
            with wp.ScopedTimer("mg pc solve", active=timer_active, synchronize=True):
                self.pc_pair.cur.zero_()
                self.pc_pair.nxt.zero_()
                self.solve_multigrid_water_3d(self.mg_solver, self.pc_pair)
        
        if not static_objects:
            rc.force.zero_()
            rc.torque.zero_()
            wp.launch(_set_fij_boundary_3d, dim = (1), inputs = [self.fu, self.fv, self.fw, 1.0, 0.0, 0.0, 0.0, h0])
            rc.compute_forces_then_integrate(self.pc_pair.cur, self.fu, self.fv, self.fw, self.u_owner, integrate = True)
            wp.launch(_set_fij_boundary_3d, dim = (1), inputs = [self.fu, self.fv, self.fw, 0.0, 0.0, 0.0, 0.0, h0])
            print(f"coupling force {rc.force.numpy()[0]}, torque {rc.torque.numpy()[0]}")

        rc.rigid_step_paired(dt)
        if time_scale != 0:
            self.save_pc()
        # wp.launch(scale_by_3d, dim = shape, inputs = [self.pc_pair.cur, 1 / h0])
    
    def divergence(self, rc: RigidCoupling):
        rc.precompute_solid_vel_3d(rc.x_pair.cur, rc.theta_pair.cur, rc.v, rc.omega, rc.rigid_geometry, self.u_rigid, self.v_rigid, self.w_rigid, self.u_owner, self.v_owner, self.w_owner) 
        wp.launch(divergence_tag, dim = shape, inputs = [self.u0, self.v0, self.w0, self.div, self.u_rigid, self.v_rigid, self.w_rigid, self.tag])
        # wp.launch(divergence_fractional_3d, dim = shape, inputs = [self.u0, self.v0, self.w0, self.div, self.fu, self.fu, self.fw, self.u_rigid, self.v_rigid, self.w_rigid])
        # wp.launch(invalidate, dim = shape, inputs = [self.div, self.phi_rigid])
        wp.launch(mask_with_tag, dim = shape, inputs = [self.div, self.tag])
        wp.launch(scale_by_3d, dim = shape, inputs = [self.div, -1.0 / h0])
        print("L inf u div = ", np.max(self.div.numpy()))

    def call_mgpcg(self, pf, tag = None):
        if tag is None:
            tag = self.tag
        # tag0 = tag.numpy()
        # tag0 = self.mgpcg.analyze(tag0)
        # tag.assign(tag0)
        
        with wp.ScopedTimer("mgpcg", active=timer_active, synchronize=True):
            self.mgpcg.set_tags(tag)
            self.mgpcg.init(self.div, -h0)
            self.mgpcg.mask_interior()
            self.mgpcg.solve(4, verbose = True, smoother = "phases")
            wp.copy(pf, self.mgpcg.x)
            wp.launch(scale_by_3d, dim = shape, inputs = [pf, h0])

    def project(self, rc: RigidCoupling = None):
        if rc is None:
            rc = self.rc
        print("\n\nbefore projection:")
        self.divergence(rc)

        # pressure solve
        self.p0.zero_()
        self.p1.zero_()
        with wp.ScopedTimer("mg solve pressure projection", active=timer_active, synchronize=True):
            if use_mgpcg:
                self.call_mgpcg(self.p0)
                # print("pmax = ", np.max(self.p0.numpy()))
        
        wp.launch(scale_by_3d, dim = shape, inputs = [self.p0, 1.0 / h0])
        # velocity update
        if use_mgpcg:
            wp.launch(u_subtract_gradient_tag, dim = shapeu, inputs = [self.u0, self.u_rigid, self.p0, self.tag, 0.0, 0.0, h0])
            wp.launch(v_subtract_gradient_tag, dim = shapev, inputs = [self.v0, self.v_rigid, self.p0, self.tag, 0.0, 0.0, h0])
            wp.launch(w_subtract_gradient_tag, dim = shapew, inputs = [self.w0, self.w_rigid, self.p0, self.tag, 0.0, 0.0, h0])
        else:
            wp.launch(u_subtract_gradient_batty_3d, dim = shapeu, inputs = [self.u0, self.p0, self.fu, self.phi_rigid, self.u_rigid, 0.0, 0.0, h0])
            wp.launch(v_subtract_gradient_batty_3d, dim = shapev, inputs = [self.v0, self.p0, self.fv, self.phi_rigid, self.v_rigid, 0.0, 0.0, h0])
            wp.launch(w_subtract_gradient_batty_3d, dim = shapew, inputs = [self.w0, self.p0, self.fw, self.phi_rigid, self.w_rigid, 0.0, 0.0, h0])

        print("after projection:")
        self.divergence(rc)
        # wp.launch(u_subtract_gradient_fractioned_3d, dim = shapeu, inputs = [self.u0, self.p0, self.fu, 0.0, 0.0, 0.0, h])
        # wp.launch(v_subtract_gradient_fractioned_3d, dim = shapev, inputs = [self.v0, self.p0, self.fv, 0.0, 0.0, 0.0, h])
        # wp.launch(w_subtract_gradient_fractioned_3d, dim = shapew, inputs = [self.w0, self.p0, self.fw, 0.0, 0.0, 0.0, h])
        if time_scale != 0:
            wp.launch(scale_by_3d, dim = shape, inputs = [self.p0, h0])

    def extrapolate(self, n_cells = 16 * res // 128):
        wp.launch(set_unknown_3d, dim=self.phi0.shape, inputs=[
            self.phi0, self.u_owner, self.v_owner, self.w_owner])
        for _ in range(n_cells):
            extrapolate_by_1(self.u0, self.v0, self.w0, self.u_owner, self.v_owner, self.w_owner)

    def step(self, rc: RigidCoupling):
        # 1. advect the fluid
        with wp.ScopedTimer("step(): global advect", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.advect()
            wp.copy(self.u2, self.u0)
            wp.copy(self.v2, self.v0)
            wp.copy(self.w2, self.w0)

            self.ext_forces()


        with wp.ScopedTimer("step(): Dh", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.Dh(rc)

        with wp.ScopedTimer("step(): coupling force", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            rc.integrate(dt)
            # 3. apply coupling force and step the rigids
            self.apply_coupling_force(rc)

        # # 5. recover solid state at (t + 1) 
        # # do nothing

        # 6. perform div-free projection using effective velocity
        with wp.ScopedTimer("step(): global project", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.project(rc)

        self.extrapolate()

        with wp.ScopedTimer("step(): particle step", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.pm.resample(self.phi0, self.u0, self.v0, self.w0)

            wp.copy(self.u1, self.u2)
            wp.copy(self.v1, self.v2)
            wp.copy(self.w1, self.w2)
            self.pm.step(self.u0, self.v0, self.w0, dt, self.p0, self.u1, self.v1, self.w1)
            if time_scale == 0:
                self.pm.out(self.phi_rigid, self.u0, self.v0, self.w0)
