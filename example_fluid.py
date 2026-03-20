import warp as wp
import numpy as np

from multigrid.mgpcg import MGPCG
from divergence import divergence_tag, mask_with_tag
from subtract_gradient import u_subtract_gradient_fractioned_3d, v_subtract_gradient_fractioned_3d, w_subtract_gradient_fractioned_3d, \
    u_subtract_gradient_tag, v_subtract_gradient_tag, w_subtract_gradient_tag
from const_param import *
from utils.util import scale_by_3d, TexPair

from adjustable import integrate_3d, init_3d

from advect import advect_3d
from geometry import PrecomputeFaceFraction
from rigid_coupling import RigidCoupling
from typing import List, Dict

class Example:
    def __init__(self):
        self.sim_fps = 60.0
        self.sim_substeps = 1
        self.iterations = 100
        self.sim_time = 0.0
        self.h = h
        self.device = wp.get_device()
        self.patch_focused = 0

        self.v0 = wp.zeros(shapev, dtype=float)
        self.v1 = wp.zeros(shapev, dtype=float)
        self.u0 = wp.zeros(shapeu, dtype=float)
        self.u1 = wp.zeros(shapeu, dtype=float)
        self.w0 = wp.zeros(shapew, dtype=float)
        self.w1 = wp.zeros(shapew, dtype=float)

        self.rho0 = wp.zeros(shape, dtype=float)
        self.rho1 = wp.zeros(shape, dtype=float)

        self.p0 = wp.zeros(shape, dtype=float)
        self.p1 = wp.zeros(shape, dtype=float)
        self.div = wp.zeros(shape, dtype=float)

        self.q0 = self.rho0
        self.q1 = self.rho1

        self.fu = wp.zeros(shapeu, dtype = float)
        self.fv = wp.zeros(shapev, dtype = float)
        self.fw = wp.zeros(shapew, dtype = float)

        self.u_rigid = wp.zeros(shapeu, dtype=float)
        self.v_rigid = wp.zeros(shapev, dtype=float)
        self.w_rigid = wp.zeros(shapew, dtype=float)
        
        self.u_owner = wp.zeros(shapeu, dtype=int)
        self.v_owner = wp.zeros(shapev, dtype=int)
        self.w_owner = wp.zeros(shapew, dtype=int)

        self.rhou = wp.zeros(shapeu, dtype = float)
        self.rhov = wp.zeros(shapev, dtype = float)
        self.rhow = wp.zeros(shapew, dtype = float)
        self.tag = wp.zeros(shape, dtype = int)
        self.tag_pc = wp.zeros(shape, dtype = int)
        self.phi_rigid = wp.zeros(shape, dtype = float)

        
        self.p_pair = TexPair(self.p0, self.p1)

        self.rc = RigidCoupling()
        self.pff = PrecomputeFaceFraction(use_mesh = False)
        self.mgpcg = MGPCG(3, shape, 3, float) if use_mgpcg else None
        self.timing_results: List[Dict] = []
        self.frame = 0

        self.max_verts = int(1e6)
        self.max_tris = int(1e6)
        self.mc = wp.MarchingCubes(gridx, gridy, gridz, self.max_verts, self.max_tris)
        
        self.reset()

    def call_mgpcg(self, pf, smoother = "fraction"):
        self.mgpcg.set_tags(self.tag)
        self.mgpcg.init(self.div, -h0)
        self.mgpcg.mask_interior()
        self.mgpcg.solve(4, verbose = True, smoother = smoother)
        wp.copy(pf, self.mgpcg.x)
        wp.launch(scale_by_3d, dim = shape, inputs = [pf, h0])

    def reset(self):
        self.frame = 0
        self.timing_results = gen_list()
        self.sim_time = 0.0
        for field in [self.rho0, self.rho1, self.p0, self.p1, 
                      self.u0, self.u1, self.v0, self.v1, self.w0, self.w1, 
                      self.fu, self.fv, self.fw, self.div, self.tag]:
            field.zero_()

        self.rc.reset()

    def advect(self):
        advect_3d(self.u0, self.v0, self.w0, self.rho0, self.u1, self.v1, self.w1, self.rho1)

        (self.u0, self.u1) = (self.u1, self.u0)
        (self.v0, self.v1) = (self.v1, self.v0)
        (self.w0, self.w1) = (self.w1, self.w0)
        (self.rho0, self.rho1) = (self.rho1, self.rho0)

    def ext_forces(self):
        _src_pos = self.rc.scene.src_pos
        src_pos = wp.vec3(_src_pos[0], _src_pos[1], _src_pos[2])

        _vel = self.rc.scene.vel
        vel = wp.vec3(_vel[0], _vel[1], _vel[2])
        if self.sim_time == 0.0 or (_vel != 0.0).any():
        # update emitters
            wp.launch(init_3d, dim=shape, inputs=[self.rho0, self.u0, self.v0, self.w0, 5, vel, src_pos, h0, 0.0, 0.0, 0.0])
        

        # force integrate
        wp.launch(integrate_3d, dim=shape, inputs=[self.v0, self.rho0, dt])

    def divergence(self, rc: RigidCoupling):
        rc.precompute_solid_vel_3d(rc.x_pair.cur, rc.theta_pair.cur, rc.v, rc.omega, rc.rigid_geometry, self.u_rigid, self.v_rigid, self.w_rigid, self.u_owner, self.v_owner, self.w_owner) 
        # wp.launch(divergence_fractional_3d, dim = shape, inputs = [self.u0, self.v0, self.w0, self.div, self.fu, self.fu, self.fw, self.u_rigid, self.v_rigid, self.w_rigid])

        wp.launch(divergence_tag, dim = shape, inputs = [self.u0, self.v0, self.w0, self.div, self.u_rigid, self.v_rigid, self.w_rigid, self.tag])
        wp.launch(mask_with_tag, dim = shape, inputs = [self.div, self.tag])

        wp.launch(scale_by_3d, dim = shape, inputs = [self.div, -1.0 / h])
        print("L inf u div = ", np.max(self.div.numpy()))
 
    def rigid_swap(self, rc: RigidCoupling):
        pff = self.pff

        rc.x_pair.swap()
        rc.theta_pair.swap()


        # 2.cont. compute face fractions for client and server
        # pff.precompute_fij_3d(self.fu, self.fv, self.fw, rc.x_pair.cur, rc.theta_pair.cur, rc.rigid_geometry, self.u_owner, self.v_owner, self.w_owner, 0.0, 0.0, 0.0, h, include_static= 0)
        # wp.launch(min_kernel, dim = shapeu, inputs= [self.fu, self.fu_static])
        # wp.launch(min_kernel, dim = shapev, inputs= [self.fv, self.fv_static])
        # wp.launch(min_kernel, dim = shapew, inputs= [self.fw, self.fw_static])
        pff.precompute_fij_with_cache(self.fu, self.fv, self.fw, self.u_owner, self.v_owner, self.w_owner, rc, 0.0, 0.0, 0.0, h0)

        self.tag.fill_(1)
        # self.pff.precompute_tag(self.tag, rc.x_pair.cur, rc.theta_pair.cur, rc.rigid_geometry)
        self.phi_rigid.fill_(1.0)
        pff.precompute_phi_with_cached_levelset(self.phi_rigid, rc, 0.0, 0.0, 0.0, h0, stagger = wp.vec3(0.5, 0.5, 0.5))
        pff.tag_neumann(self.tag, self.phi_rigid)


    def project(self, rc: RigidCoupling):
        '''prerequsite: face fractions computed on faces'''

        print("\nbefore projection: ")
        self.divergence(rc)
            
        # pressure solve
        self.p0.zero_()
        self.p1.zero_()

        with wp.ScopedTimer("global project", active=timer_active, synchronize=True):
            if use_mgpcg:
                self.call_mgpcg(self.p0)
            else:
                self.solve_multigrid_3d(self.mg_solver)
        
        wp.launch(scale_by_3d, dim = shape, inputs = [self.p0, 1.0 / h0])
        if use_mgpcg:
            wp.launch(u_subtract_gradient_tag, dim = shapeu, inputs = [self.u0, self.u_rigid, self.p0, self.tag, 0.0, 0.0, h0])
            wp.launch(v_subtract_gradient_tag, dim = shapev, inputs = [self.v0, self.v_rigid, self.p0, self.tag, 0.0, 0.0, h0])
            wp.launch(w_subtract_gradient_tag, dim = shapew, inputs = [self.w0, self.w_rigid, self.p0, self.tag, 0.0, 0.0, h0])
        else:
            # velocity update
            wp.launch(u_subtract_gradient_fractioned_3d, dim = shapeu, inputs = [self.u0, self.p0, self.fu, 0.0, 0.0, 0.0, h0])
            wp.launch(v_subtract_gradient_fractioned_3d, dim = shapev, inputs = [self.v0, self.p0, self.fv, 0.0, 0.0, 0.0, h0])
            wp.launch(w_subtract_gradient_fractioned_3d, dim = shapew, inputs = [self.w0, self.p0, self.fw, 0.0, 0.0, 0.0, h0])

        wp.launch(scale_by_3d, dim=shape, inputs=[self.p0, h0])
        print("after projection:")
        self.divergence(rc)

    def step(self):
        rc = self.rc
        # 1. advect the fluid 
        with wp.ScopedTimer("step(): global advect", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.advect()
            self.ext_forces()

        # 2. move rigid to (t + 1) state
        with wp.ScopedTimer("step(): Dh", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.rigid_swap(rc)

        # with wp.ScopedTimer("step(): coupling force", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
        #     self.apply_coupling_force(rc)

        # 5. recover solid state at (t + 1) 
        # do nothing

        # 6. perform div-free projection using effective velocity
        with wp.ScopedTimer("step(): global project", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.project(rc = rc)
        self.frame += 1

    def render(self, iso_value = 1e-2):
        v = np.zeros((0, 3), dtype = float)
        f = np.zeros((0, 3), dtype = int)
        rnp = self.rho0.numpy()
        if np.max(rnp) > iso_value and np.min(rnp) < iso_value:
            self.mc.surface(self.rho0, iso_value)
            v = self.mc.verts.numpy().reshape(-1, 3)
            f = self.mc.indices.numpy().reshape(-1, 3)
        return v, f
        
