import warp as wp
import numpy as np

from multigrid.mgpcg import MGPCG
from divergence import divergence_tag, mask_with_tag
from subtract_gradient import subtract_gradient_tag
from const_param import *
from utils.util import scale_by_3d

from adjustable import integrate_3d, init_3d

from advect import advect_3d
from geometry import PrecomputeFaceFraction
from rigid_coupling import RigidCoupling
from typing import List, Dict

# TODO: make read the scene in top level class instead of rigid coupling

class Example:
    def __init__(self):
        self.sim_time = 0.0

        self.vel0, self.vel1, self.frac, self.vel_rigid = (make_mac_grid() for _ in range(4))
        self.vel_owner = make_mac_grid(dtype = int)
        
        self.q0, self.q1, self.p0, self.p1, self.div, self.phi_rigid = (make_field(0) for _ in range(6))
        self.tag = make_field(0, dtype = int)
        self.centered_fields = [self.q0, self.q1, self.p0, self.p1, self.div, self.phi_rigid, self.tag]
        self.mac_fields = [self.vel0, self.vel1, self.frac, self.vel_rigid, self.vel_owner]

        self.rc = RigidCoupling()
        self.pff = PrecomputeFaceFraction(use_mesh = False)
        self.mgpcg = MGPCG(3, shape, 3, float) if use_mgpcg else None
        self.timing_results: List[Dict] = []
        self.frame = 0

        self.max_verts = int(1e6)
        self.max_tris = int(1e6)
        self.mc = wp.MarchingCubes(gridx, gridy, gridz, self.max_verts, self.max_tris)
        
        self.reset()
        
    def call_mgpcg(self, pf: Field, smoother = "fraction"):
        self.mgpcg.set_tags(self.tag.f)
        self.mgpcg.init(self.div.f, -h0)
        self.mgpcg.mask_interior()
        self.mgpcg.solve(4, verbose = True, smoother = smoother)
        wp.copy(pf.f, self.mgpcg.x)
        wp.launch(scale_by_3d, dim = shape, inputs = [pf, h0])

    def reset(self):
        self.frame = 0
        self.timing_results = gen_list()
        self.sim_time = 0.0
        for field in self.centered_fields:
            field.f.zero_()
        for field in self.mac_fields:
            field.u.f.zero_()
            field.v.f.zero_()
            field.w.f.zero_()

        self.rc.reset()

    def advect(self):
        advect_3d(self.vel0, self.q0, self.vel1, self.q1)

        (self.vel0, self.vel1) = (self.vel1, self.vel0)
        (self.q0, self.q1) = (self.q1, self.q0)

    def ext_forces(self):
        _src_pos = self.rc.scene.src_pos
        src_pos = wp.vec3(_src_pos[0], _src_pos[1], _src_pos[2])

        _vel = self.rc.scene.vel
        vel = wp.vec3(_vel[0], _vel[1], _vel[2])
        if self.sim_time == 0.0 or (_vel != 0.0).any():
        # update emitters
            wp.launch(init_3d, dim=shape, inputs=[self.q0, self.vel0, vel, src_pos])
        

        # force integrate
        wp.launch(integrate_3d, dim=shape, inputs=[self.vel0, self.q0, dt])

    def divergence(self):
        rc = self.rc

        rc.precompute_solid_vel_3d(rc.x_pair.cur, rc.theta_pair.cur, rc.v, rc.omega, rc.rigid_geometry, self.vel_rigid, self.vel_owner) 
        # wp.launch(divergence_fractional_3d, dim = shape, inputs = [self.u0, self.v0, self.w0, self.div, self.fu, self.fu, self.fw, self.u_rigid, self.v_rigid, self.w_rigid])

        wp.launch(divergence_tag, dim = shape, inputs = [self.vel0, self.div, self.vel_rigid, self.tag])
        wp.launch(mask_with_tag, dim = shape, inputs = [self.div, self.tag])

        wp.launch(scale_by_3d, dim = shape, inputs = [self.div, -1.0 / h])
        # print("L inf u div = ", np.max(self.div.f.numpy()))
 
    def Dh(self):
        rc = self.rc
        pff = self.pff

        rc.x_pair.swap()
        rc.theta_pair.swap()


        # 2.cont. compute face fractions for client and server
        # pff.precompute_fij_3d(self.fu, self.fv, self.fw, rc.x_pair.cur, rc.theta_pair.cur, rc.rigid_geometry, self.u_owner, self.v_owner, self.w_owner, 0.0, 0.0, 0.0, h, include_static= 0)
        # wp.launch(min_kernel, dim = shapeu, inputs= [self.fu, self.fu_static])
        # wp.launch(min_kernel, dim = shapev, inputs= [self.fv, self.fv_static])
        # wp.launch(min_kernel, dim = shapew, inputs= [self.fw, self.fw_static])
        pff.precompute_fij_with_cache(self.frac, self.vel_owner, rc)

        self.tag_dirichlet()
        self.phi_rigid.f.fill_(1.0)
        pff.precompute_phi_with_cached_levelset(self.phi_rigid.f, rc, stagger = wp.vec3(0.5, 0.5, 0.5))
        pff.tag_neumann(self.tag, self.phi_rigid)

    def tag_dirichlet(self): 
        # fill up the space, no need to set dirichlet boundary condition
        self.tag.f.fill_(1)

    def project(self):
        '''prerequsite: face fractions computed on faces'''

        print("\nbefore projection: ")
        self.divergence()
            
        # pressure solve
        self.p0.f.zero_()
        self.p1.f.zero_()

        with wp.ScopedTimer("global project", active=timer_active, synchronize=True):
            if use_mgpcg:
                self.call_mgpcg(self.p0)
        
        wp.launch(scale_by_3d, dim = shape, inputs = [self.p0, 1.0 / h0])
        if use_mgpcg:
            wp.launch(subtract_gradient_tag, dim = shapeu, inputs = [self.vel0, self.vel_rigid, self.p0, self.tag, 1])
            wp.launch(subtract_gradient_tag, dim = shapev, inputs = [self.vel0, self.vel_rigid, self.p0, self.tag, 2])
            wp.launch(subtract_gradient_tag, dim = shapew, inputs = [self.vel0, self.vel_rigid, self.p0, self.tag, 3])
        # else:
        #     # velocity update
        #     wp.launch(u_subtract_gradient_fractioned_3d, dim = shapeu, inputs = [self.u0, self.p0, self.fu, 0.0, 0.0, 0.0, h0])
        #     wp.launch(v_subtract_gradient_fractioned_3d, dim = shapev, inputs = [self.v0, self.p0, self.fv, 0.0, 0.0, 0.0, h0])
        #     wp.launch(w_subtract_gradient_fractioned_3d, dim = shapew, inputs = [self.w0, self.p0, self.fw, 0.0, 0.0, 0.0, h0])

        wp.launch(scale_by_3d, dim=shape, inputs=[self.p0, h0])
        print("after projection:")
        self.divergence()

    def step(self):
        # 1. advect the fluid 
        with wp.ScopedTimer("step(): global advect", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.advect()
            self.ext_forces()

        # 2. move rigid to (t + 1) state
        with wp.ScopedTimer("step(): Dh", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.Dh()

        # with wp.ScopedTimer("step(): coupling force", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
        #     self.apply_coupling_force(rc)

        # 5. recover solid state at (t + 1) 
        # do nothing

        # 6. perform div-free projection using effective velocity
        with wp.ScopedTimer("step(): global project", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.project()
        self.frame += 1

    def render(self, iso_value = 1e-2):
        v = np.zeros((0, 3), dtype = float)
        f = np.zeros((0, 3), dtype = int)
        rnp = self.q0.f.numpy()
        if np.max(rnp) > iso_value and np.min(rnp) < iso_value:
            self.mc.surface(self.q0.f, iso_value)
            v = self.mc.verts.numpy().reshape(-1, 3)
            f = self.mc.indices.numpy().reshape(-1, 3)
        return v, f, None
        
