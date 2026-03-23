from const_param import *
from example_fluid import Example
from advect import advect_phi, advect_uv_mc
from utils.util import center_in_fluid_3d, sample_3d, fraction_of_fluid, scale_by_3d, exclude_rigid_phi, set_unknown_3d, set_unknown
from adjustable import integrate_3d, set_phi_3d
from extrapolate import extrapolate_by_1
from particles import ParticleSystem
from rigid_coupling import RigidCoupling
import numpy as np
        
class WaterExample3d(Example):
    def __init__(self):
        # solid sdf

        # water sdf
        self.phi0, self.phi1, self.phi_static = (make_field(0) for _ in range(3))
        
        # self.phi0 = wp.zeros(shape, dtype=float)
        # self.phi1 = wp.zeros(shape, dtype=float)

        # self.phi_static = wp.zeros(shape, dtype = float)

        self.vel_p, self.weight = (make_mac_grid() for _ in range(2))

        self.pm = ParticleSystem(shape, sparsity=4)

        super().__init__()

    def reset(self):
        super().reset()
        wl = self.rc.scene.water_level
        wp.launch(set_phi_3d, dim=shape, inputs=[self.phi0, wl])


        # FIXME: why not just use precompute_phi_with_cached_levelset
        rc = self.rc
        rc.compute_rigid_phi(self.phi_static, include_static= 2)
        rc.compute_rigid_phi(self.phi_rigid)
        wp.launch(exclude_rigid_phi, dim = shape, inputs = [self.phi_rigid, self.phi0])

        self.pm.initialize_particles(self.phi0, self.vel0)

    def advect(self):     
        wp.launch(advect_phi, dim = shape, inputs = [self.vel0, self.phi0, self.phi1])
        # advect_3d(self.u0, self.v0, self.w0, self.rho0, self.u1, self.v1, self.w1, self.rho1)
        advect_uv_mc(self.vel0, self.vel1)

        (self.vel0, self.vel1) = (self.vel1, self.vel0)
        (self.phi0, self.phi1) = (self.phi1, self.phi0)

        with wp.ScopedTimer("p2g", active=timer_active, synchronize=True):
            self.pm.p2g(self.vel_p , self.weight)
        with wp.ScopedTimer("step phi", active=timer_active, synchronize=True):
            self.pm.nb_flip_step_phi(self.phi0, self.phi_rigid)
        with wp.ScopedTimer("nb combine", active=timer_active, synchronize=True):
            self.pm.nbcombine(self.vel0, self.vel_p, self.phi0)

    def ext_forces(self):
        # not updating the emitters

        # force integrate
        wp.launch(integrate_3d, dim=shape, inputs=[self.vel0, self.q0, dt])

    def tag_dirichlet(self):
        self.pff.tag_dirichlet(self.tag, self.phi0)

    def call_mgpcg(self, pf: Field, tag = None):
        if tag is None:
            tag = self.tag
        
        with wp.ScopedTimer("mgpcg", active=timer_active, synchronize=True):
            self.mgpcg.set_tags(tag.f)
            self.mgpcg.init(self.div.f, -h0)
            self.mgpcg.mask_interior()
            self.mgpcg.solve(4, verbose = True, smoother = "phases")
            wp.copy(pf.f, self.mgpcg.x)
            wp.launch(scale_by_3d, dim = shape, inputs = [pf, h0])

    def extrapolate(self, n_cells = 16 * res // 128):
        wp.launch(set_unknown, dim = shape, inputs = [self.phi0, self.vel_owner])
        for _ in range(n_cells):
            extrapolate_by_1(self.vel0, self.vel_owner)

    def step(self):
        rc = self.rc
        # 1. advect the fluid
        with wp.ScopedTimer("step(): global advect", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.advect()
            wp.copy(self.vel_p.u.f, self.vel0.u.f)
            wp.copy(self.vel_p.v.f, self.vel0.v.f)
            wp.copy(self.vel_p.w.f, self.vel0.w.f)

            self.ext_forces()


        with wp.ScopedTimer("step(): Dh", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.Dh()

        # with wp.ScopedTimer("step(): coupling force", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
        #     rc.integrate(dt)
        #     # 3. apply coupling force and step the rigids
        #     self.apply_coupling_force(rc)

        # # 5. recover solid state at (t + 1) 
        # # do nothing

        # 6. perform div-free projection using effective velocity
        with wp.ScopedTimer("step(): global project", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.project()

        self.extrapolate()

        with wp.ScopedTimer("step(): particle step", active=timer_active, synchronize=True, dict = self.timing_results[self.frame]):
            self.pm.resample(self.phi0, self.vel0)

            wp.copy(self.vel1.u.f, self.vel_p.u.f)
            wp.copy(self.vel1.v.f, self.vel_p.v.f)
            wp.copy(self.vel1.w.f, self.vel_p.w.f)
            self.pm.step(self.vel0, dt, self.p0, self.vel1)
            self.pm.out(self.phi_rigid, self.vel0)

    def render(self, iso_value = 0.0):
        v = np.zeros((0, 3), dtype = float)
        f = np.zeros((0, 3), dtype = int)
        particles = self.pm.x.numpy()[:self.pm.activated_int] * res
        
        rnp = self.phi0.f.numpy()
        if np.max(rnp) > iso_value and np.min(rnp) < iso_value:
            self.mc.surface(self.phi0.f, iso_value)
            v = self.mc.verts.numpy().reshape(-1, 3)
            f = self.mc.indices.numpy().reshape(-1, 3)
        return v, f, particles