import numpy as np
import warp as wp 
from const_param import *
from water_3d import WaterExample3d, Example
import polyscope as ps
import polyscope.imgui as gui
import os
# import pyHouGeoIO


class PSViewer:
    def __init__(self, fluids):
        self.fluids = fluids
        self.render()
        self.frame = 0
        self.ui_pause = False
        self.animate = False
        
        self.end_frame = 5000
        self.capture_interval = 1
        
    def render(self): 
        self.V, self.F, self.particles = self.fluids.render()
        if self.particles is not None:
            self.ps_particles = ps.register_point_cloud("particles", self.particles, radius = 5e-4)
        self.ps_mesh = ps.register_surface_mesh("surface", self.V, self.F)

    def callback(self):
        changed, self.ui_pause = gui.Checkbox("Pause", self.ui_pause)
        self.animate = gui.Button("Step") or not self.ui_pause
        if gui.Button("Reset"):
            self.fluids.reset()
            self.frame = self.fluids.frame
            self.ui_pause = True
            self.animate = True

        if gui.Button("Save"):
            np.save(f"output/x_{self.frame}.npy", self.V)
            print(f"output/x_{self.frame}.npy saved")
        if self.animate: 
            self.fluids.step()
            self.render()
            print("frame = ", self.frame)

            # if self.frame % self.capture_interval == 0:
            #     self.save()
        if self.frame >= self.end_frame:
            print(f"end frame = {self.frame} reached, exiting")
            quit()


    def np2density(self, np_data):
        density_np = np_data
        density = vdb.FloatGrid()
        density.name = 'density'
        density.copyFromArray(density_np)
        return density

    def save(self, i, name = "flow"):
        np_data = None
        if hasattr(example, "phi0"):
            # water
            np_data = example.phi0.numpy()
        else:
            np_data = example.rho0.numpy()
        density = self.np2density(np_data)
        vdb.write(f'{path}/{name}{i}.vdb', [density])


    def bgeo_save(self, frame):
        np_data = example.pm.x.numpy()[:example.pm.activated_int]
        if len(np_data):
            print("particle pos max, min = ", np.max(np_data), np.min(np_data))
        particles = pyHouGeoIO.HParticle()
        particles.SetPointAttributeMatXf("P", np_data)
        pyHouGeoIO.ExportHouGeo(f"{particle_path}/{frame}.bgeo", particles)
        # for j, patch in enumerate(example.patches):
        #     np_data = patch.phi_pair.cur if water else patch.rho+


if __name__ == "__main__":
    wp.init()
    water = config_file in water_config
    save_nvdb = write_npy
    save_bgeo = write_npy and water

    path = f"./dam break/res{res}" + ("water" if water else "") + "ref"
    npy_path = path + "npy"
    path += "vdb"
    particle_path = path + "_particle"

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(npy_path):
        os.mkdir(npy_path)
    if not os.path.exists(particle_path):
        os.mkdir(particle_path)

    example = WaterExample3d() if water else Example()

    ps.set_bounding_box(np.zeros(3), np.ones(3) * res)
    ps.set_ground_plane_mode("none")
    ps.init()
    viewer = PSViewer(example)
    ps.set_user_callback(viewer.callback)
    ps.show()
