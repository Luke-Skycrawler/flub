# from vis import Visualizer
import numpy as np 
from utils.mesh import MeshBase, Mesh, normalize_xz, normalize, center_axis
import json 
from typing import List
# import quaternion as quat

import os
class SceneObject(MeshBase):
    def __init__(self, path):
        super().__init__(path)
        self.vel = np.zeros((3,))
        self.com = np.zeros((3, ))
        # self.q = np.quaternion(1.0, 0.0, 0.0, 0.0)
        self.omega = np.zeros((3, ))    
        self.x_seq = None
        self.q_seq = None
        self.name = "unnamed"

    
class WarpSceneObject(Mesh):
    def __init__(self, path, normalize = False, centered_axis = [], build_mesh = True):
        super().__init__(path, normalize, centered_axis, build_mesh)
        # self.q = np.quaternion(1.0, 0.0, 0.0, 0.0)
        self.x_seq = None
        self.q_seq = None
        self.name = "unnamed"
        self.static = False
        # center of mass of the object
        self.com = np.zeros((3, ))

        # default states
        self.x = np.zeros((3,))
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        self.vel = np.zeros((3,))
        self.omega = np.zeros((3, ))    

        self.force = np.zeros((3, ))
        self.torque = np.zeros((3, ))
        self.orbit = ""

        self.shape = ""
        self.radius = 0.1
        self.density = 1.0



    def volume(self):
        if self.shape == "":
            return super().volume()
        elif self.shape == "box":
            return (2.0 * self.radius) ** 3.0
        elif self.shape == "sphere":
            return 4.0 / 3.0 * np.pi * self.radius ** 3.0
    
    # def get_vertices_transformed(self):
    #     vertices = self.V @ quat.as_rotation_matrix(self.q) + self.com.reshape((-1, 3))
    #     return vertices



class Scene:
    def __init__(self, config_file = "./scenes/wake.json"):
        self.config_file = config_file
        self.config = None
        self.vis = None
        self.x_seq = None
        self.q_seq = None
        self.frame = 0

        self.config = None
        with open(config_file) as f:
            config = json.load(f)
        
        self.scene_objects: List[WarpSceneObject] = []
        for obj in config["obj"]:
            file = obj["file"]
            so = SceneObject(file)
            if "velocity" in obj:
                so.vel = np.array(obj["velocity"])
            if "omega" in obj:
                so.omega = np.array(obj["omega"])
            if "q" in obj:
                so.q = np.quaternion(*obj["q"]).normalized()
            if "trans" in obj:
                so.com = np.array(obj["trans"])
            if "centered_axis" in obj:
                for axis in obj["centered_axis"]:
                    center_axis(so.V, axis)
            if "normalize" in obj:
                normalize(so.V)
            if "scale" in obj:
                so.V *= obj["scale"]
            if "name" in obj:
                so.name = obj["name"]
            self.scene_objects.append(so)

    def export_to_usd(self):
        pass


def log(x, axis):
    print(f"max, min = {np.max(x[: , axis])}, {np.min(x[:, axis])}")

class WarpScene:
    def __init__(self, config_file = "scenes/wake.json", build_mesh = False):

        self.top_module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"path is " + self.top_module_path)
        config_file = os.path.join(self.top_module_path, config_file)

        self.config_file = config_file
        self.config = None
        self.vis = None
        self.x_seq = None
        self.q_seq = None
        self.frame = 0
        self.src_pos = np.array([0.5, 0.1, 0.5])
        self.vel = np.array([0.0, 1.0, 0.0])
        self.water_level = 0.5

        self.config = None
        with open(config_file) as f:
            config = json.load(f)
        
        self.scene_objects: List[WarpSceneObject] = []
        for obj in config["obj"]:
            file = obj["file"]
            if file in ["box", "sphere"]:
                pass
            else:
                file = os.path.join(self.top_module_path, file)
            normalize = False
            centered_axis = []

            # direct modification on the mesh upon reading it
            if "centered_axis" in obj:
                centered_axis = obj["centered_axis"]
            if "normalize" in obj:
                normalize = obj["normalize"]
            so = WarpSceneObject(path = file, normalize = normalize, centered_axis = centered_axis, build_mesh = False)
            
            if "scale" in obj:
                so.V *= obj["scale"]
            if "trans" in obj:
                so.com = np.array(obj["trans"])
                so.V += so.com.reshape((-1, 3))

            # state variables
            if "velocity" in obj:
                so.vel = np.array(obj["velocity"])
            if "omega" in obj:
                so.omega = np.array(obj["omega"])
            if "q" in obj:
                # so.q = np.quaternion(*obj["q"]).normalized()
                so.q = np.array(obj["q"])
            if "x" in obj:
                so.x = np.array(obj["x"])
            if "force" in obj:
                so.force = np.array(obj["force"])
            if "torque" in obj:
                so.torque = np.array(obj["torque"])
            if "orbit" in obj:
                so.orbit = obj["orbit"]

            if "name" in obj:
                so.name = obj["name"]
            if "static" in obj:
                so.static = ["static"]
            if "density" in obj:
                so.density = obj["density"]
            # for _ in range(3):
            #     log(so.V, _)

                

            if file in ["box", "sphere"]:
                so.shape = file
                if "radius" in obj:
                    so.radius = obj["radius"]
                    print(f"radius = {so.radius}")
            else:
                so.build_mesh()
            self.scene_objects.append(so)
        
        if "src_pos" in config:
            self.src_pos = np.array(config["src_pos"])
        if "vel" in config:
            self.vel = np.array(config["vel"])
        if "water_level" in config:
            self.water_level = config["water_level"]
            

    def get_init_state(self):
        x, q, v, omega = [], [], [], []
        for obj in self.scene_objects:
            x.append(obj.x)
            q.append(obj.q)
            v.append(obj.vel)
            omega.append(obj.omega)

        x = np.array(x)
        q = np.array(q)
        v = np.array(v)
        omega = np.array(omega)
        return x, q, v, omega
