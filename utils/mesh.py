# import meshio
import warp as wp
import numpy as np
import igl

# from pxr import UsdGeom, Usd


def load_usd_mesh(filename, path):
    asset_stage = Usd.Stage.Open(filename)
    mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(path))

    points = np.array(mesh_geom.GetPointsAttr().Get())
    indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

    return points, indices

class MeshBase:
    def __init__(self, path = "assets/stanford-bunny.obj", usd_path = ""):
        self.V = np.zeros((0, 3))
        self.F = np.zeros((0, 3), dtype = int)

        if path in ["box", "sphere"]:
            '''predefined shapes'''
            return 
        elif path.endswith((".usd", ".usda", "usdz")):
            '''load usd'''
            vertices, indices = load_usd_mesh(path, usd_path)
            self.V =  vertices
            self.F = indices
        else:
            '''load obj'''
            vertices, indices = igl.read_triangle_mesh(path)
            self.V = vertices
            self.F = indices


    def volume(self):
        V = np.vstack([self.V, np.zeros((1, 3))])
        T = np.hstack([self.F, np.zeros((self.F.shape[0], 1), dtype = int)])
        vol = igl.volume(V, T)
        return np.abs(np.sum(vol))

            
def normalize_xz(V):
    sup = np.max(V, axis = 0)
    inf = np.min(V, axis = 0)
    
    c = (sup + inf) / 2
    V[:, 0] -= c[0]
    V[:, 2] -= c[2]

    V /= np.max(sup - inf)
    # V[:, 0] += 0.5
    # V[:, 2] += 0.5

def center_axis(V, axis = 0):
    sup = np.max(V, axis = 0)
    inf = np.min(V, axis = 0)
    c = (sup + inf) / 2
    V[:, axis] -= c[axis]
    
def normalize(V):
    '''make the mesh fit in a unit cube'''
    V /= np.max(V * 2)
    
    
class Mesh(MeshBase): 
    def __init__(self, path = "assets/stanford-bunny.obj", normalized = False, centered_axis = [], build_mesh = True):
        super().__init__(path)
        for axis in centered_axis:
            center_axis(self.V, axis)
        if normalized:
            normalize(self.V)

        self.vertices = None
        self.indices = None
        self.mesh = None
        if build_mesh:
            self.build_mesh()

    def build_mesh(self):
        self.vertices = wp.from_numpy(self.V, dtype = wp.vec3, shape = self.V.shape[0])
        self.indices = wp.from_numpy(self.F, dtype = wp.int32,  shape = (self.F.shape[0] * 3))
        self.mesh = wp.Mesh(self.vertices, self.indices)
        print(f"mesh read, vertices count = {self.V.shape[0]}, triangles count = {self.F.shape[0]}")
        
if __name__ == "__main__":
    wp.init()
    mesh = Mesh("assets/city/city_triangulated.obj", normalized = True, centered_axis= [0, 2])
    V = mesh.V * 100
    igl.write_triangle_mesh("assets/city_normalized1.obj", V, mesh.F)