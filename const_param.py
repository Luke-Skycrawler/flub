import warp as wp
import numpy as np
write_video = False
write_npy = False
use_scene = True



water_config = ["scenes/beach.json", "scenes/b2.json", "scenes/water_overload.json", "scenes/dam_break.json", "scenes/flat.json", "scenes/drop.json", "scenes/water_wake.json", "scenes/propeller.json", "scenes/water_empty.json"]


# config_file = "scenes/city.json"
config_file = "scenes/flow_pass_sphere.json"
# config_file = "scenes/wake.json"
# config_file = "scenes/empty.json"

city_config = ["scenes/city.json", "scenes/flow_pass_sphere.json"]
wake_config = ["scenes/wake.json", "scenes/empty.json"]
gravity = wp.constant(9.8) if config_file in water_config + wake_config else wp.constant(0.0)
# comm_mode = "softmin"
comm_mode = "overwrite"
# comm_mode = ""
xpbd = True
view_frames = 800 if not write_npy else 200
dt = 1e-2

# parameters for global grid
# defulat volume (1.0, 0.5, 0.25) 
use_mgpcg = True
dimension = 3
res_digits = 7

res = 256
gridx = 256
gridy = 256
gridz = 256

# res = 64
# gridx = 64
# gridy = 64
# gridz = 64
# res = 128
# gridx = 256
# gridy = 256
# gridz = 256

xtype = wp.vec2 
theta_type = float 
omega_type = float 

shape = (gridx, gridy, gridz)
shapeu = (gridx + 1, gridy, gridz)
shapev = (gridx, gridy + 1, gridz)
shapew = (gridx, gridy, gridz + 1)
n_unknowns = gridx * gridy * gridz
xtype = wp.vec3
theta_type = wp.quat
omega_type = wp.vec3
    

h = wp.constant(1.0 / float(gridx))
h0 = h

init_v0 = wp.constant(False)
rotate = wp.constant(False)
center_rigid = wp.constant(True)
obstacle_shape = 1 # quad 
obstacle_bounding_box = wp.constant(32)
radius0 = wp.constant(0.1)
damp_omega = wp.constant(2.0 / 3.0)
static_objects = False

n_particle_per_cell = wp.constant(8)

fft_blend = False
v_inflow = 0.0
n_fine_smoothing = 30
timer_active = True

gen_list = lambda: [{} for _ in range(view_frames)]
print_timing_stats = True
