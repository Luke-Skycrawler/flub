import warp as wp
# from ..utils.util import scale_by_3d
scalar = float
# int = wp.int
from warp.utils import array_inner
import math
import numpy as np

debug = True
timer_active = True
tile_size = 8
tile_shape = (tile_size, tile_size, tile_size)

# @wp.kernel
# def reduce_array_tile(
#     a: wp.array3d(dtype=float),
#     b: wp.array3d(dtype=float),
#     result: wp.array(dtype=float),
# ):
#     i, j, k = wp.tid()

#     local_val = a[i, j, k]*b[i, j, k]

#     t = wp.tile(local_val)
#     s = wp.tile_sum(t)

#     wp.tile_atomic_add(result, s)

# def array_inner(x, y, result):
#     result.zero_()
#     wp.launch(reduce_array_tile, x.shape, inputs=[x, y, result])    

@wp.kernel
def _scale_by_3d(qf: wp.array3d(dtype=scalar), scale: scalar):
    i, j, k = wp.tid()
    qf[i, j, k] = qf[i, j, k] * scale

@wp.func
def neighbor_sum(x: wp.array3d(dtype = scalar), I: wp.vec3i) -> float: 
    ret = float(0.0)
    for i in range(3):
        offset = wp.vec3i(0, 0, 0)
        offset[i] = 1

        upper = wp.vec3i(x.shape[0] - 1, x.shape[1] - 1, x.shape[2] - 1)
        lower = wp.vec3i(0, 0, 0)
        Ip = wp.min(upper, I + offset)
        Im = wp.max(lower, I - offset)
        ret += x[Ip[0], Ip[1], Ip[2]] + x[Im[0], Im[1], Im[2]] - x[I[0], I[1], I[2]] * 2.0

    return ret

@wp.kernel
def compute_Ap(Ap: wp.array3d(dtype = scalar), p: wp.array3d(dtype = scalar)):
    i, j, k = wp.tid()
    Ap[i, j, k] = neighbor_sum(p, wp.vec3i(i, j, k))
    # Ap[i, j, k] = p[i, j, k]

@wp.kernel
def compute_Ap_mcadams(Ap: wp.array3d(dtype = scalar), p: wp.array3d(dtype = scalar), tag: wp.array3d(dtype = int)):
    i, j, k = wp.tid()
    Ap[i, j, k] = neighbor_dot_tag(p, tag, wp.vec3i(i, j, k))

@wp.kernel
def _smooth(z: wp.array3d(dtype = scalar), r: wp.array3d(dtype = scalar), phase: int):
    i, j, k = wp.tid()
    res = r.shape[0]
    divdx = wp.where(i == 0 or i == res - 1, 1.0, 2.0)
    divdy = wp.where(j == 0 or j == res - 1, 1.0, 2.0) 
    divdz = wp.where(k == 0 or k == res - 1, 1.0, 2.0)
    if (i + j + k) & 1 == phase:
        z[i, j, k] = z[i, j, k] + (r[i, j, k] + neighbor_sum(z, wp.vec3i(i, j, k))) / (divdx + divdy + divdz)

@wp.kernel
def _compute_divd(tag: wp.array3d(dtype = int), divd: wp.array3d(dtype = float)):
    i, j, k = wp.tid()
    I = wp.vec3i(i, j, k)
    if tag[i, j, k] == 1:
        a = float(neighbor_sum_tag(tag, I))
        if a > 0.0: 
            a = 1.0 / a
        divd[i, j, k] = a
    else:
        divd[i, j, k] = 0.0

# @wp.kernel
# def _smooth_mcadams(z: wp.array3d(dtype = scalar), r: wp.array3d(dtype = scalar), tag: wp.array3d(dtype = int), _divd: wp.array3d(dtype = scalar), phase: int):
#     i, j, k = wp.tid()
#     if (i + j + k) & 1 == phase:
#         I = wp.vec3i(i, j, k)
#         # divd = float(neighbor_sum_tag(tag, I))
#         divd = _divd[i, j, k]
#         if divd != 0.0 and tag[i, j, k] == 1:
#             z[i, j, k] = z[i, j, k] + (r[i, j, k] + neighbor_dot_tag(z, tag, I)) / divd
#         else:
#             z[i, j, k] = 0.0

@wp.kernel
def _smooth_mcadams(z: wp.array3d(dtype = scalar), r: wp.array3d(dtype = scalar), tag: wp.array3d(dtype = int), _divd: wp.array3d(dtype = scalar), phase: int):
    i, j, _k = wp.tid()
    if (i + j) & 1 == phase:
        k = _k * 2 + 1
    else:
        k = _k * 2
    
    I = wp.vec3i(i, j, k)
    # divd = float(neighbor_sum_tag(tag, I))
    divd = _divd[i, j, k]
    z[i, j, k] += (r[i, j, k] + neighbor_dot_tag(z, tag, I)) * divd

@wp.func 
def clamp_dirichlet(zn: float, tag: float) -> float:
    return wp.where(tag == 0.0, 0.0, zn)

@wp.func 
def tdn(tag: float, zn: float) -> float:
    s = wp.where(tag < 2.0, 1., 0.)
    return s * zn

@wp.kernel
def _smooth_mcadams_tiled(z: wp.array3d(dtype = scalar), r: wp.array3d(dtype = scalar), tag: wp.array3d(dtype = int), _divd: wp.array3d(dtype = scalar)):
    i, j, k = wp.tid()

    z_tile = wp.tile_load(z, shape = (tile_size, tile_size, tile_size), offset = (i * tile_size, j * tile_size, k * tile_size), storage = "register", bounds_check=False)
    r_tile = wp.tile_load(r, shape = (tile_size, tile_size, tile_size), offset = (i * tile_size, j * tile_size, k * tile_size), storage = "register", bounds_check=False)
    for d in range(3):
        for ii in range(-1, 2, 2):
            offset = wp.vec3i(i, j, k) * tile_size
            offset[d] += ii
            
            tag_tile = wp.tile_astype(wp.tile_load(tag, shape = (tile_size, tile_size, tile_size), offset = (offset[0], offset[1], offset[2]), storage = "register", bounds_check=True), float)
            zn_tile = wp.tile_load(z, shape = (tile_size, tile_size, tile_size), offset = (offset[0], offset[1], offset[2]), storage = "register", bounds_check=True)
            
            zn_tile = wp.tile_map(clamp_dirichlet, zn_tile, tag_tile)

            r_tile += wp.tile_map(tdn, tag_tile, zn_tile - z_tile)

    divd_tile = wp.tile_load(_divd, shape = (tile_size, tile_size, tile_size), offset = (i * tile_size, j * tile_size, k * tile_size), storage="register", bounds_check=False)

    z_tile += r_tile * divd_tile
    wp.tile_store(z, z_tile, offset = (i * tile_size, j * tile_size, k * tile_size))

@wp.kernel
def _restrict(r_h: wp.array3d(dtype = scalar), z_h: wp.array3d(dtype = scalar), r_2h: wp.array3d(dtype = scalar)):
    i, j, k = wp.tid()
    res = r_h[i, j, k] - ( - neighbor_sum(z_h, wp.vec3i(i, j, k)))
    wp.atomic_add(r_2h, i // 2, j // 2, k // 2, res)

@wp.kernel
def _restrict_mcadams(r_h: wp.array3d(dtype = scalar), z_h: wp.array3d(dtype = scalar), tag_h: wp.array3d(dtype = int), r_2h: wp.array3d(dtype = scalar), tag_2h: wp.array3d(dtype = int)):
    i, j, k = wp.tid()
    res = r_h[i, j, k] - ( - neighbor_dot_tag(z_h, tag_h, wp.vec3i(i, j, k)))
    if tag_2h[i // 2, j // 2, k // 2] == 1:
        wp.atomic_add(r_2h, i // 2, j // 2, k // 2, res)
    else: 
        r_2h[i // 2, j // 2, k // 2] = 0.0

@wp.kernel
def _restrict_mcadams_int(r_h: wp.array3d(dtype = scalar), z_h: wp.array3d(dtype = scalar), tag_h: wp.array3d(dtype = int), r_2h: wp.array3d(dtype = scalar), tag_2h: wp.array3d(dtype = int)):
    i, j, k = wp.tid()
    res = r_h[i, j, k]# - ( - neighbor_dot_tag(z_h, tag_h, wp.vec3i(i, j, k)))
    if tag_2h[i // 2, j // 2, k // 2] == 1:
        wp.atomic_add(r_2h, i // 2, j // 2, k // 2, res)
    else: 
        r_2h[i // 2, j // 2, k // 2] = 0.0


@wp.kernel
def _prolongate(z_h: wp.array3d(dtype = scalar), z_2h: wp.array3d(dtype = scalar)):
    i, j, k = wp.tid()
    z_h[i, j, k] = z_h[i, j, k] + z_2h[i // 2, j // 2, k // 2]

@wp.func
def neighbor_dot_tag(x: wp.array3d(dtype = scalar), tag: wp.array3d(dtype = int), I: wp.vec3i) -> float:
    ret = float(0.0)
    upper = wp.vec3i(x.shape[0] - 1, x.shape[1] - 1, x.shape[2] - 1)
    lower = wp.vec3i(0, 0, 0)

    for d in range(3):
        for ii in range(-1, 2, 2):
            offset = wp.vec3i(0, 0, 0)
            offset[d] = ii
            Ip = wp.min(upper, I + offset)
            Ip = wp.max(lower, Ip)
            fp = wp.where(tag[Ip[0], Ip[1], Ip[2]] < 2, 1.0, 0.0)
            xp = wp.where(tag[Ip[0], Ip[1], Ip[2]] == 0, 0.0, x[Ip[0], Ip[1], Ip[2]]) 
            ret += (xp - x[I[0], I[1], I[2]]) * fp

    if tag[I[0], I[1], I[2]] != 1:
        ret = 0.0

    return ret

@wp.func
def neighbor_sum_tag(tag: wp.array3d(dtype = int), I: wp.vec3i) -> int:
    ret = int(0)
    for i in range(3):
        offset = wp.vec3i(0, 0, 0)
        offset[i] = 1

        upper = wp.vec3i(tag.shape[0] - 1, tag.shape[1] - 1, tag.shape[2] - 1)
        lower = wp.vec3i(0, 0, 0)
        Ip = wp.min(upper, I + offset)
        Im = wp.max(lower, I - offset)
        
        if tag[Ip[0], Ip[1], Ip[2]] < 2 and Ip == I + offset: 
            ret += 1
        if tag[Im[0], Im[1], Im[2]] < 2 and Im == I - offset:
            ret += 1
    return ret

@wp.kernel
def restrict_tag(tag_h: wp.array3d(dtype = int), tag_2h: wp.array3d(dtype = int)):
    i, j, k = wp.tid()
    t = int(2)
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                tag = tag_h[ii + i * 2, jj + j * 2, kk + k * 2]
                if tag == 0:
                    t = 0
                elif t > 0 and tag == 1:
                    t = 1
    tag_2h[i, j, k] = int(t)

@wp.kernel
def restrict_tag_int(tag_h: wp.array3d(dtype = int), tag_2h: wp.array3d(dtype = int)):
    i, j, k = wp.tid()
    t = int(2)
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                tag = tag_h[ii + i * 2, jj + j * 2, kk + k * 2]
                if tag == 0:
                    t = 0
                elif t > 0 and tag == 1:
                    t = 1
    tag_2h[i, j, k] = t


@wp.kernel
def _mean(qf: wp.array3d(dtype = scalar), tag: wp.array3d(dtype = int), sum: wp.array1d(dtype = scalar), n_avg: wp.array1d(dtype = int)):
    i, j, k = wp.tid()
    if tag[i, j, k] == 1:
        wp.atomic_add(sum, 0, qf[i, j, k])
        wp.atomic_add(n_avg, 0, 1)
    
@wp.kernel
def _sub(qf: wp.array3d(dtype = scalar), tag: wp.array3d(dtype = int), sum: wp.array1d(dtype = scalar), n_avg: wp.array1d(dtype = int)):
    mean = sum[0] / float(n_avg[0])
    i, j, k = wp.tid()
    if tag[i, j, k] == 1:
        qf[i, j, k] = qf[i, j, k] - mean
    else:
        qf[i, j, k] = 0.0

@wp.kernel
def _update_x(x: wp.array3d(dtype = scalar), p: wp.array3d(dtype = scalar), alpha: wp.array1d(dtype = scalar)):
    i, j, k = wp.tid()
    x[i, j, k] = x[i, j, k] + alpha[0] * p[i, j, k]

@wp.kernel
def _update_alpha(sum: wp.array1d(dtype = float), alpha: wp.array1d(dtype = float), old_zTr: wp.array1d(dtype = float), eps: float):
    pAp = sum[0]
    
    alpha[0] = old_zTr[0] / (pAp + eps)

@wp.kernel
def _update_beta(sum: wp.array1d(dtype = float), old_zTr: wp.array1d(dtype = float), beta: wp.array1d(dtype = float), eps: float):
    new_zTr = sum[0]
    beta[0] = (new_zTr / (old_zTr[0] + eps))
    old_zTr[0] = new_zTr

@wp.kernel
def _update_r(r: wp.array3d(dtype = scalar), Ap: wp.array3d(dtype = scalar), alpha: wp.array1d(dtype = scalar)):
    i, j, k = wp.tid()
    r[i, j, k] = r[i, j, k] - alpha[0] * Ap[i, j, k]

@wp.kernel
def _update_p(p: wp.array3d(dtype = scalar), z: wp.array3d(dtype = scalar), beta: wp.array1d(dtype = scalar)):
    i, j, k = wp.tid()
    p[i, j, k] = z[i, j, k] + beta[0] * p[i, j, k]

class MGPCG:
    def __init__(self, dim = 3, shape = (256, 256, 256), n_mg_levels = 6, real = float):
        
        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = 6
        self.bottom_smoothing = 32
        self.dim = dim
        self.real = real

        self.graphs = {
            "default": None,
            "phases": None,
            "fraction": None
        }
        self.r = []
        self.tag = []
        self.z = []
        self.divd = []
        for l in range(self.n_mg_levels):

            _shape = (shape[0] // 2 ** l, shape[1] // 2 ** l, shape[2] // 2 ** l)
            self.r.append(wp.empty(_shape, dtype=self.real))  # residual
            self.tag.append(wp.empty(_shape, dtype=int))  # tag
            self.z.append(wp.empty(_shape, dtype=self.real))  
            self.divd.append(wp.empty(_shape, dtype = self.real))
        
        self.shape = shape
        self.x = wp.empty(self.shape, dtype=self.real)  # solution
        self.p = wp.empty(self.shape, dtype=self.real)  # conjugate gradient
        self.Ap = wp.empty(self.shape, dtype=self.real)
        self.alpha = wp.empty(1, dtype = self.real)
        self.beta = wp.empty(1, dtype = self.real)
        self.sum = wp.empty(1, dtype = self.real)
        self.n_avg = wp.empty(1, dtype = int)
        self.old_zTr = wp.empty(1, dtype = self.real)

    def zero_centered(self):
        self.sum.zero_()
        self.n_avg.zero_()
        wp.launch(_mean, dim = self.shape, inputs = [self.r[0], self.tag[0], self.sum, self.n_avg])
        wp.launch(_sub, dim = self.shape, inputs = [self.r[0], self.tag[0], self.sum, self.n_avg])

    def init(self, r, k):
        self.r[0].zero_()
        self.z[0].zero_()

        self.Ap.zero_()
        self.p.zero_()
        self.x.zero_()
        wp.copy(self.r[0], r)
        wp.launch(_scale_by_3d, dim = self.shape, inputs = [self.r[0], k])

    def graph_precondition(self, smoother = "default"):
        if timer_active: 
            self.apply_preconditioner(smoother)
            return
        assert smoother in ["default", "phases", "fraction"]
        # self.apply_preconditioner(smoother)
        if self.graphs[smoother] is None:
            wp.capture_begin()
            self.apply_preconditioner(smoother)
            self.graphs[smoother] = wp.capture_end()
        else:
            wp.capture_launch(self.graphs[smoother])

    def apply_preconditioner(self, smoother = "default"):
        # wp.copy(self.z[0], self.r[0])
        # return 
        self.z[0].zero_()        
        
        for l in range(self.n_mg_levels - 1):
            shapel = self.r[l].shape
            shape_call = (shapel[0], shapel[1], shapel[2] // 2)

            with wp.ScopedTimer(f"MG level {l} smoother {smoother}", active = timer_active, synchronize=True):
                for i in range(self.pre_and_post_smoothing << l):
                    if smoother == "default":
                        wp.launch(_smooth, dim = self.r[l].shape, inputs = [self.z[l], self.r[l], 0])
                        wp.launch(_smooth, dim = self.r[l].shape, inputs = [self.z[l], self.r[l], 1])
                    else:
                        if not debug: 
                            wp.launch(_smooth_mcadams, dim = shape_call, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l], 0])
                            wp.launch(_smooth_mcadams, dim = shape_call, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l], 1])
                        else:
                            shape_call_tiled = (shapel[0] // tile_size, shapel[1] // tile_size, shapel[2] // tile_size)
                            wp.launch_tiled(_smooth_mcadams_tiled, dim = shape_call_tiled, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l]], block_dim = 64)

            self.z[l + 1].zero_()
            self.r[l + 1].zero_()
            with wp.ScopedTimer(f"MG level {l} restrict {smoother}", active = timer_active):
                if smoother == "default":
                    wp.launch(_restrict, dim = self.r[l].shape, inputs = [self.r[l], self.z[l], self.r[l + 1]])
                else: 
                    wp.launch(_restrict_mcadams, dim = self.r[l].shape, inputs = [self.r[l], self.z[l], self.tag[l], self.r[l + 1], self.tag[l + 1]])
            
        l = self.n_mg_levels - 1
        shapel = self.r[l].shape
        shape_call = (shapel[0], shapel[1], shapel[2] // 2)

        with wp.ScopedTimer(f"MG level {l} bottom smoother {smoother}", active = timer_active, synchronize=True):
            for i in range(self.bottom_smoothing // 2):
                if smoother == "default":
                    wp.launch(_smooth, dim = self.r[l].shape, inputs = [self.z[l], self.r[l], 0])
                    wp.launch(_smooth, dim = self.r[l].shape, inputs = [self.z[l], self.r[l], 1])
                else:
                    if not debug: 
                        wp.launch(_smooth_mcadams, dim = shape_call, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l], 0])
                        wp.launch(_smooth_mcadams, dim = shape_call, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l], 1])
                    else:
                        shape_call_tiled = (shapel[0] // tile_size, shapel[1] // tile_size, shapel[2] // tile_size)
                        wp.launch_tiled(_smooth_mcadams_tiled, dim = shape_call_tiled, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l]], block_dim = 64)

            for i in range(self.bottom_smoothing // 2):
                if smoother == "default":
                    wp.launch(_smooth, dim = self.r[l].shape, inputs = [self.z[l], self.r[l], 1])
                    wp.launch(_smooth, dim = self.r[l].shape, inputs = [self.z[l], self.r[l], 0])
                else:
                    if not debug:
                        wp.launch(_smooth_mcadams, dim = shape_call, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l], 1])
                        wp.launch(_smooth_mcadams, dim = shape_call, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l], 0])
                    else: 
                        shape_call_tiled = (shapel[0] // tile_size, shapel[1] // tile_size, shapel[2] // tile_size)
                        wp.launch_tiled(_smooth_mcadams_tiled, dim = shape_call_tiled, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l]], block_dim = 64)

        for l in reversed(range(self.n_mg_levels - 1)):
            wp.launch(_prolongate, dim = self.z[l].shape, inputs = [self.z[l], self.z[l + 1]])

            shapel = self.r[l].shape
            shape_call = (shapel[0], shapel[1], shapel[2] // 2)
            with wp.ScopedTimer(f"MG level {l} prolongate smoother {smoother}", active = timer_active, synchronize=True):
                for i in range(self.pre_and_post_smoothing << l):
                    if smoother == "default":
                        wp.launch(_smooth, dim = self.r[l].shape, inputs = [self.z[l], self.r[l], 1])
                        wp.launch(_smooth, dim = self.r[l].shape, inputs = [self.z[l], self.r[l], 0])
                    else:
                        if not debug:
                            wp.launch(_smooth_mcadams, dim = shape_call, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l], 1])
                            wp.launch(_smooth_mcadams, dim = shape_call, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l], 0])
                        else: 
                            shape_call_tiled = (shapel[0] // tile_size, shapel[1] // tile_size, shapel[2] // tile_size)
                            wp.launch_tiled(_smooth_mcadams_tiled, dim = shape_call_tiled, inputs = [self.z[l], self.r[l], self.tag[l], self.divd[l]], block_dim = 64)
        
    def update_p(self):
        wp.launch(_update_p, dim = self.shape, inputs = [self.p, self.z[0], self.beta])

    def update_r(self):
        wp.launch(_update_r, dim = self.shape, inputs = [self.r[0], self.Ap, self.alpha])

    def update_x(self):
        wp.launch(_update_x, dim = self.shape, inputs = [self.x, self.p, self.alpha])

    def set_tags(self, tag = None):
        finest = 0
        if tag is not None:
            wp.copy(self.tag[finest], tag)

        wp.launch(_compute_divd, dim = self.divd[finest].shape, inputs = [self.tag[finest], self.divd[finest]])

        for level in range(finest, self.n_mg_levels - 1):
            wp.launch(restrict_tag, dim = self.tag[level + 1].shape, inputs = [self.tag[level], self.tag[level + 1]])
            wp.launch(_compute_divd, dim = self.divd[level + 1].shape, inputs = [self.tag[level + 1], self.divd[level + 1]])
            

    def update_alpha(self, eps):
        wp.launch(_update_alpha, dim = (1), inputs = [self.sum, self.alpha, self.old_zTr, eps])

    def update_beta(self, eps): 
        wp.launch(_update_beta, dim = (1), inputs = [self.sum, self.old_zTr, self.beta, eps])

    def solve(self, max_iters = 1, eps = 0.0, abs_tol = 1e-12, rel_tol = 1e-12, verbose = False, smoother = "default"):
        """
        Solve a Poisson problem.

        :parameter max_iters: Specify the maximal iterations. -1 for no limit.
        :parameter eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter abs_tol: Specify the absolute tolerance of loss.
        :parameter rel_tol: Specify the tolerance of loss relative to initial loss.
        """

        # if smoother != "phases":
        #     self.zero_centered()
        
        array_inner(self.r[0], self.r[0], self.sum)
        initial_rTr = self.sum.numpy()[0]
        tol = max(abs_tol, initial_rTr * rel_tol)

        # self.apply_preconditioner(smoother)
        self.graph_precondition(smoother)
        self.update_p()

        # array_inner(self.z[0], self.r[0], self.sum)
        # old_zTr = self.sum.numpy()[0]
        array_inner(self.z[0], self.r[0], self.old_zTr)

        old_rTr = initial_rTr
        # initial_r = self.r[0].numpy()
        it = 0
        while max_iters == -1 or it < max_iters:
            with wp.ScopedTimer(f"PCG iter {it} smoother {smoother}", active = timer_active, synchronize=True):
                if smoother == "default":
                    # compute_Ap(self.Ap, self.p)
                    wp.launch(compute_Ap, dim = self.shape, inputs = [self.Ap, self.p])
                
                else: 
                    # compute_Ap_mcadams(self.Ap, self.p, self.tag[0])
                    wp.launch(compute_Ap_mcadams, dim = self.shape, inputs = [self.Ap, self.p, self.tag[0]] )
                
                array_inner(self.p, self.Ap, self.sum)
                # pAp = self.sum.numpy()[0]
                # self.alpha.fill_(old_zTr / (pAp + eps))
                self.update_alpha(eps)
                
                self.update_r()

                # if smoother != "phases":
                #     self.zero_centered()
                

                array_inner(self.r[0], self.r[0], self.sum)
                rTr = self.sum.numpy()[0]

            if verbose:
                if it == 0:
                    print(f"initial |residual|_2={math.sqrt(initial_rTr)}")
                print(f"after iter {it + 1}, |residual|_2={math.sqrt(rTr)}")
                # if rTr > old_rTr * 2:
                if False:
                    print("error: residual increased, stopping")

                    print("saving tags & div for diagnostics")
                    tag_np = self.tag[0].numpy()
                    div_np = initial_r
                    np.save("error_tag.npy", tag_np)
                    np.save("error_div.npy", div_np)
                    quit()
                old_rTr = rTr

            if rTr < tol:
                self.update_x()
                break

            # self.apply_preconditioner(smoother)
            self.graph_precondition(smoother)
            
            array_inner(self.z[0], self.r[0], self.sum)
            # new_zTr = self.sum.numpy()[0]
            # self.beta.fill_(new_zTr / (old_zTr + eps))
            # old_zTr = new_zTr
            self.update_beta(eps)

            self.update_x()

            self.update_p()

            it += 1
        print("finished with ", it, " iterations")

    def mask_interior(self):
        wp.launch(_mask_interior, self.tag[0].shape, inputs = [self.tag[0], self.r[0]])

    def analyze(self, tag0):
        from skimage.measure import label
    
        tag = np.maximum(tag0 - 1, np.zeros_like(tag0))
        # tag = np.zeros((3,3, 3), dtype = int)
        # tag[1,1,1] = 1
        labeled_array, num_features = label(tag, connectivity=1, return_num=True,background=1)
        print("num_features = ", num_features)
        for f in range(num_features):
            idx = np.argwhere(labeled_array == f + 1)
            print(f"component {f} length = {len(idx)}")
            if 1 < len(idx)  < 50:
                is_trouble = True
                for i in idx:
                    if tag0[i[0], i[1], i[2]] == 0:
                        is_trouble = False
                        break
                

                if is_trouble:
                    print("trouble component found")
                    for i in idx:
                        tag0[i[0], i[1], i[2]] = 2
        return tag0

@wp.kernel
def _mask_interior(
    tag: wp.array3d(dtype = int),
    r: wp.array3d(dtype = float)
):
    i, j, k = wp.tid()
    if tag[i, j, k] != 1:
        r[i, j, k] = 0.0

@wp.kernel
def _init_test(r: wp.array3d(dtype = float), N: int):
    i, j, k = wp.tid()
    r[i, j, k] = wp.cos(5.0 * wp.pi * float(i) / float(N)) * wp.cos(5.0 * wp.pi * float(j) / float(N)) * wp.cos(5.0 * wp.pi * float(k) / float(N))

@wp.kernel
def _fill_half(tag: wp.array3d(dtype = int), r: wp.array3d(dtype = float)): 
    i, j, k = wp.tid()
    if j < tag.shape[1] // 2:
        tag[i, j, k] = int(1)
    else:
        tag[i, j, k] = int(0)
        r[i, j, k] = 0.0

class MGPCG_Debug(MGPCG):
    def __init__(self):
        super().__init__(dim = 3, shape = (256, 64, 256), n_mg_levels = 3, real = float)
        # tag_np = np.load("../../error_tag.npy")        
        tag_np = self.analyze()
        div_np = np.load("../../error_div.npy")
        tag_tmp = wp.from_numpy(tag_np, dtype = int, shape = self.shape)
        div_tmp = wp.from_numpy(div_np, dtype = float, shape = self.shape)
        self.init(div_tmp, 1.0)
        self.set_tags(tag_tmp)
        self.mask_interior()

    def analyze(self):
        tag0 = np.load("../../error_tag.npy")
        return super().analyze(tag0)

class MGPCG_Test(MGPCG):
    def __init__(self):
        super().__init__(dim = 3, N = 256, n_mg_levels = 6, real = float)

    def init(self):
        self.tag[0].fill_(1)
            
        wp.launch(_init_test, dim = self.shape, inputs = [self.r[0], self.shape[0]])

        self.z[0].zero_()
        self.Ap.zero_()
        self.p.zero_()
        self.x.zero_()

    def set_tags(self):
        finest = 0
        wp.launch(_fill_half, dim = self.tag[0].shape, inputs = [self.tag[0], self.r[0]])
        for level in range(finest, self.n_mg_levels - 1):
            wp.launch(restrict_tag, dim = self.tag[level + 1].shape, inputs = [self.tag[level], self.tag[level + 1]])

    def run(self, verbose = False):
        self.init()
        self.set_tags()
        print("tags mean = ", np.mean(self.tag[0].numpy()))
        # self.mask_interior()
        # self.solve(max_iters = 400, verbose = verbose, smoother = "fraction")
        self.solve(max_iters = 400, verbose = verbose, smoother = "phases")
        # self.solve(max_iters = 400, verbose = verbose, smoother = "default")

if __name__ == "__main__":
    wp.init()
    # mg = MGPCG_Test()
    # mg.run(verbose = True)

    mg = MGPCG_Debug()
    mg.solve(20, verbose = True, smoother = "fraction")
    # mg.analyze()
    # mg.init(wp.zeros((mg.N_tot, mg.N_tot, mg.N_tot), dtype=mg.real), 1.0)
    # wp.launch(compute_Ap, dim = mg.shape, inputs = [mg.Ap])
    
    # print(mg.Ap)
    # print(mg.x)