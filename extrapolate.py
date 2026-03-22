import warp as wp
from const_param import shapeu, shapev, shapew, MACGrid, MACGridInt, get_uf
@wp.func
def is_valid_3d(unknown: wp.array3d(dtype=int), i: int, j: int, k :int):
    ret = 0 <= i and i < unknown.shape[0] and 0 <= j and j < unknown.shape[1] and 0 <= k and k < unknown.shape[2] and unknown[i, j, k] == 0
    return ret

@wp.kernel
def extrapolate_3d(
    mac: MACGrid,
    u: MACGridInt,
    d: int):
    i, j, k = wp.tid()
    qf = get_uf(mac, d).f
    unknown = get_uf(u, d).f
    if unknown[i, j, k] == 1:
        n_avg = int(0)
        q_sum = float(0)
        for ii in range(-1, 2):
            for jj in range(-1, 2):
                for kk in range(-1, 2):
                    if is_valid_3d(unknown, i + ii, j + jj, k + kk):
                        n_avg += 1
                        q_sum += qf[i + ii, j + jj, k+ kk]
        if n_avg > 0:
            qf[i, j, k] = q_sum / float(n_avg)
            unknown[i, j, k] = -1
        else:
            qf[i, j, k] = 0.0

@wp.kernel
def flush_1_3d(tag: wp.array3d(dtype=int)):
    i, j, k = wp.tid()
    if tag[i, j, k] == -1:
        tag[i, j, k] = 1


def extrapolate_by_1(vel: MACGrid, tag: MACGridInt):
    wp.launch(extrapolate_3d, dim=shapeu,
            inputs=[vel, tag, 1])
    wp.launch(extrapolate_3d, dim=shapev,
                inputs=[vel, tag, 2])
    wp.launch(extrapolate_3d, dim=shapew,
                inputs=[vel, tag, 3])
    wp.launch(flush_1_3d, dim=shapeu, inputs=[tag.u.f])
    wp.launch(flush_1_3d, dim=shapev, inputs=[tag.v.f])
    wp.launch(flush_1_3d, dim=shapew, inputs=[tag.w.f])
