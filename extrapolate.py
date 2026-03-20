import warp as wp
@wp.func
def is_valid(unknown: wp.array2d(dtype=int), i: int, j: int):
    ret = 0 <= i and i < unknown.shape[0] and 0 <= j and j < unknown.shape[1] and unknown[i, j] == 0
    return ret

@wp.kernel
def extrapolate(qf: wp.array2d(dtype=float), unknown: wp.array2d(dtype=int)):
    i, j = wp.tid()
    if unknown[i, j] == 1:
        n_avg = int(0)
        q_sum = float(0)
        for ii in range(-1, 2):
            for jj in range(-1, 2):
                if is_valid(unknown, i + ii, j + jj):
                    n_avg += 1
                    q_sum += qf[i + ii, j + jj]
        if n_avg > 0:
            qf[i, j] = q_sum / float(n_avg)
            unknown[i, j] = -1


@wp.func
def is_valid_3d(unknown: wp.array3d(dtype=int), i: int, j: int, k :int):
    ret = 0 <= i and i < unknown.shape[0] and 0 <= j and j < unknown.shape[1] and 0 <= k and k < unknown.shape[2] and unknown[i, j, k] == 0
    return ret



@wp.kernel
def extrapolate_3d(
    qf: wp.array3d(dtype=float), 
    unknown: wp.array3d(dtype=int)):
    i, j, k = wp.tid()
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
def flush_1(tag: wp.array2d(dtype=int)):
    i, j = wp.tid()
    if tag[i, j] == -1:
        tag[i, j] = 1

@wp.kernel
def flush_1_3d(tag: wp.array3d(dtype=int)):
    i, j, k = wp.tid()
    if tag[i, j, k] == -1:
        tag[i, j, k] = 1

# @wp.kernel
# def brush_unknowns(
#     uf: wp.array3d(dtype = float),
#     unknown: wp.array3d(dtype = int)
# ):
#     i, j, k = wp.tid()
#     if unknown[i, j, k] == 1:
#         uf[i, j, k] = 0.0

def extrapolate_by_1(uf, vf, wf, u_unknown, v_unknown, w_unknown):
    shapeu = u_unknown.shape
    shapev = v_unknown.shape
    shapew = w_unknown.shape
    wp.launch(extrapolate_3d, dim=shapeu,
            inputs=[uf, u_unknown])
    wp.launch(extrapolate_3d, dim=shapev,
                inputs=[vf, v_unknown])
    wp.launch(extrapolate_3d, dim=shapew,
                inputs=[wf, w_unknown])
    wp.launch(flush_1_3d, dim=shapeu, inputs=[u_unknown])
    wp.launch(flush_1_3d, dim=shapev, inputs=[v_unknown])
    wp.launch(flush_1_3d, dim=shapev, inputs=[w_unknown])
