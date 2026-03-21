import warp as wp 
from utils.util import sample_3d, fraction_of_fluid
from const_param import Field, MACGrid, get_uf, IntField
eps = 1e-5


@wp.kernel
def u_subtract_gradient_batty_3d(
        uf: wp.array3d(dtype=float),
        pf: wp.array3d(dtype=float),
        fu: wp.array3d(dtype=float),
        phi: wp.array3d(dtype=float),
        u_rigid: wp.array3d(dtype=float),
        l: float, r: float, h: float):
    i, j, k = wp.tid()
    gridx = fu.shape[0] - 1
    if 0 < i and i < gridx:
        if phi[i, j,  k] < 0.0 or phi[i - 1, j, k] < 0.0:

            if fu[i, j,  k] == 0.0:
                uf[i, j,  k] = u_rigid[i, j, k]
            else:
                pl = wp.where(phi[i - 1, j, k] >= 0,
                               0.0, sample_3d(pf, i - 1, j, k))
                pr = wp.where(phi[i, j,  k] >= 0,
                               0.0, sample_3d(pf, i, j,  k))
                theta = wp.max(0.01, fraction_of_fluid(
                    phi[i, j,  k], phi[i - 1, j, k]))
                uf[i, j,  k] = uf[i, j,  k] - (pr - pl) / theta
        else:
            uf[i, j, k] = 0.0
    else:
        if l <= eps or l + float(i) * h >= 1.0 - eps:
            uf[i, j,  k] = 0.0
            pass

@wp.kernel
def subtract_gradient_tag(
    vel: MACGrid,
    vel_rigid: MACGrid,
    pf: Field,
    tag: IntField,
    d: int):

    i, j, k = wp.tid()
    idx = wp.vec3i(i, j, k)
    ii = idx[d - 1]
    uf = get_uf(vel, d).f
    u_rigid = get_uf(vel_rigid, d).f
    gridx = uf.shape[d - 1] - 1
    
    if 0 < ii and ii < gridx:
        il = wp.vec3i(idx)
        il[d - 1] -= 1
        pl = pf.f[il[0], il[1], il[2]]
        pr = pf.f[i, j, k]

        tagl = tag.f[il[0], il[1], il[2]]
        tagr = tag.f[i, j, k]

        if tagl == 2 or tagr == 2:
            uf[i, j, k] = u_rigid[i, j, k]
        elif tagl == 1 or tagr == 1:
            uf[i, j, k] = uf[i, j, k] - (pr - pl)
        else:
            uf[i, j, k] = 0.0
    else:
        uf[i, j, k] = 0.0

@wp.kernel
def v_subtract_gradient_batty_3d(
    vf: wp.array3d(dtype=float),
    pf: wp.array3d(dtype=float),
    fv: wp.array3d(dtype=float),
    phi: wp.array3d(dtype = float),
    v_rigid: wp.array3d(dtype=float),
    b: float, t: float, h: float):
    i, j, k = wp.tid()
    gridy = fv.shape[1] - 1
    if 0 < j and j < gridy:
        if phi[i, j, k] < 0.0 or phi[i, j - 1, k] < 0.0:
            
            if fv[i, j, k] == 0.0:
                vf[i, j, k] = v_rigid[i, j, k]
            else: 
                pb = wp.where(phi[i, j - 1, k] >= 0, 0.0, sample_3d(pf, i, j - 1, k))
                pt = wp.where(phi[i, j, k] >= 0, 0.0, sample_3d(pf, i, j, k))
                theta = wp.max(0.01, fraction_of_fluid(phi[i, j, k], phi[i, j - 1, k]))
                vf[i, j, k] = vf[i, j, k] - (pt - pb) / theta
        else:
            vf[i, j, k] = 0.0
    else:
        if b <= eps or b + float(j) * h >= 1.0 - eps:
            vf[i, j,  k] = 0.0
            pass

@wp.kernel
def w_subtract_gradient_batty_3d(
    wf: wp.array3d(dtype=float),
    pf: wp.array3d(dtype=float),
    fw: wp.array3d(dtype=float),
    phi: wp.array3d(dtype = float),
    w_rigid: wp.array3d(dtype=float),
    far: float, near: float, h: float):
    i, j, k = wp.tid()
    gridz = fw.shape[2] - 1
    if 0 < k and k < gridz:
        if phi[i, j, k] < 0.0 or phi[i, j, k - 1] < 0.0:
            
            if fw[i, j, k] == 0.0:
                wf[i, j, k] = w_rigid[i, j, k]
            else: 
                pfar = wp.where(phi[i, j, k - 1] >= 0, 0.0, sample_3d(pf, i, j, k - 1))
                pnear = wp.where(phi[i, j, k] >= 0, 0.0, sample_3d(pf, i, j, k))
                theta = wp.max(0.01, fraction_of_fluid(phi[i, j, k - 1], phi[i, j, k]))
                wf[i, j, k] = wf[i, j, k] - (pnear - pfar) / theta
        else:
            wf[i, j, k] = 0.0
    else:
        if far <= eps or far + float(k) * h >= 1.0 - eps:
            wf[i, j, k] = 0.0
            pass

@wp.kernel
def u_subtract_gradient_fractioned_3d(
    uf: wp.array3d(dtype=float),
    pf: wp.array3d(dtype=float),
    fu: wp.array3d(dtype=float),
    l: float, b: float, f: float, h: float):

    
    i, j, k = wp.tid()
    t = l + float(i) * h
    if not (eps < t and t < 1.0 - eps):
        uf[i, j, k] = 0.0
    elif fu[i, j, k] > 0.0:
        pl = sample_3d(pf, i - 1, j, k)
        pr = sample_3d(pf, i, j, k)
        uf[i, j, k] = uf[i, j, k] - (pr - pl)
    # else: 
    #     uf[i, j, k] = 0.0

@wp.kernel
def v_subtract_gradient_fractioned_3d(
    vf: wp.array3d(dtype=float),
    pf: wp.array3d(dtype=float),
    fv: wp.array3d(dtype=float),
    l: float, b: float, f: float, h: float):

    i, j, k = wp.tid()
    t = b + float(j) * h
    gridy = vf.shape[1] - 1
    if not (eps < t and t < h * float(gridy) - eps):
        vf[i, j, k] = 0.0
    elif fv[i, j, k] > 0.0:
        pb = sample_3d(pf, i, j - 1, k)
        pt = sample_3d(pf, i, j, k)
        vf[i, j, k] = vf[i, j, k] - (pt - pb)
    # else: 
    #     vf[i, j, k] = 0.0

@wp.kernel
def w_subtract_gradient_fractioned_3d(
    wf: wp.array3d(dtype=float),
    pf: wp.array3d(dtype=float),
    fw: wp.array3d(dtype=float),
    l: float, b: float, f: float, h: float):

    i, j, k = wp.tid()
    t = f + float(k) * h
    gridz = fw.shape[2] - 1
    if not (eps < t and t < h * float(gridz) - eps):
        wf[i, j, k] = 0.0
    elif fw[i, j, k] > 0.0:
        pfar = sample_3d(pf, i, j, k - 1)
        pnear = sample_3d(pf, i, j, k)
        wf[i, j, k] = wf[i, j, k] - (pnear - pfar)
    # else: 
    #     wf[i, j, k] = 0.0