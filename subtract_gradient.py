import warp as wp 
from utils.util import sample_3d, fraction_of_fluid
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
def u_subtract_gradient_tag(
        uf: wp.array3d(dtype=float),
        u_rigid: wp.array3d(dtype=float),
        pf: wp.array3d(dtype=float),
        tag: wp.array3d(dtype = int),
        l: float, r: float, h: float):
    i, j, k = wp.tid()
    gridx = uf.shape[0] - 1
    if 0 < i and i < gridx:

        pl = pf[i - 1, j, k]
        pr = pf[i, j, k]

        tagl = tag[i - 1, j, k]
        tagr = tag[i, j, k]

        if tagl == 2 or tagr == 2:
            uf[i, j, k] = u_rigid[i, j, k]
        elif tagl == 1 or tagr == 1:
            uf[i, j, k] = uf[i, j, k] - (pr - pl)
        else:
            uf[i, j, k] = 0.0
    elif l == 0.0:
        uf[i, j, k] = 0.0

@wp.kernel
def u_subtract_gradient_tag_bc(
        uf: wp.array3d(dtype=float),
        u_rigid: wp.array3d(dtype=float),
        pf: wp.array3d(dtype=float),
        tag: wp.array3d(dtype = int),
        boundary_p: wp.array3d(dtype = float),
        l: float, r: float, h: float):
    i, j, k = wp.tid()
    gridx = uf.shape[0] - 1
    # if 0 < i and i < gridx:
    if True:

        pl = wp.where(i > 0, pf[i - 1, j, k], boundary_p[0, j, k])
        pr = wp.where(i < uf.shape[0] - 1, pf[i, j, k], boundary_p[1, j, k])

        tagl = wp.where(i > 0, tag[i - 1, j, k], 1)
        tagr = wp.where(i < uf.shape[0] - 1, tag[i, j, k], 1)

        if tagl == 2 or tagr == 2:
            uf[i, j, k] = u_rigid[i, j, k]
        elif tagl == 1 or tagr == 1:
            uf[i, j, k] = uf[i, j, k] - (pr - pl)
        else:
            uf[i, j, k] = 0.0

@wp.kernel
def v_subtract_gradient_tag(
        vf: wp.array3d(dtype=float),
        v_rigid: wp.array3d(dtype=float),
        pf: wp.array3d(dtype=float),
        tag: wp.array3d(dtype = int),
        l: float, r: float, h: float):
    i, j, k = wp.tid()
    gridy = vf.shape[1] - 1
    if 0 < j and j < gridy:

        pb = pf[i, j - 1, k]
        pt = pf[i, j, k]

        tagb = tag[i, j - 1, k]
        tagt = tag[i, j, k]

        if tagb == 2 or tagt == 2:
            vf[i, j, k] = v_rigid[i, j, k]
        elif tagb == 1 or tagt == 1:
            vf[i, j, k] = vf[i, j, k] - (pt - pb)
        else:
            vf[i, j, k] = 0.0
    elif l == 0.0:
        vf[i, j, k] = 0.0

@wp.kernel
def v_subtract_gradient_tag_bc(
        vf: wp.array3d(dtype=float),
        v_rigid: wp.array3d(dtype=float),
        pf: wp.array3d(dtype=float),
        tag: wp.array3d(dtype = int),
        boundary_p: wp.array3d(dtype = float),
        l: float, r: float, h: float):
    i, j, k = wp.tid()
    gridy = vf.shape[1] - 1
    # if 0 < j and j < gridy:
    if True:
        pb = wp.where(j > 0, pf[i, j - 1, k], boundary_p[2, i, k])
        pt = wp.where(j < vf.shape[1] - 1, pf[i, j, k], boundary_p[3, i, k])

        tagb = wp.where(j > 0, tag[i, j - 1, k], 1)
        tagt = wp.where(j < vf.shape[1] - 1, tag[i, j, k], 1)

        if tagb == 2 or tagt == 2:
            vf[i, j, k] = v_rigid[i, j, k]
        elif tagb == 1 or tagt == 1:
            vf[i, j, k] = vf[i, j, k] - (pt - pb)
        else:
            vf[i, j, k] = 0.0

@wp.kernel
def w_subtract_gradient_tag(
        wf: wp.array3d(dtype=float),
        w_rigid: wp.array3d(dtype=float),
        pf: wp.array3d(dtype=float),
        tag: wp.array3d(dtype = int),
        l: float, r: float, h: float):
    i, j, k = wp.tid()
    gridz = wf.shape[2] - 1
    if 0 < k and k < gridz:

        pfar = pf[i, j, k - 1]
        pnear = pf[i, j, k]

        tagfar = tag[i, j, k - 1]
        tagnear = tag[i, j, k]

        if tagfar == 2 or tagnear == 2:
            wf[i, j, k] = w_rigid[i, j, k]
        elif tagfar == 1 or tagnear == 1:
            wf[i, j, k] = wf[i, j, k] - (pnear - pfar)
        else:
            wf[i, j, k] = 0.0
    elif l == 0.0:
        wf[i, j, k] = 0.0


@wp.kernel
def w_subtract_gradient_tag_bc(
        wf: wp.array3d(dtype=float),
        w_rigid: wp.array3d(dtype=float),
        pf: wp.array3d(dtype=float),
        tag: wp.array3d(dtype = int),
        boundary_p: wp.array3d(dtype = float),
        l: float, r: float, h: float):
    i, j, k = wp.tid()
    gridz = wf.shape[2] - 1
    # if 0 < k and k < gridz:
    if True:
        pfar = wp.where(k > 0, pf[i, j, k - 1], boundary_p[4, i, j])
        pnear = wp.where(k < wf.shape[2] - 1, pf[i, j, k], boundary_p[5, i, j])

        tagfar =  wp.where(k > 0, tag[i, j, k - 1], 1)
        tagnear = wp.where(k < wf.shape[2] - 1, tag[i, j, k], 1)

        if tagfar == 2 or tagnear == 2:
            wf[i, j, k] = w_rigid[i, j, k]
        elif tagfar == 1 or tagnear == 1:
            wf[i, j, k] = wf[i, j, k] - (pnear - pfar)
        else:
            wf[i, j, k] = 0.0


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