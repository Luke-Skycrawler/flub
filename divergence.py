import warp as wp
from utils.util import sample_3d
from const_param import Field, MACGrid, IntField
@wp.kernel
def divergence_fractional_3d(
    uf: wp.array3d(dtype = float), 
    vf: wp.array3d(dtype = float), 
    wf: wp.array3d(dtype = float),
    velocity_divs: wp.array3d(dtype = float), 
    fu: wp.array3d(dtype = float), 
    fv: wp.array3d(dtype = float), 
    fw: wp.array3d(dtype = float),
    u_rigid: wp.array3d(dtype = float), 
    v_rigid: wp.array3d(dtype = float),
    w_rigid: wp.array3d(dtype = float)):
    '''3d divergence kernel'''
    i, j, k = wp.tid()
    
    vl = sample_3d(uf, i, j, k)
    vr = sample_3d(uf, i + 1, j, k)
    vb = sample_3d(vf, i, j, k)
    vt = sample_3d(vf, i, j + 1, k)
    _vf = sample_3d(wf, i, j, k)
    _vn = sample_3d(wf, i, j, k + 1)

    l = sample_3d(fu, i, j, k)
    r = sample_3d(fu, i + 1, j, k)
    b = sample_3d(fv, i, j, k)
    t = sample_3d(fv, i, j + 1, k)
    f = sample_3d(fw, i, j, k)
    n = sample_3d(fw, i, j, k + 1)

    vls = sample_3d(u_rigid, i, j, k)
    vrs = sample_3d(u_rigid, i + 1, j, k)
    vbs = sample_3d(v_rigid, i, j, k)
    vts = sample_3d(v_rigid, i, j + 1, k)
    vfs = sample_3d(w_rigid, i, j, k)
    vns = sample_3d(w_rigid, i, j, k + 1)
    
    velocity_divs[i, j, k] = (vr * r - vl * l + vt * t - vb * b + _vn * n - _vf * f) + \
        ((
            1.0 - r) * vrs - (1.0 - l) * vls + (1.0 - t) * vts - (1.0 - b) * vbs + (1.0 - n) * vns - (1.0 - f) * vfs)
@wp.kernel
def divergence_tag(
    vels: MACGrid, 
    velocity_divs: Field, 
    vel_rigid: MACGrid, 
    tag: IntField):
    '''3d divergence kernel'''

    res = tag.f.shape[0]
    i, j, k = wp.tid()

    uf = vels.u.f
    vf = vels.v.f
    wf = vels.w.f
    u_rigid = vel_rigid.u.f 
    v_rigid = vel_rigid.v.f
    w_rigid = vel_rigid.w.f
    
    vl = sample_3d(uf, i, j, k)
    vr = sample_3d(uf, i + 1, j, k)
    vb = sample_3d(vf, i, j, k)
    vt = sample_3d(vf, i, j + 1, k)
    _vf = sample_3d(wf, i, j, k)
    _vn = sample_3d(wf, i, j, k + 1)


    if i == 0:
        vl = 0.0
    elif tag.f[i - 1, j, k] == 2:
        vl = u_rigid[i, j, k]
    
    if i == res -1:
        vr = 0.0
    elif tag.f[i + 1, j, k] == 2:
        vr = u_rigid[i + 1, j, k]

    if j == 0:
        vb = 0.0
    elif tag.f[i, j - 1, k] == 2:
        vb = v_rigid[i, j, k]
    
    if j == res - 1:
        vt = 0.0
    elif tag.f[i, j + 1, k] == 2:
        vt = v_rigid[i, j + 1, k]
    
    if k == 0:
        _vf = 0.0
    elif tag.f[i, j, k - 1] == 2:
        _vf = w_rigid[i, j, k]
    
    if k == res - 1:
        _vn = 0.0
    elif tag.f[i, j, k + 1] == 2:
        _vn = w_rigid[i, j, k + 1]
    
    velocity_divs.f[i, j, k] = (vr - vl + vt - vb + _vn - _vf)

@wp.kernel
def divergence_tag_open(
    uf: wp.array3d(dtype = float), 
    vf: wp.array3d(dtype = float), 
    wf: wp.array3d(dtype = float),
    velocity_divs: wp.array3d(dtype = float), 
    u_rigid: wp.array3d(dtype = float), 
    v_rigid: wp.array3d(dtype = float),
    w_rigid: wp.array3d(dtype = float),
    tag: wp.array3d(dtype = int)):
    '''3d divergence kernel'''

    res = tag.shape[0]
    i, j, k = wp.tid()
    
    vl = sample_3d(uf, i, j, k)
    vr = sample_3d(uf, i + 1, j, k)
    vb = sample_3d(vf, i, j, k)
    vt = sample_3d(vf, i, j + 1, k)
    _vf = sample_3d(wf, i, j, k)
    _vn = sample_3d(wf, i, j, k + 1)


    if i == 0:
        # vl = 0.0
        pass
    elif tag[i - 1, j, k] == 2:
        vl = u_rigid[i, j, k]
    
    if i == res -1:
        # vr = 0.0
        pass
    elif tag[i + 1, j, k] == 2:
        vr = u_rigid[i + 1, j, k]

    if j == 0:
        # vb = 0.0
        pass
    elif tag[i, j - 1, k] == 2:
        vb = v_rigid[i, j, k]
    
    if j == res - 1:
        # vt = 0.0
        pass
    elif tag[i, j + 1, k] == 2:
        vt = v_rigid[i, j + 1, k]
    
    if k == 0:
        # _vf = 0.0
        pass
    elif tag[i, j, k - 1] == 2:
        _vf = w_rigid[i, j, k]
    
    if k == res - 1:
        # _vn = 0.0
        pass
    elif tag[i, j, k + 1] == 2:
        _vn = w_rigid[i, j, k + 1]
    
    velocity_divs[i, j, k] = (vr - vl + vt - vb + _vn - _vf)

@wp.kernel
def invalidate(
    velocity_divs: wp.array3d(dtype = float), 
    phi: wp.array3d(dtype = float)
):
    '''div = 0 for phi >= 0'''
    i, j, k = wp.tid()
    
    if phi[i, j, k] >= 0.0:
        velocity_divs[i, j, k] = 0.0

@wp.kernel
def mask_with_tag(
    div: Field,
    tag: IntField 
):
    i, j, k = wp.tid()  
    if tag.f[i, j, k] != 1:
        div.f[i, j, k] = 0.0

