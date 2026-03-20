import numpy as np
from const_param import *
from utils.util import sample_3d, fraction_of_fluid

vec6 = wp.types.vector(length = 6, dtype = float)

@wp.func
def update(phi: wp.array3d(dtype = float), _i: int, _j: int, _k: int, incx: int, incy: int, incz: int, h: float):
    i, j, k = _i, _j, _k

    if incx < 0:
        i = phi.shape[0] - 1 - _i
    if incy < 0:
        j = phi.shape[1] - 1 - _j
    if incz < 0:
        k = phi.shape[2] - 1 - _k

    is_boundary = i == 0 or i == phi.shape[0] - 1 or j == 0 or j == phi.shape[1] - 1 or k == 0 or k == phi.shape[2] - 1

    if wp.abs(phi[i, j, k]) > h and not is_boundary:
        phii = sample_3d(phi, i - incx, j, k)
        phij = sample_3d(phi, i, j - incy, k)   
        phik = sample_3d(phi, i, j, k - incz)

        phiii = sample_3d(phi, i + incx, j, k)
        phijj = sample_3d(phi, i, j + incy, k)
        phikk = sample_3d(phi, i, j, k + incz)


        sgn = float(1.0)
        if phii < 0.0:
            # phii *= -1.0
            # phij *= -1.0
            # phik *= -1.0
            phii = - wp.max(phii, phiii)
            phij = - wp.max(phij, phijj)
            phik = - wp.max(phik, phikk)
            sgn = -1.0

        else :
            phii = wp.min(phii, phiii)
            phij = wp.min(phij, phijj)
            phik = wp.min(phik, phikk)
        
        phi0 = wp.min(wp.vec3(phii, phij, phik))
        phi2 = wp.max(wp.vec3(phii, phij, phik)) 
        phi1 = phii + phij + phik - phi0 - phi2

        d = phi0 + h
        if d > phi1:
            d = 0.5 * (phi0 + phi1 + wp.sqrt(2.0 * h * h - (phi0 - phi1) ** 2.0))
            if d > phi2: 
                term = (phi0 + phi1 + phi2) ** 2.0 - 3.0 * (phi0 ** 2.0 + phi1 ** 2.0 + phi2 ** 2.0 - h ** 2.0)
                d = 1.0 / 3.0 * (phi0 + phi1 + phi2 + wp.sqrt(wp.max(0.0, term)))
        phi[i, j, k] = sgn * wp.min(d, wp.abs(phi[i, j, k]))

        
@wp.kernel
def _sweep(phi: wp.array3d(dtype = float), incx: int, incy: int, incz: int, odd: int, h: float):
    _i, _j, _k = wp.tid()

    if (_i + _j + _k) % 2 == odd:
        update(phi, _i, _j, _k, incx, incy, incz, h)
    

@wp.kernel
def _init_distance(phi: wp.array3d(dtype = float), h: float, set_boundary: int):
    i, j, k = wp.tid()
    pl = sample_3d(phi, i - 1, j, k)
    pr = sample_3d(phi, i + 1, j, k)
    pb = sample_3d(phi, i, j - 1, k)
    pt = sample_3d(phi, i, j + 1, k)
    pf = sample_3d(phi, i, j, k - 1)
    pn = sample_3d(phi, i, j, k + 1)
    pc = sample_3d(phi, i, j, k)

    if i == 0:
        pl = 0.5 * h
    if i == phi.shape[0] - 1:
        pr = 0.5 * h
    if j == 0:
        pb = 0.5 * h
    if j == phi.shape[1] - 1:
        pt = 0.5 * h
    if k == 0:
        pf = 0.5 * h
    if k == phi.shape[2] - 1:
        pn = 0.5 * h

    is_boundary = i == 0 or i == phi.shape[0] - 1 or j == 0 or j == phi.shape[1] - 1 or k == 0 or k == phi.shape[2] - 1

    if is_boundary and not set_boundary: 
        pass 
    elif pc > 0.0:
        if not is_boundary and pl < 0.0 or pr < 0.0 or pb < 0.0 or pt < 0.0 or pf < 0.0 or pn < 0.0:
            fl = 1.0 - fraction_of_fluid(pc, pl)
            fr = 1.0 - fraction_of_fluid(pc, pr)
            fb = 1.0 - fraction_of_fluid(pc, pb)
            ft = 1.0 - fraction_of_fluid(pc, pt)
            ff = 1.0 - fraction_of_fluid(pc, pf)
            fn = 1.0 - fraction_of_fluid(pc, pn)

            phi[i, j, k] = h * wp.min(vec6(fl, fr, fb, ft, ff, fn))

        else :
            phi[i, j, k] = 2.0
        
    elif pc < 0.0:
        if not is_boundary and pl > 0.0 or pr > 0.0 or pb > 0.0 or pt > 0.0 or pf > 0.0 or pn > 0.0:
            fl = fraction_of_fluid(pc, pl)
            fr = fraction_of_fluid(pc, pr)
            fb = fraction_of_fluid(pc, pb)
            ft = fraction_of_fluid(pc, pt)
            ff = fraction_of_fluid(pc, pf)
            fn = fraction_of_fluid(pc, pn)

            phi[i, j, k] = -h * wp.min(vec6(fl, fr, fb, ft, ff, fn))
        else: 
            phi[i, j, k] = -2.0

def sweep(phi, h, incx, incy, incz):
    for _i in range(2):
        wp.launch(_sweep, dim = phi.shape, inputs = [phi, incx, incy, incz, _i % 2, h])

def redistance(phi, h = h0, set_boundary = 1):
    wp.launch(_init_distance, dim = phi.shape, inputs = [phi, h, set_boundary])

    for _ in range(2):
        sweep(phi, h, 1, 1, -1)
        sweep(phi, h, -1, 1, -1)
        sweep(phi, h, -1, -1, -1)
        sweep(phi, h, 1, -1, -1)
        sweep(phi, h, 1, 1, 1)
        sweep(phi, h, -1, 1, 1)
        sweep(phi, h, -1, -1, 1)
        sweep(phi, h, 1, -1, 1)
