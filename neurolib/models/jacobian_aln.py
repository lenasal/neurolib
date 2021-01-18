import numpy as np
import numba
import logging

@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64, "idx1": numba.int64, "idy1": numba.int64})
def interpolate(sigma_f, sigmarange, ds, muf, Irange, dI, C, precalc_table):
    xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigma_f, Irange, dI, muf)
    xid1, yid1 = int(xid1), int(yid1)
    result = interpolate_values(precalc_table, xid1, yid1, dxid, dyid)
    return result

# gradient of transfer function wrt changes in sigma
@numba.njit
def der_sigma(sigma_f, sigmarange, ds, muf, Irange, dI, C, precalc_table):
    delta_s = ds * 0.01
    result0 = interpolate(sigma_f, sigmarange, ds, muf, Irange, dI, C, precalc_table)
    result1 = interpolate(sigma_f + delta_s, sigmarange, ds, muf, Irange, dI, C, precalc_table)
    
    der = ( result1 - result0) / delta_s
            
    return der

# gradient of transfer function wrt changes in mu
@numba.njit
def der_mu(sigma_f, sigmarange, ds, muf, Irange, dI, C, precalc_table):
    
    """
    if muf >= Irange[-1]:
        return der_mu(sigma_f, sigmarange, ds, Irange[-2], Irange, dI, C, precalc_table)
    if sigma_f >= sigmarange[-1]:
        return der_mu(sigmarange[-2], sigmarange, ds, Irange[-2], Irange, dI, C, precalc_table)
    """
    
    """
    lim_a = -0.6
    lim_b = 0.5
    if sigma_f <= lim_a * muf + lim_b:
        return 1e-10
    """
    
    delta_I = dI * 0.01    
    result0 = interpolate(sigma_f, sigmarange, ds, muf, Irange, dI, C, precalc_table)
    result1 = interpolate(sigma_f, sigmarange, ds, muf + delta_I, Irange, dI, C, precalc_table)
    
    der = ( result1 - result0) / delta_I
    
    """
    if np.abs(der) < 1e-10:
        if der >= 0.:
            return 1e-10
        elif der < 0.:
            return - 1e-10
    """
    
    #if (np.abs(der1 - der2) > 10-8):    
    #    print("WARNING: Large difference in der : ", der1 - der2)
            
    return der

@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64})
def interpolate_values(table, xid1, yid1, dxid, dyid):
    output = (
        table[yid1, xid1] * (1 - dxid) * (1 - dyid)
        + table[yid1, xid1 + 1] * dxid * (1 - dyid)
        + table[yid1 + 1, xid1] * (1 - dxid) * dyid
        + table[yid1 + 1, xid1 + 1] * dxid * dyid
    )
    return output

@numba.njit(locals={"xid1": numba.int64, "yid1": numba.int64, "dxid": numba.float64, "dyid": numba.float64})
def fast_interp2_opt(x, dx, xi, y, dy, yi):

    """
    Returns the values needed for interpolation:
    - bilinear (2D) interpolation within ranges,
    - linear (1D) if "one edge" is crossed,
    - corner value if "two edges" are crossed
    x     ... range of the x value
    xi    ... interpolation value on x-axis
    dx    ... grid width of x ( dx = x[1]-x[0] )
    (same for y)
    return:   xid1    ... index of the lower interpolation value
              dxid    ... distance of xi to the lower interpolation value
              (same for y)
    """
    
    #print("sigma boundaries, value : ", x[0], x[-1], xi)
    #print("current boundaries, value : ", y[0], y[-1], yi)
    
    xid1, yid1, dxid, dyid = 0., 0., 0., 0.

    # within all boundaries
    if xi >= x[0] and xi < x[-1] and yi >= y[0] and yi < y[-1]:
        #print("case 1")
        xid = (xi - x[0]) / dx
        xid1 = np.floor(xid)
        dxid = xid - xid1
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return xid1, yid1, dxid, dyid

    # outside one boundary
    if yi < y[0]:
        #print("case 2")
        yid1 = 0
        dyid = 0.0
        if xi >= x[0] and xi < x[-1]:
            xid = (xi - x[0]) / dx
            xid1 = np.floor(xid)
            dxid = xid - xid1

        elif xi < x[0]:
            xid1 = 0
            dxid = 0.0
        else:  # xi >= x(end)
            xid1 = -1
            dxid = 0.0
        return xid1, yid1, dxid, dyid

    if yi >= y[-1]:
        #print("case 3")
        yid1 = -1
        dyid = 0.0
        if xi >= x[0] and xi < x[-1]:
            xid = (xi - x[0]) / dx
            xid1 = np.floor(xid)
            dxid = xid - xid1

        elif xi < x[0]:
            xid1 = 0
            dxid = 0.0

        else:  # xi >= x(end)
            xid1 = -1
            dxid = 0.0
        return xid1, yid1, dxid, dyid

    if xi < x[0]:
        #print("case 4")
        xid1 = 0
        dxid = 0.0
        # We know that yi is within the boundaries
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return xid1, yid1, dxid, dyid

    if xi >= x[-1]:
        #print("case 5")
        xid1 = -1
        dxid = 0.0
        # We know that yi is within the boundaries
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1
        

    return xid1, yid1, dxid, dyid