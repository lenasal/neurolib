import numpy as np
import numba
import logging

def interpolate(model, sigma_f, muf, IAmin1, precalc_table):
    sigmarange = model.params.sigmarange
    ds = model.params.ds
    Irange = model.params.Irange
    dI = model.params.dI
    C = model.params.C
    if model.name == "alnSimp":
        C = 1.
            
    xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigma_f, Irange, dI, muf - IAmin1 / C)
    xid1, yid1 = int(xid1), int(yid1)
    result = interpolate_values(precalc_table, xid1, yid1, dxid, dyid)
    return result

# gradient of transfer function wrt changes in sigma
def der_sigma(model, sigma_f, muf, IAmin1, precalc_table):
    ds = model.params.ds
    result0 = interpolate(model, sigma_f, muf, IAmin1, precalc_table)
    result1 = interpolate(model, sigma_f + ds, muf, IAmin1, precalc_table)
    
    der = ( result1 - result0) / ds
            
    return der

# gradient of transfer function wrt changes in mu
def der_mu_up(model, sigma_f, muf, IAmin1, precalc_table):
    dI = model.params.dI
    
    result0 = interpolate(model, sigma_f, muf, IAmin1, precalc_table)
    result1 = interpolate(model, sigma_f, muf + dI, IAmin1, precalc_table)
    result2 = interpolate(model, sigma_f, muf - dI, IAmin1, precalc_table)
        
    der1 = ( result1 - result0) / dI
    der2 = -( result2 - result0) / dI
    
    der_analytical = 0.001 * (1./np.cosh(muf)**2)
    
    #("difference in der : ", der1 - der_analytical)
            
    return der_analytical

def der_mu_down(model, sigma_f, muf, IAmin1, precalc_table):
    dI = model.params.dI
    
    result0 = interpolate(model, sigma_f, muf, IAmin1, precalc_table)
    result1 = interpolate(model, sigma_f, muf + dI, IAmin1, precalc_table)
    result2 = interpolate(model, sigma_f, muf - dI, IAmin1, precalc_table)
        
    der1 = ( result1 - result0) / dI
    der2 = -( result2 - result0) / dI
    
    #print("difference in der : ", der1 - der2)
            
    return der2


def interpolate_values(table, xid1, yid1, dxid, dyid):
    output = (
        table[yid1, xid1] * (1 - dxid) * (1 - dyid)
        + table[yid1, xid1 + 1] * dxid * (1 - dyid)
        + table[yid1 + 1, xid1] * (1 - dxid) * dyid
        + table[yid1 + 1, xid1 + 1] * dxid * dyid
    )
    return output


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
