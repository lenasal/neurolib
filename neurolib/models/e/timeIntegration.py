import numpy as np
import numba
import logging

from . import loadDefaultParams as dp


def timeIntegration(params, control):
    """Sets up the parameters for time integration
    
    Return:
      t:          L array     : time in ms
      mufe:       N vector    : final value of mufe for each node

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """

    N = params["N"]
    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)

    # Initialization
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    rates_exc = np.zeros((N, len(t)+1))
    mufe = np.zeros((N, len(t)+1))
    tau_exc = np.zeros((N, len(t)+1))
    ext_exc_current = np.zeros((N, len(t)+1))

    rates_exc[:,0] = params["rates_exc_init"]
    mufe[:,0] = params["mufe_init"]
    ext_exc_current[:,:] = params["ext_exc_current"]
    
    dI = params["dI"]
    ds = params["ds"]
    sigmarange = params["sigmarange"]
    Irange = params["Irange"]
    
    precalc_r = params["precalc_r"]

    control_ext = control.copy()
    
    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        N,
        dt,
        t,
        rates_exc,
        mufe,
        tau_exc,
        dI,
        ds,
        sigmarange,
        Irange,
        precalc_r,
        ext_exc_current,
        control_ext,
    )


#@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64, "idx1": numba.int64, "idy1": numba.int64})
def timeIntegration_njit_elementwise(
        N,
        dt,
        t,
        rates_exc,
        mufe,
        tau_exc,
        dI,
        ds,
        sigmarange,
        Irange,
        precalc_r,
        ext_exc_current,
        control_ext,
):
    
    for i in range(1,len(t)+1):
        for no in range(N):
            
            xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, 1.5, Irange, dI, mufe[no,i-1])
            xid1, yid1 = int(xid1), int(yid1)
            rates_exc[no,i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3  # convert kHz to Hz
            
            tau_exc[no,i] = 1.
            
            mufe_rhs = (control_ext[no,0,i] + ext_exc_current[no,i] - mufe[no,i-1] ) / tau_exc[no,i]
            mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
            #rates_exc[no,i] = mufe[no,i-1]
            
    tau_exc[:,0] = 1.
              
    return t, rates_exc, mufe, tau_exc


def interpolate_values(table, xid1, yid1, dxid, dyid):
    output = (
        table[yid1, xid1] * (1 - dxid) * (1 - dyid)
        + table[yid1, xid1 + 1] * dxid * (1 - dyid)
        + table[yid1 + 1, xid1] * (1 - dxid) * dyid
        + table[yid1 + 1, xid1 + 1] * dxid * dyid
    )
    return output


#@numba.njit(locals={"xid1": numba.int64, "yid1": numba.int64, "dxid": numba.float64, "dyid": numba.float64})
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
    
    xid1, yid1, dxid, dyid = -1000, -1000, -1000, -1000

    # within all boundaries
    if xi >= x[0] and xi < x[-1] and yi >= y[0] and yi < y[-1]:
        xid = (xi - x[0]) / dx
        xid1 = np.floor(xid)
        dxid = xid - xid1
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return xid1, yid1, dxid, dyid

    # outside one boundary
    if yi < y[0]:
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
        xid1 = 0
        dxid = 0.0
        # We know that yi is within the boundaries
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return xid1, yid1, dxid, dyid

    if xi >= x[-1]:
        xid1 = -1
        dxid = 0.0
        # We know that yi is within the boundaries
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1
        
    print(xid1, yid1, dxid, dyid)

    return xid1, yid1, dxid, dyid