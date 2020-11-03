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
    
    delay = params["delay"]
    startind = 1 + delay

    rates_exc = np.zeros((N, len(t)+startind))
    mufe = np.zeros((N, len(t)+startind))
    seem = np.zeros((N, len(t)+startind))
    seev = np.zeros((N, len(t)+startind))
    sigmae_f = np.zeros((N, len(t)+startind))
    tau_exc = np.zeros((N, len(t)+startind))
    ext_exc_current = np.zeros((N, len(t)+startind))
    
    rd_exc = np.zeros((N, N))
    
    rates_exc[:,:startind] = params["rates_exc_init"]
    mufe[:,:startind] = params["mufe_init"]
    seem[:,:startind] = params["seem_init"]
    seev[:,:startind] = params["seev_init"]
    tau_exc[:,:startind] = params["mufe_init"]
    ext_exc_current[:,:] = params["ext_exc_current"]
    
    sigmae_ext = params["sigmae_ext"]
    
    dI = params["dI"]
    ds = params["ds"]
    sigmarange = params["sigmarange"]
    Irange = params["Irange"]
        
    precalc_r = params["precalc_r"]
    precalc_tau_mu = params["precalc_tau_mu"]    
    
    # recurrent coupling parameters
    Ke = params["Ke"]  # Recurrent Exc coupling. "EE = IE" assumed for act_dep_coupling in current implementation
    
    tau_se = params["tau_se"]  # Synaptic decay time constant for exc. connections "EE = IE" (ms)
    
    cee = params["cee"]  # strength of exc. connection
    
    # Recurrent connections coupling strength
    Jee_max = params["Jee_max"]  # ( mV/ms )
    
    # if params below are changed, preprocessing required
    C = params["C"]  # membrane capacitance ( pF )
    gL = params["gL"]  # Membrane conductance ( nS )    
    taum = C / gL  # membrane time constant
    

    control_ext = control.copy()
    
    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        N,
        dt,
        t,
        startind,
        delay,
        rates_exc,
        mufe,
        seem,
        seev,
        sigmae_f,
        tau_exc,
        ext_exc_current,
        sigmae_ext,
        rd_exc,
        Ke,
        tau_se,
        cee,
        Jee_max,
        taum,
        control_ext,
    )


#@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64, "idx1": numba.int64, "idy1": numba.int64})
def timeIntegration_njit_elementwise(
        N,
        dt,
        t,
        startind,
        delay,
        rates_exc,
        mufe,
        seem,
        seev,
        sigmae_f,
        tau_exc,
        ext_exc_current,
        sigmae_ext,
        rd_exc,
        Ke,
        tau_se,
        cee,
        Jee_max,
        taum,
        control_ext,
):
    
    for i in range(startind, startind + len(t)):
        for no in range(N):
            
            delay = 0
                        
            sigma_ee = (2. * Jee_max**2 * seev[no,i-1] * tau_se * taum)
            
            sigmae_f[no,i-1] = np.sqrt(seev[no,i-1] + sigmae_ext ** 2 )  # mV/sqrt(ms)
            rates_exc[no,i-1] = r_func_mu(mufe[no,i-1], sigmae_f[no,i-1]) * 1e3 # convert kHz to Hz          
            tau_exc[no,i-1] = tau_func(mufe[no,i-1])
            
            rd_exc[no,no] = rates_exc[no,i-1-delay] * 1e-3
            
            factor_ee1 = ( cee * Ke * tau_se / Jee_max )
            factor_ee2 = ( cee**2 * Ke * tau_se**2 / Jee_max**2 )
            z1ee = factor_ee1 * rd_exc[no, no]
            z2ee = factor_ee2 * rd_exc[no, no]
            
            seem_rhs = ( - seem[no,i-1] / tau_se + ( 1. - seem[no,i-1] ) * z1ee )
            seem[no,i] = seem[no,i-1] + dt * seem_rhs
            seev_rhs = 0.
            seev[no,i] = seev[no,i-1] + dt * seev_rhs
            
            mufe_rhs = (seem[no,i-1] + control_ext[no,0,i-startind] + ext_exc_current[no,i] - mufe[no,i-1]) / tau_exc[no,i-1]
            mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
            
      
    sigmae_f[no,-1] = np.sqrt(seev[no,-1] + sigmae_ext ** 2 )  # mV/sqrt(ms)
    tau_exc[no,-1] = tau_func(mufe[no,-1])
    rates_exc[no,-1] = r_func_mu(mufe[no,-1], sigmae_f[no,-1]) * 1e3
    
    return t, rates_exc, mufe, seem, seev, sigmae_f, tau_exc

def r_func_mu(mu, sigma):
    x_shift = - 2.
    x_scale = 0.6
    y_shift = 0.1
    y_scale = 0.1
    return y_shift + np.tanh(x_scale * mu + x_shift) * y_scale

def r_func_sigma(sigma):
    x_shift = -1.
    x_scale = 0.6
    y_shift = 0.07
    y_scale = 1./2500.
    return y_shift + np.cosh(x_scale * sigma + x_shift) * y_scale

def tau_func(mu):
    x_shift = -1.
    x_scale = 1.
    y_shift = 11.
    y_scale = -10.
    return y_shift + np.tanh(x_scale * mu + x_shift) * y_scale 


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