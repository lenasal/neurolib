import numpy as np
import numba
import logging

from . import loadDefaultParams as dp


def timeIntegration(params, control):
    """Sets up the parameters for time integration
    
    Return:
      rates_exc:  N*L array   : containing the exc. neuron rates in kHz time series of the N nodes
      rates_inh:  N*L array   : containing the inh. neuron rates in kHz time series of the N nodes
      t:          L array     : time in ms
      mufe:       N vector    : final value of mufe for each node
      mufi:       N vector    : final value of mufi for each node
      IA:         N vector    : final value of IA   for each node
      seem :      N vector    : final value of seem  for each node
      seim :      N vector    : final value of seim  for each node
      siem :      N vector    : final value of siem  for each node
      siim :      N vector    : final value of siim  for each node
      seev :      N vector    : final value of seev  for each node
      seiv :      N vector    : final value of seiv  for each node
      siev :      N vector    : final value of siev  for each node
      siiv :      N vector    : final value of siiv  for each node

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """

    N = params["N"]
    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)

    # ------------------------------------------------------------------------

    # Lookup tables for the transfer functions
    precalc_r = params["precalc_r"]
    precalc_tau_mu = params["precalc_tau_mu"]
    sigmae = params["sigmae_ext"]
    sigmai = params["sigmai_ext"]
    
    # parameter for the lookup tables
    dI = params["dI"]
    ds = params["ds"]
    sigmarange = params["sigmarange"]
    Irange = params["Irange"]
    
    cee = params["cee"]
    Ke = params.Ke
    Jee = params.Jee
    
    tau_se = params.tau_se
    taum = params.C / params.gL

    # Initialization
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    startind = 1
    rates_exc = np.zeros((N, startind + len(t)))
    rates_inh = np.zeros((N, startind + len(t)))
    mufe = np.zeros((N, startind + len(t)))
    mufi = np.zeros((N, startind + len(t)))
    sigmae_f = np.zeros((N, startind + len(t)))
    sigmai_f = np.zeros((N, startind + len(t)))
    ext_current_e = np.zeros((N, startind + len(t)))
    ext_current_i = np.zeros((N, startind + len(t)))
    tau_exc = np.zeros((N, startind + len(t)))
    tau_inh = np.zeros((N, startind + len(t)))
    
        
    if type(params["rates_exc_init"]) is not type(np.array([])):
        logging.error("wrong input for initial rates")
    if len(np.shape(params["rates_exc_init"])) == 1:
        logging.error("wrong input for initial rates")
        #rates_exc_init = (params["rates_exc_init"] * np.ones((1, startind))).T   # kHz
        #rates_inh_init = (params["rates_inh_init"] * np.ones((startind, 1))).T  # kHz
    # if initial values are just a Nx1 array
    if np.shape(params["rates_exc_init"])[1] == 1:
        # repeat the 1-dim value stardind times
        rates_exc_init = np.dot(params["rates_exc_init"], np.ones((startind)))  # kHz
        rates_inh_init = np.dot(params["rates_inh_init"], np.ones((startind)))  # kHz
    else:
        rates_exc_init = params["rates_exc_init"][-startind:]
        rates_inh_init = params["rates_inh_init"][-startind:]

    rates_exc[:,:startind] = rates_exc_init
    rates_inh[:,:startind] = rates_inh_init
    mufe[:,:startind] = params["mufe_init"]
    mufi[:,:startind] = params["mufi_init"]
    ext_current_e[:,:] = params["ext_exc_current"]
    ext_current_i[:,:] = params["ext_inh_current"]
    sigmae_f[:,0] = params["sigmae_ext"]
    sigmai_f[:,0] = params["sigmai_ext"]
    
    """
    xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmae_f[0,0], Irange, dI, mufe[0,0])
    xid1, yid1 = int(xid1), int(yid1)
    tau_exc[:,:startind] = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
    #tau_exc[:,:startind] = params["tau"]
    
    xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmai_f[0,0], Irange, dI, mufi[0,0])
    xid1, yid1 = int(xid1), int(yid1)
    tau_inh[:,:startind] = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
    """

    # tile external inputs to appropriate shape
    control_ext = control.copy()
    
    #mufe[:,startind-1] += control_ext[:,0,0]
    #mufi[:,startind-1] += control_ext[:,1,0]
    
    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        N,
        dt,
        duration,
        mufe,
        mufi,
        precalc_r,
        precalc_tau_mu,
        sigmae_f,
        sigmae,
        sigmai_f,
        sigmai,
        dI,
        ds,
        sigmarange,
        Irange,
        cee,
        Ke,
        Jee,
        tau_se,
        taum,
        t,
        rates_exc,
        rates_inh,
        tau_exc,
        tau_inh,
        startind,
        ext_current_e,
        ext_current_i,
        control_ext,
    )


#@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64, "idx1": numba.int64, "idy1": numba.int64})
def timeIntegration_njit_elementwise(
        N,
        dt,
        duration,
        mufe,
        mufi,
        precalc_r,
        precalc_tau_mu,
        sigmae_f,
        sigmae,
        sigmai_f,
        sigmai,
        dI,
        ds,
        sigmarange,
        Irange,
        cee,
        Ke,
        Jee,
        tau_se,
        taum,
        t,
        rates_exc,
        rates_inh,
        tau_exc,
        tau_inh,
        startind,
        ext_current_e,
        ext_current_i,
        control_ext,
):
    
    variant = 0 # as in original
    #variant = 1 # as done by Lena
    if variant == 1:
        mufe[:,0] = control_ext[:,0,1-startind]
        rates_exc[:,0] = mufe[:,0]
    
    tau_exc[:,:] = 1.
    tau_inh[:,:] = 1.
    
    seem = np.zeros((N,startind + len(t)))
    seev = np.zeros((N,startind + len(t)))
    seev[:,0] = 0.001
    
    """
    for no in range(N):
        mufe_rhs = (ext_current_e[no,startind] + control_ext[no,0,0] - mufe[no,startind-1]) / tau_exc[no,startind-1]      
        mufe[no,startind-1] = mufe[no,startind-1] + dt * mufe_rhs
        mufi_rhs = (ext_current_i[no,startind] + control_ext[no,1,0] - mufi[no,startind-1]) / tau_inh[no,startind-1]            
        mufi[no,startind-1] = mufi[no,startind-1] + dt * mufi_rhs
    """
    
    firstwarn = True
    
    for i in range(startind, startind + len(t)):
        for no in range(N):
            
            
            # use ext current and control at same time ??
            
            z1ee =  1e-3 * cee * Ke * rates_exc[no, i-1] * tau_se / Jee
            z2ee = 1e-3 * cee**2 * Ke * rates_exc[no, i-1] * tau_se**2 / Jee**2
            
           # print("r, rho ", z1ee, z2ee)
            
            sigmae_f[no,i] = np.sqrt( 2. * Jee**2 * seev[no,i-1] * tau_se * taum / ((1 + z1ee) * taum + tau_se) + sigmae**2 )
            sigmae_f[no,i] = 1.5
            #print("pref sigma e ", 2. * Jee**2 * tau_se * taum / ((1 + z1ee) * taum + tau_se), seev[no,i-1])
            #print(2. * Jee**2 * seev[no,i-1] * tau_se * taum / ((1 + z1ee) * taum + tau_se), sigmae**2)
            sigmai_f[no,i] = sigmai
            
            #print("r, rho =", rates_exc[no, i-1], z1ee, z2ee)
            #print("sigma e = ", sigmae_f[no,i])
            #print("seev = ", seev[no,i-1])
            
            
            if variant == 1:
                
                mufe_rhs = (ext_current_e[no,i] + control_ext[no,0,i-startind] - mufe[no,i-1]) / tau_exc[no,i-1] 
                mufe_rhs = control_ext[no,0,i-startind]
                mufe[no,i] = control_ext[no,0,i-startind]
                #mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
                mufi_rhs = (ext_current_i[no,i] + control_ext[no,1,i-startind] - mufi[no,i-1]) / tau_inh[no,i-1]            
                mufi[no,i] = mufi[no,i-1] + dt * mufi_rhs
        
                xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmae_f[no,i], Irange, dI, mufe[no,i])
                xid1, yid1 = int(xid1), int(yid1)
                rates_exc[no,i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3  # convert kHz to Hz
                tau_exc[no,i] = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
                rates_exc[no,i] = mufe[no,i]
                #tau_exc[no,i] = tau_exc[no,0]
                
                xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmai_f[no,i], Irange, dI, mufi[no,i])
                xid1, yid1 = int(xid1), int(yid1)
                rates_inh[no,i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3  # convert kHz to Hz
                tau_inh[no,i] = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
                
            elif variant == 0:                    
                xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmae_f[no,i-1], Irange, dI, mufe[no,i-1])
                xid1, yid1 = int(xid1), int(yid1)
                rates_exc[no,i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3  # convert kHz to Hz
                tau_exc[no,i] = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
                
                tau_exc[no,i] = 1.5
                
                #xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmai_f[no,i-1], Irange, dI, mufi[no,i-1])
                #xid1, yid1 = int(xid1), int(yid1)
                #rates_inh[no,i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3  # convert kHz to Hz
                #tau_inh[no,i] = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
                
                mufe_rhs = (ext_current_e[no,i] + control_ext[no,0,i-startind+1] - mufe[no,i-1]) / tau_exc[no,i]   
                mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
                #mufe[no,i] = 1.5
                
                if (mufe[no,i] < 0.5 or mufe[no,i] > 6.):
                    if (firstwarn):
                        print("warning: too small or too large value for reasonable backwards computation with mufe = ", mufe[no,i])
                        firstwarn = False
                    mufe[no,i] = mufe[no,i-1]
                #mufi_rhs = (ext_current_i[no,i] + control_ext[no,1,i-startind+1] - mufi[no,i-1]) / tau_inh[no,i]
                #mufi[no,i] = mufi[no,i-1] + dt * mufi_rhs
                
                seev_rhs = ( (z2ee - 2. * tau_se * (z1ee + 1)) * seev[no,i-1]) / tau_se ** 2
                seev[no,i] = seev[no,i-1] + dt * seev_rhs
                seev[no,i] = 0.
                #print("seev = ", seev[no,i])
                if seev[no,i] < 0:
                    seev[no,i] = 0.0
                
            elif variant == 2:                    
                mufe_rhs = (ext_current_e[no,i] + control_ext[no,0,i-startind] - mufe[no,i-1]) / tau_exc[no,i-1]      
                mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
                mufi_rhs = (ext_current_i[no,i] + control_ext[no,1,i-startind] - mufi[no,i-1]) / tau_inh[no,i-1]            
                mufi[no,i] = mufi[no,i-1] + dt * mufi_rhs
        
                xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmae_f[no,i], Irange, dI, mufe[no,i])
                xid1, yid1 = int(xid1), int(yid1)
                rates_exc[no,i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3  # convert kHz to Hz
                tau_exc[no,i] = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
                
                xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmai_f[no,i], Irange, dI, mufi[no,i])
                xid1, yid1 = int(xid1), int(yid1)
                rates_inh[no,i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3  # convert kHz to Hz
                tau_inh[no,i] = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
                
                mufe_rhs = (ext_current_e[no,i] + control_ext[no,0,i-startind] - mufe[no,i-1]) / tau_exc[no,i]      
                mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
                mufi_rhs = (ext_current_i[no,i] + control_ext[no,1,i-startind] - mufi[no,i-1]) / tau_inh[no,i]            
                mufi[no,i] = mufi[no,i-1] + dt * mufi_rhs
                
    tau_exc[:,:startind] = tau_exc[:,startind]
    #tau_inh[:,:startind] = tau_inh[:,startind]
            
    return t, rates_exc, mufe, seev, sigmae_f, tau_exc
    return t, rates_exc, rates_inh, mufe, mufi, seem, seev, tau_exc, tau_inh


#@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64})
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
