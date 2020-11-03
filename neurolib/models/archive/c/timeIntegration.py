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

    rates_exc[:,0] = params["rates_exc_init"]
    mufe[:,0] = params["mufe_init"]
    
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
        control_ext,
):
    
    for i in range(1,len(t)+1):
        for no in range(N):
        
            rates_exc[no,i-1] =  r_func_mu(mufe[no,i-1], 1.5)* 1e3  # convert kHz to Hz
            
            mufe_rhs = control_ext[no,0,i-1] #/ tau_exc[no,i]
            mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
            #rates_exc[no,i-1] = mufe[no,i-1]
            
    rates_exc[no,-1] = r_func_mu(mufe[no,-1], 1.5)* 1e3
    #rates_exc[no,-1] = mufe[no,-1]

              
    return t, rates_exc, mufe, tau_exc


def interpolate_values(table, xid1, yid1, dxid, dyid):
    output = (
        table[yid1, xid1] * (1 - dxid) * (1 - dyid)
        + table[yid1, xid1 + 1] * dxid * (1 - dyid)
        + table[yid1 + 1, xid1] * (1 - dxid) * dyid
        + table[yid1 + 1, xid1 + 1] * dxid * dyid
    )
    return output


def r_func_mu(mu, sigma):
    x_shift = - 2.
    x_scale = 0.6
    y_shift = 0.1
    y_scale = 0.1
    return y_shift + np.tanh(x_scale * mu + x_shift) * y_scale