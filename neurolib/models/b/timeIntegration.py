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
    sigmae_f = np.zeros((N, len(t)+1))

    rates_exc[:,0] = params["rates_exc_init"]
    mufe[:,0] = params["mufe_init"]

    control_ext = control.copy()
    
    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        N,
        dt,
        t,
        rates_exc,
        mufe,
        sigmae_f,
        control_ext,
    )


#@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64, "idx1": numba.int64, "idy1": numba.int64})
def timeIntegration_njit_elementwise(
        N,
        dt,
        t,
        rates_exc,
        mufe,
        sigmae_f,
        control_ext,
):
    
    #mufe[:,0] = control_ext[:,0,0]
    
    for i in range(1,len(t)+1):
        for no in range(N):
            
            seev = 0.
            
            #sigmae_f[no,i-1] = np.sqrt(rates_exc[no,i-1] + 1.5**2 )
            #sigmae_f[no,i-1] = np.sqrt(seev + 1.5**2 )
            sigmae_f[no,i-1] = 1e-3 * rates_exc[no,i-1] #+ 1.5
            mufe_rhs = control_ext[no,0,i]
            mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
            rates_exc[no,i] = r_func(mufe[no,i-1], sigmae_f[no,i-1]) * 1e3
            
    seev = 0.     
    #sigmae_f[no,-1] = np.sqrt(rates_exc[no,-1] + 1.5**2 )
    #sigmae_f[no,-1] = np.sqrt(seev + 1.5**2 )
    sigmae_f[no,-1] = 1e-3 * rates_exc[no,-1] #+ 1.5
  
    return t, rates_exc, mufe, sigmae_f


def r_func(mu, sigma):
    x_shift_mu = - 2.
    x_shift_sigma = -1.
    x_scale_mu = 0.6
    x_scale_sigma = 0.6
    y_shift = 0.1
    y_scale_mu = 0.1
    y_scale_sigma = 1./2500.
    return y_shift + np.tanh(x_scale_mu * mu + x_shift_mu) * y_scale_mu + np.cosh(x_scale_sigma * sigma + x_shift_sigma) * y_scale_sigma