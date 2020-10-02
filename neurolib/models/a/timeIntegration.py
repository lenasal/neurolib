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
        control_ext,
    )


#@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64, "idx1": numba.int64, "idy1": numba.int64})
def timeIntegration_njit_elementwise(
        N,
        dt,
        t,
        rates_exc,
        mufe,
        control_ext,
):
    
    for i in range(1,len(t)+1):
        for no in range(N):
            
            rates_exc[no,i] = mufe[no,i-1]
            mufe[no,i] = control_ext[no,0,i]
  
    return t, rates_exc, mufe