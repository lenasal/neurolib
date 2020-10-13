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
    
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)
    
    startind = 1

    rates_exc = np.zeros((N, len(t)+startind))
    mufe = np.zeros((N, len(t)+startind))
    seem = np.zeros((N, len(t)+startind))
    seev = np.zeros((N, len(t)+startind))
    sigmae_f = np.zeros((N, len(t)+startind))
    tau_exc = np.zeros((N, len(t)+startind))
    ext_exc_current = np.zeros((N, len(t)+startind))
    
    rd_exc = np.zeros((N, N))

    # Initialization
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    rates_exc = np.zeros((N, len(t)+1))
    mufe = np.zeros((N, len(t)+1))
    sigmae_f = np.zeros((N, len(t)+1))

    rates_exc[:,0] = params["rates_exc_init"]
    mufe[:,0] = params["mufe_init"]
    seem[:,:startind] = params["seem_init"]
    seev[:,:startind] = params["seev_init"]
    #tau_exc[:,:startind] = params["mufe_init"]
    ext_exc_current[:,:] = params["ext_exc_current"]
    
    sigmae_ext = params["sigmae_ext"]
    
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
    
    #mufe[:,0] = control_ext[:,0,0]
    
    for i in range(1,len(t)+1):
        for no in range(N):
            
            #seev = 0.
            
            #sigmae_f[no,i-1] = np.sqrt(rates_exc[no,i-1] + 1.5**2 )
            #sigmae_f[no,i-1] = np.sqrt(seev + 1.5**2 )
            sigmae_f[no,i-1] = np.sqrt( (1e-3 * rates_exc[no,i-1])**2 + sigmae_ext**2 )
            tau_exc[no,i-1] = mufe[no,i-1]
            
            rd_exc[no,no] = rates_exc[no,i-1] * 1e0
            
            factor_ee1 = 1.#( cee * Ke * tau_se / Jee_max )
            factor_ee2 = 1.#( cee**2 * Ke * tau_se**2 / Jee_max**2 )
            z1ee = factor_ee1 * rd_exc[no, no]
            z2ee = factor_ee2 * rd_exc[no, no]
            
            seem_rhs = - seem[no,i-1] / tau_se + ( 1. - seem[no,i-1] ) * z1ee
            #seem_rhs = z1ee * seem[no,i-1]
            seem[no,i] = seem[no,i-1] + dt * seem_rhs
            seev_rhs = 0.
            seev[no,i] = seev[no,i-1] + dt * seev_rhs
            
            mufe_rhs = ( seem[no,i-1] + control_ext[no,0,i] + ext_exc_current[no,i] - mufe[no,i-1] ) / tau_exc[no,i-1]
            mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
            rates_exc[no,i] = r_func(mufe[no,i-1], sigmae_f[no,i-1]) * 1e3
            
   # seev = 0.     
    #sigmae_f[no,-1] = np.sqrt(rates_exc[no,-1] + 1.5**2 )
    #sigmae_f[no,-1] = np.sqrt(seev + 1.5**2 )
    sigmae_f[no,-1] = np.sqrt( (1e-3 * rates_exc[no,-1])**2 + sigmae_ext**2 )
    tau_exc[no,-1] = mufe[no,-1]
  
    return t, rates_exc, mufe, seem, seev, sigmae_f, tau_exc


def r_func(mu, sigma):
    x_shift_mu = - 2.
    x_shift_sigma = -1.
    x_scale_mu = 0.6
    x_scale_sigma = 0.6
    y_shift = 0.1
    y_scale_mu = 0.1
    y_scale_sigma = 1./2500.
    return y_shift + np.tanh(x_scale_mu * mu + x_shift_mu) * y_scale_mu + np.cosh(x_scale_sigma * sigma + x_shift_sigma) * y_scale_sigma

def tau_func(mu):
    x_shift = -1.
    x_scale = 1.
    y_shift = 11.
    y_scale = -10.
    return y_shift + np.tanh(x_scale * mu + x_shift) * y_scale