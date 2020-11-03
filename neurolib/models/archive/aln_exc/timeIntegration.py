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
    IA = np.zeros((N, startind + len(t)))
    
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

    rates_exc[:,:startind] = params["rates_exc_init"]
    mufe[:,:startind] = params["mufe_init"]
    IA[:,:startind] = params["IA_init"]
    seem[:,:startind] = params["seem_init"]
    seev[:,:startind] = params["seev_init"]
    tau_exc[:,:startind] = params["mufe_init"]
    ext_exc_current[:,:] = params["ext_exc_current"]
    
    sigmae_ext = params["sigmae_ext"]
    
    # neuron model parameters
    a = params["a"]
    b = params["b"]
    EA = params["EA"]
    tauA = params["tauA"]
    
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
        rates_exc,
        mufe,
        IA,
        seem,
        seev,
        sigmae_f,
        tau_exc,
        ext_exc_current,
        sigmae_ext,
        rd_exc,
        a,
        b,
        EA,
        tauA,
        Ke,
        tau_se,
        cee,
        Jee_max,
        taum,
        C,
        control_ext,
    )


#@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64, "idx1": numba.int64, "idy1": numba.int64})
def timeIntegration_njit_elementwise(
        N,
        dt,
        t,
        startind,
        rates_exc,
        mufe,
        IA,
        seem,
        seev,
        sigmae_f,
        tau_exc,
        ext_exc_current,
        sigmae_ext,
        rd_exc,
        a,
        b,
        EA,
        tauA,
        Ke,
        tau_se,
        cee,
        Jee_max,
        taum,
        C,
        control_ext,
):
    
    factor_ee1 = ( cee * Ke * tau_se / Jee_max )
    factor_ee2 = ( cee**2 * Ke * tau_se**2 / Jee_max**2 )
    
    Vmean_exc = np.zeros((N, startind + len(t)))
    
    for i in range(startind, startind + len(t)):
        for no in range(N):
            
            rd_exc[no,no] = rates_exc[no,i-1] * 1e-3
        
            z1ee = factor_ee1 * rd_exc[no, no]
            z2ee = factor_ee2 * rd_exc[no, no]
            
            sig_ee = seev[no,i-1] * ( 2. * Jee_max**2 * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)
            
            sigmae_f[no,i-1] = np.sqrt( sig_ee + sigmae_ext**2 )
            tau_exc[no,i-1] = tau_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1])
            
            seem_rhs = ( - seem[no,i-1]  + ( 1. - seem[no,i-1] ) * z1ee ) / tau_se
            seem[no,i] = seem[no,i-1] + dt * seem_rhs
            seev_rhs = ( (1. - seem[no,i-1])**2 * z2ee + seev[no,i-1] * (z2ee - 2. * tau_se * ( z1ee + 1.) ) ) / tau_se**2 
            seev[no,i] = seev[no,i-1] + dt * seev_rhs
            
            mufe_rhs = ( Jee_max * seem[no,i-1] + control_ext[no,0,i] + ext_exc_current[no,i] - mufe[no,i-1] ) / tau_exc[no,i-1]
            mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
            rates_exc[no,i] = r_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1]) * 1e3
            Vmean_exc[no,i] = V_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1])
            
            IA_rhs =  - b * rates_exc[no,i] * 1e-3 + ( a * ( Vmean_exc[no,i] - EA ) - IA[no,i-1] ) / tauA
            IA[no,i] = IA[no,i-1] + dt * IA_rhs
            
    rd_exc[no,no] = rates_exc[no,-1] * 1e-3
        
    z1ee = factor_ee1 * rd_exc[no, no]
    z2ee = factor_ee2 * rd_exc[no, no]  
    
    sig_ee = seev[no,-1] * ( 2. * Jee_max**2 * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)

    sigmae_f[no,-1] = np.sqrt( sig_ee + sigmae_ext**2 )
    tau_exc[no,-1] = tau_func(mufe[no,-1], sigmae_f[no,-1])
    
    Vmean_exc[:,:startind] = Vmean_exc[:,startind]
  
    return t, rates_exc, mufe, IA, seem, seev, sigmae_f, Vmean_exc, tau_exc


def r_func(mu, sigma):
    x_shift_mu = - 2.
    x_shift_sigma = -1.
    x_scale_mu = 0.6
    x_scale_sigma = 0.6
    y_shift = 0.1
    y_scale_mu = 0.1
    y_scale_sigma = 1./2500.
    return y_shift + np.tanh(x_scale_mu * mu + x_shift_mu) * y_scale_mu + np.cosh(x_scale_sigma * sigma + x_shift_sigma) * y_scale_sigma

def tau_func(mu, sigma):
    mu_shift = - 1.1
    mu_scale = - 10.
    y_shift = 15.
    sigma_shift = 1.4
    #return mu + sigma
    return (mu_shift + mu) * sigma + mu_scale * mu + y_shift + np.exp( mu_scale * (mu_shift + mu) / (sigma + sigma_shift) )

def V_func(mu, sigma):
    y_scale1 = 30.
    mu_shift1 = 1.
    y_shift = - 85.
    y_scale2 = 2.
    mu_shift2 = 0.5
    return mu + sigma
    return y_shift + y_scale1 * np.tanh( mu + mu_shift1 ) + y_scale2 * np.exp( - ( mu - mu_shift2 )**2 ) / sigma