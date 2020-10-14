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
    
    sigmae_ext = params["sigmae_ext"]  # External exc input standard deviation ( mV/sqrt(ms) )
    sigmai_ext = params["sigmai_ext"]  # External inh input standard deviation ( mV/sqrt(ms) )
    
    # recurrent coupling parameters
    Ke = params["Ke"]  # Recurrent Exc coupling. "EE = IE" assumed for act_dep_coupling in current implementation
    Ki = params["Ki"]  # Recurrent Exc coupling. "EI = II" assumed for act_dep_coupling in current implementation
    
    tau_se = params["tau_se"]  # Synaptic decay time constant for exc. connections "EE = IE" (ms)
    tau_si = params["tau_si"]  # Synaptic decay time constant for inh. connections  "EI = II" (ms)
    
    cee = params["cee"]  # strength of exc. connection
    #  -> determines ePSP magnitude in state-dependent way (in the original model)
    cie = params["cie"]  # strength of inh. connection
    #   -> determines iPSP magnitude in state-dependent way (in the original model)
    cei = params["cei"]
    cii = params["cii"]

    # Recurrent connections coupling strength
    Jee_max = params["Jee_max"]  # ( mV/ms )
    Jei_max = params["Jei_max"]  # ( mV/ms )
    Jie_max = params["Jie_max"]  # ( mV/ms )
    Jii_max = params["Jii_max"]  # ( mV/ms )
    
    # neuron model parameters
    a = params["a"]  # Adaptation coupling term ( nS )
    b = params["b"]  # Spike triggered adaptation ( pA )
    EA = params["EA"]  # Adaptation reversal potential ( mV )
    tauA = params["tauA"]  # Adaptation time constant ( ms )
    # if params below are changed, preprocessing required
    C = params["C"]  # membrane capacitance ( pF )
    gL = params["gL"]  # Membrane conductance ( nS )
    EL = params["EL"]  # Leak reversal potential ( mV )
    DeltaT = params["DeltaT"]  # Slope factor ( EIF neuron ) ( mV )
    VT = params["VT"]  # Effective threshold (in exp term of the aEIF model)(mV)
    Vr = params["Vr"]  # Membrane potential reset value (mV)
    Vs = params["Vs"]  # Cutoff or spike voltage value, determines the time of spike (mV)
    Tref = params["Tref"]  # Refractory time (ms)
    taum = C / gL  # membrane time constant
    
    # ------------------------------------------------------------------------
    
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)
    
    startind = 1
    
    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data
    rates_exc = np.zeros((N, startind + len(t)))
    rates_inh = np.zeros((N, startind + len(t)))
    IA = np.zeros((N, startind + len(t)))
    
    mufe  = np.zeros((N, startind + len(t)))
    mufi  = np.zeros((N, startind + len(t)))
    
    seem = np.zeros((N, startind + len(t)))
    seim = np.zeros((N, startind + len(t)))
    seev = np.zeros((N, startind + len(t)))
    seiv = np.zeros((N, startind + len(t)))
    siim = np.zeros((N, startind + len(t)))
    siem = np.zeros((N, startind + len(t)))
    siiv = np.zeros((N, startind + len(t)))
    siev = np.zeros((N, startind + len(t)))
    
    mue_ou = np.zeros((N, startind + len(t)))
    mui_ou = np.zeros((N, startind + len(t)))
    
    # ------------------------------------------------------------------------
    # Set initial values
    mufe[:,:startind] = params["mufe_init"].copy()  # Filtered mean input (mu) for exc. population
    mufi[:,:startind] = params["mufi_init"].copy()  # Filtered mean input (mu) for inh. population
    IA_init = params["IA_init"].copy()  # Adaptation current (pA)
    seem[:,:startind] = params["seem_init"].copy()  # Mean exc synaptic input
    seim[:,:startind] = params["seim_init"].copy()
    seev[:,:startind] = params["seev_init"].copy()  # Exc synaptic input variance
    seiv[:,:startind] = params["seiv_init"].copy()
    siim[:,:startind] = params["siim_init"].copy()  # Mean inh synaptic input
    siem[:,:startind] = params["siem_init"].copy()
    siiv[:,:startind] = params["siiv_init"].copy()  # Inh synaptic input variance
    siev[:,:startind] = params["siev_init"].copy()

    mue_ou[:,0] = params["mue_ou"].copy()  # Mean of external exc OU input (mV/ms)
    mui_ou[:,0] = params["mui_ou"].copy()  # Mean of external inh ON inout (mV/ms)
    
    
    sigmae_f = np.zeros((N, len(t)+startind))
    tau_exc = np.zeros((N, len(t)+startind))
    ext_exc_current = np.zeros((N, len(t)+startind))
    ext_inh_current = np.zeros((N, len(t)+startind))
    
    rd_exc = np.zeros((N, N))  # kHz  rd_exc(i,j): Connection from jth node to ith
    rd_inh = np.zeros(N)

    sigmae_f = np.zeros((N, len(t)+1))

    rates_exc[:,:startind] = params["rates_exc_init"]
    IA[:,:startind] = params["IA_init"]
    ext_exc_current[:,:] = params["ext_exc_current"]
    ext_inh_current[:,:] = params["ext_inh_current"]

    control_ext = control.copy()
    
    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        dt,
        duration,
        sigmae_ext,
        sigmai_ext,
        Ke,
        Ki,
        tau_se,
        tau_si,
        cee,
        cie,
        cii,
        cei,
        Jee_max,
        Jei_max,
        Jie_max,
        Jii_max,
        a,
        b,
        EA,
        tauA,
        C,
        taum,
        mufe,
        mufi,
        IA,
        seem,
        seim,
        seev,
        seiv,
        siim,
        siem,
        siiv,
        siev,
        N,
        t,
        rates_exc,
        rates_inh,
        rd_exc,
        rd_inh,
        startind,
        mue_ou,
        mui_ou,
        ext_exc_current,
        ext_inh_current,
        control_ext,
    )


#@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64, "idx1": numba.int64, "idy1": numba.int64})
def timeIntegration_njit_elementwise(
        dt,
        duration,
        sigmae_ext,
        sigmai_ext,
        Ke,
        Ki,
        tau_se,
        tau_si,
        cee,
        cie,
        cii,
        cei,
        Jee_max,
        Jei_max,
        Jie_max,
        Jii_max,
        a,
        b,
        EA,
        tauA,
        C,
        taum,
        mufe,
        mufi,
        IA,
        seem,
        seim,
        seev,
        seiv,
        siim,
        siem,
        siiv,
        siev,
        N,
        t,
        rates_exc,
        rates_inh,
        rd_exc,
        rd_inh,
        startind,
        mue_ou,
        mui_ou,
        ext_exc_current,
        ext_inh_current,
        control_ext,
):
    
    factor_ee1 = ( cee * Ke * tau_se / Jee_max )
    factor_ee2 = ( cee**2 * Ke * tau_se**2 / Jee_max**2 )
    
    factor_ii1 = ( cii * Ki * tau_si / Jii_max )
    factor_ii2 = ( cii**2 * Ki * tau_si**2 / Jii_max**2 )
    
    sigmae_f = np.zeros((N, startind + len(t)))
    sigmai_f = np.zeros((N, startind + len(t)))
    Vmean_exc = np.zeros((N, startind + len(t)))
    tau_exc = np.zeros((N, startind + len(t)))
    tau_inh = np.zeros((N, startind + len(t)))
    
    for i in range(startind, startind + len(t)):
        for no in range(N):
            
            rd_exc[no,no] = rates_exc[no,i-1] * 1e-3
            rd_inh[no] = rates_inh[no,i-1] * 1e-3
        
            z1ee = factor_ee1 * rd_exc[no,no]
            z2ee = factor_ee2 * rd_exc[no,no]
            
            z1ii = factor_ii1 * rd_inh[no]
            z2ii = factor_ii2 * rd_inh[no]
            
            sig_ee = seev[no,i-1] * ( 2. * Jee_max**2 * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)
            
            sigmae_f[no,i-1] = np.sqrt( sig_ee + sigmae_ext**2 )
            tau_exc[no,i-1] = tau_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1])
            
            sig_ii = siiv[no,i-1] * ( 2. * Jii_max**2 * tau_si * taum ) * ( (1 + z1ii) * taum + tau_si )**(-1)
            
            sigmai_f[no,i-1] = np.sqrt( sig_ii + sigmai_ext**2 )
            tau_inh[no,i-1] = tau_func(mufi[no,i-1], sigmai_f[no,i-1])
            
            seem_rhs = ( - seem[no,i-1]  + ( 1. - seem[no,i-1] ) * z1ee ) / tau_se
            seem[no,i] = seem[no,i-1] + dt * seem_rhs
            seev_rhs = ( (1. - seem[no,i-1])**2 * z2ee + seev[no,i-1] * (z2ee - 2. * tau_se * ( z1ee + 1.) ) ) / tau_se**2 
            seev[no,i] = seev[no,i-1] + dt * seev_rhs
            
            siim_rhs = ( - siim[no,i-1]  + ( 1. - siim[no,i-1] ) * z1ii ) / tau_si
            siim[no,i] = siim[no,i-1] + dt * siim_rhs
            siiv_rhs = ( (1. - siim[no,i-1])**2 * z2ii + siiv[no,i-1] * (z2ii - 2. * tau_si * ( z1ii + 1.) ) ) / tau_si**2 
            siiv[no,i] = siiv[no,i-1] + dt * siiv_rhs
            
            mufe_rhs = ( Jee_max * seem[no,i-1] + control_ext[no,0,i] + ext_exc_current[no,i] - mufe[no,i-1] ) / tau_exc[no,i-1]
            mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
            rates_exc[no,i] = r_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1]) * 1e3
            Vmean_exc[no,i] = V_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1])
            
            mufi_rhs = ( Jii_max * siim[no,i-1] + control_ext[no,1,i] + ext_inh_current[no,i] - mufi[no,i-1] ) / tau_inh[no,i-1]
            mufi[no,i] = mufi[no,i-1] + dt * mufi_rhs
            rates_inh[no,i] = r_func(mufi[no,i-1], sigmai_f[no,i-1]) * 1e3
            
            IA_rhs =  - b * rates_exc[no,i] * 1e-3 + ( a * ( Vmean_exc[no,i] - EA ) - IA[no,i-1] ) / tauA
            IA[no,i] = IA[no,i-1] + dt * IA_rhs
            
    rd_exc[no,no] = rates_exc[no,-1] * 1e-3
        
    z1ee = factor_ee1 * rd_exc[no, no]
    z2ee = factor_ee2 * rd_exc[no, no]  
    
    sig_ee = seev[no,-1] * ( 2. * Jee_max**2 * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)

    sigmae_f[no,-1] = np.sqrt( sig_ee + sigmae_ext**2 )
    tau_exc[no,-1] = tau_func(mufe[no,-1], sigmae_f[no,-1])
    
    Vmean_exc[:,:startind] = Vmean_exc[:,startind]
  
    return t, rates_exc, rates_inh, mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv, mue_ou, mui_ou, sigmae_f, sigmai_f, Vmean_exc, tau_exc, tau_inh


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