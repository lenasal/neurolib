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
    
     # external input parameters:
    tau_ou = params["tau_ou"]  # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    mue_ext_mean = params["mue_ext_mean"]  # Mean external excitatory input (OU process) (mV/ms)
    mui_ext_mean = params["mui_ext_mean"]  # Mean external inhibitory input (OU process) (mV/ms)
    sigmae_ext = params["sigmae_ext"]  # External exc input standard deviation ( mV/sqrt(ms) )
    sigmai_ext = params["sigmai_ext"]  # External inh input standard deviation ( mV/sqrt(ms) )
    
    # recurrent coupling parameters
    Ke = params["Ke"]  # Recurrent Exc coupling. "EE = IE" assumed for act_dep_coupling in current implementation
    Ki = params["Ki"]  # Recurrent Exc coupling. "EI = II" assumed for act_dep_coupling in current implementation
    
    # Recurrent connection delays
    de = params["de"]  # Local constant delay "EE = IE" (ms)
    di = params["di"]  # Local constant delay "EI = II" (ms)
    
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
    sqrt_dt = np.sqrt(dt)
    
    ndt_de = np.around(de / dt).astype(int)
    ndt_di = np.around(di / dt).astype(int)

    rd_exc = np.zeros((N, N))  # kHz  rd_exc(i,j): Connection from jth node to ith
    rd_inh = np.zeros(N)

    # Already done above when Dmat_ndt is built
    # for l in range(N):
    #    Dmat_ndt[l, l] = ndt_de  # if no distributed, this is a fixed value (E-E coupling)

    max_global_delay = max(ndt_de, ndt_di)
    startind = int(max_global_delay + 1)
    
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

    if ( type(params["mue_ou"]) == np.float64 or type(params["mue_ou"]) == float ):
        mue_ou[:,:startind] = params["mue_ou"].copy() # Mean of external exc OU input (mV/ms)
        mui_ou[:,:startind] = params["mui_ou"].copy()  # Mean of external inh ON inout (mV/ms)
    elif params["mue_ou"].ndim == 2:
        mue_ou[:,:startind] = params["mue_ou"].copy()[:,-1] # Mean of external exc OU input (mV/ms)
        mui_ou[:,:startind] = params["mui_ou"].copy()[:,-1]  # Mean of external inh ON inout (mV/ms)
    else:
        mue_ou[:,:startind] = params["mue_ou"].copy() # Mean of external exc OU input (mV/ms)
        mui_ou[:,:startind] = params["mui_ou"].copy()  # Mean of external inh ON inout (mV/ms)
    
    sigmae_f = np.zeros((N, len(t)+startind))
    sigmai_f = np.zeros((N, len(t)+startind))
    tau_exc = np.zeros((N, len(t)+startind))
    tau_inh = np.zeros((N, len(t)+startind))
    Vmean_exc = np.zeros((N, startind + len(t)))
    ext_exc_current = np.zeros((N, len(t)+startind))
    ext_inh_current = np.zeros((N, len(t)+startind))
    
    rd_exc = np.zeros((N, N))  # kHz  rd_exc(i,j): Connection from jth node to ith
    rd_inh = np.zeros(N)
    
    # Save the noise in the rates array to save memory
    rates_exc[:, startind:] = np.zeros( (N, len(t)) ) #np.random.standard_normal( (N, len(t)) )
    rates_inh[:, startind:] = np.zeros( (N, len(t)) ) #np.random.standard_normal( (N, len(t)) )

    rates_exc[:,:startind] = params["rates_exc_init"]
    rates_inh[:,:startind] = params["rates_inh_init"]
    IA[:,:startind] = params["IA_init"]
    
    noise_exc = np.zeros((N,))
    noise_inh = np.zeros((N,))
    
    ext_exc_current[:,:] = params["ext_exc_current"]
    ext_inh_current[:,:] = params["ext_inh_current"]

    control_ext = control.copy()
    
    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        dt,
        duration,
        tau_ou,
        sigma_ou,
        mue_ext_mean,
        mui_ext_mean,
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
        sqrt_dt,
        startind,
        ndt_de,
        ndt_di,
        mue_ou,
        mui_ou,
        sigmae_f,
        sigmai_f,
        tau_exc,
        tau_inh,
        Vmean_exc,
        ext_exc_current,
        ext_inh_current,
        noise_exc,
        noise_inh,
        control_ext,
    )


#@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64, "idx1": numba.int64, "idy1": numba.int64})
def timeIntegration_njit_elementwise(
        dt,
        duration,
        tau_ou,
        sigma_ou,
        mue_ext_mean,
        mui_ext_mean,
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
        sqrt_dt,
        startind,
        ndt_de,
        ndt_di,
        mue_ou,
        mui_ou,
        sigmae_f,
        sigmai_f,
        tau_exc,
        tau_inh,
        Vmean_exc,
        ext_exc_current,
        ext_inh_current,
        noise_exc,
        noise_inh,
        control_ext,
):
    
    factor_ee1 = ( cee * Ke * tau_se / np.abs(Jee_max ) )
    factor_ee2 = ( cee**2 * Ke * tau_se**2 / Jee_max**2 )
    
    factor_ei1 = ( cei * Ki * tau_si / np.abs(Jei_max ) )
    factor_ei2 = ( cei**2 * Ki * tau_si**2 / Jei_max**2 )
    
    factor_ie1 = ( cie * Ke * tau_se / np.abs(Jie_max ) )
    factor_ie2 = ( cie**2 * Ke * tau_se**2 / Jie_max**2 )
    
    factor_ii1 = ( cii * Ki * tau_si / np.abs(Jii_max ) )
    factor_ii2 = ( cii**2 * Ki * tau_si**2 / Jii_max**2 )
    
    
    # time t = 0
    for no in range(N):
        rd_exc[no,no] = rates_exc[no,0] * 1e-3
        rd_inh[no] = rates_inh[no,0] * 1e-3
            
        z1ee = factor_ee1 * rd_exc[no,no]
        z2ee = factor_ee2 * rd_exc[no,no]  
        
        z1ei = factor_ei1 * rd_inh[no]
        z2ei = factor_ei2 * rd_inh[no]
        
        z1ie = factor_ie1 * rd_exc[no,no]
        z2ie = factor_ie2 * rd_exc[no,no] 
        
        z1ii = factor_ii1 * rd_inh[no]
        z2ii = factor_ii2 * rd_inh[no]
        
        sig_ee = seev[no,0] * ( 2. * Jee_max**2 * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)
        sig_ei = seiv[no,0] * ( 2. * Jei_max**2 * tau_si * taum ) * ( (1 + z1ei) * taum + tau_si )**(-1)
    
        sigmae_f[no,0] = np.sqrt( sig_ee + sig_ei + sigmae_ext**2 )
        tau_exc[no,0] = tau_func(mufe[no,0] - IA[no,0] / C, sigmae_f[no,0])
        
        sig_ii = siiv[no,0] * ( 2. * Jii_max**2 * tau_si * taum ) * ( (1 + z1ii) * taum + tau_si )**(-1)
        sig_ie = siev[no,0] * ( 2. * Jie_max**2 * tau_se * taum ) * ( (1 + z1ie) * taum + tau_se )**(-1)
                
        sigmai_f[no,0] = np.sqrt( sig_ii + sig_ie + sigmai_ext**2 )
        tau_inh[no,0] = tau_func(mufi[no,-1], sigmai_f[no,-1])
        
    
    for i in range(startind, startind + len(t)):
        for no in range(N):
            
            noise_exc[no] = rates_exc[no,i]
            noise_inh[no] = rates_inh[no,i]
            
            rd_exc[no,no] = rates_exc[no,i-1] * 1e-3
            rd_inh[no] = rates_inh[no,i-1] * 1e-3
        
            z1ee = factor_ee1 * rd_exc[no,no]
            z2ee = factor_ee2 * rd_exc[no,no]
            
            z1ei = factor_ei1 * rd_inh[no]
            z2ei = factor_ei2 * rd_inh[no]
            
            z1ie = factor_ie1 * rd_exc[no,no]
            z2ie = factor_ie2 * rd_exc[no,no]
            
            z1ii = factor_ii1 * rd_inh[no]
            z2ii = factor_ii2 * rd_inh[no]
                                    
            sig_ee = seev[no,i-1] * ( 2. * Jee_max**2 * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)
            sig_ei = seiv[no,i-1] * ( 2. * Jei_max**2 * tau_si * taum ) * ( (1 + z1ei) * taum + tau_si )**(-1)
            
            sigmae_f[no,i-1] = np.sqrt( sig_ee + sig_ei + sigmae_ext**2 )
            tau_exc[no,i-1] = tau_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1])
                        
            sig_ii = siiv[no,i-1] * ( 2. * Jii_max**2 * tau_si * taum ) * ( (1 + z1ii) * taum + tau_si )**(-1)
            sig_ie = siev[no,i-1] * ( 2. * Jie_max**2 * tau_se * taum ) * ( (1 + z1ie) * taum + tau_se )**(-1)
            
            sigmai_f[no,i-1] = np.sqrt( sig_ii + sig_ie + sigmai_ext**2 )
            tau_inh[no,i-1] = tau_func(mufi[no,i-1], sigmai_f[no,i-1])
            
            #-----------------------------------------------------
                        
            seem_rhs = ( - seem[no,i-1]  + ( 1. - seem[no,i-1] ) * z1ee ) / tau_se
            seem[no,i] = seem[no,i-1] + dt * seem_rhs
            seev_rhs = ( (1. - seem[no,i-1])**2 * z2ee + seev[no,i-1] * (z2ee - 2. * tau_se * ( z1ee + 1.) ) ) / tau_se**2 
            seev[no,i] = seev[no,i-1] + dt * seev_rhs
            
            seim_rhs = ( - seim[no,i-1]  + ( 1. - seim[no,i-1] ) * z1ei ) / tau_si
            seim[no,i] = seim[no,i-1] + dt * seim_rhs
            seiv_rhs = ( (1. - seim[no,i-1])**2 * z2ei + seiv[no,i-1] * (z2ei - 2. * tau_si * ( z1ei + 1.) ) ) / tau_si**2 
            seiv[no,i] = seiv[no,i-1] + dt * seiv_rhs
            
            siem_rhs = ( - siem[no,i-1]  + ( 1. - siem[no,i-1] ) * z1ie ) / tau_se
            siem[no,i] = siem[no,i-1] + dt * siem_rhs
            siev_rhs = ( (1. - siem[no,i-1])**2 * z2ie + siev[no,i-1] * (z2ie - 2. * tau_se * ( z1ie + 1.) ) ) / tau_se**2 
            siev[no,i] = siev[no,i-1] + dt * siev_rhs
            
            siim_rhs = ( - siim[no,i-1]  + ( 1. - siim[no,i-1] ) * z1ii ) / tau_si
            siim[no,i] = siim[no,i-1] + dt * siim_rhs
            siiv_rhs = ( (1. - siim[no,i-1])**2 * z2ii + siiv[no,i-1] * (z2ii - 2. * tau_si * ( z1ii + 1.) ) ) / tau_si**2 
            siiv[no,i] = siiv[no,i-1] + dt * siiv_rhs
            
            # Ensure the variance does not get negative for low activity
            if seev[no,i] < 0:
                seev[no,i] = 0.0

            if siev[no,i] < 0:
                siev[no,i] = 0.0

            if seiv[no,i] < 0:
                seiv[no,i] = 0.0

            if siiv[no,i] < 0:
                siiv[no,i] = 0.0
            
            #-----------------------------------------------------
            
            mufe_rhs = ( Jee_max * seem[no,i-1] + Jei_max * seim[no,i-1] + control_ext[no,0,i-startind+1] + ext_exc_current[no,i] + mue_ou[no,i-1]
                        - mufe[no,i-1] ) / tau_exc[no,i-1]
            mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
            rates_exc[no,i] = r_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1]) * 1e3
            Vmean_exc[no,i] = V_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1])
                        
            mufi_rhs = ( Jii_max * siim[no,i-1] + Jie_max * siem[no,i-1] + control_ext[no,1,i-startind+1] + ext_inh_current[no,i] + mui_ou[no,i-1]
                        - mufi[no,i-1] ) / tau_inh[no,i-1]
            mufi[no,i] = mufi[no,i-1] + dt * mufi_rhs
            rates_inh[no,i] = r_func(mufi[no,i-1], sigmai_f[no,i-1]) * 1e3
            
            IA_rhs =  b * rates_exc[no,i] * 1e-3 + ( a * ( Vmean_exc[no,i] - EA ) - IA[no,i-1] ) / tauA
            IA[no,i] = IA[no,i-1] + dt * IA_rhs
            
            # ornstein-uhlenbeck process
            mue_ou_rhs = (mue_ext_mean - mue_ou[no,i-1]) / tau_ou + sigma_ou * sqrt_dt * noise_exc[no] / dt
            mue_ou[no,i] = mue_ou[no,i-1] + dt * mue_ou_rhs # mV/ms
            mui_ou_rhs = (mui_ext_mean - mui_ou[no,i-1]) / tau_ou + sigma_ou * sqrt_dt * noise_inh[no] / dt
            mui_ou[no,i] = mui_ou[no,i-1] + dt * mui_ou_rhs # mV/ms
            
            
            
    rd_exc[no,no] = rates_exc[no,-1] * 1e-3
    rd_inh[no] = rates_inh[no,-1] * 1e-3
        
    z1ee = factor_ee1 * rd_exc[no, no]
    z2ee = factor_ee2 * rd_exc[no, no]  
    
    z1ei = factor_ei1 * rd_inh[no]
    z2ei = factor_ei2 * rd_inh[no]
    
    z1ie = factor_ie1 * rd_exc[no, no]
    z2ie = factor_ie2 * rd_exc[no, no] 
    
    z1ii = factor_ii1 * rd_inh[no]
    z2ii = factor_ii2 * rd_inh[no]
    
    sig_ee = seev[no,-1] * ( 2. * Jee_max**2 * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)
    sig_ei = seiv[no,-1] * ( 2. * Jei_max**2 * tau_si * taum ) * ( (1 + z1ei) * taum + tau_si )**(-1)

    sigmae_f[no,-1] = np.sqrt( sig_ee + sig_ei + sigmae_ext**2 )
    tau_exc[no,-1] = tau_func(mufe[no,-1] - IA[no,-1] / C, sigmae_f[no,-1])
    
    sig_ii = siiv[no,-1] * ( 2. * Jii_max**2 * tau_si * taum ) * ( (1 + z1ii) * taum + tau_si )**(-1)
    sig_ie = siev[no,-1] * ( 2. * Jie_max**2 * tau_se * taum ) * ( (1 + z1ie) * taum + tau_se )**(-1)
            
    sigmai_f[no,-1] = np.sqrt( sig_ii + sig_ie + sigmai_ext**2 )
    tau_inh[no,-1] = tau_func(mufi[no,-1], sigmai_f[no,-1])
    
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
    sigma_scale = 0.5
    mu_scale = - 10
    mu_scale1 = - 3
    y_shift = 15.
    sigma_shift = 1.4
    return sigma_scale * ( mu_shift + mu ) * sigma + mu_scale1 * mu + y_shift + np.exp( mu_scale * ( mu_shift + mu ) / ( sigma + sigma_shift ) )    
   

def V_func(mu, sigma):
    y_scale1 = 30.
    mu_shift1 = 1.
    y_shift = - 85.
    y_scale2 = 2.
    mu_shift2 = 0.5
    return y_shift + y_scale1 * np.tanh( mu + mu_shift1 ) + y_scale2 * np.exp( - ( mu - mu_shift2 )**2 ) / sigma