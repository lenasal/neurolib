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

    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)
    RNGseed = params["seed"]  # seed for RNG

    # set to 0 for faster computation

    # ------------------------------------------------------------------------
    # global coupling parameters

    # Connectivity matric
    # Interareal relative coupling strengths (values between 0 and 1), Cmat(i,j) connnection from jth to ith
    Cmat = params["Cmat"]
    c_gl = params["c_gl"]  # EPSP amplitude between areas
    Ke_gl = params["Ke_gl"]  # number of incoming E connections (to E population) from each area

    N = len(Cmat)  # Number of areas

    # Interareal connection delay
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]

    if N == 1:
        Dmat = np.ones((N, N)) * params["de"]
    else:
        Dmat = dp.computeDelayMatrix(
            lengthMat, signalV
        )  # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat[np.eye(len(Dmat)) == 1] = np.ones(len(Dmat)) * params["de"]

    params["Dmat"] = Dmat
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt

    # ------------------------------------------------------------------------

    # local network (area) parameters [identical for all areas for now]

    ### model parameters
    filter_sigma = params["filter_sigma"]

    # distributed delay between areas, not tested, but should work
    # distributed delay is implemented by a convolution with the delay kernel
    # the convolution is represented as a linear ODE with the timescale that
    # corresponds to the width of the delay distribution
    distr_delay = params["distr_delay"]

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
    tau_de = params["tau_de"]
    tau_di = params["tau_di"]

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

    # rescales c's here: multiplication with tau_se makes
    # the increase of s subject to a single input spike invariant to tau_se
    # division by J ensures that mu = J*s will result in a PSP of exactly c
    # for a single spike!

    cee = cee * tau_se / Jee_max  # ms
    cie = cie * tau_se / Jie_max  # ms
    cei = cei * tau_si / abs(Jei_max)  # ms
    cii = cii * tau_si / abs(Jii_max)  # ms
    c_gl = c_gl * tau_se / Jee_max  # ms

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

    # Lookup tables for the transfer functions
    precalc_r, precalc_V, precalc_tau_mu, precalc_tau_sigma = (
        params["precalc_r"],
        params["precalc_V"],
        params["precalc_tau_mu"],
        params["precalc_tau_sigma"],
    )

    # parameter for the lookup tables
    dI = params["dI"]
    ds = params["ds"]
    sigmarange = params["sigmarange"]
    Irange = params["Irange"]

    # Initialization
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)
    sqrt_dt = np.sqrt(dt)

    ndt_de = np.around(de / dt).astype(int)
    ndt_di = np.around(di / dt).astype(int)

    rd_exc = np.zeros((N, N))  # kHz  rd_exc(i,j): Connection from jth node to ith
    rd_inh = np.zeros(N)

    # Already done above when Dmat_ndt is built
    # for l in range(N):
    #    Dmat_ndt[l, l] = ndt_de  # if no distributed, this is a fixed value (E-E coupling)

    max_global_delay = max(np.max(Dmat_ndt), ndt_de, ndt_di)
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

    mue_ou[:,:startind] = params["mue_ou"].copy()  # Mean of external exc OU input (mV/ms)
    mui_ou[:,:startind] = params["mui_ou"].copy()  # Mean of external inh ON inout (mV/ms)

    # Set the initial firing rates.
    # if initial values are just N array:
    if type(params["rates_exc_init"]) is not type(np.array([])):
        logging.error("wrong input for initial rates")
    elif len(np.shape(params["rates_exc_init"])) == 1:
        logging.error("wrong input for initial rates")
        #rates_exc_init = (params["rates_exc_init"] * np.ones((1, startind))).T   # kHz
        #rates_inh_init = (params["rates_inh_init"] * np.ones((startind, 1))).T  # kHz
    # if initial values are just a Nx1 array
    elif np.shape(params["rates_exc_init"])[1] == 1:
        # repeat the 1-dim value stardind times
        rates_exc_init = np.dot(params["rates_exc_init"], np.ones((1, startind)))  # kHz
        rates_inh_init = np.dot(params["rates_inh_init"], np.ones((1, startind)))  # kHz
        # set initial adaptation current
    # if initial values are a Nxt array
    else:
        rates_exc_init = params["rates_exc_init"][:, -startind:]
        rates_inh_init = params["rates_inh_init"][:, -startind:]

    if type(params["IA_init"]) is not type(np.array([])):
        logging.error("wrong input for initial adaptation current")
    elif len(np.shape(params["IA_init"])) == 1:
        A_init = np.ones((1, startind)) * params["IA_init"][0]
    # if initial values are just a Nx1 array
    elif np.shape(params["IA_init"])[1] == 1:
        # repeat the 1-dim value stardind times
        IA_init = np.dot(params["IA_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        IA_init = params["IA_init"][:, -startind:] 

    if RNGseed:
        np.random.seed(RNGseed)

    # Save the noise in the rates array to save memory
    rates_exc[:, startind:] = np.random.standard_normal((N, len(t)))
    rates_inh[:, startind:] = np.random.standard_normal((N, len(t)))

    # Set the initial conditions
    rates_exc[:, :startind] = rates_exc_init
    rates_inh[:, :startind] = rates_inh_init
    IA[:, :startind] = IA_init

    noise_exc = np.zeros((N,))
    noise_inh = np.zeros((N,))

    # tile external inputs to appropriate shape
    ext_exc_current = adjust_shape(params["ext_exc_current"], rates_exc)
    ext_inh_current = adjust_shape(params["ext_inh_current"], rates_exc)
    ext_exc_rate = adjust_shape(params["ext_exc_rate"], rates_exc)
    ext_inh_rate = adjust_shape(params["ext_inh_rate"], rates_exc)
    
    
    control_ext = control.copy()

    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        dt,
        duration,
        distr_delay,
        filter_sigma,
        Cmat,
        Dmat,
        c_gl,
        Ke_gl,
        tau_ou,
        sigma_ou,
        mue_ext_mean,
        mui_ext_mean,
        sigmae_ext,
        sigmai_ext,
        Ke,
        Ki,
        de,
        di,
        tau_se,
        tau_si,
        tau_de,
        tau_di,
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
        gL,
        EL,
        DeltaT,
        VT,
        Vr,
        Vs,
        Tref,
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
        precalc_r,
        precalc_V,
        precalc_tau_mu,
        precalc_tau_sigma,
        dI,
        ds,
        sigmarange,
        Irange,
        N,
        Dmat_ndt,
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
        ext_exc_rate,
        ext_inh_rate,
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
    distr_delay,
    filter_sigma,
    Cmat,
    Dmat,
    c_gl,
    Ke_gl,
    tau_ou,
    sigma_ou,
    mue_ext_mean,
    mui_ext_mean,
    sigmae_ext,
    sigmai_ext,
    Ke,
    Ki,
    de,
    di,
    tau_se,
    tau_si,
    tau_de,
    tau_di,
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
    gL,
    EL,
    DeltaT,
    VT,
    Vr,
    Vs,
    Tref,
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
    precalc_r,
    precalc_V,
    precalc_tau_mu,
    precalc_tau_sigma,
    dI,
    ds,
    sigmarange,
    Irange,
    N,
    Dmat_ndt,
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
    ext_exc_rate,
    ext_inh_rate,
    ext_exc_current,
    ext_inh_current,
    noise_exc,
    noise_inh,
    control_ext,
):

    # squared Jee_max
    sq_Jee_max = Jee_max ** 2
    sq_Jei_max = Jei_max ** 2
    sq_Jie_max = Jie_max ** 2
    sq_Jii_max = Jii_max ** 2

    # initialize so we don't get an error when returning
    rd_exc_rhs = 0.0
    rd_inh_rhs = 0.0
    sigmae_f_rhs = 0.0
    sigmai_f_rhs = 0.0
    
    sigmae_f = np.zeros((N, startind + len(t)))
    sigmai_f = np.zeros((N, startind + len(t)))
    Vmean_exc = np.zeros((N, startind + len(t)))
    tau_exc = np.zeros((N, startind + len(t)))
    tau_inh = np.zeros((N, startind + len(t)))

    if filter_sigma:
        sigmae_f[:,:startind] = sigmae_ext
        sigmai_f[:,:startind] = sigmai_ext
    
    """
    for no in range(N):           
        mue = (Jee_max * seem[no,0] + Jei_max * seim[no,0] + mue_ou[no,0] + ext_exc_current[no,0] + control_ext[no,0,0])
        mui = (Jie_max * siem[no,0] + Jii_max * siim[no,0] + mui_ou[no,0] + ext_inh_current[no,0] + control_ext[no,1,0])
        
        mufe_rhs = ( mue - mufe[no,startind - min(startind, 2)] ) / tau_exc[no,0]
        mufi_rhs = ( mui - mufi[no,startind - min(startind, 2)] ) / tau_inh[no,0]

        mufe[no,startind] = mufe[no,startind - min(startind, 2)] + dt * mufe_rhs
        mufi[no,startind] = mufi[no,startind - min(startind, 2)] + dt * mufi_rhs
    """
        
    ### integrate ODE system:
    for i in range(startind, startind + len(t)):
        
        if not distr_delay:
            # Get the input from one node into another from the rates at time t - connection_delay - 1
            # remark: assume Kie == Kee and Kei == Kii
            for no in range(N):
                # interareal coupling
                for l in range(N):
                    # rd_exc(i,j) delayed input rate from population j to population i
                    rd_exc[l, no] = rates_exc[no, i - Dmat_ndt[l, no] - 1] * 1e-3  # convert Hz to kHz
                # Warning: this is a vector and not a matrix as rd_exc
                rd_inh[no] = rates_inh[no, i - ndt_di - 1] * 1e-3  # convert Hz to kHz
               # print("excitatory and inhibitory rates : ", rates_exc[no, i - Dmat_ndt[l, no] - 1], " index ",  no, i - Dmat_ndt[l, no] - 1)
                #print("inh =", rates_inh[no, i - ndt_di - 1], " index ", no, i - ndt_di - 1)

        # loop through all the nodes
        for no in range(N):

            # To save memory, noise is saved in the rates array
            noise_exc[no] = rates_exc[no, i]
            noise_inh[no] = rates_inh[no, i]

            # subtract startind from control, as initial conditions are not set.
            mue = (Jee_max * seem[no,i-1] + Jei_max * seim[no,i-1] + mue_ou[no,i-1] + ext_exc_current[no, i]
                   + control_ext[no, 0, i-startind+1]
                   )
            mui = (Jie_max * siem[no,i-1] + Jii_max * siim[no,i-1] + mui_ou[no,i-1] + ext_inh_current[no, i]
                   + control_ext[no, 1, i-startind+1]
                   )
            #if (i in range(startind, startind + 3,1)):
            #print("mue computation: ",no, i-startind, control_ext[no, 0, i-startind], control_ext[no, 1, i-startind])

            # compute row sum of Cmat*rd_exc and Cmat**2*rd_exc
            rowsum = 0
            rowsumsq = 0
            for col in range(N):
                rowsum = rowsum + Cmat[no, col] * rd_exc[no, col]
                rowsumsq = rowsumsq + Cmat[no, col] ** 2 * rd_exc[no, col]
                
            #print("input via coupling: time, node, rowsum = ", i, no, rowsum)

            # z1: weighted sum of delayed rates, weights=c*K
            z1ee = (
                cee * Ke * rd_exc[no, no] + c_gl * Ke_gl * rowsum + c_gl * Ke_gl * ext_exc_rate[no, i]
            )  # rate from other regions + exc_ext_rate
            z1ei = cei * Ki * rd_inh[no]
            z1ie = (
                cie * Ke * rd_exc[no, no] + c_gl * Ke_gl * ext_inh_rate[no, i]
            )  # first test of external rate input to inh. population
            z1ii = cii * Ki * rd_inh[no]
            #print("parameters of calculation: rd_exc[no, no], rd_inh[no]", rd_exc[no, no], rd_inh[no])
            # z2: weighted sum of delayed rates, weights=c^2*K (see thesis last ch.)
            z2ee = (
                cee ** 2 * Ke * rd_exc[no, no] + c_gl ** 2 * Ke_gl * rowsumsq + c_gl ** 2 * Ke_gl * ext_exc_rate[no, i]
            )
            #print("parts of z2ee: ", cee ** 2 * Ke * rd_exc[no, no],  cee ** 2 * Ke, rd_exc[no, no])
            z2ei = cei ** 2 * Ki * rd_inh[no]
            z2ie = (
                cie ** 2 * Ke * rd_exc[no, no] + c_gl ** 2 * Ke_gl * ext_inh_rate[no, i]
            )  # external rate input to inh. population
            z2ii = cii ** 2 * Ki * rd_inh[no]

            sigmae = np.sqrt(
                2 * sq_Jee_max * seev[no,i-1] * tau_se * taum / ((1 + z1ee) * taum + tau_se)
                + 2 * sq_Jei_max * seiv[no,i-1] * tau_si * taum / ((1 + z1ei) * taum + tau_si)
                + sigmae_ext ** 2
            )  # mV/sqrt(ms)
            
            ratio_exc = sigmae/sigmae_ext - 1.
            if (ratio_exc < 1e-4 and ratio_exc != 0.):
                print("ratio of sigma exc : ",  ratio_exc)
            
            sigmai = np.sqrt(
                2 * sq_Jie_max * siev[no,i-1] * tau_se * taum / ((1 + z1ie) * taum + tau_se)
                + 2 * sq_Jii_max * siiv[no,i-1] * tau_si * taum / ((1 + z1ii) * taum + tau_si)
                + sigmai_ext ** 2
            )  # mV/sqrt(ms)

            if not filter_sigma:
                sigmae_f[no,i-1] = sigmae
                sigmai_f[no,i-1] = sigmai

            # Read the transfer function from the lookup table
            # -------------------------------------------------------------

            # ------- excitatory population
            rates_exc[no,i] = r_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1]) * 1e3
            Vmean_exc[no,i] = V_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1])

            tau_exc[no,i-1] = tau_func(mufe[no,i-1] - IA[no,i-1] / C, sigmae_f[no,i-1])


            # ------- inhibitory population
            rates_inh[no,i] = r_func(mufi[no,i-1], sigmai_f[no,i-1]) * 1e3
            
            tau_inh[no,i-1] = tau_func(mufi[no,i-1], sigmai_f[no,i-1])


            # -------------------------------------------------------------

            # now everything available for r.h.s:

            mufe_rhs = (mue - mufe[no,i-1] 
                        #+ control_ext[no, 0, i-startind+1]
                        ) / tau_exc[no,i-1]
            
            mufi_rhs = (mui - mufi[no,i-1] 
                        #+ control_ext[no, 1, i-startind+1]
                        ) / tau_inh[no,i-1]

            # rate has to be kHz
            IA_rhs = (a * (Vmean_exc[no,i] - EA) - IA[no, i - 1] + tauA * b * rates_exc[no, i] * 1e-3) / tauA
            
            # EQ. 4.43
            if distr_delay:
                rd_exc_rhs = (rates_exc[no, i] * 1e-3 - rd_exc[no, no]) / tau_de
                rd_inh_rhs = (rates_inh[no, i] * 1e-3 - rd_inh[no]) / tau_di

            # integration of synaptic input (eq. 4.36)
            
            seem_rhs = ((1 - seem[no,i-1]) * z1ee - seem[no,i-1]) / tau_se
            seim_rhs = ((1 - seim[no,i-1]) * z1ei - seim[no,i-1]) / tau_si
            siem_rhs = ((1 - siem[no,i-1]) * z1ie - siem[no,i-1]) / tau_se
            siim_rhs = ((1 - siim[no,i-1]) * z1ii - siim[no,i-1]) / tau_si
            seev_rhs = ((1 - seem[no,i-1]) ** 2 * z2ee + (z2ee - 2 * tau_se * (z1ee + 1)) * seev[no,i-1]) / tau_se ** 2
            seiv_rhs = ((1 - seim[no,i-1]) ** 2 * z2ei + (z2ei - 2 * tau_si * (z1ei + 1)) * seiv[no,i-1]) / tau_si ** 2
            siev_rhs = ((1 - siem[no,i-1]) ** 2 * z2ie + (z2ie - 2 * tau_se * (z1ie + 1)) * siev[no,i-1]) / tau_se ** 2
            siiv_rhs = ((1 - siim[no,i-1]) ** 2 * z2ii + (z2ii - 2 * tau_si * (z1ii + 1)) * siiv[no,i-1]) / tau_si ** 2

            # -------------- integration --------------

            mufe[no,i] = mufe[no,i-1] + dt * mufe_rhs
            mufi[no,i] = mufi[no,i-1] + dt * mufi_rhs
            IA[no,i] = IA[no, i - 1] + dt * IA_rhs

            seem[no,i] = seem[no,i-1] + dt * seem_rhs
            seim[no,i] = seim[no,i-1] + dt * seim_rhs
            siem[no,i] = siem[no,i-1] + dt * siem_rhs
            siim[no,i] = siim[no,i-1] + dt * siim_rhs
            seev[no,i] = seev[no,i-1] + dt * seev_rhs
            seiv[no,i] = seiv[no,i-1] + dt * seiv_rhs
            siev[no,i] = siev[no,i-1] + dt * siev_rhs
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
                  
            if (not filter_sigma and i == startind + len(t) - 1):
                sigmae_f[no,i] = np.sqrt(
                2 * sq_Jee_max * seev[no,i] * tau_se * taum / ((1 + z1ee) * taum + tau_se)
                + 2 * sq_Jei_max * seiv[no,i] * tau_si * taum / ((1 + z1ei) * taum + tau_si)
                + sigmae_ext ** 2)  # mV/sqrt(ms)

                sigmai_f[no,i] = np.sqrt(
                2 * sq_Jie_max * siev[no,i] * tau_se * taum / ((1 + z1ie) * taum + tau_se)
                + 2 * sq_Jii_max * siiv[no,i] * tau_si * taum / ((1 + z1ii) * taum + tau_si)
                + sigmai_ext ** 2)

            # ornstein-uhlenbeck process
            mue_ou[no,i] = (
                mue_ou[no,i-1] + (mue_ext_mean - mue_ou[no,i-1]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_exc[no]
            )  # mV/ms
            mui_ou[no,i] = (
                mui_ou[no,i-1] + (mui_ext_mean - mui_ou[no,i-1]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_inh[no]
            )  # mV/ms
                        
    #sigmae_f[:,:startind] = sigmae_f[:,startind]
    #sigmai_f[:,:startind] = sigmai_f[:,startind]
    #Vmean_exc[:,:startind] = Vmean_exc[:,startind]
    if a == 0.:
        Vmean_exc[:,:startind] = Vmean_exc[:,startind]
    else:
        Vmean_exc[:,:startind] = EA + ( 1./a ) * ( tauA * ( IA[:,startind] - IA[:,startind-1] ) / dt - tauA * b * rates_exc[:,startind] * 1e-3 + IA[:,startind-1]) 
    
    if not distr_delay:
    # Get the input from one node into another from the rates at time t - connection_delay - 1
    # remark: assume Kie == Kee and Kei == Kii
        for no in range(N):
            # interareal coupling
            for l in range(N):
                # rd_exc(i,j) delayed input rate from population j to population i
                rd_exc[l,no] = rates_exc[no,-Dmat_ndt[l, no]-1] * 1e-3  # convert Hz to kHz
            # Warning: this is a vector and not a matrix as rd_exc
            rd_inh[no] = rates_inh[no,-ndt_di-1] * 1e-3  # convert Hz to kHz
    
    # compute row sum of Cmat*rd_exc and Cmat**2*rd_exc
    rowsum = 0
    rowsumsq = 0
    for col in range(N):
        rowsum = rowsum + Cmat[no,col] * rd_exc[no,col]
        rowsumsq = rowsumsq + Cmat[no,col] ** 2 * rd_exc[no,col]

    # z1: weighted sum of delayed rates, weights=c*K
    z1ee = (
        cee * Ke * rd_exc[no,no] + c_gl * Ke_gl * rowsum + c_gl * Ke_gl * ext_exc_rate[no,-1]
    )  # rate from other regions + exc_ext_rate
    z1ei = cei * Ki * rd_inh[no]
    z1ie = (
        cie * Ke * rd_exc[no, no] + c_gl * Ke_gl * ext_inh_rate[no,-1]
    )  # first test of external rate input to inh. population
    z1ii = cii * Ki * rd_inh[no]
    #print("parameters of calculation: rd_exc[no, no], rd_inh[no]", rd_exc[no, no], rd_inh[no])
    # z2: weighted sum of delayed rates, weights=c^2*K (see thesis last ch.)
    z2ee = (
        cee ** 2 * Ke * rd_exc[no, no] + c_gl ** 2 * Ke_gl * rowsumsq + c_gl ** 2 * Ke_gl * ext_exc_rate[no,-1]
    )
    #print("parts of z2ee: ", cee ** 2 * Ke * rd_exc[no, no],  cee ** 2 * Ke, rd_exc[no, no])
    z2ei = cei ** 2 * Ki * rd_inh[no]
    z2ie = (
        cie ** 2 * Ke * rd_exc[no, no] + c_gl ** 2 * Ke_gl * ext_inh_rate[no,-1]
    )  # external rate input to inh. population
    z2ii = cii ** 2 * Ki * rd_inh[no]

    sigmae = np.sqrt(
        2 * sq_Jee_max * seev[no,-1] * tau_se * taum / ((1 + z1ee) * taum + tau_se)
        + 2 * sq_Jei_max * seiv[no,-1] * tau_si * taum / ((1 + z1ei) * taum + tau_si)
        + sigmae_ext ** 2
    )  # mV/sqrt(ms)
    
    ratio_exc = sigmae/sigmae_ext - 1.
    if (ratio_exc < 1e-4 and ratio_exc != 0.):
                print("ratio of sigma exc : ",  ratio_exc)
    
    sigmai = np.sqrt(
        2 * sq_Jie_max * siev[no,-1] * tau_se * taum / ((1 + z1ie) * taum + tau_se)
        + 2 * sq_Jii_max * siiv[no,-1] * tau_si * taum / ((1 + z1ii) * taum + tau_si)
        + sigmai_ext ** 2
    )  # mV/sqrt(ms)

    if not filter_sigma:
        sigmae_f[no,-1] = sigmae
        sigmai_f[no,-1] = sigmai
    
    tau_exc[no,-1] = tau_func(mufe[no,-1], sigmae_f[no,-1])

    tau_inh[no,-1] = tau_func(mufi[no,-1], sigmai_f[no,-1])

    return t, rates_exc, rates_inh, mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv, mue_ou, mui_ou, sigmae_f, sigmai_f, Vmean_exc, tau_exc, tau_inh


def adjust_shape(original, target):
    """
    Tiles and then cuts an array (or list or float) such that
    it has the same shape as target at the end.
    This is used to make sure that any input parameter like external current has
    the same shape as the rate array.
    """

    # make an ext_exc_current ARRAY from a LIST or INT
    if not hasattr(original, "__len__"):
        original = [original]
    original = np.array(original)

    # repeat original in y until larger (or same size) as target

    # tile until N

    # either (x,) shape or (y,x) shape
    if len(original.shape) == 1:
        # if original.shape[0] > 1:
        rep_y = target.shape[0]
    elif target.shape[0] > original.shape[0]:
        rep_y = int(target.shape[0] / original.shape[0]) + 1
    else:
        rep_y = 1

    # tile once so the array has shape (N,1)
    original = np.tile(original, (rep_y, 1))

    # tile until t

    if target.shape[1] > original.shape[1]:
        rep_x = int(target.shape[1] / original.shape[1]) + 1
    else:
        rep_x = 1
    original = np.tile(original, (1, rep_x))

    # cut from end because the beginning can be initial condition
    original = original[: target.shape[0], -target.shape[1] :]

    return original

def r_func(mu, sigma):
    x_scale_sigma = 0.6
    return ( mu + sigma )* 1e-2

def tau_func(mu, sigma):
    return (1. + mu + sigma) * 1e0
   

def V_func(mu, sigma):
    return (mu + sigma) * 1e-2