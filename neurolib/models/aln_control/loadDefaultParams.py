import os
import numpy as np
import h5py

from ...utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, lookupTableFileName=None):
    """Load default parameters for a network of aLN nodes.
    :param Cmat: Structural connectivity matrix (adjacency matrix) of coupling strengths, will be normalized to 1. If not given, then a single node simulation will be assumed, defaults to None
    :type Cmat: numpy.ndarray, optional
    :param Dmat: Fiber length matrix, will be used for computing the delay matrix together with the signal transmission speed parameter `signalV`, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param lookUpTableFileName: Filename of lookup table with aln non-linear transfer functions and other precomputed quantities., defaults to aln-precalc/quantities_cascade.h
    :type lookUpTableFileName: str, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional
    
    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = dotdict({})

    # Todo: Model metadata
    # recently added for easier simulation of aln and brian in pypet
    params.model = "aln-control"
    params.name = "aln-control"
    params.description = "build up aln for control from mean field model of excitatory node"
    
    params.N = 1
    params.Cmat = np.zeros((1, 1))
    params.lengthMat = np.zeros((1, 1))
    params.Dmat = np.zeros((1, 1))

    # runtime parameters
    params.dt = 0.1  # ms 0.1ms is reasonable
    params.duration = 2000  # Simulation duration (ms)
    
    # ------------------------------------------------------------------------
    # local E-I node parameters
    # ------------------------------------------------------------------------

    # external input parameters:
    params.tau_ou = 5.0  # ms timescale of ornstein-uhlenbeck (OU) noise
    params.sigma_ou = 0.0 # mV/ms/sqrt(ms) intensity of OU oise
    params.mue_ext_mean = np.array( [[0.4]] )  # mV/ms mean external input current to E
    params.mui_ext_mean = np.array( [[0.3]])  # mV/ms mean external input current to I

    # Ornstein-Uhlenbeck noise state variables, set to mean input
    # mue_ou will fluctuate around mue_ext_mean (mean of the OU process)
    params.mue_ou = params.mue_ext_mean * np.ones((params.N,))  # np.zeros((params.N,))
    params.mui_ou = params.mui_ext_mean * np.ones((params.N,))  # np.zeros((params.N,))
    
    # externaln input currents, same as mue_ext_mean but can be time-dependent!
    params.ext_exc_current = 0.0  # external excitatory input current [mV/ms], C*[]V/s=[]nA
    params.ext_inh_current = 0.0  # external inhibiroty input current [mV/ms]
    
    # Fokker Planck noise (for N->inf)
    params.sigmae_ext = 1.5  # mV/sqrt(ms) (fixed, for now) [1-5] (Internal noise due to random coupling)
    params.sigmai_ext = 1.5  # mV/sqrt(ms) (fixed, for now) [1-5]  
    
    # recurrent coupling parameters
    params.Ke = 800.0  # Number of excitatory inputs per neuron
    params.Ki = 200.0  # Number of inhibitory inputs per neuron
    
    # synaptic delays
    params.de = 4.0  # ms local constant delay "EE = IE"
    params.di = 2.0  # ms local constant delay "EI = II"
    
    # synaptic time constants
    params.tau_se = 2.0  # ms  "EE = IE", for fixed delays
    params.tau_si = 5.0  # ms  "EI = II"
    
    # neuron model parameters
    params.a = 15.0  # nS, 15.0
    params.b = 40.0  # pA, 40.0
    params.EA = -80.0  # mV, -80.
    params.tauA = 200.  # ms, 200.0
    
    # PSC amplitudes
    params.cee = 0.3  # mV/ms
    params.cie = 0.3  # AMPA
    params.cei = 0.5  # GABA BrunelWang2003
    params.cii = 0.5 

    # Coupling strengths used in Cakan2020
    params.Jee_max = 2.43  # mV/ms
    params.Jie_max = 2.60  # mV/ms
    params.Jei_max = -3.3  # mV/ms [0-(-10)]
    params.Jii_max = -1.64  # mV/ms

    # single neuron paramters - if these are changed, new transfer functions must be precomputed!
    params.C = 200.0  # pF
    params.gL = 10.0  # nS
    params.EL = -65.0  # mV
    params.DeltaT = 1.5  # mV
    params.VT = -50.0  # mV
    params.Vr = -70.0  # mV
    params.Vs = -40.0  # mV
    params.Tref = 1.5  # ms
    
    params.rates_exc_init = 0.01 * 0.5 * np.ones((params.N,))
    
    #params.rates_exc_init = np.array( [[0.01 * 0.5 ]] )
    params.rates_inh_init = np.array( [[0.01 * 0.5 ]] )
    params.mufe_init = np.array( [[3. * 0.5 ]] )  # mV/ms
    params.mufi_init = np.array( [[3. * 0.5 ]] )  # mV/ms
    params.IA_init = np.array( [[200. * 0.5 ]] )  # pA
    params.seem_init = np.array( [[0.5 * 0.5 ]] )
    params.seim_init = np.array( [[0.5 * 0.5 ]] )   
    params.seev_init = np.array( [[0.01 * 0.5 ]] )
    params.seiv_init = np.array( [[0.01 * 0.5 ]] )
    params.siim_init = np.array( [[0.5 * 0.5 ]] )
    params.siem_init = np.array( [[0.5 * 0.5 ]] )
    params.siiv_init = np.array( [[0.01 * 0.5 ]] )
    params.siev_init = np.array( [[0.01 * 0.5 ]] )

    return params

