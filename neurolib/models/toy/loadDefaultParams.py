import numpy as np

from ...utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None):
    """Load default parameters for the Toy model"""

    params = dotdict({})

    ### runtime parameters
    params.dt = 0.1  # ms 0.1ms is reasonable
    params.duration = 2000  # Simulation duration (ms)

    # signal transmission speec between areas
    params.signalV = 1.0
    params.K_gl = 1.0  # global coupling strength
    params.sigma_ou = 0.0

    if Cmat is None:
        params.N = 1
        params.Cmat = np.zeros((1, 1))
        params.lengthMat = np.zeros((1, 1))

    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(params.Cmat, 0)  # no self connections
        params.N = len(params.Cmat)  # number of nodes
        params.lengthMat = Dmat

    params.signalV = 1.0

    # ------------------------------------------------------------------------
    # local node parameters
    # ------------------------------------------------------------------------
    params.tau = 10.0
    params.alpha = 1.0
    # ------------------------------------------------------------------------

    params.x_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.y_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.x_ext = np.zeros((params.N,))
    params.y_ext = np.zeros((params.N,))
    params.x_ext_baseline = 0.0
    params.y_ext_baseline = 0.0

    return params
