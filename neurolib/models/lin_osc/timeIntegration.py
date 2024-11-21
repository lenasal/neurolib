import numpy as np
import numba

from ...utils import model_utils as mu


def timeIntegration(params):
    """
    setting up parameters for time integration

    :param params: Parameter dictionary of the model
    :type params: dict

    :return: Integrated activity of the model
    :rtype: (numpy.ndarray, )
    """
    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)
    RNGseed = params["seed"]  # seed for RNG

    np.random.seed(RNGseed)

    # ------------------------------------------------------------------------
    # model parameters
    # ------------------------------------------------------------------------

    N = params["N"]  # number of oscillators

    omega = params["omega"]  # frequencies of oscillators

    # ------------------------------------------------------------------------
    # global coupling parameters
    # ------------------------------------------------------------------------

    # Connectivity matrix and Delay
    Cmat = params["Cmat"]

    # Interareal connection delay
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]

    if N == 1:
        Dmat = np.zeros((N, N))
    else:
        # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat = mu.computeDelayMatrix(lengthMat, signalV)

        # no self-feedback delay
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    Dmat = Dmat.astype(int)
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt

    # ------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------

    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    max_global_delay = np.max(Dmat_ndt)  # maximum global delay
    startind = int(max_global_delay + 1)  # start simulation after delay

    # Placeholders
    theta = np.zeros((N, startind + len(t)))

    theta_ext = mu.adjustArrayShape(params["theta_ext"], theta)

    # ------------------------------------------------------------------------
    # initial values
    # ------------------------------------------------------------------------

    if params["theta_init"].shape[1] == 1:
        theta_init = np.dot(params["theta_init"], np.ones((1, startind)))
    else:
        theta_init = params["theta_init"][:, -startind:]

    # put noise to instantiated array to save memory
    theta[:, :startind] = theta_init
    theta[:, startind:] = np.random.standard_normal((N, len(t)))

    # ------------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        N,
        omega,
        theta,
        theta_ext,
    )


@numba.njit
def timeIntegration_njit_elementwise(
    startind,
    t,
    dt,
    N,
    omega,
    theta,
    theta_ext,
):
    """
    Kuramoto Model
    """
    for i in range(startind, startind + len(t)):
        # Kuramoto model
        for no in range(N):

            theta_rhs = omega[no] + theta_ext[no, i - 1]

            # time integration
            theta[no, i] = theta[no, i - 1] + dt * theta_rhs

    return t, theta


@numba.njit
def compute_hx(
    N,
    V,
    T,
):
    hx = np.zeros((N, T, V, V))

    return hx


@numba.njit
def compute_hx_nw(
    N,
    T,
    V,
):

    hx_nw = np.zeros((N, N, T, V, V))

    return hx_nw

    return -hx_nw


@numba.njit
def Duh(
    N,
    V_in,
    V_vars,
    T,
    sv,
):
    """Jacobian of systems dynamics wrt. external inputs (control signals).

    :param model_params:    Tuple of parameters in the WC Model in order
    :type model_params:     tuple of float
    :param N:               Number of nodes in the network.
    :type N:                int
    :param V_in:            Number of input variables.
    :type V_in:             int
    :param V_vars:          Number of system variables.
    :type V_vars:           int
    :param T:               Length of simulation (time dimension).
    :type T:                int
    :param  nw_e:           N x T input of network into each node's 'exc'
    :type  nw_e:            np.ndarray
    :param ue:              N x T array of the total input received by 'exc' population in every node at any time.
    :type ue:               np.ndarray
    :param ui:              N x T array of the total input received by 'inh' population in every node at any time.
    :type ui:               np.ndarray
    :param e:               Value of the E-variable for each node and timepoint
    :type e:                np.ndarray
    :param i:               Value of the I-variable for each node and timepoint
    :type i:                np.ndarray
    :param K_gl:            global coupling strength
    :type K_gl              float
    :param cmat:            coupling matrix
    :type cmat:             np.ndarray
    :param dmat_ndt:        delay index matrix
    :type dmat_ndt:         np.ndarray
    :param exc_values:      N x T array containing values of 'exc' of all nodes through time.
    :type exc_values:       np.ndarray
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :rtype:     np.ndarray of shape N x V x V x T
    """

    duh = np.zeros((N, V_vars, V_in, T))
    for t in range(T):
        for n in range(N):
            duh[n, sv["theta"], sv["theta"], t] = -1.0
    return duh


@numba.njit
def Dxdoth(N, V):
    """Derivative of system dynamics wrt x dot

    :param N:       Number of nodes in the network.
    :type N:        int
    :param V:       Number of system variables.
    :type V:        int

    :return:        N x V x V matrix.
    :rtype:         np.ndarray
    """
    dxdoth = np.zeros((N, V, V))
    for n in range(N):
        for v in range(V):
            dxdoth[n, v, v] = 1.0

    return dxdoth
