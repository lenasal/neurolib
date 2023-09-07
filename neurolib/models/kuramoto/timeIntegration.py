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

    # ornstein uhlenbeck noise param
    tau_ou = params["tau_ou"]  # noise time constant
    sigma_ou = params["sigma_ou"]  # noise strength

    # ------------------------------------------------------------------------
    # global coupling parameters
    # ------------------------------------------------------------------------

    # Connectivity matrix and Delay
    Cmat = params["Cmat"]

    # Interareal connection delay
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]
    k = params["k"]  # coupling strength

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
    sqrt_dt = np.sqrt(dt)

    max_global_delay = np.max(Dmat_ndt)  # maximum global delay
    startind = int(max_global_delay + 1)  # start simulation after delay

    # Placeholders
    theta_ou = params["theta_ou"].copy()
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

    k_n = k / N  # auxiliary variable

    # ------------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        sqrt_dt,
        N,
        omega,
        k_n,
        Cmat,
        Dmat,
        theta,
        theta_ext,
        tau_ou,
        sigma_ou,
        theta_ou,
    )


@numba.njit
def timeIntegration_njit_elementwise(
    startind,
    t,
    dt,
    sqrt_dt,
    N,
    omega,
    k_n,
    Cmat,
    Dmat,
    theta,
    theta_ext,
    tau_ou,
    sigma_ou,
    theta_ou,
):
    """
    Kuramoto Model
    """
    for i in range(startind, startind + len(t)):
        # Kuramoto model
        for no in range(N):
            noise_theta = theta[no, i]
            theta_input_d = 0.0

            # adding input from other nodes
            for m in range(N):
                theta_input_d += k_n * Cmat[no, m] * np.sin(theta[m, i - 1 - Dmat[no, m]] - theta[no, i - 1])

            theta_rhs = omega[no] + theta_input_d + theta_ou[no] + theta_ext[no, i - 1]

            # time integration
            theta[no, i] = theta[no, i - 1] + dt * theta_rhs

            # phase reset
            theta[no, i] = np.mod(theta[no, i], 2 * np.pi)

            # Ornstein-Uhlenbeck
            theta_ou[no] = theta_ou[no] - theta_ou[no] * dt / tau_ou + sigma_ou * sqrt_dt * noise_theta

    return t, theta, theta_ou


@numba.njit
def compute_hx(
    k_n,
    N,
    V,
    T,
    theta,
    cmat,
    dmat_ndt,
    sv,
):
    """Jacobians  of FHN model wrt. its 'state_vars' at each time step.

    :param model_params:    Ordered tuple of parameters in the FHN Model in order
    :type model_params:     tuple of float
    :param N:                   Number of nodes in the network.
    :type N:                    int
    :param V:                   Number of system variables.
    :type V:                    int
    :param T:                   Length of simulation (time dimension).
    :type T:                    int
    :param dyn_vars:            Values of the 'x' and 'y' variable of FHN of all nodes through time.
    :type dyn_vars:             np.ndarray of shape N x 2 x T
    :param cmat:                Connectivity matrix
    :type cmat:                 np.ndarray
    :param dmat_ndt:            Delay matrix in time steps
    :type dmat_ndt:             np.ndarray of ints
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                    Array that contains Jacobians for all nodes in all time steps.
    :rtype:                     np.ndarray of shape N x T x v X v
    """
    hx = np.zeros((N, T, V, V))
    nw_input = compute_nw_input_cos(N, T, theta, cmat, dmat_ndt, k_n)

    for n in range(N):  # Iterate through nodes.
        for t in range(T):
            hx[n, t, sv["theta"], sv["theta"]] = -nw_input[n, t]

    return hx


@numba.njit
def compute_nw_input_cos(N, T, theta, cmat, dmat_ndt, k_n):
    nw_input = np.zeros((N, T))

    for n in range(N):
        for l in range(N):
            if cmat[n, l] == 0.0:
                continue
            if n == l and cmat[n, l] != 0.0:
                print("WARNING: Cmat diagonal not zero.")
            for t in range(T):
                nw_input[n, t] += cmat[n, l] * np.cos(theta[l, t - 1 - dmat_ndt[n, l]] - theta[n, t - 1])

    return k_n * nw_input


@numba.njit
def compute_hx_nw(
    k_n,
    N,
    V,
    T,
    theta,
    cmat,
    dmat_ndt,
    sv,
):
    """Jacobians for network connectivity in all time steps.

    :param K_gl:     Model parameter of global coupling strength.
    :type K_gl:      float
    :param cmat:     Model parameter, connectivity matrix.
    :type cmat:      ndarray
    :param N:        Number of nodes in the network.
    :type N:         int
    :param V:        Number of system variables.
    :type V:         int
    :param T:        Length of simulation (time dimension).
    :type T:         int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:         Jacobians for network connectivity in all time steps.
    :rtype:          np.ndarray of shape N x N x T x 2 x 2
    """
    hx_nw = np.zeros((N, N, T, V, V))

    for n1 in range(N):
        nw_der = compute_nw_input_cos(N, T, theta, cmat, dmat_ndt, k_n)
        for n2 in range(N):
            for t in range(T):
                if t + dmat_ndt[n1, n2] >= T:
                    continue
                hx_nw[n1, n2, t, sv["theta"], sv["theta"]] = k_n * np.cos(
                    theta[n2, t] - theta[n1, t + dmat_ndt[n1, n2]]
                )

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

    :param N:               Number of nodes in the network.
    :type N:                int
    :param V_in:            Number of input variables.
    :type V_in:             int
    :param V_vars:          Number of system variables.
    :type V_vars:           int
    :param T:               Length of simulation (time dimension).
    :type T:                int
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
            dxdoth[n, v, v] = 1

    return dxdoth
