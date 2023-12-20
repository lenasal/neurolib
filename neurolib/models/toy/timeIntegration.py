import numpy as np
import numba

from ...utils import model_utils as mu


def timeIntegration(params):
    """Sets up the parameters for time integration

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """

    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)

    # ------------------------------------------------------------------------
    # local parameters
    alpha = params["alpha"]
    tau = params["tau"]

    # ------------------------------------------------------------------------
    # global coupling parameters

    # Connectivity matrix
    # Interareal relative coupling strengths (values between 0 and 1), Cmat(i,j) connnection from jth to ith
    Cmat = params["Cmat"]
    N = len(Cmat)  # Number of nodes
    K_gl = params["K_gl"]  # global coupling strength
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
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt

    # ------------------------------------------------------------------------

    # Initialization
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)

    max_global_delay = np.max(Dmat_ndt)
    startind = int(max_global_delay + 1)  # timestep to start integration at

    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data
    xs = np.zeros((N, startind + len(t)))
    x_ext = mu.adjustArrayShape(params["x_ext"], xs)
    x_ext_baseline = params["x_ext_baseline"]
    ys = np.zeros((N, startind + len(t)))
    y_ext = mu.adjustArrayShape(params["y_ext"], xs)
    y_ext_baseline = params["y_ext_baseline"]

    # ------------------------------------------------------------------------
    # Set initial values
    if np.shape(params["x_init"])[1] == 1:
        x_init = np.dot(params["x_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        x_init = params["x_init"][:, -startind:]

    if np.shape(params["y_init"])[1] == 1:
        y_init = np.dot(params["y_init"], np.ones((1, startind)))
    # if initial values are a Nxt array
    else:
        y_init = params["y_init"][:, -startind:]

    # Save the noise in the activity array to save memory
    xs[:, :startind] = x_init
    ys[:, :startind] = y_init

    # ------------------------------------------------------------------------

    return timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        duration,
        N,
        Cmat,
        Dmat,
        K_gl,
        signalV,
        Dmat_ndt,
        xs,
        x_ext,
        x_ext_baseline,
        ys,
        y_ext,
        y_ext_baseline,
        alpha,
        tau,
    )


@numba.njit
def timeIntegration_njit_elementwise(
    startind,
    t,
    dt,
    duration,
    N,
    Cmat,
    Dmat,
    K_gl,
    signalV,
    Dmat_ndt,
    xs,
    x_ext,
    x_ext_baseline,
    ys,
    y_ext,
    y_ext_baseline,
    alpha,
    tau,
):
    ### integrate ODE system:
    for i in range(startind, startind + len(t)):
        # loop through all the nodes
        for no in range(N):
            # delayed input to each node
            xs_input_d = 0

            for l in range(N):
                xs_input_d += K_gl * Cmat[no, l] * (xs[l, i - Dmat_ndt[no, l] - 1])

            x_rhs = ys[no, i - 1] + x_ext[no, i - 1] + xs_input_d + x_ext_baseline
            y_rhs = (
                -xs[no, i - 1] / tau
                - alpha * ys[no, i - 1]
                + y_ext[no, i - 1]
                + y_ext_baseline
            )
            # Euler integration
            xs[no, i] = xs[no, i - 1] + dt * x_rhs
            ys[no, i] = ys[no, i - 1] + dt * y_rhs

    return t, xs, ys


@numba.njit
def jacobian_toy(
    model_params,
    x,
    nw,
    u,
    V,
    sv,
):
    """Jacobian of a single node of the FHN dynamical system wrt. its 'state_vars' ('x', 'y', 'x_ou', 'y_ou'). The
       Jacobian of the FHN systems dynamics depends only on the constant model parameters and the values of the 'x'-
       population.

    :param model_params:    Ordered tuple of parameters in the FHN Model in order
    :type model_params:     tuple of float
    :param x:                   Value of the 'x'-population in the FHN node at a specific time step.
    :type x:                    float
    :param V:                   Number of system variables.
    :type V:                    int
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict


    :return:                    V x V Jacobian matrix.
    :rtype:                     np.ndarray
    """
    (
        alpha,
        tau,
        x_ext_baseline,
    ) = model_params
    jacobian = np.zeros((V, V))
    jacobian[sv["x"], sv["y"]] = -1.0
    jacobian[sv["y"], sv["x"]] = 1.0 / tau
    jacobian[sv["y"], sv["y"]] = alpha
    return jacobian


@numba.njit
def compute_hx(
    model_params,
    K_gl,
    cmat,
    dmat_ndt,
    N,
    V,
    T,
    dyn_vars,
    dyn_vars_delay,
    control,
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
    :param sv:                  dictionary of state vars and respective indices
    :type sv:                   dict

    :return:                    Array that contains Jacobians for all nodes in all time steps.
    :rtype:                     np.ndarray of shape N x T x v X v
    """
    hx = np.zeros((N, T, V, V))
    nw = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, dyn_vars_delay[:, sv["x"], :])

    for n in range(N):  # Iterate through nodes.
        for t in range(T):
            hx[n, t, :, :] = jacobian_toy(
                model_params, dyn_vars[n, sv["x"], t], nw[n, t], control[n, :, t], V, sv
            )
    return hx


@numba.njit
def compute_nw_input(N, T, K_gl, cmat, dmat_ndt, x):
    nw_input = np.zeros((N, T))

    for t in range(1, T):
        for n in range(N):
            for l in range(N):
                nw_input[n, t] += K_gl * cmat[n, l] * (x[l, t - dmat_ndt[n, l] - 1])
    return nw_input


@numba.njit
def compute_hx_nw(
    model_params,
    K_gl,
    cmat,
    dmat_ndt,
    N,
    V,
    T,
    sv,
    dyn_vars,
    dyn_vars_delay,
    control,
):
    (
        alpha,
        tau,
        x_ext_baseline,
    ) = model_params
    hx_nw = np.zeros((N, N, T, V, V))

    nw = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, dyn_vars_delay[:, sv["x"], :])

    for n1 in range(N):
        for n2 in range(N):
            for t in range(T - 1):
                exp = np.exp(-control[n1, sv["x"], t] - x_ext_baseline - nw[n1, t])
                hx_nw[n1, n2, t, sv["x"], sv["x"]] = K_gl * cmat[n1, n2]

    return -hx_nw


@numba.njit
def Duh(
    N,
    V_in,
    V_vars,
    T,
    sv,
    x,
    control,
    model_params,
):
    (
        alpha,
        tau,
        x_ext_baseline,
    ) = model_params

    duh = np.zeros((N, V_vars, V_in, T))
    for t in range(T):
        for n in range(N):
            exp = np.exp(-control[n, sv["x"], t] - x_ext_baseline)
            duh[n, sv["x"], sv["x"], t] = -1.0
            duh[n, sv["y"], sv["y"], t] = -1.0
    return duh


@numba.njit
def Dxdoth(N, V):
    dxdoth = np.zeros((N, V, V))
    for n in range(N):
        for v in range(V):
            dxdoth[n, v, v] = 1

    return dxdoth
