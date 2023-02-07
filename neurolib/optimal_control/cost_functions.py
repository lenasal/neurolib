import numpy as np
import numba

@numba.njit
def accuracy_cost(x, x_target, weights, precision_matrix, dt, interval=(0, None)):
    """Total cost related to the accuracy, weighted sum of contributions.

    :param x:           State of dynamical system.
    :type x:            np.ndarray
    :param x_target:    Target state.  
    :type x_target:     np.darray
    :param weights:     Dictionary of weights.
    :type weights:      dictionary
    :param dt:          Time step.
    :type dt:           float
    :return:            Accuracy cost of the control.
    :rtype:             float
    """

    cost_timeseries = np.zeros((x_target.shape))

    # timeseries of control vector is weighted sum of contributing cost functionals
    if weights["w_p"] != 0.0:
        cost_timeseries += weights["w_p"] * precision_cost(x_target, x)

    cost = 0.0
    # integrate over nodes, channels, and time
    if weights["w_p"] != 0.0:
        for n in range(u.shape[0]):
            for v in range(u.shape[1]):
                for t in range(interval[0], interval[1]):
                    cost += precision_matrix[n, v] * cost_timeseries[n, v, t] * dt

    return cost


@numba.njit
def derivative_accuracy_cost(x, x_target, weights, precision_matrix, interval=(0,None)):
    """Derivative of the 'accuracy_cost' wrt. to the control 'u'.

    :param u:           Control-dimensions x T array. Control signals.
    :type u:            np.ndarray
    :param weights:     Dictionary of weights.
    :type weights:      dictionary
    :param dt:          Time step.
    :type dt:           float
    :return:    Control-dimensions x T array of L2-cost gradients.
    :rtype:     np.ndarray
    """

    der = np.zeros((u.shape))

    if weights["w_p"] != 0.0:
        der += weights["w_p"] * derivative_precision_cost(u)

    return der


@numba.njit
def precision_cost(x_target, x_sim):
    """Summed squared difference between target and simulation within specified time interval weighted by w_p.
       Penalizes deviation from the target.

    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray
    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray
    :param w_p:         Weight that is multiplied with the precision cost.
    :type w_p:          float
    :param precision_matrix: N x V binary matrix that defines nodes and channels of precision measurement. Defaults to
                             None.
    :type precision_matrix:  np.ndarray
    :param dt:          Time step.
    :type dt:           float
    :param interval:    (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                        dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:     tuple
    :return:            Precision cost for time interval.
    :rtype:             float
    """

    cost = np.zeros((x_target.shape))

    for n in range(x_target.shape[0]):
        for v in range(x_target.shape[1]):
            for t in range(interval[0], interval[1]):
                cost[n,v,t] = 0.5 * (x_target[n, v, t] - x_sim[n, v, t]) ** 2

    return cost


@numba.njit
def derivative_precision_cost(x_target, x_sim, w_p, precision_matrix, interval):
    """Derivative of 'precision_cost' wrt. to 'x_sim'.

    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray
    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray
    :param w_p:         Weight that is multiplied with the precision cost.
    :type w_p:          float
    :param precision_matrix: N x V binary matrix that defines nodes and channels of precision measurement, defaults to
                                 None
    :type precision_matrix:  np.ndarray
    :param interval:    (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                        dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:     tuple
    :return:            Control-dimensions x T array of precision cost gradients.
    :rtype:             np.ndarray
    """

    derivative = np.zeros(x_target.shape)

    for n in range(x_target.shape[0]):
        for v in range(x_target.shape[1]):
            for t in range(interval[0], interval[1]):
                derivative[n, v, t] = np.multiply(-w_p * (x_target[n, v, t] - x_sim[n, v, t]), precision_matrix[n, v])

    return derivative


@numba.njit
def control_strength_cost(u, weights, dt):
    """Total cost related to the control strength, weighted sum of contributions.

    :param u:           Control-dimensions x T array. Control signals.
    :type u:            np.ndarray
    :param weights:     Dictionary of weights.
    :type weights:      dictionary
    :param dt:          Time step.
    :type dt:           float
    :return:            control strength cost of the control.
    :rtype:             float
    """

    cost_timeseries = np.zeros((u.shape))

    # timeseries of control vector is weighted sum of contributing cost functionals
    if weights["w_2"] != 0.0:
        cost_timeseries += weights["w_2"] * L2_cost(u)
    if weights["w_1"] != 0.0:
        cost_timeseries += weights["w_1"] * L1_cost(u)

    cost = 0.0
    # integrate over nodes, channels, and time
    if weights["w_2"] != 0.0 or weights["w_1"] != 0.0:
        for n in range(u.shape[0]):
            for v in range(u.shape[1]):
                for t in range(u.shape[2]):
                    cost += cost_timeseries[n, v, t] * dt

    if weights["w_1T"] != 0.0:
        cost += weights["w_1T"] * L1T_cost_integral(u, dt)
    if weights["w_1D"] != 0.0:
        cost += weights["w_1D"] * L1D_cost_integral(u, dt)

    return cost


@numba.njit
def derivative_control_strength_cost(u, weights, dt):
    """Derivative of the 'control_strength_cost' wrt. to the control 'u'.

    :param u:           Control-dimensions x T array. Control signals.
    :type u:            np.ndarray
    :param weights:     Dictionary of weights.
    :type weights:      dictionary
    :param dt:          Time step.
    :type dt:           float
    :return:    Control-dimensions x T array of L2-cost gradients.
    :rtype:     np.ndarray
    """

    der = np.zeros((u.shape))

    if weights["w_2"] != 0.0:
        der += weights["w_2"] * derivative_L2_cost(u)
    if weights["w_1"] != 0.0:
        der += weights["w_1"] * derivative_L1_cost(u)
    if weights["w_1T"] != 0.0:
        der += weights["w_1T"] * derivative_L1T_cost(u, dt)
    if weights["w_1D"] != 0.0:
        der += weights["w_1D"] * derivative_L1D_cost(u, dt)

    return der


@numba.njit
def L2_cost(u):
    """'Energy' or 'L2' cost. Penalizes for control strength.

    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray
    :return:    L2 cost of the control.
    :rtype:     float
    """

    return 0.5 * u**2.0


@numba.njit
def derivative_L2_cost(u):
    """Derivative of the 'L2_cost' wrt. to the control 'u'.

    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray
    :return:    Control-dimensions x T array of L2-cost gradients.
    :rtype:     np.ndarray
    """
    return u


@numba.njit
def L1_cost(u):
    """'Sparsity' or 'L1' cost. Penalizes for control strength.

    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray
    :return:    L1 cost of the control.
    :rtype:     float
    """
    return np.abs(u)


@numba.njit
def derivative_L1_cost(u):
    """Derivative of the 'L1_cost' wrt. to the control 'u'.

    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray
    :return :   Control-dimensions x T array of L1-cost gradients.
    :rtype:     np.ndarray
    """
    return np.sign(u)


@numba.njit
def L1T_cost_integral(u, dt):
    """'Temporal sparsity' or 'L1T' cost integrated over time. Penalizes for control strength.
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray
    :param dt:  Time step.
    :type dt:   float
    :return:    L1T cost of the control.
    :rtype:     float
    """

    return np.sqrt(np.sum(np.sum(np.sum(np.abs(u), axis=1), axis=0) ** 2) * dt)


@numba.njit
def derivative_L1T_cost(u, dt):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray
    :param dt:  Time step.
    :type dt:   float
    :return :   Control-dimensions x T array of L1T-cost gradients.
    :rtype:     np.ndarray
    """

    denominator = L1T_cost_integral(u, dt)
    if denominator == 0.0:
        return np.zeros((u.shape))

    return np.sum(np.sum(np.abs(u), axis=1), axis=0) * np.sign(u) / denominator


@numba.njit
def L1D_cost_integral(u, dt):
    """'Directional sparsity' or 'L1D' cost integrated over time. Penalizes for control strength.
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray
    :param dt:  Time step.
    :type dt:   float
    :return:    L1D cost of the control.
    :rtype:     float
    """

    return np.sum(np.sum(np.sqrt(np.sum(u**2, axis=2) * dt), axis=1), axis=0)


@numba.njit
def derivative_L1D_cost(u, dt):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray
    :param dt:  Time step.
    :type dt:   float
    :return :   Control-dimensions x T array of L1D-cost gradients.
    :rtype:     np.ndarray
    """

    denominator = np.sqrt(np.sum(u**2, axis=2) * dt)
    der = np.zeros((u.shape))
    for n in range(der.shape[0]):
        for v in range(der.shape[1]):
            if denominator[n, v] != 0.0:
                der[n, v, :] = u[n, v, :] / denominator[n, v]

    return der
