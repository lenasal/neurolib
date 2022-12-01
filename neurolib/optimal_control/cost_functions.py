import numpy as np
import numba


@numba.njit
def precision_cost(x_target, x_sim, w_p, precision_matrix, dt, interval=(0, None)):
    """Summed squared difference between target and simulation within specified time interval weighted by w_p.

    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray

    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray

    :param w_p:         Weight that is multiplied with the precision cost.
    :type w_p:          float

    :param precision_matrix: N x V binary matrix that defines nodes and channels of precision measurement. Defaults to
                                 None.
    :type precision_matrix:  np.ndarray

    :param dt:  Time step.
    :type dt:   float

    :param interval:    (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                        dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:     tuple

    :return:            Precision cost for time interval.
    :rtype:             float

    """

    cost = 0.0
    for n in range(x_target.shape[0]):
        for v in range(x_target.shape[1]):
            for t in range(interval[0], interval[1]):
                cost += precision_matrix[n, v] * (x_target[n, v, t] - x_sim[n, v, t]) ** 2

    return w_p * 0.5 * cost * dt


@numba.njit
def derivative_precision_cost(x_target, x_sim, w_p, precision_matrix, interval):
    """Derivative of precision cost wrt. to x_sim.

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
def energy_cost(u, w_2, dt):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_2: Weight that is multiplied with the L2 ("energy") cost.
    :type w_2:  float

    :param dt:  Time step.
    :type dt:   float

    :return:    L2 cost of the control.
    :rtype:     float
    """
    return w_2 * 0.5 * np.sum(u**2.0) * dt


@numba.njit
def derivative_energy_cost(u, w_2):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_2: Weight that is multiplied with the L2 ("energy") cost.
    :type w_2:  float

    :return :   Control-dimensions x T array of L2-cost gradients.
    :rtype:     np.ndarray
    """
    return w_2 * u


@numba.njit
def L1_cost(u, w_1, dt):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_1: Weight that is multiplied with the L1 ("sparsity") cost.
    :type w_1:  float

    :param dt:  Time step.
    :type dt:   float

    :return:    L1 cost of the control.
    :rtype:     float
    """
    return w_1 * np.sum(np.abs(u)) * dt


@numba.njit
def derivative_L1_cost(u, w_1):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_1: Weight that is multiplied with the L1 ("sparsity") cost.
    :type w_1:  float

    :return :   Control-dimensions x T array of L1-cost gradients.
    :rtype:     np.ndarray
    """
    return w_1 * np.sign(u)


@numba.njit
def L1T_cost(u, w_1T, dt):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_1T: Weight that is multiplied with the L1T ("temporal sparsity") cost.
    :type w_1T:  float

    :param dt:  Time step.
    :type dt:   float

    :return:    L1T cost of the control.
    :rtype:     float
    """

    return w_1T * np.sqrt(np.sum(np.sum(np.sum(np.abs(u), axis=1), axis=0) ** 2) * dt)


@numba.njit
def derivative_L1T_cost(u, w_1T, dt):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_1T: Weight that is multiplied with the L1T ("temporal sparsity") cost.
    :type w_1T:  float

    :param dt:  Time step.
    :type dt:   float

    :return :   Control-dimensions x T array of L1T-cost gradients.
    :rtype:     np.ndarray
    """

    denominator = L1T_cost(u, w_1T, dt) / w_1T
    if denominator == 0.0:
        return np.zeros((u.shape))

    return w_1T * np.sum(np.sum(np.abs(u), axis=1), axis=0) * np.sign(u) / denominator


@numba.njit
def L1D_cost(u, w_1D, dt):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_1D: Weight that is multiplied with the L1D ("directional sparsity") cost.
    :type w_1D:  float

    :param dt:  Time step.
    :type dt:   float

    :return:    L1D cost of the control.
    :rtype:     float
    """

    return w_1D * np.sum(np.sum(np.sqrt(np.sum(u**2, axis=2) * dt), axis=1), axis=0)


@numba.njit
def derivative_L1D_cost(u, w_1D, dt):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_1D: Weight that is multiplied with the L1D ("directional sparsity") cost.
    :type w_1D:  float

    :param dt:  Time step.
    :type dt:   float

    :return :   Control-dimensions x T array of L1D-cost gradients.
    :rtype:     np.ndarray
    """

    denominator = np.sqrt(np.sum(u**2, axis=2) * dt) / w_1D
    der = np.zeros((u.shape))
    for n in range(der.shape[0]):
        for v in range(der.shape[1]):
            if denominator[n, v] != 0.0:
                der[n, v, :] = w_1D * u[n, v, :] / denominator[n, v]

    return der
