import numpy as np
import numba


@numba.njit
def accuracy_cost(
    x,
    target_timeseries,
    target_period,
    weights,
    cost_matrix,
    dt,
    interval=numba.typed.List([0, -1]),
):
    """Total cost related to the accuracy, weighted sum of contributions.

    :param x:                   State of dynamical system.
    :type x:                    np.ndarray
    :param target_timeseries:    Target state.
    :type target_timeseries:     np.darray
    :param target_period:       Target oscillation period
    :type target_period:        float
    :param weights:             Dictionary of weights.
    :type weights:              dictionary
    :param cost_matrix:         Matrix of channels to take into account
    :type cost_matrix:          ndarray
    :param dt:                  Time step.
    :type dt:                   float
    :param interval:            (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                                dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:             tuple, optional

    :return:                    Accuracy cost.
    :rtype:                     float
    """

    cost_timeseries = np.zeros((target_timeseries.shape))

    # timeseries of control vector is weighted sum of contributing cost functionals
    if weights["w_p"] != 0.0:
        cost_timeseries += weights["w_p"] * precision_cost(x, target_timeseries, cost_matrix, interval)

    cost = 0.0
    # integrate over nodes, channels, and time
    if weights["w_p"] != 0.0:
        for n in range(x.shape[0]):
            for v in range(x.shape[1]):
                for t in range(interval[0], interval[1]):
                    cost += cost_timeseries[n, v, t] * dt

    if weights["w_f_osc"] != 0.0:
        fc = fourier_cost_osc(x, dt, target_period, cost_matrix, interval)
        for n in range(x.shape[0]):
            for v in range(x.shape[1]):
                cost += weights["w_f_osc"] * fc[n, v]

    if weights["w_f_sync"] != 0.0:
        fc = weights["w_f_sync"] * fourier_cost_sync(x, dt, target_period, cost_matrix, interval)
        for v in range(x.shape[1]):
            cost += fc[v]

    if weights["w_var"] != 0.0:
        fvar = weights["w_var"] * var_cost(x, cost_matrix, interval, dt)
        for v in range(x.shape[1]):
            for t in range(interval[0], interval[1]):
                cost += fvar[v, t] * dt

    if weights["w_cc"] != 0.0:
        fcc = weights["w_cc"] * cc_cost(x, cost_matrix, interval, dt)
        for v in range(x.shape[1]):
            for t in range(interval[0], interval[1]):
                cost += fcc[v, t] * dt

    return cost


@numba.njit
def derivative_accuracy_cost(
    x,
    target_timeseries,
    target_period,
    weights,
    cost_matrix,
    dt,
    interval=numba.typed.List([0, -1]),
):
    """Derivative of the 'accuracy_cost' wrt. the state 'x'.

    :param x:               State of dynamical system.
    :type x:                np.ndarray
    :param target_timeseries:    Target state.
    :type target_timeseries:     np.darray
    :param target_period:       Target oscillation period
    :type target_period:        float
    :param weights:         Dictionary of weights.
    :type weights:          dictionary
    :param cost_matrix:     Matrix of channels to take into account
    :type cost_matrix:      ndarray
    :param interval:        (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                            dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:         tuple, optional

    :return:                Accuracy cost derivative.
    :rtype:                 ndarray
    """

    der = np.zeros((cost_matrix.shape[0], cost_matrix.shape[1], x.shape[2]))

    if weights["w_p"] != 0.0:
        der += weights["w_p"] * derivative_precision_cost(x, target_timeseries, cost_matrix, interval)
    if weights["w_f_osc"] != 0.0:
        der += weights["w_f_osc"] * derivative_fourier_cost_osc(x, dt, target_period, cost_matrix, interval)
    if weights["w_f_sync"] != 0.0:
        der += weights["w_f_sync"] * derivative_fourier_cost_sync(x, dt, target_period, cost_matrix, interval)
    if weights["w_var"] != 0.0:
        der += weights["w_var"] * derivative_var_cost(x, cost_matrix, interval, dt)
    if weights["w_cc"] != 0.0:
        der += weights["w_cc"] * derivative_cc_cost(x, dt, cost_matrix, interval)

    return der


@numba.njit
def precision_cost(
    x_sim,
    x_target,
    cost_matrix,
    interval=(0, None),
):
    """Summed squared difference between target and simulation within specified time interval weighted by w_p.
       Penalizes deviation from the target.

    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray
    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray
    :param cost_matrix: N x V binary matrix that defines nodes and channels of precision measurement. Defaults to
                             None.
    :type cost_matrix:  np.ndarray
    :param interval:    (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                        dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:     tuple

    :return:            Precision cost for time interval.
    :rtype:             float
    """

    cost = np.zeros((x_target.shape))

    # integrate over nodes, channels, and time
    for n in range(x_target.shape[0]):
        for v in range(x_target.shape[1]):
            for t in range(interval[0], interval[1]):
                cost[n, v, t] = 0.5 * cost_matrix[n, v] * (x_target[n, v, t] - x_sim[n, v, t]) ** 2

    return cost


@numba.njit
def derivative_precision_cost(
    x_sim,
    x_target,
    cost_matrix,
    interval,
):
    """Derivative of 'precision_cost' wrt. 'x_sim'.

    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray
    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray
    :param cost_matrix: N x V binary matrix that defines nodes and channels of precision measurement, defaults to
                        None
    :type cost_matrix:  np.ndarray
    :param interval:    (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                        dimension. Only 'int' positive index-notation allowed (i.e. no negative indices or 'None').
    :type interval:     tuple

    :return:            Control-dimensions x T array of precision cost gradients.
    :rtype:             np.ndarray
    """

    derivative = np.zeros(x_target.shape)

    # integrate over nodes, variables, and time
    for n in range(x_target.shape[0]):
        for v in range(x_target.shape[1]):
            for t in range(interval[0], interval[1]):
                derivative[n, v, t] = -cost_matrix[n, v] * (x_target[n, v, t] - x_sim[n, v, t])

    return derivative


@numba.njit
def compute_fourier_component(
    X,
    target_period,
    dt,
    T,
):
    res = 0.0
    omega = -2.0 * np.pi * dt / target_period
    for t in range(T):
        res += X[t] * np.exp(omega * complex(0, 1) * t) * dt
    return np.abs(res)


@numba.njit
def fourier_cost_osc(
    data,
    dt,
    target_period,
    cost_matrix,
    interval,
):
    cost = np.zeros((cost_matrix.shape[0], cost_matrix.shape[1]))
    T = len(data[0, 0, interval[0] : interval[1]])

    for n in range(data.shape[0]):
        for v in range(data.shape[1]):
            if cost_matrix[n, v] == 0.0:
                continue

            fc = compute_fourier_component(data[n, v, interval[0] : interval[1]], target_period, dt, T)
            cost[n, v] -= fc**2 / ((T * dt) ** 2 * data.shape[0])

    return cost


@numba.njit
def derivative_fourier_cost_osc(
    data,
    dt,
    target_period,
    cost_matrix,
    interval,
):
    derivative = np.zeros((data.shape))
    T = len(data[0, 0, interval[0] : interval[1]])
    omega = -2.0 * np.pi * dt / target_period

    for n in range(data.shape[0]):
        for v in range(data.shape[1]):
            if cost_matrix[n, v] == 0.0:
                continue

            for t in range(interval[0], interval[1]):
                for t1 in range(interval[0], interval[1]):
                    derivative[n, v, t] += data[n, v, t1] * np.cos(omega * (t1 - t)) * dt
                derivative[n, v, t] *= -2.0 / ((T * dt) ** 2 * data.shape[0])

    return derivative


@numba.njit
def fourier_cost_sync(
    data,
    dt,
    target_period,
    cost_matrix,
    interval,
):
    cost = np.zeros((cost_matrix.shape[1]))
    T = len(data[0, 0, interval[0] : interval[1]])

    for v in range(cost_matrix.shape[1]):
        data_nodesum = np.zeros((data.shape[2]))

        for n in range(data.shape[0]):
            if cost_matrix[n, v] != 0.0:
                data_nodesum += data[n, v, :]

        fc = compute_fourier_component(data_nodesum[interval[0] : interval[1]], target_period, dt, T)
        cost[v] -= fc**2 / ((T * dt) ** 2 * data.shape[0] ** 2)

    return cost


@numba.njit
def derivative_fourier_cost_sync(
    data,
    dt,
    target_period,
    cost_matrix,
    interval,
):
    derivative = np.zeros((data.shape))
    T = len(data[0, 0, interval[0] : interval[1]])

    omega = -2.0 * np.pi * dt / target_period

    for v in range(cost_matrix.shape[1]):
        data_nodesum = np.zeros((data.shape[2]))

        for n in range(data.shape[0]):
            if cost_matrix[n, v] == 0.0:
                continue
            data_nodesum += data[n, v, :]

        for n in range(data.shape[0]):
            for t in range(interval[0], interval[1]):
                for t1 in range(interval[0], interval[1]):
                    derivative[n, v, t] += data_nodesum[t1] * np.cos(omega * (t1 - t)) * dt

                derivative[n, v, t] *= -2.0 / ((T * dt) ** 2 * data.shape[0] ** 2)

    return derivative


@numba.njit
def getmean_vt(
    x,
):
    xmean = np.zeros((x.shape[1], x.shape[2]))
    for v in range(x.shape[1]):
        for t in range(x.shape[2]):
            for n in range(x.shape[0]):
                xmean[v, t] += x[n, v, t]
            xmean[v, t] /= x.shape[0]
    return xmean


@numba.njit
def var_cost(
    x_sim,
    cost_matrix,
    interval,
    dt,
):
    cost = np.zeros((x_sim.shape[1], x_sim.shape[2]))
    xmean = getmean_vt(x_sim)

    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):
            for n in range(x_sim.shape[0]):
                if cost_matrix[n, v] == 0.0:
                    continue
                cost[v, t] += (x_sim[n, v, t] - xmean[v, t]) ** 2

    cost /= x_sim.shape[0] * (interval[1] - interval[0]) * dt

    return cost


@numba.njit
def derivative_var_cost(
    x_sim,
    cost_matrix,
    interval,
    dt,
):
    derivative = np.zeros(x_sim.shape)
    xmean = getmean_vt(x_sim)

    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):

            for n in range(x_sim.shape[0]):
                if cost_matrix[n, v] != 0.0:
                    derivative[n, v, t] = +2.0 * (x_sim[n, v, t] - xmean[v, t])

    derivative /= x_sim.shape[0] * (interval[1] - interval[0]) * dt

    return derivative


@numba.njit
def getmean_nv(
    x,
    cost_matrix,
):
    xmean = np.zeros((x.shape[0], x.shape[1]))
    for n in range(x.shape[0]):
        for v in range(x.shape[1]):
            if cost_matrix[n, v] == 0.0:
                continue

            for t in range(x.shape[2]):
                xmean[n, v] += x[n, v, t]

    xmean /= x.shape[2]

    return xmean


@numba.njit
def getstd_nv(
    x,
    xmean,
    cost_matrix,
):
    xstd = np.zeros((x.shape[0], x.shape[1]))
    for n in range(x.shape[0]):
        for v in range(x.shape[1]):
            if cost_matrix[n, v] == 0.0:
                continue

            for t in range(x.shape[2]):
                xstd[n, v] += (x[n, v, t] - xmean[n, v]) ** 2

            if xstd[n, v] == 0.0:
                xstd[n, v] = 1e-6

    xstd /= x.shape[2]

    return np.sqrt(xstd)


@numba.njit
def cc_cost(
    x_sim,
    cost_matrix,
    interval,
    dt,
):
    cost = np.zeros((x_sim.shape[1], x_sim.shape[2]))

    xmean = getmean_nv(x_sim[:, :, interval[0] : interval[1]], cost_matrix)
    xstd = getstd_nv(x_sim[:, :, interval[0] : interval[1]], xmean, cost_matrix)

    for v in range(x_sim.shape[1]):
        for n in range(x_sim.shape[0]):
            if cost_matrix[n, v] == 0.0:
                continue
            for k in range(n + 1, x_sim.shape[0]):
                if cost_matrix[k, v] == 0.0:
                    continue
                for t in range(interval[0], interval[1]):
                    cost[v, t] -= (
                        (x_sim[n, v, t] - xmean[n, v]) * (x_sim[k, v, t] - xmean[k, v]) / (xstd[k, v] * xstd[n, v])
                    )

    cost /= x_sim.shape[0] ** 2 * (interval[1] - interval[0]) * dt

    return cost


@numba.njit
def get_mn_int(
    x,
    xmean,
    cost_matrix,
):
    mnint = np.zeros((x.shape[0], x.shape[0], x.shape[1]))

    for v in range(x.shape[1]):
        for n in range(x.shape[0]):
            if cost_matrix[n, v] == 0.0:
                continue
            for k in range(x.shape[0]):
                if cost_matrix[k, v] == 0.0:
                    continue

                for t in range(x.shape[2]):
                    mnint[n, k, v] += (x[n, v, t] - xmean[n, v]) * (x[k, v, t] - xmean[k, v])

    return mnint


@numba.njit
def derivative_cc_cost(
    x_sim,
    dt,
    cost_matrix,
    interval,
):
    derivative = np.zeros(x_sim.shape)

    xmean = getmean_nv(x_sim[:, :, interval[0] : interval[1]], cost_matrix)
    xstd = getstd_nv(x_sim[:, :, interval[0] : interval[1]], xmean, cost_matrix)

    T = interval[1] - interval[0]
    mnint = get_mn_int(x_sim, xmean, cost_matrix)

    for v in range(x_sim.shape[1]):
        for n in range(x_sim.shape[0]):
            if cost_matrix[n, v] == 0.0:
                continue
            for k in range(x_sim.shape[0]):
                if cost_matrix[k, v] == 0.0:
                    continue
                if k == n:
                    continue

                for t in range(interval[0], interval[1]):
                    sumand1 = -(x_sim[n, v, t] - xmean[n, v]) * mnint[n, k, v] / (T * xstd[n, v] ** 3 * xstd[k, v])
                    sumand2 = (x_sim[k, v, t] - xmean[k, v]) / (xstd[n, v] * xstd[k, v])
                    derivative[n, v, t] -= sumand1 + sumand2

    derivative /= (x_sim.shape[0]) ** 2 * (interval[1] - interval[0]) * dt
    return derivative


@numba.njit
def control_strength_cost(
    u,
    weights,
    dt,
):
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

    cost = 0.0
    # integrate over nodes, channels, and time
    if weights["w_2"] != 0.0:
        for n in range(u.shape[0]):
            for v in range(u.shape[1]):
                for t in range(u.shape[2]):
                    cost += cost_timeseries[n, v, t] * dt

    if weights["w_1D"] != 0.0:
        cost += weights["w_1D"] * L1D_cost_integral(u, dt)

    return cost


@numba.njit
def derivative_control_strength_cost(
    u,
    weights,
    dt,
):
    """Derivative of the 'control_strength_cost' wrt. the control 'u'.

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
    if weights["w_1D"] != 0.0:
        der += weights["w_1D"] * derivative_L1D_cost(u, dt)

    return der


@numba.njit
def L2_cost(
    u,
):
    """'Energy' or 'L2' cost. Penalizes for control strength.

    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :return:    L2 cost of the control.
    :rtype:     float
    """

    return 0.5 * u**2.0


@numba.njit
def derivative_L2_cost(
    u,
):
    """Derivative of the 'L2_cost' wrt. the control 'u'.

    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :return:    Control-dimensions x T array of L2-cost gradients.
    :rtype:     np.ndarray
    """
    return u


@numba.njit
def L1D_cost_integral(
    u,
    dt,
):
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
def derivative_L1D_cost(
    u,
    dt,
):
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
