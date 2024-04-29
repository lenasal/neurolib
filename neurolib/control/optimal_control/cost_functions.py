import numpy as np
import numba

# global PHASE_VARIATION_LIMIT_DER, X_VARIATION_LIMIT
# PHASE_VARIATION_LIMIT_DER = 1e-4
# X_VARIATION_LIMIT = 1e-2

# global FOURIER_DX, FOURIER_TOL, FOURIER_LIM
# FOURIER_DX = 1e-4


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

    if weights["w_f"] != 0.0:
        fc = fourier_cost(x, dt, target_period, cost_matrix, interval)
        for n in range(x.shape[0]):
            for v in range(x.shape[1]):
                cost += weights["w_f"] * fc[n, v]

    if weights["w_f_sync"] != 0.0:
        fc = weights["w_f_sync"] * fourier_cost_sync(x, dt, target_period, cost_matrix, interval)
        for v in range(x.shape[1]):
            cost += fc[v]

    if weights["w_var"] != 0.0:
        fvar = weights["w_var"] * var_cost(x, cost_matrix, interval)
        for v in range(x.shape[1]):
            for t in range(interval[0], interval[1]):
                cost += fvar[v, t] * dt

    if weights["w_cc"] != 0.0:
        fcc = weights["w_cc"] * cc_cost(x, cost_matrix, interval)
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

    der = np.zeros((cost_matrix.shape[0], cost_matrix.shape[1], x.shape[2]))

    if weights["w_p"] != 0.0:
        der += weights["w_p"] * derivative_precision_cost(x, target_timeseries, cost_matrix, interval)
    if weights["w_f"] != 0.0:
        der += weights["w_f"] * derivative_fourier_cost(x, dt, target_period, cost_matrix, interval)
    if weights["w_f_sync"] != 0.0:
        der += weights["w_f_sync"] * derivative_fourier_cost_sync(x, dt, target_period, cost_matrix, interval)
    if weights["w_var"] != 0.0:
        der += weights["w_var"] * derivative_var_cost(x, cost_matrix, interval)
    if weights["w_cc"] != 0.0:
        der += weights["w_cc"] * derivative_cc_cost(x, dt, cost_matrix, interval)

    return der


@numba.njit
def precision_cost(x_sim, x_target, cost_matrix, interval):
    """Summed squared difference between target and simulation within specified time interval weighted by w_p.
       Penalizes deviation from the target.

    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray
    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray
    :param w_p:         Weight that is multiplied with the precision cost.
    :type w_p:          float
    :param cost_matrix: N x V binary matrix that defines nodes and channels of precision measurement. Defaults to
                             None.
    :type cost_matrix:  np.ndarray
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
                cost[n, v, t] = 0.5 * cost_matrix[n, v] * (x_target[n, v, t] - x_sim[n, v, t]) ** 2

    return cost


@numba.njit
def derivative_precision_cost(x_sim, x_target, cost_matrix, interval):
    """Derivative of 'precision_cost' wrt. 'x_sim'.

    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray
    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray
    :param w_p:         Weight that is multiplied with the precision cost.
    :type w_p:          float
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

    for n in range(x_target.shape[0]):
        for v in range(x_target.shape[1]):
            for t in range(interval[0], interval[1]):
                derivative[n, v, t] = -cost_matrix[n, v] * (x_target[n, v, t] - x_sim[n, v, t])

    return derivative


@numba.njit
def compute_fourier_component(X, target_period, dt, T):
    res = 0.0
    k = numba.uint16(np.around(T * dt / target_period, 0))
    exponent = -2.0 * complex(0, 1) * np.pi * k / T
    for t in range(T):
        res += X[t] * np.exp(exponent * t)
    return np.abs(res)


@numba.njit
def fourier_cost(data, dt, target_period, cost_matrix, interval):
    cost = np.zeros((cost_matrix.shape[0], cost_matrix.shape[1]))
    T = len(data[0, 0, interval[0] : interval[1]])

    for n in range(data.shape[0]):
        for v in range(data.shape[1]):
            if cost_matrix[n, v] == 0.0:
                continue

            fc = compute_fourier_component(data[n, v, interval[0] : interval[1]], target_period, dt, T)

            # cost[n, v] -= 2.0 * fc / T
            cost[n, v] -= fc**2 / T**2

    return cost


@numba.njit
def derivative_fourier_cost(data, dt, target_period, cost_matrix, interval):
    derivative = np.zeros((data.shape))
    T = len(data[0, 0, interval[0] : interval[1]])

    k = numba.uint16(np.around(T * dt / target_period, 0))
    argument = -2.0 * np.pi * k / T

    for n in range(data.shape[0]):
        for v in range(data.shape[1]):
            if cost_matrix[n, v] == 0.0:
                continue

            for t in range(interval[0], interval[1]):
                for t1 in range(interval[0], interval[1]):
                    derivative[n, v, t] += data[n, v, t1] * np.cos(argument * (t1 - t))

                if np.abs(derivative[n, v, t]) < 1e-14:
                    derivative[n, v, t] = 0.0

                # derivative[n, v, t] *= 4.0 / (fcost[n, v] * T**2)
                derivative[n, v, t] *= -2.0 / (T**2 * dt)

    return derivative


@numba.njit
def fourier_cost_sync(data, dt, target_period, cost_matrix, interval):
    cost = np.zeros((cost_matrix.shape[1]))
    T = len(data[0, 0, interval[0] : interval[1]])

    for v in range(cost_matrix.shape[1]):
        data_nodesum = np.zeros((data.shape[2]))

        for n in range(data.shape[0]):
            if cost_matrix[n, v] != 0.0:
                data_nodesum += data[n, v, :]

        fc = compute_fourier_component(data_nodesum[interval[0] : interval[1]], target_period, dt, T)
        cost[v] -= fc**2 / T**2

    return cost


@numba.njit
def derivative_fourier_cost_sync(data, dt, target_period, cost_matrix, interval):
    derivative = np.zeros((data.shape))
    T = len(data[0, 0, interval[0] : interval[1]])

    k = numba.uint16(np.around(T * dt / target_period, 0))
    argument = -2.0 * np.pi * k / T

    for v in range(cost_matrix.shape[1]):
        data_nodesum = np.zeros((data.shape[2]))

        for n in range(data.shape[0]):
            if cost_matrix[n, v] == 0.0:
                continue
            data_nodesum += data[n, v, :]

        for n in range(data.shape[0]):
            for t in range(interval[0], interval[1]):
                for t1 in range(interval[0], interval[1]):
                    derivative[n, v, t] += data_nodesum[t1] * np.cos(argument * (t1 - t))

                derivative[n, v, t] *= -2.0 / (T**2 * dt)

    return derivative


@numba.njit
def var_cost(x_sim, cost_matrix, interval):
    cost = np.zeros((x_sim.shape[1], x_sim.shape[2]))
    xmean = np.zeros((x_sim.shape[1], x_sim.shape[2]))
    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):
            for n in range(x_sim.shape[0]):
                xmean[v, t] += x_sim[n, v, t]
            xmean[v, t] /= x_sim.shape[0]

    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):
            for n in range(x_sim.shape[0]):
                if cost_matrix[n, v] == 0.0:
                    continue
                cost[v, t] += (x_sim[n, v, t] - xmean[v, t]) ** 2

    cost /= x_sim.shape[0]

    return cost


@numba.njit
def derivative_var_cost(x_sim, cost_matrix, interval):
    derivative = np.zeros(x_sim.shape)
    xmean = np.zeros((x_sim.shape[1], x_sim.shape[2]))
    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):
            for n in range(x_sim.shape[0]):
                xmean[v, t] += x_sim[n, v, t]
            xmean[v, t] /= x_sim.shape[0]

    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):
            # xsum = 0.0
            # for n in range(x_sim.shape[0]):
            # if cost_matrix[n, v] != 0.0:
            #    xsum += x_sim[n, v, t] - xmean[v, t]

            for n in range(x_sim.shape[0]):
                if cost_matrix[n, v] != 0.0:
                    derivative[n, v, t] = (
                        # 2.0 * x_sim[n, v, t] * xsum
                        +2.0
                        * (x_sim[n, v, t] - xmean[v, t])
                        # * (1.0 - 1.0 / x_sim.shape[0])
                    )

    derivative /= x_sim.shape[0]

    return derivative


@numba.njit
def getmean_nv(x, cost_matrix):
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
def getstd_nv(x, xmean, cost_matrix):
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
def cc_cost(x_sim, cost_matrix, interval):
    cost = np.zeros((x_sim.shape[1], x_sim.shape[2]))

    xmean = getmean_nv(x_sim[:, :, interval[0] : interval[1]], cost_matrix)
    xstd = getstd_nv(x_sim[:, :, interval[0] : interval[1]], xmean, cost_matrix)

    # print(xmean)

    for v in range(x_sim.shape[1]):
        for n in range(x_sim.shape[0]):
            if cost_matrix[n, v] == 0.0:
                continue
            for t in range(interval[0], interval[1]):
                for k in range(n + 1, x_sim.shape[0]):
                    if cost_matrix[k, v] == 0.0:
                        continue
                    cost[v, t] -= (
                        (x_sim[n, v, t] - xmean[n, v]) * (x_sim[k, v, t] - xmean[k, v]) / (xstd[k, v] * xstd[n, v])
                    )

    cost /= x_sim.shape[0] ** 2

    return cost


@numba.njit
def get_mn_int(x, xmean, cost_matrix):
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
def derivative_cc_cost(x_sim, dt, cost_matrix, interval):
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

    derivative /= (x_sim.shape[0]) ** 2
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
    """Derivative of the 'L2_cost' wrt. the control 'u'.

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


"""
###########################################################7
# the following cost functionals were not effective and can be removed


@numba.njit
def cc0_cost(x_sim, cost_matrix, interval):
    cost = np.zeros((x_sim.shape[1], x_sim.shape[2]))
    xmean = np.zeros((x_sim.shape[0], x_sim.shape[1]))
    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):
            for t in range(interval[0], interval[1]):
                xmean[n, v] += x_sim[n, v, t]
            xmean[n, v] /= interval[1] - interval[0]


    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):
            for n in range(x_sim.shape[0]):
                if cost_matrix[n, v] == 0.0:
                    continue
                for k in range(x_sim.shape[0]):
                    if cost_matrix[k, v] == 0.0:
                        continue
                    cost[v, t] -= (x_sim[n, v, t] - xmean[n, v]) * (x_sim[k, v, t] - xmean[k, v])

            cost[v, t] /= x_sim.shape[0] ** 2

    return cost


@numba.njit
def derivative_cc0_cost____(x_sim, cost_matrix, interval):
    N = x_sim.shape[0]

    derivative = np.zeros(x_sim.shape)
    xmean = np.zeros((x_sim.shape[0], x_sim.shape[1]))
    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):
            for t in range(interval[0], interval[1]):
                xmean[n, v] += x_sim[n, v, t]
            xmean[n, v] /= interval[1] - interval[0]

    sum1 = np.zeros((x_sim.shape[1], x_sim.shape[2]))
    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):
            for n in range(N):
                sum1[v, t] += x_sim[n, v, t] - xmean[n, v]

    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):
            for n in range(N):
                if cost_matrix[n, v] != 0.0:
                    derivative[n, v, t] = (
                        -(sum1[v, t] + (x_sim[n, v, t] - xmean[n, v]) - (2.0 * N - 1.0) * sum1[v, t]) / N**2
                    )

    return derivative


@numba.njit
def derivative_cc0_cost(x_sim, dt, cost_matrix, interval):
    x_sim_dx = x_sim.copy()
    cost0 = cc_cost(x_sim, cost_matrix, interval)
    derivative = np.zeros(x_sim.shape)
    dx = 1e-2

    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):
            if cost_matrix[n, v] != 0.0:
                for t in range(1, x_sim.shape[2]):
                    x_sim_dx[n, v, t] += dx
                    cost1 = cc_cost(x_sim_dx, cost_matrix, interval)
                    x_sim_dx[n, v, t] -= dx
                    derivative[n, v, t] = (np.sum(cost1, axis=1)[v] - np.sum(cost0, axis=1)[v]) / (dx * dt)

    return derivative


@numba.njit
def osc_phase_sum(phase_array):
    co = 0.0
    si = 0.0
    for p in phase_array:
        co += np.cos(p)
        si += np.sin(p)

    return co**2 + si**2


@numba.njit
def phase_cost(x_sim, x_analytic, target_period, dt, cost_matrix, interval):
    ind_tau = int(target_period / dt)
    cost = np.zeros((x_sim.shape))

    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):
            if cost_matrix[n, v] != 0.0:
                y = x_analytic[n, v, :].imag
                x = x_sim[n, v, :]

                p = np.arctan2(y, x)

                for t in range(interval[0] + ind_tau, interval[1] - ind_tau):
                    p_array = np.zeros((3))
                    p_array[0] = np.arctan2(y[t - ind_tau], x[t - ind_tau])
                    p_array[2] = np.arctan2(y[t + ind_tau], x[t + ind_tau])
                    p_array[1] = p[t]

                    cost[n, v, t] = -cost_matrix[n, v] * osc_phase_sum(p_array)

                check_variation(cost[n, v, :], x, ind_tau, dt)

    return cost


@numba.njit
def check_variation(cost, phase, ind_tau, dt):
    phase_der = np.zeros((len(phase)))

    for t in range(1, len(phase_der)):
        phase_der[t] = (phase[t] - phase[t - 1]) / dt

    phase_der[-1] = phase_der[-2]

    for t in range(len(phase_der) - ind_tau):
        if (np.amax(phase[t : t + ind_tau]) - np.amin(phase[t : t + ind_tau])) < PHASE_VARIATION_LIMIT_DER:
            cost[t] = 0.0

    for t in range(len(phase_der) - ind_tau, len(phase_der)):
        if (np.amax(phase[t - ind_tau : t]) - np.amin(phase[t - ind_tau : t])) < PHASE_VARIATION_LIMIT_DER:
            cost[t] = 0.0

    return


@numba.njit
def derivative_phase_cost(x_sim, x_analytic, target_period, dt, cost_matrix, interval):
    ind_tau = int(target_period / dt)
    derivative = np.zeros(x_sim.shape)

    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):
            if cost_matrix[n, v] != 0.0:
                y = x_analytic[n, v, :].imag
                x = x_sim[n, v, :]

                p = np.arctan2(y, x)

                for t in range(interval[0] + ind_tau, interval[1] - ind_tau):
                    p_m = np.arctan2(y[t - ind_tau], x[t - ind_tau])
                    p_p = np.arctan2(y[t + ind_tau], x[t + ind_tau])

                    cos_sum = np.cos(p_m) + np.cos(p[t]) + np.cos(p_p)
                    sin_sum = np.sin(p_m) + np.sin(p[t]) + np.sin(p_p)

                    prefactor = 2.0 * (cos_sum * (-np.sin(p[t])) + sin_sum * np.cos(p[t]))

                    dfx = 0.0
                    denominator = x[t] ** 2 + y[t] ** 2
                    if denominator != 0.0:
                        dfx = -y[t] / denominator

                    derivative[n, v, t] = -cost_matrix[n, v] * prefactor * dfx

    return derivative


@numba.njit
def KO_cost(x_sim, x_analytic, target_period, dt, cost_matrix, interval):
    ind_tau = int(target_period / dt)
    cost = np.zeros((x_sim.shape[1], x_sim.shape[2]))

    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):
            p_array = []

            for n in range(x_sim.shape[0]):
                if cost_matrix[n, v] == 0.0:
                    continue

                testx = x_sim[n, v, max(0, int(t - ind_tau / 2)) : min(x_sim.shape[2] - 1, int(t + ind_tau / 2))]
                if np.abs(np.amax(testx) - np.amin(testx)) < X_VARIATION_LIMIT:
                    continue

                y = x_analytic[n, v, t].imag
                x = x_sim[n, v, t]
                p_array.append(np.arctan2(y, x))

            if len(p_array) != 0:
                cost[v, t] = -osc_phase_sum(p_array)

    return cost


@numba.njit
def derivative_KO_cost(x_sim, x_analytic, cost_matrix, interval):
    derivative = np.zeros(x_sim.shape)

    for v in range(x_sim.shape[1]):
        for t in range(interval[0], interval[1]):
            cos_sum = 0.0
            sin_sum = 0.0
            for n in range(x_sim.shape[0]):
                if cost_matrix[n, v] != 0.0:
                    y = x_analytic[n, v, t].imag
                    x = x_sim[n, v, t]
                    p = np.arctan2(y, x)
                    cos_sum += np.cos(p)
                    sin_sum += np.sin(p)

            for n in range(x_sim.shape[0]):
                if cost_matrix[n, v] != 0.0:
                    y = x_analytic[n, v, t].imag
                    x = x_sim[n, v, t]
                    p = np.arctan2(y, x)

                    derivative[n, v, t] = -2.0 * (-np.sin(p) * cos_sum + np.cos(p) * sin_sum) * (-y) / (x**2 + y**2)

    return derivative


@numba.njit
def ac_cost(x_sim, target_period, dt, cost_matrix, interval):
    cost = np.zeros(x_sim.shape)
    ind_tau = int(target_period / dt)

    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):
            if cost_matrix[n, v] != 0.0:
                x_mean = np.mean(x_sim[n, v, interval[0] + ind_tau : interval[1] - ind_tau])
                x_mean_m = np.mean(x_sim[n, v, interval[0] : interval[1] - 2 * ind_tau])
                x_mean_p = np.mean(x_sim[n, v, interval[0] + 2 * ind_tau : interval[1]])

                for t in range(interval[0] + ind_tau, interval[1] - ind_tau):
                    cost[n, v, t] = -cost_matrix[
                        n,
                        v,
                    ] * (
                        (x_sim[n, v, t] - x_mean) * (x_sim[n, v, t - ind_tau] - x_mean_m)
                        + (x_sim[n, v, t] - x_mean) * (x_sim[n, v, t + ind_tau] - x_mean_p)
                    )

    return cost


@numba.njit
def derivative_ac_cost(x_sim, target_period, dt, cost_matrix, interval):
    ind_tau = int(target_period / dt)
    derivative = np.zeros(x_sim.shape)

    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):
            if cost_matrix[n, v] != 0.0:
                x_mean_m = np.mean(x_sim[n, v, interval[0] : interval[1] - 2 * ind_tau])
                x_mean_p = np.mean(x_sim[n, v, interval[0] + 2 * ind_tau : interval[1]])

                for t in range(interval[0] + ind_tau, interval[1] - ind_tau):
                    derivative[n, v, t] = -cost_matrix[n, v] * (
                        (x_sim[n, v, t - ind_tau] - x_mean_m) + (x_sim[n, v, t + ind_tau] - x_mean_p)
                    )

    return derivative


@numba.njit
def var_osc_cost(x_sim, cost_matrix, interval):
    cost = np.zeros(x_sim.shape)
    xmean = np.zeros((x_sim.shape[0], x_sim.shape[1]))
    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):
            for t in range(interval[0], interval[1]):
                xmean[n, v] += x_sim[n, v, t]
            xmean[n, v] /= interval[1] - interval[0]

            for t in range(interval[0], interval[1]):
                if cost_matrix[n, v] != 0.0:
                    cost[n, v, t] -= (x_sim[n, v, t] - xmean[n, v]) ** 2

    return cost


@numba.njit
def derivative_var_osc_cost(x_sim, dt, cost_matrix, interval):
    derivative = np.zeros(x_sim.shape)
    xmean = np.zeros((x_sim.shape[0], x_sim.shape[1]))
    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):
            for t in range(interval[0], interval[1]):
                xmean[n, v] += x_sim[n, v, t]
            xmean[n, v] /= interval[1] - interval[0]

            for t in range(interval[0], interval[1]):
                if cost_matrix[n, v] != 0.0:
                    derivative[n, v, t] = (
                        -2.0 * (x_sim[n, v, t] - xmean[n, v]) * (1.0 - 1.0 / (dt * (interval[1] - interval[0])))
                    )

    return derivative


######################
# redundant numerical computation

@numba.njit
def derivative_cc_cost_num(x_sim, dt, cost_matrix, interval):
    x_sim_dx = x_sim.copy()
    cost0 = cc_cost(x_sim, cost_matrix, interval)
    c0sum = np.sum(cost0, axis=1)
    derivative = np.zeros(x_sim.shape)
    dx = 1e-2

    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):
            if cost_matrix[n, v] != 0.0:
                for t in range(1, x_sim.shape[2]):
                    x_sim_dx[n, v, t] += dx
                    cost1 = cc_cost(x_sim_dx, cost_matrix, interval)
                    x_sim_dx[n, v, t] -= dx
                    # derivative[n, v, t] = (np.sum(cost1, axis=1)[v] - c0sum[v]) / (dx)
                    derivative[n, v, t] = (cost1[v, t] - cost0[v, t]) / (dx)

    return derivative


    
@numba.njit
def derivative_fourier_cost_sync_num(data, dt, target_period, cost_matrix, interval, dx=FOURIER_DX):
    data_dx = data.copy()
    cost0 = fourier_cost_sync(data, dt, target_period, cost_matrix, interval)
    derivative = np.zeros((data_dx.shape))

    for n in range(data.shape[0]):
        for v in range(data.shape[1]):
            if cost_matrix[n, v] != 0.0:
                for i in range(1, data.shape[2]):
                    data_dx[n, v, i] += dx
                    cost1 = fourier_cost_sync(data_dx, dt, target_period, cost_matrix, interval)
                    data_dx[n, v, i] -= dx
                    derivative[n, v, i] = (cost1[v] - cost0[v]) / dx

    return derivative


    @numba.njit
def derivative_fourier_cost_num(data, dt, target_period, cost_matrix, interval, dx=FOURIER_DX):
    data_dx = data.copy()
    cost0 = fourier_cost(data, dt, target_period, cost_matrix, interval)
    derivative = np.zeros((data_dx.shape))

    for n in range(data.shape[0]):
        for v in range(data.shape[1]):
            if cost_matrix[n, v] == 0.0:
                continue

            for i in range(1, data.shape[2]):
                data_dx[n, v, i] += dx
                cost1 = fourier_cost(data_dx, dt, target_period, cost_matrix, interval)
                data_dx[n, v, i] -= dx
                derivative[n, v, i] = (cost1[n, v] - cost0[n, v]) / dx

    return derivative



@numba.njit
def derivative_fourier_cost_num(data, dt, target_period, cost_matrix, interval, dx=1e-2):
    data_dx = data.copy()
    cost0 = fourier_cost(data, dt, target_period, cost_matrix, interval)
    derivative = np.zeros((data_dx.shape))

    for n in range(data.shape[0]):
        for v in range(data.shape[1]):
            if cost_matrix[n, v] == 0.0:
                continue

            for i in range(1, data.shape[2]):
                data_dx[n, v, i] += dx
                cost1 = fourier_cost(data_dx, dt, target_period, cost_matrix, interval)
                data_dx[n, v, i] -= dx
                derivative[n, v, i] = (cost1[n, v] - cost0[n, v]) / dx

    return derivative

"""
