import numpy as np
import numba

global PHASE_VARIATION_LIMIT_DER
PHASE_VARIATION_LIMIT_DER = 2.0 * 1e-1


@numba.njit
def accuracy_cost(x, x_analytic, target_timeseries, target_period, weights, cost_matrix, dt, interval=(0, None)):
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
    if weights["w_phase"] != 0.0:
        cost_timeseries += weights["w_phase"] * phase_cost(x, x_analytic, target_period, dt, cost_matrix, interval)
    if weights["w_ac"] != 0.0:
        cost_timeseries += weights["w_ac"] * ac_cost(x, target_period, dt, cost_matrix, interval)

    cost = 0.0
    # integrate over nodes, channels, and time
    if weights["w_p"] != 0.0 or weights["w_phase"] != 0.0 or weights["w_ac"] != 0.0:
        for n in range(x.shape[0]):
            for v in range(x.shape[1]):
                for t in range(interval[0], interval[1]):
                    cost += cost_timeseries[n, v, t] * dt

    if weights["w_f"] != 0.0:
        fc = fourier_cost(x, dt, target_period, cost_matrix, interval)
        for n in range(x.shape[0]):
            for v in range(x.shape[1]):
                cost += fc[n, v]

    return cost


@numba.njit
def derivative_accuracy_cost(
    x, x_analytic, target_timeseries, target_period, weights, cost_matrix, dt, interval=(0, None)
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
    if weights["w_phase"] != 0.0:
        der += weights["w_phase"] * derivative_phase_cost(x, x_analytic, target_period, dt, cost_matrix, interval)
    if weights["w_ac"] != 0.0:
        der += weights["w_ac"] * derivative_ac_cost(x, target_period, dt, cost_matrix, interval)

    return der


@numba.njit
def precision_cost(x_sim, x_target, cost_matrix, interval=(0, None)):
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
def derivative_precision_cost(x_target, x_sim, cost_matrix, interval):
    """Derivative of 'precision_cost' wrt. to 'x_sim'.

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
def compute_fft(x):
    l = len(x) // 2
    with numba.objmode(
        y="complex128[:]",
        freqs="float64[:]",
    ):

        y = np.fft.fft(x)[:l]
        freqs = np.fft.fftfreq(len(x))[:l]

    y = 2.0 / len(x) * np.abs(y)
    return y, freqs


@numba.njit
def fourier_cost(data, dt, target_period, cost_matrix, interval, f_tol=0.1, f_lim_percent=1e-3):

    cost = np.zeros((data.shape[0], data.shape[1]))

    sampling_rate = 1.0 / dt
    target_f = 1.0 / target_period

    for n in range(data.shape[0]):
        for v in range(data.shape[1]):
            if cost_matrix[n, v] != 0.0:
                fft_data_scaled, freqs = compute_fft(data[n, v, interval[0] : interval[1]])

                f_lim = f_lim_percent * np.amax(fft_data_scaled[1:])
                max_freq = freqs[-1] * sampling_rate
                max_harmonic = np.int(np.floor(max_freq / target_f))

                max_harmonic = 1

                for i in range(len(fft_data_scaled)):
                    for k in range(1, max_harmonic + 1, 1):
                        if fft_data_scaled[i] > f_lim:
                            if freqs[i] * sampling_rate > k * target_f * (1.0 - f_tol) and freqs[
                                i
                            ] * sampling_rate < k * target_f * (1.0 + f_tol):
                                cost[n, v] -= fft_data_scaled[i]

    return cost


@numba.njit
def derivative_fourier_cost(data, dt, target_period, cost_matrix, interval, dx=0.1, f_tol=0.1, f_lim_percent=1e-3):
    data_dx = data.copy()
    cost0 = fourier_cost(data, dt, target_period, cost_matrix, interval, f_tol=f_tol, f_lim_percent=f_lim_percent)
    derivative = np.zeros((data_dx.shape))

    for n in range(data.shape[0]):
        for v in range(data.shape[1]):
            if cost_matrix[n, v] != 0.0:

                for i in range(1, data.shape[2]):
                    data_dx[n, v, i] += dx
                    cost1 = fourier_cost(
                        data_dx, dt, target_period, cost_matrix, interval, f_tol=f_tol, f_lim_percent=f_lim_percent
                    )
                    data_dx[n, v, i] -= dx
                    derivative[n, v, i] = (cost1[n, v] - cost0[n, v]) / dx

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
    L = int(np.floor((interval[1] - interval[0]) / ind_tau))
    cost = np.zeros((x_sim.shape))

    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):

            if cost_matrix[n, v] != 0.0:

                y = x_analytic[n, v, :].imag
                x = x_sim[n, v, :]

                p = np.arctan2(y, x)

                for t in range(interval[1] - (L - 1) * ind_tau, interval[1] - ind_tau):

                    p_array = np.zeros((3))
                    p_array[0] = np.arctan2(y[t - ind_tau], x[t - ind_tau])
                    p_array[2] = np.arctan2(y[t + ind_tau], x[t + ind_tau])
                    p_array[1] = p[t]

                    cost[n, v, t] = -cost_matrix[n, v] * osc_phase_sum(p_array)

                check_phase_variation(cost[n, v, :], p, ind_tau, dt)

    return cost


@numba.njit
def check_phase_variation(cost, phase, ind_tau, dt):
    phase_der = np.zeros((len(phase)))

    for t in range(1, len(phase_der)):
        phase_der[t] = (phase[t] - phase[t - 1]) / dt

    phase_der[-1] = phase_der[-2]

    for t in range(len(phase_der) - ind_tau):
        if np.amax(np.abs(phase_der[t : t + ind_tau])) < PHASE_VARIATION_LIMIT_DER:
            cost[t] = 0.0

    for t in range(len(phase_der) - ind_tau, len(phase_der)):
        if np.amax(np.abs(phase_der[t - ind_tau : t])) < PHASE_VARIATION_LIMIT_DER:
            cost[t] = 0.0


@numba.njit
def derivative_phase_cost(x_sim, x_analytic, target_period, dt, cost_matrix, interval):

    ind_tau = int(target_period / dt)
    L = int(np.floor((interval[1] - interval[0]) / ind_tau))
    derivative = np.zeros(x_sim.shape)

    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):

            if cost_matrix[n, v] != 0.0:

                y = x_analytic[n, v, :].imag
                x = x_sim[n, v, :]

                p = np.arctan2(y, x)

                for t in range(interval[1] - (L - 1) * ind_tau, interval[1] - ind_tau):

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
def absdiff(x):
    diff = np.amax(x) - np.amin(x)
    if diff == 0.0:
        diff = 1.0

    return diff


@numba.njit
def ac_cost(x_sim, target_period, dt, cost_matrix, interval):

    cost = np.zeros((x_sim.shape))
    ind_tau = int(target_period / dt)
    L = int(np.floor((interval[1] - interval[0]) / ind_tau))

    if L < 2:
        print("L too small")

    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):

            if cost_matrix[n, v] != 0.0:

                x_mean = np.mean(x_sim[n, v, interval[1] - (L - 1) * ind_tau : interval[1] - ind_tau])
                x_mean_m = np.mean(x_sim[n, v, interval[1] - L * ind_tau : interval[1] - ind_tau])
                x_mean_p = np.mean(x_sim[n, v, interval[1] - (L - 2) * ind_tau : interval[1]])

                for t in range(interval[1] - (L - 1) * ind_tau, interval[1] - ind_tau):
                    cost[n, v, t] -= cost_matrix[n, v,] * (
                        (x_sim[n, v, t] - x_mean) * (x_sim[n, v, t - ind_tau] - x_mean_m)
                        + (x_sim[n, v, t] - x_mean) * (x_sim[n, v, t + ind_tau] - x_mean_p)
                    )

    return cost


@numba.njit
def derivative_ac_cost(x_sim, target_period, dt, cost_matrix, interval):

    ind_tau = int(target_period / dt)
    derivative = np.zeros(x_sim.shape)
    L = int(np.floor((interval[1] - interval[0]) / ind_tau))

    if L < 2:
        print("L too small")

    for n in range(x_sim.shape[0]):
        for v in range(x_sim.shape[1]):

            if cost_matrix[n, v] != 0.0:

                x_mean_m = np.mean(x_sim[n, v, interval[1] - L * ind_tau : interval[1] - ind_tau])
                x_mean_p = np.mean(x_sim[n, v, interval[1] - (L - 2) * ind_tau : interval[1]])

                for t in range(interval[1] - (L - 1) * ind_tau, interval[1] - ind_tau):
                    derivative[n, v, t] -= cost_matrix[n, v] * (
                        (x_sim[n, v, t - ind_tau] - x_mean_m) + (x_sim[n, v, t + ind_tau] - x_mean_p)
                    )

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
