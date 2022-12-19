import matplotlib.pyplot as plt
import numpy as np
import os, scipy, matplotlib
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


cmap_red = matplotlib.cm.get_cmap("OrRd")
cmap_blue = matplotlib.cm.get_cmap("PuBu")


def plot_oc_singlenode(
    duration, dt, state, target, control, orig_input=None, cost_array=(), color_x="red", color_y="blue"
):
    """Plot target and controlled dynamics for a network with a single node.
    :param duration:    Duration of simulation (in ms).
    :param dt:          Time discretization (in ms).
    :param state:       The state of the system controlled with the found oc-input.
    :param target:      The target state.
    :param control:     The control signal found by the oc-algorithm.
    :param orig_input:  The inputs that were used to generate target time series.
    :param cost_array:  Array of costs in optimization iterations.
    :param color_x:     Color used for plots of x-population variables.
    :param color_y:     Color used for plots of y-population variables.
    """
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)

    # Plot the target (dashed line) and unperturbed activity
    t_array = np.arange(0, duration + dt, dt)

    ax[0].plot(t_array, state[0, 0, :], label="x", color=color_x)
    ax[0].plot(t_array, state[0, 1, :], label="y", color=color_y)
    ax[0].plot(t_array, target[0, 0, :], linestyle="dashed", label="Target x", color=color_x)
    ax[0].plot(t_array, target[0, 1, :], linestyle="dashed", label="Target y", color=color_y)
    ax[0].legend()
    ax[0].set_title("Activity without stimulation and target activity")

    # Plot the target control signal (dashed line) and "initial" zero control signal
    ax[1].plot(t_array, control[0, 0, :], label="stimulation x", color=color_x)
    ax[1].plot(t_array, control[0, 1, :], label="stimulation y", color=color_y)
    if type(orig_input) != type(None):
        ax[1].plot(t_array, orig_input[0, 0, :], linestyle="dashed", label="input x", color=color_x)
        ax[1].plot(t_array, orig_input[0, 1, :], linestyle="dashed", label="input y", color=color_y)
    ax[1].legend()
    ax[1].set_title("Active stimulation and input stimulation")

    ax[2].plot(cost_array)
    ax[2].set_title("Cost throughout optimization.")

    plt.show()


def plot_oc_network(
    N, duration, dt, state, target, control, orig_input, cost_array=(), step_array=(), color_x="red", color_y="blue"
):
    """Plot target and controlled dynamics for a network with a single node.
    :param N:           Number of nodes in the network.
    :param duration:    Duration of simulation (in ms).
    :param dt:          Time discretization (in ms).
    :param state:       The state of the system controlled with the found oc-input.
    :param target:      The target state.
    :param control:     The control signal found by the oc-algorithm.
    :param orig_input:  The inputs that were used to generate target time series.
    :param cost_array:  Array of costs in optimization iterations.
    :param step_array:  Number of iterations in the step-size algorithm in each optimization iteration.
    :param color_x:     Color used for plots of x-population variables.
    :param color_y:     Color used for plots of y-population variables.
    """

    t_array = np.arange(0, duration + dt, dt)
    fig, ax = plt.subplots(3, N, figsize=(12, 8), constrained_layout=True)

    for n in range(N):
        ax[0, n].plot(t_array, state[n, 0, :], label="x", color=color_x)
        ax[0, n].plot(t_array, state[n, 1, :], label="y", color=color_y)
        ax[0, n].plot(t_array, target[n, 0, :], linestyle="dashed", label="Target x", color=color_x)
        ax[0, n].plot(t_array, target[n, 1, :], linestyle="dashed", label="Target y", color=color_y)
        ax[0, n].legend()
        ax[0, n].set_title(f"Activity and target, node %s" % (n))

        # Plot the target control signal (dashed line) and "initial" zero control signal
        ax[1, n].plot(t_array, control[n, 0, :], label="stimulation x", color=color_x)
        ax[1, n].plot(t_array, control[n, 1, :], label="stimulation y", color=color_y)
        ax[1, n].plot(t_array, orig_input[n, 0, :], linestyle="dashed", label="input x", color=color_x)
        ax[1, n].plot(t_array, orig_input[n, 1, :], linestyle="dashed", label="input y", color=color_y)
        ax[1, n].legend()
        ax[1, n].set_title(f"Stimulation and input, node %s" % (n))

    ax[2, 0].plot(cost_array)
    ax[2, 0].set_title("Cost throughout optimization.")

    ax[2, 1].plot(step_array)
    ax[2, 1].set_title("Step size throughout optimization.")

    ax[2, 1].set_ylim(bottom=0, top=None)

    plt.show()


def ops_plotall(c_array, ymax, duration, dt, vline, filename, path):

    fig, ax = plt.subplots(len(c_array), 1, figsize=(5, len(c_array)))

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)

    time = np.arange(0, duration + dt, dt)

    for i in range(len(c_array)):
        ax[i].plot(time, c_array[i][0, 0, :], color="red")
        ax[i].plot(time, c_array[i][0, 1, :], color="blue")
        ax[i].set_ylim([-ymax, ymax])
        ax[i].set_title(str(np.round(i * 2.0 / (len(c_array) - 1), 2)) + " * pi")
        ax[i].set_xlim(0, duration)
        ax[i].vlines(vline, -ymax, ymax, color="grey")
        if i != len(c_array) - 1:
            ax[i].set_xticklabels([])

    plt.savefig(os.path.join(path, filename))
    plt.show()


def ops_plotsubset(s_array, c_array, symax, cymax, duration, dt, vline, filename, path):

    fig, ax = plt.subplots(14, 1, figsize=(5, 15))

    time = np.arange(0, duration + dt, dt)

    i_range = [5, 10, 15, 20, 25, 30, 35]

    for j in range(len(i_range)):
        i = i_range[j]

        ax[2 * j].plot(time, s_array[i][0, 0, :], color="red")
        ax[2 * j].plot(time, s_array[i][0, 1, :], color="blue")
        ax[2 * j].set_ylim([0, symax])
        ax[2 * j].set_title(str(np.round(i * 2.0 / (len(c_array) - 1), 2)) + " * pi")
        ax[2 * j].set_xlim(0, duration)
        ax[2 * j].vlines(vline, 0, symax, color="grey")
        ax[2 * j].set_xticklabels([])

        ax[2 * j + 1].plot(time, c_array[i][0, 0, :], color="red")
        ax[2 * j + 1].plot(time, c_array[i][0, 1, :], color="blue")
        ax[2 * j + 1].set_ylim([-cymax, cymax])
        ax[2 * j + 1].set_xlim(0, duration)
        ax[2 * j + 1].vlines(vline, -cymax, cymax, color="grey")

        if j != 6:
            ax[2 * j + 1].set_xticklabels([])

    plt.savefig(os.path.join(path, filename))
    plt.show()


def ops_plot_acp_inits(res0, res1, period, limdiff_percent, dt, filename, path):
    n_points = len(res0["total_cost"][0])

    init_costs = np.zeros((n_points))
    for i in range(len(init_costs)):
        init_costs[i] = res0["initial_cost"][0][i]

    custom_legend_amp = [
        Line2D([0], [0], color="red", linestyle="", marker="x"),
        Line2D([0], [0], color="blue", linestyle="", marker="x"),
        Line2D([0], [0], color="red", linestyle="", marker="o", fillstyle="none"),
        Line2D([0], [0], color="blue", linestyle="", marker="o", fillstyle="none"),
    ]

    custom_legend_cost = [
        Line2D([0], [0], color="grey", linestyle="", marker="x"),
        Line2D([0], [0], color="grey", linestyle="", marker="o", fillstyle="none"),
        Line2D([0], [0], color="red", linestyle="", marker="x"),
        Line2D([0], [0], color="red", linestyle="", marker="o", fillstyle="none"),
        Line2D([0], [0], color="grey", linestyle="--"),
    ]

    r, c = 3, 1
    fig, ax = plt.subplots(r, c, sharey="row", figsize=(5, 7))

    ax[0].set_ylabel("Max/ min amplitude")
    ax[1].set_ylabel("Cost")
    ax[2].set_ylabel("Period duration")

    ax[-1].set_xlabel(r"Phase shift $\cdot \pi$")

    for i in range(r - 1):
        ax[i].set_xticks([])

    for i in range(r):
        ax[i].set_xlim([0, 2])

    ind_w = 1

    for i_shift in range(n_points):
        period_shift = i_shift * 2.0 / (n_points - 1)

        ax[0].plot(
            period_shift,
            np.amax(res0["control"][ind_w][i_shift][0, 0, :]),
            marker="x",
            fillstyle="none",
            color="red",
        )
        ax[0].plot(
            period_shift,
            np.amin(res0["control"][ind_w][i_shift][0, 0, :]),
            marker="x",
            fillstyle="none",
            color="red",
        )
        ax[0].plot(
            period_shift,
            np.amax(res0["control"][ind_w][i_shift][0, 1, :]),
            marker="x",
            fillstyle="none",
            color="blue",
        )
        ax[0].plot(
            period_shift,
            np.amin(res0["control"][ind_w][i_shift][0, 1, :]),
            marker="x",
            fillstyle="none",
            color="blue",
        )

        ax[0].plot(
            period_shift,
            np.amax(res1["control"][ind_w][i_shift][0, 0, :]),
            marker="o",
            fillstyle="none",
            color="red",
        )
        ax[0].plot(
            period_shift,
            np.amin(res1["control"][ind_w][i_shift][0, 0, :]),
            marker="o",
            fillstyle="none",
            color="red",
        )
        ax[0].plot(
            period_shift,
            np.amax(res1["control"][ind_w][i_shift][0, 1, :]),
            marker="o",
            fillstyle="none",
            color="blue",
        )
        ax[0].plot(
            period_shift,
            np.amin(res1["control"][ind_w][i_shift][0, 1, :]),
            marker="o",
            fillstyle="none",
            color="blue",
        )

        ax[0].legend(
            custom_legend_amp,
            [
                r"$u_E$, I1",
                r"$u_I$, I1",
                r"$u_E$, I2",
                r"$u_I$, I2",
            ],
            loc="upper left",
        )

        ax[1].plot(period_shift, res0["precision_cost"][ind_w][i_shift], marker="x", fillstyle="none", color="grey")
        ax[1].plot(period_shift, res1["precision_cost"][ind_w][i_shift], marker="o", fillstyle="none", color="grey")

        if "w1" in res0["filename"][0]:
            ax[1].plot(
                period_shift, res0["L1_cost_unweighted"][ind_w][i_shift], marker="x", fillstyle="none", color="red"
            )
            ax[1].plot(
                period_shift, res1["L1_cost_unweighted"][ind_w][i_shift], marker="o", fillstyle="none", color="red"
            )
            ax[1].legend(
                custom_legend_cost,
                [
                    r"$F_P$, I1",
                    r"$F_P$, I2",
                    r"$F_1$, I1",
                    r"$F_1$, I2",
                    "Uncontrolled\nsystem",
                ],
                loc="upper left",
            )

        peaks_e = scipy.signal.find_peaks(np.abs(res0["state"][ind_w][i_shift][0, 0, :]))[0]
        periods_e = np.zeros((len(peaks_e) - 1))

        for i in range(len(peaks_e) - 1):
            periods_e[i] = (peaks_e[i + 1] - peaks_e[i]) * dt
            if np.abs(periods_e[i] - period) > limdiff_percent * period:
                ax[2].plot(period_shift, periods_e[i], marker="x", fillstyle="none", color="red")

        peaks_i = scipy.signal.find_peaks(np.abs(res0["state"][ind_w][i_shift][0, 1, :]))[0]
        periods_i = np.zeros((len(peaks_i) - 1))

        for i in range(len(peaks_i) - 1):
            periods_i[i] = (peaks_i[i + 1] - peaks_i[i]) * dt
            if np.abs(periods_i[i] - period) > limdiff_percent * period:
                ax[2].plot(period_shift, periods_i[i], marker="x", fillstyle="none", color="blue")

        peaks_e = scipy.signal.find_peaks(np.abs(res1["state"][ind_w][i_shift][0, 0, :]))[0]
        periods_e = np.zeros((len(peaks_e) - 1))

        for i in range(len(peaks_e) - 1):
            periods_e[i] = (peaks_e[i + 1] - peaks_e[i]) * dt
            if np.abs(periods_e[i] - period) > limdiff_percent * period:
                ax[2].plot(period_shift, periods_e[i], marker="o", fillstyle="none", color="red")

        peaks_i = scipy.signal.find_peaks(np.abs(res1["state"][ind_w][i_shift][0, 1, :]))[0]
        periods_i = np.zeros((len(peaks_i) - 1))

        for i in range(len(peaks_i) - 1):
            periods_i[i] = (peaks_i[i + 1] - peaks_i[i]) * dt
            if np.abs(periods_i[i] - period) > limdiff_percent * period:
                ax[2].plot(period_shift, periods_i[i], marker="o", fillstyle="none", color="blue")

        ax[2].legend(
            custom_legend_amp,
            [
                "E, I1",
                "I, I1",
                "E, I2",
                "I, I2",
            ],
            loc="upper left",
        )
        ax[2].text(
            0.8,
            0.1,
            r"deviation > {:.1f} %".format(limdiff_percent * 100),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax[2].transAxes,
            fontweight=100,
            fontsize=8,
        )

        ax[1].plot(np.linspace(0, 2.0, n_points, endpoint=True), init_costs, color="grey", linestyle="--")
        ax[2].hlines(period, 0, 2.0, color="grey", linestyle="--")

    plt.savefig(os.path.join(path, filename))
    plt.show()


def ops_plot_timing_inits(res0, res1, dt, rrange, rdtick, filename, path):

    n_points = len(res0["total_cost"][0])
    r, c = 2, 2
    t_str = ["Init1 <br> <br>E+/ I-", "Init2 <br> <br>E+/ I-", "E-/ I+", "E-/ I+"]

    fig = make_subplots(
        rows=r,
        cols=c,
        specs=[[{"type": "polar"}] * c] * r,
        subplot_titles=t_str,
    )

    i_w = 1
    for i_shift in range(n_points):

        peaks0 = scipy.signal.find_peaks(np.abs(res0["state"][i_w][i_shift][0, 0, :]))[0]
        periods0 = np.zeros((len(peaks0) - 1))
        for i in range(len(peaks0) - 1):
            periods0[i] = (peaks0[i + 1] - peaks0[i]) * dt

        peaks1 = scipy.signal.find_peaks(np.abs(res1["state"][i_w][i_shift][0, 0, :]))[0]
        periods1 = np.zeros((len(peaks1) - 1))
        for i in range(len(peaks1) - 1):
            periods1[i] = (peaks1[i + 1] - peaks1[i]) * dt

        for e_i in [0, 1]:

            if e_i == 1 and "w1" in res0["filename"][0]:
                continue

            if np.amax(np.abs(res0["control"][i_w][i_shift][0, e_i, :])) < 1e-3:
                continue

            if e_i == 0:
                col = cmap_red(0.2 + i_shift * 0.8 / (n_points - 1))
            elif e_i == 1:
                col = cmap_blue(0.2 + i_shift * 0.8 / (n_points - 1))
            col_str = "rgb(" + str(col[0]) + ", " + str(col[1]) + ", " + str(col[2]) + ")"

            i = 2
            if e_i == 0:
                i = 1

            radii, timing, height = get_rth_plus(res0, e_i, i_w, i_shift, peaks0, periods0, dt)
            addtrace(fig, radii, timing, col_str, height, i, 1)

            radii, timing, height = get_rth_plus(res1, e_i, i_w, i_shift, peaks1, periods1, dt)
            addtrace(fig, radii, timing, col_str, height, i, 2)

            i = 2
            if e_i == 1:
                i = 1

            radii, timing, height = get_rth_minus(res0, e_i, i_w, i_shift, peaks0, periods0, dt)
            addtrace(fig, radii, timing, col_str, height, i, 1)

            radii, timing, height = get_rth_minus(res1, e_i, i_w, i_shift, peaks1, periods1, dt)
            addtrace(fig, radii, timing, col_str, height, i, 2)

    fig.update_layout(
        showlegend=False,
        width=600,
        height=750,
        polar=polardict(rrange, rdtick),
        polar2=polardict(rrange, rdtick),
        polar3=polardict(rrange, rdtick),
        polar4=polardict(rrange, rdtick),
    )

    fig.layout.annotations[0].update(y=1.05)
    fig.layout.annotations[1].update(y=1.05)
    fig.layout.annotations[2].update(y=0.45)
    fig.layout.annotations[3].update(y=0.45)

    pio.write_image(fig, os.path.join(path, str(filename + ".png")), format="png")
    fig.show()


def get_rth_plus(res, ind, iw, ish, peaks, periods, dt):
    cplus = np.zeros((res["control"][iw][ish][0, ind, :].shape))
    for t in range(len(cplus)):
        cplus[t] = max(0.0, res["control"][iw][ish][0, ind, t])
    max_p = np.amax(cplus)
    peaks_control_p = scipy.signal.find_peaks(cplus, height=0.5 * max_p)[0]

    height_plus = np.zeros((len(peaks_control_p)))
    for i in range(len(height_plus)):
        height_plus[i] = np.abs(res["control"][iw][ish][0, ind, peaks_control_p[i]])

    timing, radii = [], []
    for i in range(len(peaks) - 1):
        for pce in peaks_control_p:
            if pce < peaks[i + 1] and pce >= peaks[i]:
                timing.append(2.0 * np.pi * (pce - peaks[i]) * dt / periods[i])
                radii.append(pce * dt)

    return radii, timing, height_plus


def get_rth_minus(res, ind, iw, ish, peaks, periods, dt):
    cminus = np.zeros((res["control"][iw][ish][0, ind, :].shape))
    for t in range(len(cminus)):
        cminus[t] = max(0.0, -res["control"][iw][ish][0, ind, t])
    max_m = np.amax(cminus)
    peaks_control_m = scipy.signal.find_peaks(cminus, height=0.5 * max_m)[0]

    height_minus = np.zeros((len(peaks_control_m)))
    for i in range(len(height_minus)):
        height_minus[i] = np.abs(res["control"][iw][ish][0, ind, peaks_control_m[i]])

    timing, radii = [], []
    for i in range(len(peaks) - 1):
        for pce in peaks_control_m:
            if pce < peaks[i + 1] and pce >= peaks[i]:
                timing.append(2.0 * np.pi * (pce - peaks[i]) * dt / periods[i])
                radii.append(pce * dt)

    return radii, timing, height_minus


def addtrace(fig, radii, theta, cstr, height, i, j):
    fig.add_trace(
        go.Scatterpolar(
            r=radii,
            theta=theta,
            thetaunit="radians",
            mode="lines+markers",
            name="Figure 8",
            line_color=cstr,
            line_width=1,
            marker_size=4.0 + height * 40.0,
        ),
        i,
        j,
    )


def polardict(rrange, rdtick):
    return dict(
        radialaxis=dict(angle=0, tickangle=0, range=rrange, dtick=rdtick),
        angularaxis=dict(thetaunit="radians", dtick=np.pi * 0.25),
    )
