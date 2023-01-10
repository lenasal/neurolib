import matplotlib.pyplot as plt
import numpy as np
import os, scipy, matplotlib
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

global fs_, PROM, TITLELIST
fs_ = 16
PROM = [
    [2.0 * 1e-3, 4.0 * 1e-3, 2.0 * 1e-3],
    [1.0 * 1e-3, 3.0 * 1e-3, 1.0 * 1e-3],
    [2.0 * 1e-3, 8.0 * 1e-4, 8.0 * 1e-4],
    [1.0 * 1e-3, 1.0 * 1e-3, 1.0 * 1e-3],
]  # [3.0 * 1e-2, 9.0 * 1e-3, 4.0 * 1e-3]
TITLELIST = ["(A)", "(B)", "(C)", "(D)"]


def setplotparams():
    plt.rcParams["axes.titlesize"] = fs_
    plt.rcParams["axes.labelsize"] = fs_
    plt.rcParams["lines.markersize"] = 8
    plt.rcParams.update({"font.size": fs_})
    matplotlib.rc("xtick", labelsize=fs_)
    matplotlib.rc("ytick", labelsize=fs_)
    plt.rc("legend", fontsize=fs_)
    return


plt.rcParams["axes.titlesize"] = fs_
plt.rcParams["axes.labelsize"] = fs_
plt.rcParams["lines.markersize"] = 8
plt.rcParams.update({"font.size": fs_})
matplotlib.rc("xtick", labelsize=fs_)
matplotlib.rc("ytick", labelsize=fs_)
plt.rc("legend", fontsize=fs_)

global A_EPLUS, A_EMIN, A_IPLUS, A_IMIN
global b_EPLUS, B_EMIN, B_IPLUS, B_IMIN
global C_EPLUS, C_EMIN, C_IPLUS, C_IMIN
global D_EPLUS, D_EMIN, D_IPLUS, D_IMIN

A_EPLUS = [1.0247, 1.3688, 1.21276163, 0.0085, 0.5726, 0.0]
A_IMIN = [1.7868, 0.4263, 1.74598659, 0.8242, 1.4794, 1.13039355]
A_EMIN = [1.9126, 0.3461, 1.89121452, 1.2306, 0.9592, 1.43602249]
A_IPLUS = [0.0, 0.0, 0.0, 1.8291, 0.4263, 1.87604145]

B_EPLUS = [1.2022, 0.5119, 1.21173503, 0.3296, 1.1886, 0.51081906]
B_IMIN = [0.9494, 0.2421, 0.94942254, 0.1151, 1.3373, 0.28461436]
B_EMIN = [0.2139, 1.2651, 0.30691624, 1.2415, 0.5692, 1.3561662]
B_IPLUS = [1.9999, 0.0, 0.0, 1.1209, 0.6000, 1.1819992]

C_EPLUS = [0.8397, 0.8569, 1.01038161, 1.8311, 1.1259, 1.38383332]
C_IMIN = [0.6329, 0.8534, 0.4570774, 1.6318, 1.1076, 1.43882008]
C_EMIN = [1.8357, 1.1259, 1.45714899, 0.9531, 0.6346, 1.14441183]
C_IPLUS = [1.5831, 1.0377, 1.30249875, 0.7240, 0.6782, 0.87176917]

D_EPLUS = [0.9995, 1.0521, 1.11481531, 2.0948, 0.6692, 0.0]
D_IMIN = [0.3594, 0.1667, 0.3594, 1.8528, 0.6785, 0.0]
D_EMIN = [1.9221, 0.6812, 1.91892252, 1.2495, 0.6027, 1.38151754]
D_IPLUS = [1.56021, 0.1947, 1.53753834, 0.8815, 1.1068, 1.07347646]

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

    fig, ax = plt.subplots(14, 1, figsize=(8, 25))
    fig.subplots_adjust(hspace=0.6)

    time = np.arange(0, duration + dt, dt)

    i_range = [5, 10, 15, 20, 25, 30, 35]

    for j in range(len(i_range)):
        i = i_range[j]

        ax[2 * j].plot(time, s_array[i][0, 0, :], color="red")
        ax[2 * j].plot(time, s_array[i][0, 1, :], color="blue")
        ax[2 * j].set_ylim([0, symax])
        ax[2 * j].set_title(str(np.round(i * 2.0 / (len(c_array) - 1), 2)) + r" $ \cdot \pi$")
        ax[2 * j].set_xlim(0, duration)
        ax[2 * j].vlines(vline, 0, symax, color="grey")
        ax[2 * j].set_xticklabels([])
        ax[2 * j].set_ylabel("Activity")
        ax[2 * j].yaxis.set_label_coords(-0.15, 0.5)

        ax[2 * j + 1].plot(time, c_array[i][0, 0, :], color="red")
        ax[2 * j + 1].plot(time, c_array[i][0, 1, :], color="blue")
        ax[2 * j + 1].set_ylim([-cymax, cymax])
        ax[2 * j + 1].set_xlim(0, duration)
        ax[2 * j + 1].vlines(vline, -cymax, cymax, color="grey")
        ax[2 * j + 1].set_ylabel("Control")
        ax[2 * j + 1].yaxis.set_label_coords(-0.15, 0.5)

        if j != 6:
            ax[2 * j + 1].set_xticklabels([])

    ax[-1].set_xlabel("Time")

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

    custom_legend_cost2 = [
        Line2D([0], [0], color="grey", linestyle="", marker="x"),
        Line2D([0], [0], color="grey", linestyle="", marker="o", fillstyle="none"),
        Line2D([0], [0], color="red", linestyle="", marker="x"),
        Line2D([0], [0], color="red", linestyle="", marker="o", fillstyle="none"),
        Line2D([0], [0], color="blue", linestyle="", marker="x"),
        Line2D([0], [0], color="blue", linestyle="", marker="o", fillstyle="none"),
        Line2D([0], [0], color="grey", linestyle="--"),
    ]

    r, c = 2, 1
    i_dur = r - 1
    fig, ax = plt.subplots(r, c, sharey="row", figsize=(8, 8))

    ax[0].set_ylabel("Amplitude")
    # ax[1].set_ylabel("Cost")
    ax[1].set_ylabel("Period duration")

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

        if res1 != None:
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

        """
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
        elif "w2" in res0["filename"][0]:
            ax[1].plot(
                period_shift, res0["L2_cost_e_unweighted"][ind_w][i_shift], marker="x", fillstyle="none", color="red"
            )
            ax[1].plot(
                period_shift, res0["L2_cost_i_unweighted"][ind_w][i_shift], marker="x", fillstyle="none", color="blue"
            )
            ax[1].plot(
                period_shift, res1["L2_cost_e_unweighted"][ind_w][i_shift], marker="o", fillstyle="none", color="red"
            )
            ax[1].plot(
                period_shift, res1["L2_cost_i_unweighted"][ind_w][i_shift], marker="o", fillstyle="none", color="blue"
            )
            ax[1].legend(
                custom_legend_cost2,
                [
                    r"$F_P$, I1",
                    r"$F_P$, I2",
                    r"$F_{2,e}$, I1",
                    r"$F_{2,i}$, I1",
                    r"$F_{2,e}$, I2",
                    r"$F_{2,i}$, I2",
                    "Uncontrolled\nsystem",
                ],
                loc="upper left",
            )
        """

        peaks_e = scipy.signal.find_peaks(np.abs(res0["state"][ind_w][i_shift][0, 0, :]))[0]
        periods_e = np.zeros((len(peaks_e) - 1))

        for i in range(len(peaks_e) - 1):
            periods_e[i] = (peaks_e[i + 1] - peaks_e[i]) * dt
            if np.abs(periods_e[i] - period) > limdiff_percent * period:
                ax[i_dur].plot(period_shift, periods_e[i], marker="x", fillstyle="none", color="red")

        peaks_i = scipy.signal.find_peaks(np.abs(res0["state"][ind_w][i_shift][0, 1, :]))[0]
        periods_i = np.zeros((len(peaks_i) - 1))

        for i in range(len(peaks_i) - 1):
            periods_i[i] = (peaks_i[i + 1] - peaks_i[i]) * dt
            if np.abs(periods_i[i] - period) > limdiff_percent * period:
                ax[i_dur].plot(period_shift, periods_i[i], marker="o", fillstyle="none", color="blue")

        if res1 != None:
            peaks_e = scipy.signal.find_peaks(np.abs(res1["state"][ind_w][i_shift][0, 0, :]))[0]
            periods_e = np.zeros((len(peaks_e) - 1))

            for i in range(len(peaks_e) - 1):
                periods_e[i] = (peaks_e[i + 1] - peaks_e[i]) * dt
                if np.abs(periods_e[i] - period) > limdiff_percent * period:
                    ax[i_dur].plot(period_shift, periods_e[i], marker="o", fillstyle="none", color="red")

            peaks_i = scipy.signal.find_peaks(np.abs(res1["state"][ind_w][i_shift][0, 1, :]))[0]
            periods_i = np.zeros((len(peaks_i) - 1))

            for i in range(len(peaks_i) - 1):
                periods_i[i] = (peaks_i[i + 1] - peaks_i[i]) * dt
                if np.abs(periods_i[i] - period) > limdiff_percent * period:
                    ax[i_dur].plot(period_shift, periods_i[i], marker="o", fillstyle="none", color="blue")

        ax[i_dur].legend(
            custom_legend_amp,
            [
                "E, I1",
                "I, I1",
                "E, I2",
                "I, I2",
            ],
            loc="upper left",
        )
        ax[i_dur].text(
            0.8,
            0.1,
            r"deviation > {:.1f} %".format(limdiff_percent * 100),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax[i_dur].transAxes,
            fontweight=100,
            fontsize=fs_,
        )

        # ax[1].plot(np.linspace(0, 2.0, n_points, endpoint=True), init_costs, color="grey", linestyle="--")
        ax[i_dur].hlines(period, 0, 2.0, color="grey", linestyle="--")

    plt.savefig(os.path.join(path, filename))
    plt.show()


def ops_ap_scenes(res_list, period, dt, filename, path):
    n_points = len(res_list[0]["total_cost"][0])

    markerlist = ["x", "o", "*"]
    al = [0.4, 0.6, 0.8]

    custom_legend_amp = [
        Line2D([0], [0], color="red", linestyle="", marker=markerlist[0]),
        Line2D([0], [0], color="blue", linestyle="", marker=markerlist[0]),
        Line2D([0], [0], color="red", linestyle="", marker=markerlist[1], fillstyle="none"),
        Line2D([0], [0], color="blue", linestyle="", marker=markerlist[1], fillstyle="none"),
        Line2D([0], [0], color="red", linestyle="", marker=markerlist[2], fillstyle="none"),
        Line2D([0], [0], color="blue", linestyle="", marker=markerlist[2], fillstyle="none"),
    ]

    r, c = 2, 1
    fig, ax = plt.subplots(r, c, sharey="row", figsize=(8, 8))

    ax[0].set_ylabel(r"Amplitude")
    ax[1].set_ylabel(r"Max. deviation from" + "\n" + r"natural oscillation period [%]")

    ax[-1].set_xlabel(r"Phase shift $\cdot \pi$")

    for i in range(r - 1):
        ax[i].set_xticks([])

    for i in range(r):
        ax[i].set_xlim([0, 2])

    ind_w = 1

    for ind in range(3):
        res = res_list[ind]

        plte = []

        for i_shift in range(n_points):
            period_shift = i_shift * 2.0 / (n_points - 1)

            ax[0].plot(
                period_shift,
                np.amax(res["control"][ind_w][i_shift][0, 0, :]),
                marker=markerlist[ind],
                fillstyle="none",
                color="red",
            )
            ax[0].plot(
                period_shift,
                np.amin(res["control"][ind_w][i_shift][0, 0, :]),
                marker=markerlist[ind],
                fillstyle="none",
                color="red",
            )
            if "w2" in filename:
                ax[0].plot(
                    period_shift,
                    np.amax(res["control"][ind_w][i_shift][0, 1, :]),
                    marker=markerlist[ind],
                    fillstyle="none",
                    color="blue",
                )
                ax[0].plot(
                    period_shift,
                    np.amin(res["control"][ind_w][i_shift][0, 1, :]),
                    marker=markerlist[ind],
                    fillstyle="none",
                    color="blue",
                )

            peaks_e = scipy.signal.find_peaks(np.abs(res["state"][ind_w][i_shift][0, 0, :]))[0]
            periods_e = np.zeros((len(peaks_e) - 1))
            periodsdev_e = periods_e.copy()

            if i_shift in [0, 40]:
                plte.append(0)
                continue

            for i in range(len(peaks_e) - 1):
                periods_e[i] = (peaks_e[i + 1] - peaks_e[i]) * dt
                periodsdev_e[i] = ((periods_e[i] / period) - 1.0) * 100.0

            if np.amin(periodsdev_e) <= -np.amax(periodsdev_e):
                plte.append(np.amin(periodsdev_e))
                ax[1].plot(period_shift, np.amin(periodsdev_e), marker=markerlist[ind], fillstyle="none", color="red")
            else:
                plte.append(np.amax(periodsdev_e))
                ax[1].plot(period_shift, np.amax(periodsdev_e), marker=markerlist[ind], fillstyle="none", color="red")

            peaks_i = scipy.signal.find_peaks(np.abs(res["state"][ind_w][i_shift][0, 1, :]))[0]
            periods_i = np.zeros((len(peaks_i) - 1))
            periodsdev_i = periods_i.copy()

            for i in range(len(peaks_i) - 1):
                periods_i[i] = (peaks_i[i + 1] - peaks_i[i]) * dt
                periodsdev_i[i] = ((periods_i[i] / period) - 1.0) * 100.0

            if np.amin(periodsdev_i) <= -np.amax(periodsdev_i):
                ax[1].plot(period_shift, np.amin(periodsdev_i), marker=markerlist[ind], fillstyle="none", color="blue")
            else:
                ax[1].plot(period_shift, np.amax(periodsdev_i), marker=markerlist[ind], fillstyle="none", color="blue")

        limind = np.where(np.array(plte[:-5]) < 0.0)[0][-1]
        ax[1].fill_between(
            np.arange(0.0, (limind) * 0.05 + 0.01, 0.05), plte[: limind + 1], color="yellowgreen", alpha=al[ind]
        )
        ax[1].fill_between(
            np.arange((limind + 1) * 0.05, 2.01, 0.05), plte[limind + 1 :], color="darkgreen", alpha=al[ind]
        )

        ax[0].legend(
            custom_legend_amp,
            [
                r"$u_E$, S1",
                r"$u_I$, S1",
                r"$u_E$, S2",
                r"$u_I$, S2",
                r"$u_E$, S3",
                r"$u_I$, S3",
            ],
            loc="upper left",
        )

        ax[1].legend(
            custom_legend_amp,
            [
                "E, S1",
                "I, S1",
                "E, S2",
                "I, S2",
                "E, S3",
                "I, S3",
            ],
            loc="upper left",
        )

        ax[0].hlines(0, 0, 2, color="gray", linewidth=1)
        ax[1].hlines(0, 0, 2, color="gray", linewidth=1)

    plt.savefig(os.path.join(path, filename))
    plt.show()


def ops_plot_timing_inits(res0, res1, dt, rrange, rdtick, filename, path):

    n_points = len(res0["total_cost"][0])
    r, c = 2, 1
    t_str = [
        "<b>Init1 <br> <br>E+</b>",
        # "<b>Init2 <br> <br>E+</b>",
        "<b>E-</b>",
        # "<b>E-</b>",
    ]
    if res1 != None:
        c = 2
        t_str = [
            "<b>Init1 <br> <br>E+</b>",
            "<b>Init2 <br> <br>E+</b>",
            "<b>E-</b>",
            "<b>E-</b>",
        ]

    fig = make_subplots(
        rows=r,
        cols=c,
        specs=[[{"type": "polar"}] * c] * r,
        subplot_titles=t_str,
    )

    i_w = 1

    if "2-5_0" in path:
        background(fig, rrange[1], 1, B_EPLUS)
        background(fig, rrange[1], 2, B_EMIN)

    elif "1-75_0" in path:
        background(fig, rrange[1], 1, D_EPLUS)
        background(fig, rrange[1], 2, D_EMIN)

    elif "2-6_1-2" in path:
        background(fig, rrange[1], 1, C_EPLUS)
        background(fig, rrange[1], 2, C_EMIN)
    else:  # done
        background(fig, rrange[1], 1, A_EPLUS)
        background(fig, rrange[1], 2, A_EMIN)

    for i_shift in range(n_points):

        peaks0 = scipy.signal.find_peaks(np.abs(res0["state"][i_w][i_shift][0, 0, :]))[0]
        periods0 = np.zeros((len(peaks0) - 1))
        for i in range(len(peaks0) - 1):
            periods0[i] = (peaks0[i + 1] - peaks0[i]) * dt

        if res1 != None:
            peaks1 = scipy.signal.find_peaks(np.abs(res1["state"][i_w][i_shift][0, 0, :]))[0]
            periods1 = np.zeros((len(peaks1) - 1))
            for i in range(len(peaks1) - 1):
                periods1[i] = (peaks1[i + 1] - peaks1[i]) * dt

        e_i = 0

        if np.amax(np.abs(res0["control"][i_w][i_shift][0, e_i, :])) < 1e-3:
            continue

        col = cmap_red(0.2 + i_shift * 0.8 / (n_points - 1))
        col_str = "rgb(" + str(col[0]) + ", " + str(col[1]) + ", " + str(col[2]) + ")"

        radii, timing, height = get_rth_plus(res0, e_i, i_w, i_shift, peaks0, periods0, dt)
        addtrace(fig, radii, timing, col_str, height, 1, 1)

        radii, timing, height = get_rth_minus(res0, e_i, i_w, i_shift, peaks0, periods0, dt)
        addtrace(fig, radii, timing, col_str, height, 2, 1)

        if res1 != None:
            radii, timing, height = get_rth_plus(res1, e_i, i_w, i_shift, peaks1, periods1, dt)
            addtrace(fig, radii, timing, col_str, height, 1, 2)

            radii, timing, height = get_rth_minus(res1, e_i, i_w, i_shift, peaks1, periods1, dt)
            addtrace(fig, radii, timing, col_str, height, 2, 2)

    fig.update_layout(
        showlegend=False,
        width=500,
        height=750,
        polar=polardict(rrange, rdtick),
        polar2=polardict(rrange, rdtick),
        # polar3=polardict(rrange, rdtick),
        # polar4=polardict(rrange, rdtick),
    )

    fig.layout.annotations[0].update(y=1.05)
    # fig.layout.annotations[1].update(y=1.05)
    fig.layout.annotations[1].update(y=0.45)
    # fig.layout.annotations[3].update(y=0.45)

    pio.write_image(fig, os.path.join(path, str(filename + ".png")), format="png")
    fig.show()


def ops_plot_timing_inits_scenes(res_list, dt, rrange, rdtick, filename, path):

    n_points = len(res_list[0]["total_cost"][0])
    r, c = 2, 1
    t_str = [
        "<b>E<sup>+</sup></b>",
        "<b>E<sup>-</sup></b>",
    ]

    fig = make_subplots(
        rows=r,
        cols=c,
        specs=[[{"type": "polar"}] * c] * r,
        subplot_titles=t_str,
    )

    i_w = 1

    if "2-5_0" in path:
        background(fig, rrange[1], 1, B_EPLUS)
        background(fig, rrange[1], 2, B_EMIN)

    elif "1-75_0" in path:
        background(fig, rrange[1], 1, D_EPLUS)
        background(fig, rrange[1], 2, D_EMIN)

    elif "2-6_1-2" in path:
        background(fig, rrange[1], 1, C_EPLUS)
        background(fig, rrange[1], 2, C_EMIN)
    else:  # done
        background(fig, rrange[1], 1, A_EPLUS)
        background(fig, rrange[1], 2, A_EMIN)

    if "1_0" in path:
        point = 0
    elif "2-5_0" in path:
        point = 1
    elif "2-6_1-2" in path:
        point = 2
    elif "1-75_0" in path:
        point = 3

    for res_ind in range(len(res_list)):
        res = res_list[res_ind]

        for i_shift in range(n_points):

            peaks0 = scipy.signal.find_peaks(np.abs(res["state"][i_w][i_shift][0, 0, :]))[0]
            periods0 = np.zeros((len(peaks0) - 1))
            for i in range(len(peaks0) - 1):
                periods0[i] = (peaks0[i + 1] - peaks0[i]) * dt

            e_i = 0

            if np.amax(np.abs(res["control"][i_w][i_shift][0, e_i, :])) < 1e-3:
                continue

            col = cmap_red(0.2 + i_shift * 0.8 / (n_points - 1))
            col_str = "rgb(" + str(col[0]) + ", " + str(col[1]) + ", " + str(col[2]) + ")"

            radii, timing, height = get_rth_plus(res, e_i, i_w, i_shift, res_ind, peaks0, periods0, dt, point)
            addtrace(fig, radii, timing, col_str, height, 1, 1)

            radii, timing, height = get_rth_minus(res, e_i, i_w, i_shift, res_ind, peaks0, periods0, dt, point)
            addtrace(fig, radii, timing, col_str, height, 2, 1)

    fig.update_layout(
        showlegend=False,
        width=500,
        height=1000,
        polar=polardict(rrange, rdtick),
        polar2=polardict(rrange, rdtick),
        font=dict(
            size=fs_,
        ),
    )

    fig.layout.annotations[0].update(y=1.08, font_size=fs_)
    fig.layout.annotations[1].update(y=0.45, font_size=fs_)

    pio.write_image(fig, os.path.join(path, str(filename + ".png")), format="png")
    fig.show()


def background(fig, r, k, input):
    for j in range(1, 2):
        if input[1] > 0.0:
            addbackground(fig, r, input[0], input[1], k, j, "yellowgreen")
            addline(fig, r, np.pi * input[2], "yellowgreen", k, j)
        if input[4] > 0.0:
            addbackground(fig, r, input[3], input[4], k, j, "darkgreen")
            addline(fig, r, np.pi * input[5], "darkgreen", k, j)


def ops_plot_timing_inits_L2(res0, res1, dt, rrange, rdtick, filename, path):

    n_points = len(res0["total_cost"][0])
    r, c = 4, 1
    t_str = [
        "<b>Init1 <br> <br>E</b>",
        "<b>I-</b>",
        "<b>E-</b>",
        "<b>I+</b>",
    ]
    if res1 != None:
        r, c = 4, 2
        t_str = [
            "<b>Init1 <br> <br>E+</b>",
            "<b>Init2 <br> <br>E+</b>",
            "<b>I-</b>",
            "<b>I-</b>",
            "<b>E-</b>",
            "<b>E-</b>",
            "<b>I+</b>",
            "<b>I+</b>",
        ]

    fig = make_subplots(
        rows=r,
        cols=c,
        specs=[[{"type": "polar"}] * c] * r,
        subplot_titles=t_str,
    )

    i_w = 1

    if "2-5_0" in path:
        background(fig, rrange[1], 1, B_EPLUS)
        background(fig, rrange[1], 2, B_IMIN)
        background(fig, rrange[1], 3, B_EMIN)
        background(fig, rrange[1], 4, B_IPLUS)
    elif "1-75_0" in path:
        background(fig, rrange[1], 1, D_EPLUS)
        background(fig, rrange[1], 2, D_IMIN)
        background(fig, rrange[1], 3, D_EMIN)
        background(fig, rrange[1], 4, D_IPLUS)
    elif "2-6_1-2" in path:
        background(fig, rrange[1], 1, C_EPLUS)
        background(fig, rrange[1], 2, C_IMIN)
        background(fig, rrange[1], 3, C_EMIN)
        background(fig, rrange[1], 4, C_IPLUS)
    else:  # done
        background(fig, rrange[1], 1, A_EPLUS)
        background(fig, rrange[1], 2, A_IMIN)
        background(fig, rrange[1], 3, A_EMIN)
        background(fig, rrange[1], 4, A_IPLUS)

    for i_shift in range(n_points):

        peaks0 = scipy.signal.find_peaks(np.abs(res0["state"][i_w][i_shift][0, 0, :]))[0]
        periods0 = np.zeros((len(peaks0) - 1))
        for i in range(len(peaks0) - 1):
            periods0[i] = (peaks0[i + 1] - peaks0[i]) * dt

        if res1 != None:
            peaks1 = scipy.signal.find_peaks(np.abs(res1["state"][i_w][i_shift][0, 0, :]))[0]
            periods1 = np.zeros((len(peaks1) - 1))
            for i in range(len(peaks1) - 1):
                periods1[i] = (peaks1[i + 1] - peaks1[i]) * dt

        for e_i in [0, 1]:

            if np.amax(np.abs(res0["control"][i_w][i_shift][0, e_i, :])) < 1e-3:
                continue

            col = cmap_red(0.2 + i_shift * 0.8 / (n_points - 1))
            if e_i == 1:
                col = cmap_blue(0.2 + i_shift * 0.8 / (n_points - 1))
            col_str = "rgb(" + str(col[0]) + ", " + str(col[1]) + ", " + str(col[2]) + ")"

            i, j = 1, 3
            if e_i == 1:
                i, j = 4, 2

            radii, timing, height = get_rth_plus(res0, e_i, i_w, i_shift, peaks0, periods0, dt)
            addtrace(fig, radii, timing, col_str, height, i, 1)

            radii, timing, height = get_rth_minus(res0, e_i, i_w, i_shift, peaks0, periods0, dt)
            addtrace(fig, radii, timing, col_str, height, j, 1)

            if res1 != None:
                radii, timing, height = get_rth_plus(res1, e_i, i_w, i_shift, peaks1, periods1, dt)
                addtrace(fig, radii, timing, col_str, height, i, 2)

                radii, timing, height = get_rth_minus(res1, e_i, i_w, i_shift, peaks1, periods1, dt)
                addtrace(fig, radii, timing, col_str, height, j, 2)

    fig.update_layout(
        showlegend=False,
        width=500,
        height=1300,
        polar=polardict(rrange, rdtick),
        polar2=polardict(rrange, rdtick),
        polar3=polardict(rrange, rdtick),
        polar4=polardict(rrange, rdtick),
        # polar5=polardict(rrange, rdtick),
        # polar6=polardict(rrange, rdtick),
        # polar7=polardict(rrange, rdtick),
        # polar8=polardict(rrange, rdtick),
    )

    fig.layout.annotations[0].update(y=1.03)
    # fig.layout.annotations[1].update(y=1.03)
    fig.layout.annotations[1].update(y=0.75)
    # fig.layout.annotations[3].update(y=0.75)
    fig.layout.annotations[2].update(y=0.47)
    # fig.layout.annotations[5].update(y=0.47)
    fig.layout.annotations[3].update(y=0.19)
    # fig.layout.annotations[7].update(y=0.19)

    pio.write_image(fig, os.path.join(path, str(filename + ".png")), format="png")
    fig.show()


def ops_plot_timing_inits_L2_scenes(res_list, dt, rrange, rdtick, filename, path):

    n_points = len(res_list[0]["total_cost"][0])
    r, c = 4, 1
    t_str = [
        "<b>E<sup>+</sub></b>",
        "<b>I<sup>-</sub></b>",
        "<b>E<sup>-</sub></b>",
        "<b>I<sup>+</sub></b>",
    ]

    fig = make_subplots(
        rows=r,
        cols=c,
        specs=[[{"type": "polar"}] * c] * r,
        subplot_titles=t_str,
    )

    i_w = 1

    if "2-5_0" in path:
        background(fig, rrange[1], 1, B_EPLUS)
        background(fig, rrange[1], 2, B_IMIN)
        background(fig, rrange[1], 3, B_EMIN)
        background(fig, rrange[1], 4, B_IPLUS)
    elif "1-75_0" in path:
        background(fig, rrange[1], 1, D_EPLUS)
        background(fig, rrange[1], 2, D_IMIN)
        background(fig, rrange[1], 3, D_EMIN)
        background(fig, rrange[1], 4, D_IPLUS)
    elif "2-6_1-2" in path:
        background(fig, rrange[1], 1, C_EPLUS)
        background(fig, rrange[1], 2, C_IMIN)
        background(fig, rrange[1], 3, C_EMIN)
        background(fig, rrange[1], 4, C_IPLUS)
    else:  # done
        background(fig, rrange[1], 1, A_EPLUS)
        background(fig, rrange[1], 2, A_IMIN)
        background(fig, rrange[1], 3, A_EMIN)
        background(fig, rrange[1], 4, A_IPLUS)

    if "1_0" in path:
        point = 0
    elif "2-5_0" in path:
        point = 1
    elif "2-6_1-2" in path:
        point = 2
    elif "1-75_0" in path:
        point = 3

    for res_ind in range(len(res_list)):
        res = res_list[res_ind]

        for i_shift in range(n_points):

            peaks0 = scipy.signal.find_peaks(np.abs(res["state"][i_w][i_shift][0, 0, :]))[0]
            periods0 = np.zeros((len(peaks0) - 1))
            for i in range(len(peaks0) - 1):
                periods0[i] = (peaks0[i + 1] - peaks0[i]) * dt

            for e_i in [0, 1]:

                if np.amax(np.abs(res["control"][i_w][i_shift][0, e_i, :])) < 1e-3:
                    continue

                col = cmap_red(0.2 + i_shift * 0.8 / (n_points - 1))
                if e_i == 1:
                    col = cmap_blue(0.2 + i_shift * 0.8 / (n_points - 1))
                col_str = "rgb(" + str(col[0]) + ", " + str(col[1]) + ", " + str(col[2]) + ")"

                i, j = 1, 3
                if e_i == 1:
                    i, j = 4, 2

                radii, timing, height = get_rth_plus(res, e_i, i_w, i_shift, res_ind, peaks0, periods0, dt, point)
                addtrace(fig, radii, timing, col_str, height, i, 1)

                radii, timing, height = get_rth_minus(res, e_i, i_w, i_shift, res_ind, peaks0, periods0, dt, point)
                addtrace(fig, radii, timing, col_str, height, j, 1)

    fig.update_layout(
        showlegend=False,
        width=500,
        height=2000,
        polar=polardict(rrange, rdtick),
        polar2=polardict(rrange, rdtick),
        polar3=polardict(rrange, rdtick),
        polar4=polardict(rrange, rdtick),
        font=dict(
            size=fs_,
        ),
    )

    fig.layout.annotations[0].update(y=1.03, font_size=fs_)
    fig.layout.annotations[1].update(y=0.75, font_size=fs_)
    fig.layout.annotations[2].update(y=0.47, font_size=fs_)
    fig.layout.annotations[3].update(y=0.19, font_size=fs_)

    pio.write_image(fig, os.path.join(path, str(filename + ".png")), format="png")
    fig.show()


def get_rth_plus(res, ind, iw, ish, isc, peaks, periods, dt, point):
    cplus = np.zeros((res["control"][iw][ish][0, ind, :].shape))
    for t in range(len(cplus)):
        cplus[t] = max(0.0, res["control"][iw][ish][0, ind, t])
    max_p = np.amax(cplus)
    minheight = max(0.5 * max_p, 0.1 * np.amax(np.abs(res["control"][iw][ish][0, ind, :])))
    peaks_control_p = scipy.signal.find_peaks(cplus, height=minheight)[0]

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


def get_rth_minus(res, ind, iw, ish, isc, peaks, periods, dt, point):
    cminus = np.zeros((res["control"][iw][ish][0, ind, :].shape))
    for t in range(len(cminus)):
        cminus[t] = max(0.0, -res["control"][iw][ish][0, ind, t])
    max_m = np.amax(cminus)
    minheight = max(0.5 * max_m, 0.1 * np.amax(np.abs(res["control"][iw][ish][0, ind, :])))
    peaks_control_m = scipy.signal.find_peaks(cminus, height=minheight)[0]

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
            mode="markers",
            line_color=cstr,
            line_width=1,
            marker_size=5.0 + height * 50.0,
        ),
        i,
        j,
    )


def addline(fig, r, theta, col, i, j):
    fig.add_trace(
        go.Scatterpolar(
            r=[0, r],
            theta=[theta, theta],
            thetaunit="radians",
            mode="lines",
            line_color=col,
            line_width=3,
        ),
        i,
        j,
    )


def addbackground(fig, r, theta, width, i, j, col, op=0.2):
    fig.add_trace(
        go.Barpolar(
            r=[r],
            theta=[theta * np.pi],
            thetaunit="radians",
            width=[width * np.pi],
            marker_color=col,
            marker_line_color="black",
            marker_line_width=0,
            opacity=op,
        ),
        i,
        j,
    )


def polardict(rrange, rdtick):
    return dict(
        radialaxis=dict(angle=0, tickangle=0, range=rrange, dtick=rdtick),
        angularaxis=dict(thetaunit="radians", dtick=np.pi * 0.25),
    )


def get_prc(model, point, da, wa, dta, norm=True):

    duration = 200.0
    period = np.zeros((len(da)))

    model.params.sigma_ou = 0.0
    model.params.duration = duration

    shift_ep = []
    shift_em = []
    shift_ip = []
    shift_im = []

    for i_de in range(len(da)):

        print(i_de)

        dt = dta[i_de]
        zero_input = np.zeros(((1, np.around(duration / dt).astype(int) + 1)))
        model.params["exc_ext"] = zero_input + point[0]
        model.params["inh_ext"] = zero_input + point[1]

        model.params.dt = dt
        model.run()

        peaks0 = scipy.signal.find_peaks(model.exc[0, :])[0]
        p_list = []
        for i in range(4, len(peaks0)):
            p_list.append(peaks0[i] - peaks0[i - 1])
        period[i_de] = np.mean(p_list) * dt

        i0, i1 = 0, 0
        for i in range(len(peaks0)):
            if peaks0[i] > 100 / dt:
                i0 = peaks0[i]
                i1 = peaks0[i + 1]
                break

        ind_range = range(i0, i1 + 1, 1)
        p0 = peaks0[-1]

        shift_ep.append(np.zeros((len(ind_range))))
        shift_em.append(np.zeros((len(ind_range))))
        # shift_ip.append(np.zeros((len(ind_range))))
        # shift_im.append(np.zeros((len(ind_range))))

        de = da[i_de]
        shift = np.around(0.5 * wa[i_de] / dt).astype(int)

        for i in range(len(ind_range)):
            t = ind_range[i]

            model.params.exc_ext[0, t - shift : t + shift] += de
            model.run()
            model.params.exc_ext[0, t - shift : t + shift] -= de

            p = scipy.signal.find_peaks(model.exc[0, :])[0][-1]
            shift_ep[-1][i] = (p - p0) * dt

            model.params.exc_ext[0, t - shift : t + shift] += -de
            model.run()
            model.params.exc_ext[0, t - shift : t + shift] -= -de

            p = scipy.signal.find_peaks(model.exc[0, :])[0][-1]
            shift_em[-1][i] = (p - p0) * dt

            """
            model.params.inh_ext[0, t - shift : t + shift] += de
            model.run()
            model.params.inh_ext[0, t - shift : t + shift] -= de

            p = scipy.signal.find_peaks(model.exc[0, :])[0][-1]
            shift_ip[i_de, i] = (p - p0) * dt

            model.params.inh_ext[0, t - shift : t + shift] += -de
            model.run()
            model.params.inh_ext[0, t - shift : t + shift] -= -de

            p = scipy.signal.find_peaks(model.exc[0, :])[0][-1]
            shift_im[i_de, i] = (p - p0) * dt
            """

        if norm:
            shift_ep[-1] *= 1.0 / np.amax(np.abs(shift_ep[-1]))  # 2.0 * np.pi / (period * de * dt)
            shift_em[-1] *= 1.0 / np.amax(np.abs(shift_em[-1]))
            # shift_ip *= 1.0 / np.amax(np.abs(shift_ip))
            # shift_im *= 1.0 / np.amax(np.abs(shift_im))

    return shift_ep, shift_em, shift_ip, shift_im, period


def plot_prc(resdict, dist_array, filename, path):

    setplotparams()

    fig, ax = plt.subplots(1, 4, figsize=(20, 5), sharex=True, constrained_layout=True)

    custom_legend_0 = [
        Line2D([0], [0], color="red"),
        Line2D([0], [0], color="red", linestyle=":"),
        Line2D([0], [0], color="blue"),
        Line2D([0], [0], color="blue", linestyle=":"),
    ]

    for ip in range(4):

        for i in range(len(dist_array)):
            phase = np.linspace(0, 2, len(resdict["ep"][ip][i]))
            rmax = 1.1 * max(
                np.amax(np.abs(resdict["ep"][ip][i])),
                np.amax(np.abs(resdict["em"][ip][i])),
                0,  # np.amax(np.abs(resdict["ip"][ip][i])),
                0,  # np.amax(np.abs(resdict["im"][ip][i])),
            )

            ax[ip].plot(phase, resdict["ep"][ip][i][:], color="red", linewidth=1)
            ax[ip].plot(phase, resdict["em"][ip][i][:], color="red", linestyle=":", linewidth=1)
            # ax[ip].plot(phase, resdict["ip"][ip][i][:], color="blue", linewidth=1)
            # ax[ip].plot(phase, resdict["im"][ip][i][:], color="blue", linestyle=":", linewidth=1)

            ax[ip].set_title(TITLELIST[ip], fontweight="bold")
            ax[ip].hlines(0.0, 0, 2.0, color="grey")
            ax[ip].set_xlim(0, 2.0)
            ax[ip].set_xlabel(r"Phase shift $\cdot \pi$")
            ax[ip].set_ylim(-rmax, rmax)
            ax[ip].text(
                0.5,
                0.65,
                "phase delay",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax[ip].transAxes,
            )
            ax[ip].text(
                0.5,
                0.35,
                "phase advance",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax[ip].transAxes,
            )

        ax[0].legend(custom_legend_0, ["E+", "E-", "I+", "I-"], loc="lower left")

    fig.savefig(os.path.join(path, filename))
    plt.show()


def ops_plot_traces(res1, dt, filename, path):

    fig, ax = plt.subplots(
        14,
        3,
        figsize=(24, 14),
        sharex="col",
        gridspec_kw={"width_ratios": [1.4, 3.0, 5.0]},
        constrained_layout=True,
    )
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)

    custom_legend = [
        Line2D([0], [0], color="red"),
        Line2D([0], [0], color="blue"),
    ]

    i_range = [5, 10, 15, 20, 25, 30, 35]
    title_list = [
        "S1",
        "S2",
        "S3",
    ]

    p_str = []
    for p in i_range:
        p_str.append(r"$ p = $" + str(np.round(p / 20.0, 2)) + r" $ \cdot \pi$")

    act_range = [0.0, 1.1 * np.amax(res1[0]["state"][1][1])]

    for ind_sc in range(3):

        duration = (res1[ind_sc]["state"][1][1].shape[2] - 1) * dt
        time = np.arange(0, duration + dt, dt)

        c_range = [
            1.1 * np.amin(res1[ind_sc]["control"][1][:]),
            1.1 * np.amax(res1[ind_sc]["control"][1][:]),
        ]

        ax[0, ind_sc].set_title(title_list[ind_sc], fontweight="bold")

        for j in range(len(i_range)):
            i = i_range[j]

            ax[2 * j, ind_sc].plot(time, res1[ind_sc]["state"][1][i][0, 0, :], color="red")
            ax[2 * j, ind_sc].plot(time, res1[ind_sc]["state"][1][i][0, 1, :], color="blue")

            ax[2 * j, ind_sc].set_ylim(act_range)
            ax[2 * j, ind_sc].set_xlim(0, duration)

            ax[2 * j + 1, ind_sc].plot(time, res1[ind_sc]["control"][1][i][0, 0, :], color="red")
            ax[2 * j + 1, ind_sc].plot(time, res1[ind_sc]["control"][1][i][0, 1, :], color="blue")

            ax[2 * j + 1, ind_sc].set_ylim(c_range)
            ax[2 * j + 1, ind_sc].set_xlim(0, duration)

            ax[2 * j, ind_sc].vlines(duration - 100.0, act_range[0], act_range[1], color="grey")
            ax[2 * j + 1, ind_sc].vlines(duration - 100.0, c_range[0], c_range[1], color="grey")

            ax[2 * j, ind_sc].text(
                0.0,
                act_range[1],
                p_str[j],
                verticalalignment="top",
                horizontalalignment="left",
                fontsize=fs_,
                bbox=dict(facecolor="white", alpha=1.0, pad=0.0),
            )

            if ind_sc == 0:
                ax[2 * j, ind_sc].set_ylabel(r"$E, I$")
                ax[2 * j, ind_sc].yaxis.set_label_coords(-0.3, 0.5)
                ax[2 * j + 1, ind_sc].set_ylabel(r"$u_{E,I}$")
                ax[2 * j + 1, ind_sc].yaxis.set_label_coords(-0.3, 0.5)

        ax[-1, ind_sc].set_xlabel("Time")

    if "w2" in filename:
        ax[0, 0].legend(
            custom_legend,
            [r"$E$", r"$I$"],
            loc="upper right",
            labelspacing=0.1,
            handletextpad=0.1,
            handlelength=1,
            borderaxespad=0.15,
        )
        ax[1, 0].legend(
            custom_legend,
            [r"$u_E$", r"$u_I$"],
            loc="upper right",
            labelspacing=0.1,
            handletextpad=0.1,
            handlelength=1,
            borderaxespad=0.15,
        )
    else:
        ax[0, 0].legend(
            custom_legend,
            [r"$E$"],
            loc="upper right",
            labelspacing=0.1,
            handletextpad=0.1,
            handlelength=1,
            borderaxespad=0.15,
        )
        ax[1, 0].legend(
            custom_legend,
            [r"$u_E$"],
            loc="upper right",
            labelspacing=0.1,
            handletextpad=0.1,
            handlelength=1,
            borderaxespad=0.15,
        )

    plt.savefig(os.path.join(path, filename))
    plt.show()


def ops_ap_scenes_allpoints(reslist, periods, dt, filename, path):

    setplotparams()

    n_points = len(reslist[0]["total_cost"][0])

    markerlist = ["x", "o", "*"]
    al = [0.3, 0.5, 0.7]

    custom_legend_L1 = [
        Line2D([0], [0], color="red", linestyle="", marker=markerlist[0]),
        Line2D([0], [0], color="red", linestyle="", marker=markerlist[1], fillstyle="none"),
        Line2D([0], [0], color="red", linestyle="", marker=markerlist[2], fillstyle="none"),
    ]

    custom_legend_L2 = [
        Line2D([0], [0], color="blue", linestyle="", marker=markerlist[0]),
        Line2D([0], [0], color="blue", linestyle="", marker=markerlist[1], fillstyle="none"),
        Line2D([0], [0], color="blue", linestyle="", marker=markerlist[2], fillstyle="none"),
    ]

    r, c = 2, 4
    fig, ax = plt.subplots(r, c, figsize=(20, 6), sharex=True, constrained_layout=True)

    ax[0, 0].set_ylabel(r"Amplitude")
    ax[1, 0].set_ylabel(r"Max. deviation from" + "\n" + r"natural period [%]")
    ax[0, 0].yaxis.set_label_coords(-0.2, 0.5)
    ax[1, 0].yaxis.set_label_coords(-0.2, 0.5)

    for ind_p in range(4):
        ax[-1, ind_p].set_xlabel(r"Phase shift $\cdot \pi$")

        ax[0, ind_p].set_title(TITLELIST[ind_p], fontweight="bold")

        ax[0, ind_p].set_xlim([0, 2])
        ax[1, ind_p].set_xlim([0, 2])

        for ind in range(3):
            res = reslist[ind_p * 3 + ind]

            plte = []

            for i_shift in range(n_points):
                period_shift = i_shift * 2.0 / (n_points - 1)

                ax[0, ind_p].plot(
                    period_shift,
                    np.amax(res["control"][1][i_shift][0, 0, :]),
                    marker=markerlist[ind],
                    fillstyle="none",
                    color="red",
                )
                ax[0, ind_p].plot(
                    period_shift,
                    np.amin(res["control"][1][i_shift][0, 0, :]),
                    marker=markerlist[ind],
                    fillstyle="none",
                    color="red",
                )
                if "w2" in filename:
                    ax[0, ind_p].plot(
                        period_shift,
                        np.amax(res["control"][1][i_shift][0, 1, :]),
                        marker=markerlist[ind],
                        fillstyle="none",
                        color="blue",
                    )
                    ax[0, ind_p].plot(
                        period_shift,
                        np.amin(res["control"][1][i_shift][0, 1, :]),
                        marker=markerlist[ind],
                        fillstyle="none",
                        color="blue",
                    )

                peaks_e = scipy.signal.find_peaks(np.abs(res["state"][1][i_shift][0, 0, :]))[0]
                periods_e = np.zeros((len(peaks_e) - 1))
                periodsdev_e = periods_e.copy()

                if i_shift in [0, 40]:
                    plte.append(0)
                    continue

                for i in range(len(peaks_e) - 1):
                    periods_e[i] = (peaks_e[i + 1] - peaks_e[i]) * dt
                    periodsdev_e[i] = ((periods_e[i] / periods[ind_p]) - 1.0) * 100.0

                if np.amin(periodsdev_e) <= -np.amax(periodsdev_e):
                    plte.append(np.amin(periodsdev_e))
                    ax[1, ind_p].plot(
                        period_shift, np.amin(periodsdev_e), marker=markerlist[ind], fillstyle="none", color="red"
                    )
                else:
                    plte.append(np.amax(periodsdev_e))
                    ax[1, ind_p].plot(
                        period_shift, np.amax(periodsdev_e), marker=markerlist[ind], fillstyle="none", color="red"
                    )

                peaks_i = scipy.signal.find_peaks(np.abs(res["state"][1][i_shift][0, 1, :]))[0]
                periods_i = np.zeros((len(peaks_i) - 1))
                periodsdev_i = periods_i.copy()

                for i in range(len(peaks_i) - 1):
                    periods_i[i] = (peaks_i[i + 1] - peaks_i[i]) * dt
                    periodsdev_i[i] = ((periods_i[i] / periods[ind_p]) - 1.0) * 100.0

                if np.amin(periodsdev_i) <= -np.amax(periodsdev_i):
                    ax[1, ind_p].plot(
                        period_shift, np.amin(periodsdev_i), marker=markerlist[ind], fillstyle="none", color="blue"
                    )
                else:
                    ax[1, ind_p].plot(
                        period_shift, np.amax(periodsdev_i), marker=markerlist[ind], fillstyle="none", color="blue"
                    )

            limind = np.where(np.array(plte[:-5]) < 0.0)[0][-1]
            ax[1, ind_p].fill_between(
                np.arange(0.0, (limind) * 0.05 + 0.01, 0.05), plte[: limind + 1], color="yellowgreen", alpha=al[ind]
            )
            ax[1, ind_p].fill_between(
                np.arange((limind + 1) * 0.05, 2.01, 0.05), plte[limind + 1 :], color="darkgreen", alpha=al[ind]
            )

            ax[0, ind_p].hlines(0, 0, 2, color="gray", linewidth=1)
            ax[1, ind_p].hlines(0, 0, 2, color="gray", linewidth=1)

    legL1 = ax[0, 0].legend(
        custom_legend_L1,
        [
            r"$u_E$, S1",
            r"$u_E$, S2",
            r"$u_E$, S3",
        ],
        loc="lower left",
        labelspacing=0.2,
        handletextpad=0.1,
        borderaxespad=0.3,
    )
    if "w2" in filename:
        ax[0, 0].legend(
            custom_legend_L2,
            [
                r"$u_I$, S1",
                r"$u_I$, S2",
                r"$u_I$, S3",
            ],
            loc="lower right",
            labelspacing=0.2,
            handletextpad=0.1,
            borderaxespad=0.3,
        )
        ax[0, 0].add_artist(legL1)

    plt.savefig(os.path.join(path, filename))
    plt.show()
