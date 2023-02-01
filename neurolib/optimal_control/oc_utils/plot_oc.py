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
TITLELIST = ["(A)", "(B)", "(C)", "(D)", "(E)"]


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


def ops_plotall(c_array, ymax, duration, dt, vline, filename, path=None):

    fig, ax = plt.subplots(len(c_array), 1, figsize=(8, 1.0 * len(c_array)), constrained_layout=True)

    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.5)

    time = np.arange(0, duration + dt, dt)

    for i in range(len(c_array)):
        print(i, np.amax(c_array[i][0, 0, :]), np.amin(c_array[i][0, 0, :]))

        ax[i].plot(time, c_array[i][0, 0, :], color="red")
        ax[i].plot(time, c_array[i][0, 1, :], color="blue")
        ax[i].set_ylim([-ymax, ymax])
        ax[i].set_title(str(np.round(i * 2.0 / (len(c_array) - 1), 2)) + " * pi")
        ax[i].set_xlim(0, duration)
        ax[i].vlines(vline, -ymax, ymax, color="grey")
        if i != len(c_array) - 1:
            ax[i].set_xticklabels([])

    if path != None:
        plt.savefig(os.path.join(path, filename))
    plt.show()


def ops_plotsubset(res1, dt, filename, path):

    fig, ax = plt.subplots(
        14,
        1,
        figsize=(10, 16),
        sharex="col",
        constrained_layout=True,
    )

    custom_legend = [
        Line2D([0], [0], color="red"),
        Line2D([0], [0], color="blue"),
    ]

    i_range = [5, 10, 15, 20, 25, 30, 35]

    p_str = []
    for p in i_range:
        p_str.append(r"$ p = $" + str(np.round(p / 20.0, 2)) + r" $ \cdot \pi$")

    act_range = [0.0, 1.1 * np.amax(res1["state"][1][1])]

    duration = (res1["state"][1][1].shape[2] - 1) * dt
    time = np.arange(0, duration + dt, dt)

    c_range = [
        1.1 * np.amin(res1["control"][1][:]),
        1.1 * np.amax(res1["control"][1][:]),
    ]

    for j in range(len(i_range)):
        i = i_range[j]

        ax[2 * j].plot(time, res1["state"][1][i][0, 0, :], color="red")
        ax[2 * j].plot(time, res1["state"][1][i][0, 1, :], color="blue")

        ax[2 * j].set_ylim(act_range)
        ax[2 * j].set_xlim(0, duration)

        ax[2 * j + 1].plot(time, res1["control"][1][i][0, 0, :], color="red")
        ax[2 * j + 1].plot(time, res1["control"][1][i][0, 1, :], color="blue")

        ax[2 * j + 1].set_ylim(c_range)
        ax[2 * j + 1].set_xlim(0, duration)

        ax[2 * j].vlines(duration - 100.0, act_range[0], act_range[1], color="grey")
        ax[2 * j + 1].vlines(duration - 100.0, c_range[0], c_range[1], color="grey")

        ax[2 * j].text(
            0.0,
            act_range[1],
            p_str[j],
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=fs_,
            bbox=dict(facecolor="white", alpha=1.0, pad=0.0),
        )

        ax[2 * j].set_ylabel(r"Activity")
        ax[2 * j].yaxis.set_label_coords(-0.1, 0.5)
        ax[2 * j + 1].set_ylabel(r"Control")
        ax[2 * j + 1].yaxis.set_label_coords(-0.1, 0.5)

    ax[-1].set_xlabel("Time")

    ax[0].legend(
        custom_legend,
        [r"$E$", r"$I$"],
        loc="upper right",
        labelspacing=0.1,
        handletextpad=0.1,
        handlelength=1,
        borderaxespad=0.15,
    )
    ax[1].legend(
        custom_legend,
        [r"$u_E$", r"$u_I$"],
        loc="upper right",
        labelspacing=0.1,
        handletextpad=0.1,
        handlelength=1,
        borderaxespad=0.15,
    )

    plt.savefig(os.path.join(path, filename))
    plt.show()


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

        from scipy.signal import find_peaks

        peaks0 = find_peaks(model.exc[0, :])[0]
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
        shift_ip.append(np.zeros((len(ind_range))))
        shift_im.append(np.zeros((len(ind_range))))

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

            model.params.inh_ext[0, t - shift : t + shift] += de
            model.run()
            model.params.inh_ext[0, t - shift : t + shift] -= de

            p = scipy.signal.find_peaks(model.exc[0, :])[0][-1]
            shift_ip[-1][i] = (p - p0) * dt

            model.params.inh_ext[0, t - shift : t + shift] += -de
            model.run()
            model.params.inh_ext[0, t - shift : t + shift] -= -de

            p = scipy.signal.find_peaks(model.exc[0, :])[0][-1]
            shift_im[-1][i] = (p - p0) * dt

        if norm:
            shift_ep[-1] *= 1.0 / np.amax(np.abs(shift_ep[-1]))  # 2.0 * np.pi / (period * de * dt)
            shift_em[-1] *= 1.0 / np.amax(np.abs(shift_em[-1]))
            shift_ip[-1] *= 1.0 / np.amax(np.abs(shift_ip[-1]))
            shift_im[-1] *= 1.0 / np.amax(np.abs(shift_im[-1]))
        else:
            shift_ep[-1] *= 2.0 * np.pi / period[-1]
            shift_em[-1] *= 2.0 * np.pi / period[-1]
            shift_ip[-1] *= 2.0 * np.pi / period[-1]
            shift_im[-1] *= 2.0 * np.pi / period[-1]

    return shift_ep, shift_em, shift_ip, shift_im, period


def plot_prc(resdict, dist_array, filename, path):

    setplotparams()

    npoints = len(resdict["point"])
    fig, ax = plt.subplots(1, npoints, figsize=(20, 5), sharex=True, constrained_layout=True)

    custom_legend_0 = [
        Line2D([0], [0], color="red"),
        Line2D([0], [0], color="red", linestyle=":"),
        Line2D([0], [0], color="blue"),
        Line2D([0], [0], color="blue", linestyle=":"),
    ]

    for ip in range(npoints):

        print(ip)

        for i in range(len(dist_array)):
            phase = np.linspace(0, 2, len(resdict["ep"][ip][i]))
            rmax = 1.1 * max(
                np.amax(np.abs(resdict["ep"][ip][i])),
                np.amax(np.abs(resdict["em"][ip][i])),
                np.amax(np.abs(resdict["ip"][ip][i])),
                np.amax(np.abs(resdict["im"][ip][i])),
            )

            ax[ip].plot(phase, resdict["ep"][ip][i][:], color="red", linewidth=1)
            ax[ip].plot(phase, resdict["em"][ip][i][:], color="red", linestyle=":", linewidth=1)
            ax[ip].plot(phase, resdict["ip"][ip][i][:], color="blue", linewidth=1)
            ax[ip].plot(phase, resdict["im"][ip][i][:], color="blue", linestyle=":", linewidth=1)

            if len(TITLELIST) == npoints:
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

    if type(path) != type(None):
        fig.savefig(os.path.join(path, filename))
    plt.show()


def ops_plot_traces(res1, dt, filename, path):

    i_range = [5, 10, 15, 20, 25, 30, 35]

    if "_b" in filename or "_c" in filename or "_d" in filename or "e_l2" in filename:
        i_range = [10, 15, 20, 25, 30]

    fig, ax = plt.subplots(
        2 * len(i_range),
        3,
        figsize=(24, 2.4 * len(i_range)),
        sharex="col",
        gridspec_kw={"width_ratios": [1.4, 3.0, 5.0]},
        constrained_layout=True,
    )
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)

    custom_legend = [
        Line2D([0], [0], color="red"),
        Line2D([0], [0], color="blue"),
    ]

    title_list = [
        "S1",
        "S2",
        "S3",
    ]

    dlabel = 0.25
    if "_c" in filename:
        dlabel = 0.3

    distticks = [0.2, 0.04, 0.02]
    if "_a_l2" in filename:
        distticks = [0.1, 0.02, 0.01]
    if "_b" in filename:
        distticks = [0.1, 0.02, 0.01]
    if "_c." in filename:
        distticks = [0.04, 0.004, 0.002]
    if "_c_l2" in filename:
        distticks = [0.02, 0.004, 0.004]
    if "_d." in filename:
        distticks = [0.2, 0.04, 0.02]
    if "d_l2" in filename:
        distticks = [0.4, 0.04, 0.02]
    if "_e." in filename:
        distticks = [0.1, 0.04, 0.02]
    if "e_l2" in filename:
        distticks = [0.1, 0.04, 0.02]

    p_str = []
    for p in i_range:
        p_str.append(r"$ p = $" + str(np.round(p / 20.0, 2)) + r" $ \cdot \pi$")

    act_range = [0.0, 1.1 * np.amax(res1[0]["state"][1][1])]

    for ind_sc in range(3):

        duration = (res1[ind_sc]["state"][1][1].shape[2] - 1) * dt
        time = np.arange(0, duration + dt, dt)

        clim = 1.1 * max(
            np.amax(np.abs(res1[ind_sc]["control"][1][20])),
            np.amax(np.abs(res1[ind_sc]["control"][1][25])),
            np.amax(np.abs(res1[ind_sc]["control"][1][30])),
        )

        ax[0, ind_sc].set_title(title_list[ind_sc], fontweight="bold")

        for j in range(len(i_range)):
            ax[2 * j, ind_sc].set_yticks(np.arange(0, act_range[1] + 0.2, 0.2))
            ax[2 * j + 1, ind_sc].set_yticks(np.arange(-0.4, 0.41, distticks[ind_sc]))

            i = i_range[j]

            ax[2 * j, ind_sc].plot(time, res1[ind_sc]["state"][1][i][0, 0, :], color="red")
            ax[2 * j, ind_sc].plot(time, res1[ind_sc]["state"][1][i][0, 1, :], color="blue")

            ax[2 * j, ind_sc].set_ylim(act_range)
            ax[2 * j, ind_sc].set_xlim(0, duration)

            ax[2 * j + 1, ind_sc].plot(time, res1[ind_sc]["control"][1][i][0, 0, :], color="red")
            ax[2 * j + 1, ind_sc].plot(time, res1[ind_sc]["control"][1][i][0, 1, :], color="blue")

            ax[2 * j + 1, ind_sc].set_ylim(-clim, clim)
            ax[2 * j + 1, ind_sc].set_xlim(0, duration)

            ax[2 * j, ind_sc].vlines(duration - 100.0, act_range[0], act_range[1], color="grey")
            ax[2 * j + 1, ind_sc].vlines(duration - 100.0, -clim, clim, color="grey")

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
                ax[2 * j, ind_sc].set_ylabel(r"Activity")
                ax[2 * j, ind_sc].yaxis.set_label_coords(-dlabel, 0.5)
                ax[2 * j + 1, ind_sc].set_ylabel(r"Control")
                ax[2 * j + 1, ind_sc].yaxis.set_label_coords(-dlabel, 0.5)

        ax[-1, ind_sc].set_xlabel("Time")

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

    plt.savefig(os.path.join(path, filename))
    plt.show()


def ops_ap_scenes_allpoints(reslist, periods, dt, filename, path):

    setplotparams()

    n_points = len(reslist[0]["total_cost"][0])

    markerlist = ["x", "o", "+"]
    al = [0.3, 0.5, 0.7]

    custom_legend_L1 = [
        Line2D([0], [0], color="red", linestyle="", marker=markerlist[0]),
        Line2D([0], [0], color="red", linestyle="", marker=markerlist[1], fillstyle="none"),
        Line2D([0], [0], color="red", linestyle="", marker=markerlist[2], fillstyle="none"),
    ]

    custom_legend_amp = [
        Line2D([0], [0], color="grey", linestyle="", marker=markerlist[0]),
        Line2D([0], [0], color="grey", linestyle="", marker=markerlist[1], fillstyle="none"),
        Line2D([0], [0], color="grey", linestyle="", marker=markerlist[2], fillstyle="none"),
    ]

    custom_legend_L2 = [
        Line2D([0], [0], color="blue", linestyle="", marker=markerlist[0]),
        Line2D([0], [0], color="blue", linestyle="", marker=markerlist[1], fillstyle="none"),
        Line2D([0], [0], color="blue", linestyle="", marker=markerlist[2], fillstyle="none"),
    ]

    r, c = 2, 5
    fig, ax = plt.subplots(r, c, figsize=(20, 6), sharex=True, constrained_layout=True)

    ms_ = 5

    ax[0, 0].set_ylabel(r"Amplitude")
    ax[1, 0].set_ylabel(r"Period change $\frac{\Pi}{\Pi_{(P)}}$ in %")
    ax[0, 0].yaxis.set_label_coords(-0.2, 0.5)
    ax[1, 0].yaxis.set_label_coords(-0.2, 0.5)

    for ind_p in range(c):
        ax[-1, ind_p].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
        labels = [item.get_text() for item in ax[-1, ind_p].get_xticklabels()]
        labels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
        ax[-1, ind_p].set_xticklabels(labels)

        ax[-1, ind_p].set_xlabel(r"Phase shift")

        ax[0, ind_p].set_title(TITLELIST[ind_p], fontweight="bold")

        ax[0, ind_p].set_xlim([0, 2])
        ax[1, ind_p].set_xlim([0, 2])

        plte = []
        plt0 = []

        for ind in range(3):

            res = reslist[ind_p * 3 + ind]

            for i_shift in range(n_points):
                period_shift = i_shift * 2.0 / (n_points - 1)

                maxe, mine = np.amax(res["control"][1][i_shift][0, 0, :]), np.amin(res["control"][1][i_shift][0, 0, :])
                plotres = maxe
                if np.abs(maxe) < np.abs(mine):
                    plotres = mine

                if ind_p == 4 and ind == 0 and i_shift == 30:
                    print(maxe, mine, plotres)
                    print(np.argmin(res["control"][1][i_shift][0, 0, :]))
                    print(res["control"][1][i_shift][0, 0, 200:210])

                if ind_p in [0, 1, 2, 3] or "l2" in filename:

                    ax[0, ind_p].plot(
                        period_shift,
                        plotres,
                        marker=markerlist[ind],
                        fillstyle="none",
                        color="red",
                        markersize=ms_,
                    )

                plt0.append(np.abs(plotres))

                maxi, mini = np.amax(res["control"][1][i_shift][0, 1, :]), np.amin(res["control"][1][i_shift][0, 1, :])
                plotres = maxi
                if np.abs(maxi) < np.abs(mini):
                    plotres = mini

                if ind_p in [4] or "l2" in filename:

                    ax[0, ind_p].plot(
                        period_shift,
                        plotres,
                        marker=markerlist[ind],
                        fillstyle="none",
                        color="blue",
                        markersize=ms_,
                    )

                plt0.append(np.abs(plotres))

                peaks_e = scipy.signal.find_peaks(np.abs(res["state"][1][i_shift][0, 0, :]))[0]
                periods_e = np.zeros((len(peaks_e) - 1))
                periodsdev_e = periods_e.copy()

                if i_shift in [0, 40]:
                    continue

                for i in range(len(peaks_e) - 1):
                    periods_e[i] = (peaks_e[i + 1] - peaks_e[i]) * dt
                    periodsdev_e[i] = ((periods_e[i] / periods[ind_p]) - 1.0) * 100.0

                plte.append(np.amax(np.abs(periodsdev_e)))

                if np.amin(periodsdev_e) <= -np.amax(periodsdev_e):
                    ax[1, ind_p].plot(
                        period_shift,
                        np.amin(periodsdev_e),
                        marker=markerlist[ind],
                        fillstyle="none",
                        color="grey",
                        markersize=ms_,
                    )
                else:
                    ax[1, ind_p].plot(
                        period_shift,
                        np.amax(periodsdev_e),
                        marker=markerlist[ind],
                        fillstyle="none",
                        color="grey",
                        markersize=ms_,
                    )

        ylim = np.amax(plt0) * 1.1
        ax[0, ind_p].set_ylim(-ylim, ylim)
        ylim = np.amax(plte) * 1.1
        ax[1, ind_p].set_ylim(-ylim, ylim)

        ax[0, ind_p].hlines(0, 0, 2, color="gray", linewidth=1)
        ax[1, ind_p].hlines(0, 0, 2, color="gray", linewidth=1)

    legL1 = ax[0, 0].legend(
        custom_legend_L1,
        [
            r"$a_E$, S1",
            r"$a_E$, S2",
            r"$a_E$, S3",
        ],
        loc="upper left",
        labelspacing=0.1,
        handletextpad=0.0,
        borderaxespad=0.2,
        handlelength=1,
    )
    ax[0, 4].legend(
        custom_legend_L2,
        [
            r"$a_I$, S1",
            r"$a_I$, S2",
            r"$a_I$, S3",
        ],
        loc="upper right",
        labelspacing=0.1,
        handletextpad=0.0,
        borderaxespad=0.2,
        handlelength=1,
    )
    ax[1, 0].legend(
        custom_legend_amp,
        [
            r"S1",
            r"S2",
            r"S3",
        ],
        loc="upper left",
        labelspacing=0.1,
        handletextpad=0.0,
        borderaxespad=0.2,
        handlelength=1,
    )
    ax[0, 0].add_artist(legL1)

    plt.savefig(os.path.join(path, filename))
    plt.show()
