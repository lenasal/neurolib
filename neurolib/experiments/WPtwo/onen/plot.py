import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
from scipy.optimize import fsolve
from sigfig import round


global label_size
label_size = 20


def set_labels():
    mpl.rcParams["xtick.labelsize"] = label_size
    mpl.rcParams["ytick.labelsize"] = label_size
    mpl.rcParams["font.size"] = label_size
    mpl.rcParams["axes.titlesize"] = label_size
    mpl.rcParams["axes.titlesize"] = label_size
    mpl.rcParams["figure.titlesize"] = label_size
    mpl.rcParams["lines.linewidth"] = 1
    return


def findroots(array):
    roots = []
    if np.amax(array) - np.amin(array) < 1e-2:
        return roots

    for t in range(1, len(array)):
        if np.sign(array[t]) != np.sign(array[t - 1]):
            if np.abs(array[t] < np.abs(array[t - 1])):
                roots.append(t)
            else:
                roots.append(t - 1)

    return roots


def findroots_w1(array):
    roots = np.arange(250, 2000, 125)
    return roots

    roots = []
    t = 1
    while t < len(array):
        if np.sign(array[t]) < -1e-2:
            if np.abs(array[t] < np.abs(array[t - 1])):
                roots.append(t)
            else:
                roots.append(t - 1)
            t += 150
        else:
            t += 1

    return roots


cmap = plt.cm.get_cmap("jet")


def plot_1n_osc(state_list, control_list, dur, dt, w_array_list, indzoom, filename):
    rows, cols = 4, 4
    time = np.arange(0, dur + dt, dt)

    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(22, 10),
        gridspec_kw={"height_ratios": [14, 14, 14, 1]},
        constrained_layout=True,
    )  # , sharey="row")

    for c in range(cols):
        states = state_list[c].copy()
        controls = control_list[c].copy()
        w_array = w_array_list[c].copy()

        maxdur = 0

        for w in range(len(w_array)):
            index = float(w / (len(w_array) - 1))
            col = cmap(index)
            ax[0, c].plot(time, states[w][0, 0, :], color=col)
            ax[1, c].plot(time, controls[w][0, 0, :], color=col)

            roots = findroots(controls[w][0, 0, :])

            if len(roots) > 4:
                ind = int(indzoom[c, w])
                i0, i1 = roots[-ind] - 10, roots[-ind + 2] + 10
                if i1 - i0 > maxdur:
                    maxdur = i1 - i0
                time_osc = np.linspace(0, (i1 - i0) * dt, i1 - i0, endpoint=True)
                ax[2, c].plot(time_osc, controls[w][0, 0, i0:i1], color=col)

        grad = np.linspace(w_array[0], w_array[-1], len(w_array))
        ax[3, c].imshow(np.vstack((grad, grad)), aspect="auto", cmap=cmap)
        ax[3, c].set_yticks([])
        ax[3, c].set_xticks(np.arange(0, len(w_array), 1))
        wlabel = []

        if "wc" in filename:
            for w in w_array:
                wlabel.append(str(round(1e3 * w, sigfigs=3)))
        else:
            for w in w_array:
                wlabel.append(str(round(w, sigfigs=3)))

        ax[2, c].set_xlim(0, maxdur * dt)

        ax[3, c].set_xticklabels(wlabel, rotation=90)

        ax[0, c].set_xlim(0, dur)
        ax[0, c].set_xticks([])
        ax[1, c].set_xlim(0, dur)
        ax[2, c].set_xlabel("Time")

        if "wc" in filename:
            ax[0, c].set_ylim(0, 0.55)
        else:
            ax[0, c].set_ylim(0, 150)

    ax[0, 0].set_ylabel("Activity")
    ax[1, 0].set_ylabel("Control")
    ax[2, 0].set_ylabel("Control")
    if "wc" in filename:
        if "w2" in filename or "w_2" in filename:
            ax[3, 0].set_ylabel(r"$10^3 w_2$")
        elif "w1D" in filename or "w_1D" in filename:
            ax[3, 0].set_ylabel(r"$10^3 w_{1D}$")
        else:
            ax[3, 0].set_ylabel(r"$10^3 w_{1}$")
    else:
        if "w2" in filename or "w_2" in filename:
            ax[3, 0].set_ylabel(r"$w_2$")

    ax[0, 0].set_title(r"Point (A): down $\rightarrow$ osc")
    ax[0, 1].set_title(r"Point (B): down $\rightarrow$ osc")
    ax[0, 2].set_title(r"Point (C): up $\rightarrow$ osc")
    ax[0, 3].set_title(r"Point (D): up $\rightarrow$ osc")
    ax[3, 0].set_title("  ", fontsize=20)

    fig.align_ylabels(ax[:, 0])
    # fig.constrained_layout()

    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    plt.savefig(filename)
    plt.show()


def plot_1n_osc_noisy(
    state,
    control,
    cost,
    state_det,
    control_det,
    cost_det,
    state_det_opt,
    control_det_opt,
    cost_det_opt,
    state_det_det,
    state_det_opt_det,
    dur,
    dt,
    sigma_array,
    M,
    filename,
):
    rows, cols = 2, 3
    time = np.arange(0, dur + dt, dt)
    roundint = 2

    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(22, 10),
        constrained_layout=True,
        sharex=True,
        sharey="row",
    )

    for s in range(len(sigma_array)):
        index = float(s / max(len(sigma_array) - 1, 1))
        col = cmap(index)

        ax[0, 0].plot(time, state[s][0, 0, :], color=col, label=r"$F=$" + str(np.around(cost[s], roundint)))
        ax[0, 1].plot(time, state_det[s][0, 0, :], color=col, label=r"$F=$" + str(np.around(cost_det[s], roundint)))
        ax[0, 2].plot(
            time, state_det_opt[s][0, 0, :], color=col, label=r"$F=$" + str(np.around(cost_det_opt[s], roundint))
        )
        ax[0, 2].plot(time, state_det_opt_det[s][0, 0, :], color=col, linestyle=":")

        ax[1, 0].plot(time, control[s][0, 0, :], color=col, label=r"$\sigma=$" + str(sigma_array[s]))
        ax[1, 2].plot(time, control_det_opt[s][0, 0, :], color=col, label=r"$\sigma=$" + str(sigma_array[s]))

    # ax[0, 1].plot(time, state_det_det[0, 0, :], color="grey")
    ax[1, 1].plot(time, control_det[0, 0, :], color="grey")

    if "wc" in filename or "WC" in filename:
        ax[0, 0].set_ylim(0, 0.6)
    else:
        ax[0, 0].set_ylim(0, 180)

    ax[0, 0].set_ylabel("Activity")
    ax[1, 0].set_ylabel("Control")
    ax[0, 0].set_xlim(0, dur)
    ax[1, 0].set_xlim(0, dur)

    for c in range(cols):
        ax[1, c].set_xlabel("Time")

    for axs in [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0]]:
        axs.legend(
            ncol=3,
            loc="upper left",
            borderpad=0.2,
            labelspacing=0.3,
            handlelength=1.0,
            handletextpad=0.4,
            borderaxespad=0.3,
            columnspacing=1.0,
        )

    set_labels()

    ax[0, 0].set_title(r"Noisy OC")
    ax[0, 1].set_title(r"Deterministic OC")
    ax[0, 2].set_title(r"OC with deterministic initialization")

    fig.align_ylabels(ax[:, 0])

    fig.suptitle(r"Point (D)" + "\n")

    plt.savefig(filename)
    plt.show()
