import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import fsolve


def findroots(array):
    roots = []
    for t in range(1, len(array)):
        if np.sign(array[t]) != np.sign(array[t - 1]):
            if np.abs(array[t] < np.abs(array[t - 1])):
                roots.append(t)
            else:
                roots.append(t - 1)

    return roots


cmap = plt.cm.get_cmap("jet")


def plot_1n_osc(state_list, control_list, dur, dt, w2array_list, indzoom, filename):
    rows, cols = 4, 4
    time = np.arange(0, dur + dt, dt)

    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(24, 12),
        gridspec_kw={"height_ratios": [10, 10, 10, 1]},
        constrained_layout=True,
    )  # , sharey="row")

    for c in range(cols):
        states = state_list[c].copy()
        controls = control_list[c].copy()
        w2_array = w2array_list[c].copy()
        for w in range(len(w2_array)):
            index = float(w / (len(w2_array) - 1))
            col = cmap(index)
            ax[0, c].plot(time, states[w][0, 0, :], color=col)
            ax[1, c].plot(time, controls[w][0, 0, :], color=col)

            roots = findroots(controls[w][0, 0, :])
            maxdur = 0

            if len(roots) > 4:
                ind = int(indzoom[c, w])
                i0, i1 = roots[-ind] - 10, roots[-ind + 2] + 10
                if i1 - i0 > maxdur:
                    maxdur = i1 - i0
                time_osc = np.linspace(0, (i1 - i0) * dt, i1 - i0, endpoint=True)
                # print(i0, i1, time_osc.shape)
                ax[2, c].plot(time_osc, controls[w][0, 0, i0:i1], color=col)

        grad = np.linspace(w2_array[0], w2_array[-1], len(w2_array))
        ax[3, c].imshow(np.vstack((grad, grad)), aspect="auto", cmap=cmap)
        ax[3, c].set_yticks([])
        ax[3, c].set_xticks(np.arange(0, 10, 1))
        ax[3, c].set_xticklabels(
            np.around(1e3 * w2_array, 3),
            rotation=90,
        )

        ax[0, c].set_xlim(0, dur)
        ax[0, c].set_xticks([])
        ax[1, c].set_xlim(0, dur)
        ax[2, c].set_xlim(0, maxdur * dt)
        ax[2, c].set_xlabel("Time [ms]")

        ax[0, c].set_ylim(0, 0.5)

    ax[0, 0].set_ylabel("Activity")
    ax[1, 0].set_ylabel("Control")
    ax[2, 0].set_ylabel("Control")
    ax[3, 0].set_ylabel(r"$w_2$")

    ax[0, 0].set_title(r"Point (A): down $\rightarrow$ osc")
    ax[0, 1].set_title(r"Point (B): down $\rightarrow$ osc")
    ax[0, 2].set_title(r"Point (C): up $\rightarrow$ osc")
    ax[0, 3].set_title(r"Point (D): up $\rightarrow$ osc")
    ax[3, 0].set_title("  ")

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
    roundint = 3

    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(24, 12),
        constrained_layout=True,
        sharex=True,
        sharey="row",
    )

    for s in range(len(sigma_array)):
        index = float(s / (len(sigma_array) - 1))
        col = cmap(index)

        ax[0, 0].plot(time, state[s][0, 0, :], color=col, label=r"$F=$" + str(np.around(cost[s], roundint)))
        ax[0, 1].plot(time, state_det[s][0, 0, :], color=col, label=r"$F=$" + str(np.around(cost_det[s], roundint)))
        ax[0, 2].plot(
            time, state_det_opt[s][0, 0, :], color=col, label=r"$F=$" + str(np.around(cost_det_opt[s], roundint))
        )
        ax[0, 2].plot(time, state_det_opt_det[s][0, 0, :], color=col, linestyle=":")

        ax[1, 0].plot(time, control[s][0, 0, :], color=col, label=r"$\sigma=$" + str(sigma_array[s]))
        ax[1, 2].plot(time, control_det_opt[s][0, 0, :], color=col, label=r"$\sigma=$" + str(sigma_array[s]))

    ax[0, 1].plot(time, state_det_det[0, 0, :], color="grey")
    ax[1, 1].plot(time, control_det[0, 0, :], color="grey")

    ax[0, 0].set_ylim(0, 0.6)

    ax[0, 0].set_ylabel("Activity")
    ax[1, 0].set_ylabel("Control")
    ax[0, 0].set_xlim(0, dur)
    ax[1, 0].set_xlim(0, dur)

    for c in range(cols):
        ax[1, c].set_xlabel("Time [ms]")

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

    ax[0, 0].set_title(r"Noisy OC")
    ax[0, 1].set_title(r"Deterministic OC")
    ax[0, 2].set_title(r"OC with deterministic initialization")

    fig.align_ylabels(ax[:, 0])

    plt.savefig(filename)
    plt.show()
