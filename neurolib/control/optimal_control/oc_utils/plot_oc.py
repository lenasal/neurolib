import matplotlib.pyplot as plt
import numpy as np

colors = ["red", "blue", "green", "orange"]


def plot_oc_singlenode(
    duration,
    dt,
    state,
    target,
    control,
    orig_input,
    cost_array=(),
    plot_state_vars=[0, 1],
    plot_control_vars=[0, 1],
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
    rows = 3
    if cost_array == None:
        rows = 2
    fig, ax = plt.subplots(rows, 1, figsize=(8, 2 * rows), constrained_layout=True)

    # Plot the target (dashed line) and unperturbed activity
    t_array = np.arange(0, duration + dt, dt)

    for v in plot_state_vars:
        ax[0].plot(t_array, state[0, v, :], label="state var " + str(v), color=colors[v])
        ax[0].plot(t_array, target[0, v, :], linestyle="dashed", label="target var " + str(v), color=colors[v])
    ax[0].legend(loc="upper right")
    ax[0].set_title("Activity without stimulation and target activity")

    # Plot the target control signal (dashed line) and "initial" zero control signal
    for v in plot_control_vars:
        ax[1].plot(t_array, control[0, v, :], label="stimulation var " + str(v), color=colors[v])
        ax[1].plot(t_array, orig_input[0, v, :], linestyle="dashed", label="input var " + str(v), color=colors[v])
    ax[1].legend(loc="upper right")
    ax[1].set_title("Active stimulation and input stimulation")

    if cost_array != None:
        ax[2].plot(cost_array)
        ax[2].set_title("Cost throughout optimization.")

    if savepath != None:
        plt.savefig(savepath)

    plt.show()


def plot_oc_network(
    N,
    duration,
    dt,
    state,
    target,
    control,
    orig_input,
    cost_array=(),
    step_array=(),
    plot_state_vars=[0, 1],
    plot_control_vars=[0, 1],
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
    rows = 2
    if cost_array is not None:
        rows = 3
    fig, ax = plt.subplots(rows, N, figsize=(12, 3 * rows), constrained_layout=True)

    for n in range(N):
        for v in plot_state_vars:
            ax[0, n].plot(t_array, state[n, v, :], label="state var " + str(v), color=colors[v])
            ax[0, n].plot(t_array, target[n, v, :], linestyle="dashed", label="target var " + str(v), color=colors[v])
        # ax[0, n].legend(loc="upper right")
        ax[0, n].set_title(f"Activity and target, node %s" % (n))

        # Plot the target control signal (dashed line) and "initial" zero control signal
        for v in plot_control_vars:
            ax[1, n].plot(t_array, control[n, v, :], label="stimulation var " + str(v), color=colors[v])
            ax[1, n].plot(
                t_array, orig_input[n, v, :], linestyle="dashed", label="input var " + str(v), color=colors[v]
            )
        # ax[1, n].legend(loc="upper right")
        ax[1, n].set_title(f"Stimulation and input, node %s" % (n))

    for n in range(N):
        ax[0].plot(t_array, state[n, 0, :], label=f"Node %s" % (n))
        ax[1].plot(t_array, control[n, 0, :], label=f"Node %s" % (n))
        if isinstance(target, float):
            k = int(np.ceil(duration / target))
            for k_ in range(2, k, 2):
                ax[0].axvspan(duration - k_ * target, duration - (k_ - 1) * target, color="lightgrey", alpha=0.3)

    if N < 5:
        ax[0].legend(loc="upper left", fontsize=FS)
    ax[0].set_ylabel("Activity", fontsize=FS)
    # ax[1].legend(loc="upper left", fontsize=FS)
    ax[1].set_ylabel("Control", fontsize=FS)

    ax[0].tick_params(axis="both", which="major", labelsize=FS)
    ax[1].tick_params(axis="both", which="major", labelsize=FS)
    ax[0].set_xticks([])
    ax[0].set_xlim(0, duration)
    ax[0].set_ylim(0, 0.6)
    ax[1].set_xlim(0, duration)
    ax[1].set_xlabel("Time", fontsize=FS)

    if filename is not None:
        plt.savefig(filename, dpi=300)

    plt.show()


def plot_oc_nw(
    N,
    duration,
    dt,
    state,
    target,
    control,
    filename=None,
):
    t_array = np.arange(0, duration + dt, dt)
    rows = 2
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), constrained_layout=True)

    for n in range(N):
        ax[0].plot(t_array, state[n, 0, :], label=f"Node %s" % (n))
        ax[1].plot(t_array, control[n, 0, :], label=f"Node %s" % (n))
        if isinstance(target, float):
            k = int(np.ceil(duration / target))
            for k_ in range(2, k, 2):
                ax[0].axvspan(duration - k_ * target, duration - (k_ - 1) * target, color="lightgrey", alpha=0.3)

    if N < 5:
        ax[0].legend(loc="upper left", fontsize=FS)
    ax[0].set_ylabel("Activity", fontsize=FS)
    # ax[1].legend(loc="upper left", fontsize=FS)
    ax[1].set_ylabel("Control", fontsize=FS)

    ax[0].tick_params(axis="both", which="major", labelsize=FS)
    ax[1].tick_params(axis="both", which="major", labelsize=FS)
    ax[0].set_xticks([])
    ax[0].set_xlim(0, duration)
    ax[1].set_xlim(0, duration)
    ax[1].set_xlabel("Time", fontsize=FS)

    if filename is not None:
        plt.savefig(filename, dpi=300)

    plt.show()
