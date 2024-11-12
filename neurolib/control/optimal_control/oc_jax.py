import jax
import jax.numpy as jnp
import numpy as np
import copy
from neurolib.models.jax.wc import WCModel
from neurolib.models.jax.wc.timeIntegration import timeIntegration_args, timeIntegration_elementwise
from neurolib.optimize.autodiff.wc_optimizer import args_names

import logging
from neurolib.control.optimal_control.oc import getdefaultweights


def update_control_with_limit(N, dim_in, T, control, step, gradient, u_max):
    return control + step * gradient


class OcWc:
    def __init__(self, model, target, opt_params=["exc_ext"]):
        self.model = copy.deepcopy(model)
        self.target = target
        self.opt_params = opt_params
        self.weights = getdefaultweights()
        self.M = 1

        args_values = timeIntegration_args(self.model.params)
        self.args = dict(zip(args_names, args_values))

        self.loss = self.get_loss()
        self.compute_gradient = jax.jit(jax.grad(self.loss))
        self.T = len(self.args["t"]) + 1
        self.startind = self.model.getMaxDelay() + 1
        self.control = jnp.zeros((self.model.params.N, self.T), dtype=float)  # TODO: depend on opt_params

        self.step = 10.0  # Initial step size in first optimization iteration.
        self.count_noisy_step = 10
        self.count_step = 30

        self.factor_down = 0.5  # Factor for adaptive step size reduction.
        self.factor_up = 2.0  # Factor for adaptive step size increment.

        self.cost_history = []
        self.step_sizes_history = []
        self.step_sizes_loops_history = []

        self.dim_vars = len(self.model.state_vars)
        self.dim_in = 1
        self.dim_out = len(self.model.output_vars)
        self.maximum_control_strength = 0

        self.print_array = []
        self.zero_step_encountered = False  # deterministic gradient descent cannot further improve
        self.channelwise_optimization = False
        self.grad_method = 0
        self.factors = jnp.array([1.0, 1.0])

    def simulate(self, control):
        args_local = self.args.copy()
        args_local.update(dict(zip(self.opt_params, [control])))
        return timeIntegration_elementwise(**args_local)

    def get_loss(self):
        @jax.jit
        def loss(control):
            t, exc, inh, exc_ou, inh_ou = self.simulate(control)
            return self.compute_total_cost(control, exc[:, self.startind - 1 :])

        return loss

    def accuracy_cost(self, exc):
        accuracy_cost = 0.0
        if self.weights["w_p"] != 0.0:
            accuracy_cost += self.weights["w_p"] * 0.5 * self.model.params.dt * jnp.sum((exc - self.target) ** 2)
        if self.weights["w_cc"] != 0.0:
            accuracy_cost += self.weights["w_cc"] * self.compute_cc_cost(exc)
        return accuracy_cost

    def control_strength_cost(self, control):
        control_strength_cost = 0.0
        if self.weights["w_2"] != 0.0:
            control_strength_cost += self.weights["w_2"] * 0.5 * self.model.params.dt * jnp.sum(control**2)
        if self.weights["w_1D"] != 0.0:
            control_strength_cost += self.compute_ds_cost(control)
        return control_strength_cost

    def compute_ds_cost(self, control):
        return jnp.sum(jnp.sqrt(jnp.sum(control**2, axis=1) * self.model.params.dt), axis=0)

    def compute_cc_cost(self, exc):
        xmean = jnp.stack([jnp.mean(exc, axis=1)] * exc.shape[1]).T
        xstd = jnp.stack([jnp.std(exc, axis=1)] * exc.shape[1]).T
        N = self.model.params.N

        xvec = (exc - xmean) / xstd

        costmat = jnp.einsum("ik,jk->ijk", xvec, xvec)
        diag = jnp.einsum("ij,ij->j", xvec, xvec)
        cost = np.sum(np.sum(np.sum(costmat, axis=0), axis=0) - diag, axis=0) * self.model.params.dt / 2.0
        cost *= -2.0 / (N * (N - 1) * (self.T) * self.model.params.dt)

        return cost

    def compute_total_cost(self, control, exc):
        """Compute the total cost as weighted sum precision of all contributing cost terms.
        :rtype: float
        """
        accuracy_cost = self.accuracy_cost(jnp.array(exc))
        control_strength_cost = self.control_strength_cost(control)
        return accuracy_cost + control_strength_cost

    def optimize_deterministic(self, n_max_iterations):
        """Compute the optimal control signal for noise averaging method 0 (deterministic, M=1).

        :param n_max_iterations: maximum number of iterations of gradient descent
        :type n_max_iterations: int
        """

        # (I) forward simulation
        t, exc, inh, exc_ou, inh_ou = self.simulate(self.control)  # yields x(t)

        cost = self.compute_total_cost(self.control, exc[:, self.startind - 1 :])
        print(f"Cost in iteration 0: %s" % (cost))
        if len(self.cost_history) == 0:  # add only if control model has not yet been optimized
            self.cost_history.append(cost)

        for i in range(1, n_max_iterations + 1):
            self.gradient = self.compute_gradient(self.control)

            if self.channelwise_optimization:
                self.step_size_nv(-self.gradient)
            else:
                self.step_size(-self.gradient)
            t, exc, inh, exc_ou, inh_ou = self.simulate(self.control)

            cost = self.compute_total_cost(self.control, exc[:, self.startind - 1 :])
            if i in self.print_array:
                print(f"Cost in iteration %s: %s" % (i, cost))
            self.cost_history.append(cost)

            if self.zero_step_encountered:
                print(f"Converged in iteration %s with cost %s" % (i, cost))
                break

        print(f"Final cost : %s" % (cost))

    def step_size(self, cost_gradient):
        """Adaptively choose a step size for control update.

        :param cost_gradient:   N x V x T gradient of the total cost wrt. control.
        :type cost_gradient:    np.ndarray

        :return:    Step size that got multiplied with the 'cost_gradient'.
        :rtype:     float
        """
        if self.M > 1:
            noisy = True
        else:
            noisy = False

        t, exc, inh, exc_ou, inh_ou = self.simulate(self.control)
        if noisy:
            cost0 = self.compute_cost_noisy(self.M)
        else:
            cost0 = self.compute_total_cost(
                self.control, exc[:, self.startind - 1 :]
            )  # Current cost without updating the control according to the "cost_gradient".

        step = self.step  # Load step size of last optimization-iteration as initial guess.

        control0 = self.control  # Memorize unchanged control throughout step-size computation.

        while True:  # Reduce the step size, if numerical instability occurs in the forward-simulation.
            # inplace updating of models control bc. forward-sim relies on models parameters
            self.control = update_control_with_limit(
                self.model.params.N, self.dim_in, self.T, control0, step, cost_gradient, self.maximum_control_strength
            )
            ##self.update_input()

            # Input signal might be too high and produce diverging values in simulation.
            t, exc, inh, exc_ou, inh_ou = self.simulate(self.control)

            # TODO
            """
            if np.isnan(self.get_xs()).any():  # Detect numerical instability due to too large control update.
                step *= self.factor_down**2  # Double the step for faster search of stable region.
                self.step = step
                print(f"Diverging model output, decrease step size to {step}.")
            else:
                break
            """
            break

        if noisy:
            cost = self.compute_cost_noisy(self.M)
        else:
            cost = self.compute_total_cost(
                self.control, exc[:, self.startind - 1 :]
            )  # Cost after applying control update according to gradient with first valid
        # step size (numerically stable).
        # print(cost, cost0)
        if (
            cost > cost0
        ):  # If the cost choosing the first (stable) step size is no improvement, reduce step size by bisection.
            step, counter = self.decrease_step(cost, cost0, step, control0, self.factor_down, cost_gradient)

        elif (
            cost < cost0
        ):  # If the cost is improved with the first (stable) step size, search for larger steps with even better
            # reduction of cost.

            step, counter = self.increase_step(cost, cost0, step, control0, self.factor_up, cost_gradient)

        else:  # Remark: might be included as part of adaptive search for further improvement.
            step = 0.0  # For later analysis only.
            counter = 0
            self.zero_step_encountered = True

        self.step = step  # Memorize the last step size for the next optimization step with next gradient.

        self.step_sizes_loops_history.append(counter)
        self.step_sizes_history.append(step)

        return step, counter, cost

    def decrease_step(self, cost, cost0, step, control0, factor_down, cost_gradient):
        """Find a step size which leads to improved cost given the gradient. The step size is iteratively decreased.
        The control-inputs are updated in place according to the found step size via the
        "####self.update_input()" call.

        :param cost:    Cost after applying control update according to gradient with first valid step size (numerically
                        stable).
        :type cost:     float
        :param cost0:   Cost without updating the control.
        :type cost0:    float
        :param step:    Step size initial to the iterative decreasing.
        :type step:     float
        :param control0:    The unchanged control signal.
        :type control0:     np.ndarray N x V x T
        :param factor_down:  Factor the step size is scaled with in each iteration until cost is improved.
        :type factor_down:   float
        :param cost_gradient:   Gradient of the total cost wrt. the control signal.
        :type cost_gradient:    np.ndarray of shape N x V x T

        :return:    The selected step size and the count-variable how often step-adjustment-loop was executed.
        :rtype:     tuple[float, int]
        """
        if self.M > 1:
            noisy = True
        else:
            noisy = False

        counter = 0

        while cost > cost0:  # Decrease the step size until first step size is found where cost is improved.
            step *= factor_down  # Decrease step size.
            counter += 1
            # print(step, cost, cost0)

            # Inplace updating of models control bc. forward-sim relies on models parameters.
            self.control = update_control_with_limit(
                self.model.params.N, self.dim_in, self.T, control0, step, cost_gradient, self.maximum_control_strength
            )
            # self.update_input()

            # Simulate with control updated according to new step and evaluate cost.
            t, exc, inh, exc_ou, inh_ou = self.simulate(self.control)

            if noisy:
                cost = self.compute_cost_noisy(self.M)
            else:
                cost = self.compute_total_cost(self.control, exc[:, self.startind - 1 :])

            if counter == self.count_step:  # Exit if the maximum search depth is reached without improvement of
                # cost.
                step = 0.0  # For later analysis only.
                self.control = update_control_with_limit(
                    self.model.params.N,
                    self.dim_in,
                    self.T,
                    control0,
                    0.0,
                    jnp.zeros_like(control0, dtype=float),
                    self.maximum_control_strength,
                )
                # self.update_input()

                self.zero_step_encountered = True
                break

        return step, counter

    def increase_step(self, cost, cost0, step, control0, factor_up, cost_gradient):
        """Find the largest step size which leads to the biggest improvement of cost given the gradient. The step size is
        iteratively increased. The control-inputs are updated in place according to the found step size via the
        "self.update_input()" call.

        :param cost:    Cost after applying control update according to gradient with first valid step size (numerically
                        stable).
        :type cost:     float
        :param cost0:   Cost without updating the control.
        :type cost0:    float
        :param step:    Step size initial to the iterative decreasing.
        :type step:     float
        :param control0:    The unchanged control signal.
        :type control0:     np.ndarray N x V x T
        :param factor_up:  Factor the step size is scaled with in each iteration while the cost keeps improving.
        :type factor_up:   float
        :param cost_gradient:   Gradient of the total cost wrt. the control signal.
        :type cost_gradient:    np.ndarray of shape N x V x T

        :return:    The selected step size and the count-variable how often step-adjustment-loop was executed.
        :rtype:     tuple[float, int]
        """
        if self.M > 1:
            noisy = True
        else:
            noisy = False

        cost_prev = cost0
        counter = 0

        while cost < cost_prev:  # Increase the step size as long as the cost is improving.
            step *= factor_up
            counter += 1

            # Inplace updating of models control bc. forward-sim relies on models parameters
            self.control = update_control_with_limit(
                self.model.params.N, self.dim_in, self.T, control0, step, cost_gradient, self.maximum_control_strength
            )
            # self.update_input()

            t, exc, inh, exc_ou, inh_ou = self.simulate(self.control)
            # TODO
            """
            if np.isnan(self.get_xs()).any():  # Go back to last step (that was numerically stable and improved cost)
                # and exit.
                logging.info("Increasing step encountered NAN.")
                step /= factor_up  # Undo the last step update by inverse operation.
                self.control = update_control_with_limit(
                    self.model.params.N, self.dim_in, self.T, control0, step, cost_gradient, self.maximum_control_strength
                )
                #self.update_input()
                break

            else:
            """
            if noisy:
                cost = self.compute_cost_noisy(self.M)
            else:
                cost = self.compute_total_cost(self.control, exc[:, self.startind - 1 :])

            if cost > cost_prev:  # If the cost increases: go back to last step (that resulted in best cost until
                # then) and exit.
                step /= factor_up  # Undo the last step update by inverse operation.
                self.control = update_control_with_limit(
                    self.model.params.N,
                    self.dim_in,
                    self.T,
                    control0,
                    step,
                    cost_gradient,
                    self.maximum_control_strength,
                )
                # self.update_input()
                break

            else:
                cost_prev = cost  # Memorize cost with this step size for comparison in next step-update.

            if counter == self.count_step:
                # Terminate step size search at count limit, exit with the best performing step size.
                break

        return step, counter

    def step_size_nv(self, cost_gradient):
        control0 = self.control.copy()
        step0 = self.step

        stepall, counterall, costall = self.step_size(cost_gradient)
        zerostepall = self.zero_step_encountered
        self.zero_step_encountered = False

        minind = -1
        mincost = costall

        energy = np.sum(control0**2, axis=1)
        energy_sort = np.argsort(energy)
        searchind = 5

        steps = np.zeros((searchind))
        costs = steps.copy()
        counters = steps.copy()
        zerosteps = steps.copy()

        for ind, n in enumerate(energy_sort[-searchind:]):

            # print("search step ", n)
            self.control = control0.copy()
            self.step = step0
            grad = np.zeros((cost_gradient.shape))
            grad[n, :] = cost_gradient[n, :]
            steps[ind], counters[ind], costs[ind] = self.step_size(grad)

            if costs[ind] < mincost:
                mincost = costs[ind]
                minind = ind
            if self.zero_step_encountered:
                zerosteps[ind] = 1
                self.zero_step_encountered = False

        if zerostepall and np.amin(zerosteps) >= 1.0:
            # all options ended with maximum counter
            step, counter = 0.0, self.count_step
            self.zero_step_encountered = True
            grad = cost_gradient.copy()

        else:
            if minind == -1:
                grad = cost_gradient.copy()
                step, counter, cost = stepall, counterall, costall
                self.zero_step_encountered = False
            else:
                grad = np.zeros((cost_gradient.shape))
                grad[minind, :] = cost_gradient[minind, :]
                step, counter, cost = (
                    steps[minind],
                    counters[minind],
                    costs[minind],
                )
                grad = jnp.asarray(grad, dtype=float)
                self.zero_step_encountered = False

        self.step = step  # Memorize the last step size for the next optimization step with next gradient.
        self.step_sizes_loops_history.append(counter)
        self.step_sizes_history.append(step)

        self.control = update_control_with_limit(
            self.model.params.N,
            self.dim_in,
            self.T,
            control0,
            step,
            grad,
            self.maximum_control_strength,
        )
