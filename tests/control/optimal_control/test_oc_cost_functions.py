import unittest
import numpy as np
from neurolib.control.optimal_control import cost_functions
from neurolib.control.optimal_control.oc import getdefaultweights
from neurolib.utils.stimulus import ZeroInput
from neurolib.models.fhn import FHNModel
from neurolib.control.optimal_control import oc_fhn
from scipy.signal import hilbert

global DT
DT = 1.0


class TestCostFunctions(unittest.TestCase):
    @staticmethod
    def get_arbitrary_array():
        return np.array([[1, -10, 5.555], [-1, 3.3333, 9]])[
            np.newaxis, :, :
        ]  # an arbitrary vector with positive and negative entries

    def test_precision_cost_full_timeseries(self):
        print(" Test precision cost full timeseries")
        N = 1
        cost_mat = np.ones((N, 2))
        dt = 0.1
        x_target = self.get_arbitrary_array()
        interval = (0, x_target.shape[2])
        weights = getdefaultweights()

        self.assertAlmostEqual(
            cost_functions.accuracy_cost(x_target, hilbert(x_target), x_target, 0.0, weights, cost_mat, dt, interval),
            0,
            places=8,
        )  # target and simulation coincide

        x_sim = np.copy(x_target)
        x_sim[:, 0] = -x_sim[:, 0]  # create setting where result depends only on this first entries
        self.assertAlmostEqual(
            cost_functions.accuracy_cost(x_sim, hilbert(x_sim), x_target, 0.0, weights, cost_mat, dt, interval),
            weights["w_p"] / 2 * np.sum((2 * x_target[:, 0]) ** 2) * dt,
            places=8,
        )

    def test_precision_cost_nodes_channels(self):
        print(" Test precision cost full timeseries for node and channel selection.")
        N = 2
        x_target0 = self.get_arbitrary_array()
        x_target1 = 2.0 * self.get_arbitrary_array()
        target = np.concatenate((x_target0, x_target1), axis=0)
        cost_mat = np.zeros((N, 2))
        dt = 0.1
        interval = (0, target.shape[2])
        zerostate = np.zeros((target.shape))
        weights = getdefaultweights()

        self.assertAlmostEqual(
            cost_functions.accuracy_cost(target, hilbert(target), zerostate, 0.0, weights, cost_mat, dt, interval),
            0.0,
            places=8,
        )  # no cost if precision matrix is zero

        for i in range(N):
            for j in range(N):
                cost_mat[i, j] = 1
                result = weights["w_p"] * 0.5 * sum((target[i, j, :] ** 2)) * dt
                self.assertAlmostEqual(
                    cost_functions.accuracy_cost(
                        target, hilbert(target), zerostate, 0.0, weights, cost_mat, dt, interval
                    ),
                    result,
                    places=8,
                )
                cost_mat[i, j] = 0

    def test_derivative_precision_cost_full_timeseries(self):
        print(" Test precision cost derivative full timeseries")
        weights = getdefaultweights()
        N = 1
        cost_mat = np.ones((N, 2))
        x_target = self.get_arbitrary_array()
        x_sim = np.copy(x_target)
        x_sim[0, :, 0] = -x_sim[0, :, 0]  # create setting where result depends only on this first entries
        interval = (0, x_target.shape[2])

        derivative_p_c = cost_functions.derivative_accuracy_cost(
            x_target, hilbert(x_target), x_sim, 0.0, weights, cost_mat, DT, interval
        )

        self.assertTrue(np.all(derivative_p_c[0, :, 1::] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 0] == 2 * weights["w_p"] * x_target[0, :, 0]))

    def test_derivative_precision_cost_full_timeseries_nodes_channels(self):
        print(" Test precision cost derivative full timeseries for node and channel selection")
        weights = getdefaultweights()
        N = 2
        x_target0 = self.get_arbitrary_array()
        x_target1 = 2.0 * self.get_arbitrary_array()
        target = np.concatenate((x_target0, x_target1), axis=0)
        cost_mat = np.zeros((N, 2))  # ToDo: overwrites previous definition, bug?
        zerostate = np.zeros((target.shape))
        interval = (0, target.shape[2])

        derivative_p_c = cost_functions.derivative_accuracy_cost(
            target, hilbert(target), zerostate, 0.0, weights, cost_mat, DT, interval
        )
        self.assertTrue(np.all(derivative_p_c == 0))

        for i in range(N):
            for j in range(N):
                cost_mat[i, j] = 1
                derivative_p_c = cost_functions.derivative_accuracy_cost(
                    target, hilbert(target), zerostate, 0.0, weights, cost_mat, DT, interval
                )
                result = weights["w_p"] * np.einsum("ijk,ij->ijk", target, cost_mat)
                self.assertTrue(np.all(derivative_p_c - result == 0))
                cost_mat[i, j] = 0

    def test_precision_cost_in_interval(self):
        """This test is analogous to the 'test_precision_cost'. However, the signal is repeated twice, but only
        the second interval is to be taken into account.
        """
        print(" Test precision cost in time interval")
        N = 1
        cost_mat = np.ones((N, 2))
        dt = 0.1
        x_target = np.concatenate((self.get_arbitrary_array(), self.get_arbitrary_array()), axis=2)
        x_sim = -np.copy(x_target)
        interval = (3, x_target.shape[2])
        weights = getdefaultweights()
        precision_cost = cost_functions.accuracy_cost(
            x_target, hilbert(x_target), x_sim, 0.0, weights, cost_mat, dt, interval
        )
        # Result should only depend on second half of the timeseries.

        self.assertAlmostEqual(
            precision_cost, weights["w_p"] / 2 * np.sum((2 * x_target[0, :, 3:]) ** 2) * dt, places=8
        )

    def test_derivative_precision_cost_in_interval(self):
        """This test is analogous to the 'test_derivative_precision_cost'. However, the signal is repeated twice, but
        only the second interval is to be taken into account.
        """
        print(" Test precision cost derivative in time interval")
        weights = getdefaultweights()
        N = 1
        cost_mat = np.ones((N, 2))
        x_target = np.concatenate((self.get_arbitrary_array(), self.get_arbitrary_array()), axis=2)
        x_sim = np.copy(x_target)
        x_sim[0, :, 3] = -x_sim[0, :, 3]  # create setting where result depends only on this first entries
        interval = (3, x_target.shape[2])
        derivative_p_c = cost_functions.derivative_accuracy_cost(
            x_target, hilbert(x_target), x_sim, 0.0, weights, cost_mat, DT, interval
        )

        self.assertTrue(np.all(derivative_p_c[0, :, 0:3] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 4::] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 3] == 2 * weights["w_p"] * x_target[0, :, 3]))

    def test_L2_cost(self):
        print(" Test L2 cost")
        dt = 0.1
        reference_result = 112.484456945 * dt
        weights = getdefaultweights()
        weights["w_2"] = 1.0
        u = self.get_arbitrary_array()
        L2_cost = cost_functions.control_strength_cost(u, weights, dt)
        self.assertAlmostEqual(L2_cost, reference_result, places=8)

    def test_derivative_L2_cost(self):
        print(" Test L2 cost derivative")
        u = self.get_arbitrary_array()
        desired_output = u
        self.assertTrue(np.all(cost_functions.derivative_L2_cost(u) == desired_output))

    def test_L1_cost(self):
        print(" Test L1 cost")
        dt = 0.1
        reference_result = 29.8883 * dt
        weights = getdefaultweights()
        weights["w_1"] = 1.0
        u = self.get_arbitrary_array()
        L1_cost = cost_functions.control_strength_cost(u, weights, dt)
        self.assertAlmostEqual(L1_cost, reference_result, places=8)

    def test_derivative_L1_cost(self):
        print(" Test L1 cost derivative")
        u = self.get_arbitrary_array()
        desired_output = np.sign(u)
        self.assertTrue(np.all(cost_functions.derivative_L1_cost(u) == desired_output))

    def test_L1T_cost(self):
        print(" Test L1T cost")
        dt = 0.1
        # np.array([[1, -10, 5.555], [-1, 3.3333, 9]])
        reference_result = np.sqrt(393.62491388999996 * dt)
        weights = getdefaultweights()
        weights["w_1T"] = 1.0
        u = self.get_arbitrary_array()
        L1T_cost = cost_functions.control_strength_cost(u, weights, dt)
        self.assertAlmostEqual(L1T_cost, reference_result, places=8)

    def test_derivative_L1T_cost(self):
        print(" Test L1T cost derivative")
        u = self.get_arbitrary_array()
        dt = 0.1
        denominator = np.sqrt(393.62491388999996 * dt)
        desired_output = np.zeros((u.shape))
        desired_output[0, 0, :] = [2.0, -13.3333, 14.555] / denominator
        desired_output[0, 1, :] = [-2.0, 13.3333, 14.555] / denominator

        self.assertTrue(np.all(cost_functions.derivative_L1T_cost(u, dt) == desired_output))

    def test_L1D_cost(self):
        print(" Test L1D cost")
        dt = 0.1
        reference_result = np.sqrt((101.0 + 5.555**2) * dt) + np.sqrt((82.0 + 3.3333**2) * dt)
        weights = getdefaultweights()
        weights["w_1D"] = 1.0
        u = self.get_arbitrary_array()
        L1D_cost = cost_functions.control_strength_cost(u, weights, dt)
        self.assertAlmostEqual(L1D_cost, reference_result, places=8)

    def test_derivative_L1D_cost(self):
        print(" Test L1D cost derivative")
        u = self.get_arbitrary_array()
        dt = 0.1
        desired_output = np.zeros((u.shape))
        denominator = np.sqrt((101.0 + 5.555**2) * dt)
        desired_output[0, 0, :] = u[0, 0, :] / denominator
        denominator = np.sqrt((82.0 + 3.3333**2) * dt)
        desired_output[0, 1, :] = u[0, 1, :] / denominator

        self.assertTrue(np.all(cost_functions.derivative_L1D_cost(u, dt) == desired_output))


if __name__ == "__main__":
    unittest.main()
