import unittest
from neurolib.optimal_control import cost_functions
import numpy as np


class TestCostFunctions(unittest.TestCase):
    @staticmethod
    def get_arbitrary_array():
        array = np.ones(((2, 2, 4)))  # an arbitrary vector with positive and negative entries

        array[0, 0, 0] = -1.0
        array[0, 0, 1] = 2.0
        array[0, 0, 2] = 0.1
        array[0, 0, 3] = 10.0

        array[0, 1, 0] = 1.0
        array[0, 1, 1] = -2.0
        array[0, 1, 2] = 3.0
        array[0, 1, 3] = -1.0

        array[1, 0, 0] = 0.0
        array[1, 0, 1] = 0.5
        array[1, 0, 2] = -0.1
        array[1, 0, 3] = 1.0

        array[1, 1, 0] = -2.0
        array[1, 1, 1] = -3.0
        array[1, 1, 2] = 0.5
        array[1, 1, 3] = 4.0

        return array

    def test_precision_cost_full_timeseries(self):
        print(" Test precision cost full timeseries")
        w_p = 1
        dt = 0.1
        target = self.get_arbitrary_array()
        N = target.shape[0]
        precision_cost_matrix = np.ones((N, 2))
        interval = (0, target.shape[2])

        self.assertEqual(
            cost_functions.precision_cost(target, target, w_p, precision_cost_matrix, dt, interval),
            0,
        )  # target and simulation coincide

        x_sim = np.copy(target)
        x_sim[:, :, 0] = -x_sim[:, :, 0]  # create setting where result depends only on this first entries
        self.assertEqual(
            cost_functions.precision_cost(target, target, w_p, precision_cost_matrix, dt, interval),
            0,
        )
        self.assertEqual(
            cost_functions.precision_cost(target, x_sim, w_p, precision_cost_matrix, dt, interval),
            w_p / 2 * np.sum(np.sum((2 * target[:, :, 0]) ** 2, axis=1), axis=0) * dt,
        )

    def test_precision_cost_nodes_channels(self):
        print(" Test precision cost full timeseries for node and channel selection.")
        w_p = 1
        target = self.get_arbitrary_array()
        N = target.shape[0]
        precision_cost_matrix = np.zeros((N, 2))
        dt = 0.1
        interval = (0, target.shape[2])
        zerostate = np.zeros((target.shape))

        self.assertTrue(
            np.abs(cost_functions.precision_cost(target, zerostate, w_p, precision_cost_matrix, dt, interval)) < 1e-20,
        )  # no cost if precision matrix is zero

        for i in range(N):
            for j in range(N):
                precision_cost_matrix[i, j] = 1
                result = w_p * 0.5 * sum((target[i, j, :] ** 2)) * dt
                self.assertEqual(
                    cost_functions.precision_cost(target, zerostate, w_p, precision_cost_matrix, dt, interval),
                    result,
                )
                precision_cost_matrix[i, j] = 0

    def test_derivative_precision_cost_full_timeseries(self):
        print(" Test precision cost derivative full timeseries")
        w_p = 1
        target = self.get_arbitrary_array()
        N = target.shape[0]
        precision_cost_matrix = np.ones((N, 2))
        x_sim = np.copy(target)
        x_sim[0, :, 0] = -x_sim[0, :, 0]  # create setting where result depends only on this first entries
        interval = (0, target.shape[2])

        derivative_p_c = cost_functions.derivative_precision_cost(target, x_sim, w_p, precision_cost_matrix, interval)

        self.assertTrue(np.all(derivative_p_c[0, :, 1::] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 0] == 2 * (-w_p * target[0, :, 0])))

    def test_derivative_precision_cost_full_timeseries_nodes_channels(self):
        print(" Test precision cost derivative full timeseries for node and channel selection")
        w_p = 1
        target = self.get_arbitrary_array()
        N = target.shape[0]
        precision_cost_matrix = np.zeros((N, 2))  # ToDo: overwrites previous definition, bug?
        zerostate = np.zeros((target.shape))
        interval = (0, target.shape[2])

        derivative_p_c = cost_functions.derivative_precision_cost(
            target, zerostate, w_p, precision_cost_matrix, interval
        )
        self.assertTrue(np.all(derivative_p_c == 0))

        for i in range(N):
            for j in range(N):
                precision_cost_matrix[i, j] = 1
                derivative_p_c = cost_functions.derivative_precision_cost(
                    target, zerostate, w_p, precision_cost_matrix, interval
                )
                result = -w_p * np.einsum("ijk,ij->ijk", target, precision_cost_matrix)
                self.assertTrue(np.all(derivative_p_c - result == 0))
                precision_cost_matrix[i, j] = 0

    def test_precision_cost_in_interval(self):
        """This test is analogous to the 'test_precision_cost'. However, the signal is repeated twice, but only
        the second interval is to be taken into account.
        """
        print(" Test precision cost in time interval")
        w_p = 1
        dt = 0.1
        target = self.get_arbitrary_array()
        x_sim = np.copy(target)
        x_sim[0, :, 3] = -x_sim[0, :, 3]
        N = target.shape[0]
        precision_cost_matrix = np.ones((N, 2))
        interval = (3, target.shape[2])
        precision_cost = cost_functions.precision_cost(target, x_sim, w_p, precision_cost_matrix, dt, interval)
        # Result should only depend on second half of the timeseries.
        self.assertEqual(precision_cost, w_p / 2 * np.sum((2 * target[0, :, 3]) ** 2) * dt)

    def test_derivative_precision_cost_in_interval(self):
        """This test is analogous to the 'test_derivative_precision_cost'. However, the signal is repeated twice, but
        only the second interval is to be taken into account.
        """
        print(" Test precision cost derivative in time interval")
        w_p = 1
        target = self.get_arbitrary_array()
        N = target.shape[0]
        precision_cost_matrix = np.ones((N, 2))
        x_sim = np.copy(target)
        x_sim[0, :, 3] = -x_sim[0, :, 3]  # create setting where result depends only on this first entries
        interval = (3, target.shape[2])
        derivative_p_c = cost_functions.derivative_precision_cost(target, x_sim, w_p, precision_cost_matrix, interval)

        self.assertTrue(np.all(derivative_p_c[0, :, 0:3] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 4::] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 3] == 2 * (-w_p * target[0, :, 3])))

    def test_energy_cost(self):
        print(" Test energy cost")
        dt = 0.1
        reference_result = 150.52 * dt * 0.5
        w_2 = 1
        u = self.get_arbitrary_array()
        energy_cost = cost_functions.energy_cost(u, w_2, dt)
        self.assertEqual(energy_cost, reference_result)

    def test_derivative_energy_cost(self):
        print(" Test energy cost derivative")
        w_2 = -0.9995
        u = self.get_arbitrary_array()
        desired_output = w_2 * u
        self.assertTrue(np.all(cost_functions.derivative_energy_cost(u, w_2) == desired_output))

    def test_L1_cost(self):
        print(" Test L1 cost")
        dt = 0.1
        reference_result = 31.2 * dt
        w_1 = 1
        u = self.get_arbitrary_array()
        L1_cost = np.round(cost_functions.L1_cost(u, w_1, dt), 6)
        self.assertEqual(L1_cost, reference_result)

    def test_derivative_L1_cost(self):
        print(" Test L1 cost derivative")
        w_1 = -0.9995
        u = self.get_arbitrary_array()
        desired_output = w_1 * np.sign(u)
        self.assertTrue(np.all(cost_functions.derivative_L1_cost(u, w_1) == desired_output))

    def test_L1T_cost(self):
        print(" Test L1T cost")
        dt = 0.1
        reference_result = np.round(18.491619723539635 * np.sqrt(dt), 10)
        w_1T = 1
        u = self.get_arbitrary_array()
        L1T_cost = np.round(cost_functions.L1T_cost(u, w_1T, dt), 10)
        self.assertEqual(L1T_cost, reference_result)

    def test_derivative_L1T_cost(self):
        print(" Test L1T cost derivative")
        w_1T = -0.9995
        dt = 0.1
        u = self.get_arbitrary_array()
        arr = np.array([(1.0 + 1.0 + 2.0), (2.0 + 2.0 + 0.5 + 3.0), (0.1 + 3.0 + 0.1 + 0.5), (10.0 + 1.0 + 1.0 + 4.0)])
        denominator = cost_functions.L1T_cost(u, w_1T, dt)
        desired_output = w_1T * np.sign(u) * arr / denominator
        self.assertTrue(np.all(cost_functions.derivative_L1T_cost(u, w_1T, dt) == desired_output))

    def test_L1D_cost(self):
        print(" Test L1D cost")
        dt = 0.1
        reference_result = np.round(20.651246179814354 * np.sqrt(dt), 20)
        w_1D = 1
        u = self.get_arbitrary_array()
        L1D_cost = np.round(cost_functions.L1D_cost(u, w_1D, dt), 20)
        self.assertEqual(L1D_cost, reference_result)

    def test_derivative_L1D_cost(self):
        print(" Test L1D cost derivative")
        w_1D = -0.9995
        dt = 0.1
        u = self.get_arbitrary_array()
        denominator = np.sqrt(np.sum(u**2, axis=2)) * np.sqrt(dt)
        multiplier = 1.0 / denominator
        desired_output = w_1D * u * multiplier[:, :, np.newaxis]
        self.assertTrue(np.all(np.abs(cost_functions.derivative_L1D_cost(u, w_1D, dt) - desired_output) < 1e-14))


if __name__ == "__main__":
    unittest.main()
