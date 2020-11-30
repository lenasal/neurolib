import unittest
import numpy as np

from neurolib.models.fhn import FHNModel
from neurolib.utils import costFunctions as cost

tolerance_digits = 8

class TestCostFunctions(unittest.TestCase):

    """
    def test_setGetParameters(self):
        testip = 2.0
        testie = 1.0
        testis = 0.5
        cost.setParams(testip, testie, testis)
        params = cost.getParams()
        self.assertEqual(params[0], testip)
        self.assertEqual(params[1], testie)
        self.assertEqual(params[2], testis)
        
    def test_setNegativeParameters(self):
        testip = 2.0
        testie = -1.0
        testis = 0.5
        cost.setParams(testip, testie, testis)
        params = cost.getParams()
        self.assertEqual(params[0], testip)
        self.assertNotEqual(params[1], testie)
        self.assertEqual(params[2], testis)
    """
        
    def test_sparsity(self):
        T = 1.
        dt = 0.1
        N = 1
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(1 + np.around(T/dt) ) ))
        result_cost = N * var * test_control_value * np.sqrt(T)
        result_gradient = test_control_value / np.sqrt( test_control_value**2 * T )
        
        i_s_ = 1.
        
        cost.setParams(0., 0., i_s_)
        
        for t in range(test_control.shape[2]):
            test_control[:,:,t] = test_control_value
            
        print(cost.f_cost_sparsity_int(N, T, dt, i_s_, test_control, va_ = [0,1]), result_cost)
            
        self.assertAlmostEqual(cost.f_cost_sparsity_int(N, T, dt, i_s_, test_control, va_ = [0,1]), result_cost, tolerance_digits)
        
        gradient = cost.f_cost_sparsity_gradient(N, var, dt, test_control)
        
        for n in range(test_control.shape[0]):
            for v in range(test_control.shape[1]):
                for t in range(test_control.shape[2]):
                    self.assertAlmostEqual(gradient[n,v,t], result_gradient, tolerance_digits)

if __name__ == '__main__':
    unittest.main()