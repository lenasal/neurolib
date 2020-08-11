import unittest
import numpy as np

from neurolib.models.fhn import FHNModel
from neurolib.utils import costFunctions as cost

tolerance_digits = 8

class TestCostFunctions(unittest.TestCase):

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
        
    def test_sparsityInt_1_1(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = 80.
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[:,:,t] = test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity1_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_1_2(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = 80.
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[:,0,t] = 2. * test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity1_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_1_3(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = 80.
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[0,:,t] = 2. * test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity1_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_2_1(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = np.sqrt(640.)
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[:,:,t] = test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity2_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_2_2(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = np.sqrt(640.)
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[:,0,t] = 2. * test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity2_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_2_3(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = np.sqrt(640.)
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[0,:,t] = 2. * test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity2_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_3_1(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = 4. * np.sqrt(40.)
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[:,:,t] = test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity3_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_3_2(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = 4. * np.sqrt(40.)
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[:,0,t] = 2. * test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity3_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_3_3(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = 4. * np.sqrt(40.)
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[0,:,t] = 2. * test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity3_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_4_1(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = 2.
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[:,:,t] = test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity4_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_3_2(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = 2.
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[:,0,t] = 2. * test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity4_int(dt, test_control), result, tolerance_digits)
        
    def test_sparsityInt_3_3(self):
        T = 10.
        dt = 0.1
        N = 2
        var = 2
        test_control_value = 2.
        test_control = np.zeros(( N, var, int(T/dt) ))
        result = 1.
        
        cost.setParams(0., 0., 1.)
        
        for t in range(test_control.shape[2]):
            test_control[0,:,t] = 2. * test_control_value
            
        self.assertAlmostEqual(cost.f_cost_sparsity4_int(dt, test_control), result, tolerance_digits)

if __name__ == '__main__':
    unittest.main()