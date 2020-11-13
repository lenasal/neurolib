import unittest
import numpy as np
import random

from neurolib.models.fhn import FHNModel
from neurolib.models.aln import ALNModel
from neurolib.models.aln_control import Model_ALN_control

from neurolib.utils import costFunctions as cost
import test_control_functions as func

assertion_tolerance = 2
        
controlmin, controlmax = -2., 2.
algorithm_tolerance = 1e-16
max_iteration = 5 * int(1e4)
start_step = 100.

duration = 1.
dur_pre = 0.5
dur_post = 0.5

#tests = ["fhn1", "aln1", "fhn2", "aln2", "fhn2delay", "aln1delay", "aln2delay"]
tests = ["aln1"]#, "aln-control"]

np.set_printoptions(precision=16)

class TestA1(unittest.TestCase): 
# set init vars zero everywhere or nowhere
    
    """
    def test_A1PrecisionCostOnly_InhControlExcCost(self):
                        
        print("test_A1PrecisionCostOnly_InhControlExcCost for model ", testcaseind)
        
        target_vars, output_vars, init_vars = model.target_output_vars, model.output_vars, model.init_vars
        c_scheme, u_mat, u_scheme = func.getSchemes(model)
            
        func.setInitVarsZero(model, init_vars)
            
        model.params.duration = duration + dur_pre
            
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
        
        variables = [0]
        delay_ndt = func.getDelay_ndt(model)
            
        control1 = func.getRandomControl(model, cntrl_zeros_pre, controlmin, controlmax, variables_ = variables)
        control1[:,:,-4-delay_ndt:] = 0.

        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
                
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
            
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, controlmin, controlmax, variables_ = variables)
            
        testip, testie, testis = 1., 0., 0.
        cost.setParams(testip, testie, testis)
            
        func.setInitVarsZero(model, init_vars)
               
        A1_bestControl, A1_bestState, A1_cost, A1_runtime = model.A1(control2, target, c_scheme, u_mat,
                            u_scheme, max_iteration_ = max_iteration, tolerance_ = algorithm_tolerance, startStep_ = start_step,
                            max_control_ = 1e5 * controlmax, t_sim_ = duration, t_sim_pre_ = dur_pre, t_sim_post_ = dur_post,
                            CGVar = None, variables_ = variables)        
            
        self.assertEqual(A1_bestControl.shape[2], cntrl_len)
                    
        for n in range(A1_bestControl.shape[0]):
            for v in range(A1_bestControl.shape[1]):
                for t in range(1, control1.shape[2] - 4 - delay_ndt):
                    self.assertAlmostEqual(A1_bestControl[n, v, t], control1[n, v, t], assertion_tolerance) 
                    
        for t in range(len(A1_runtime)-1):
            if (A1_runtime[t+1] == 0.):
                break
                self.assertLessEqual(A1_cost[t+1], A1_cost[t])
    """
    
                
    def test_A1PrecisionCostOnly_ExcControlInhCost(self):
                        
        print("test_A1PrecisionCostOnly_ExcControlInhCost for model ", testcaseind)
        
        target_vars, output_vars, init_vars = model.target_output_vars, model.output_vars, model.init_vars
        c_scheme, u_mat, u_scheme = func.getSchemes(model)
            
        func.setInitVarsZero(model, init_vars)
            
        model.params.duration = duration + dur_pre
            
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
        
        variables = [1]
        delay_ndt = func.getDelay_ndt(model)
            
        control1 = func.getRandomControl(model, cntrl_zeros_pre, controlmin, controlmax, variables_ = variables) 
        control1[:,:,-4-delay_ndt:] = 0.

        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
                
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
            
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, controlmin, controlmax, variables_ = variables)
            
        testip, testie, testis = 1., 0., 0.
        cost.setParams(testip, testie, testis)
            
        func.setInitVarsZero(model, init_vars)
               
        A1_bestControl, A1_bestState, A1_cost, A1_runtime = model.A1(control2, target, c_scheme, u_mat,
                            u_scheme, max_iteration_ = max_iteration, tolerance_ = algorithm_tolerance, startStep_ = start_step,
                            max_control_ = 1e5 * controlmax, t_sim_ = duration, t_sim_pre_ = dur_pre, t_sim_post_ = dur_post,
                            CGVar = None, variables_ = variables)        
            
        self.assertEqual(A1_bestControl.shape[2], cntrl_len)
        
                            
        print("setting control to zero for timesteps before end: ", 4+delay_ndt)
        print("delays = ", model.params.di, model.params.signalV, model.params.de)
                    
        for n in range(A1_bestControl.shape[0]):
            for v in range(A1_bestControl.shape[1]):
                for t in range(1, control1.shape[2] - 4 - delay_ndt):
                    print(n, v, t)
                    print(A1_bestControl[n, v, t] - control1[n, v, t])
                    self.assertAlmostEqual(A1_bestControl[n, v, t], control1[n, v, t], assertion_tolerance) 
                    
        for t in range(len(A1_runtime)-1):
            if (A1_runtime[t+1] == 0.):
                break
                self.assertLessEqual(A1_cost[t+1], A1_cost[t])
    
if __name__ == '__main__':
    
    runs = 0
    errors = 0
    failures = 0
    success = True
    result = []
    failedTests = []
    
    for testcaseind in tests:
        print(testcaseind)
        model = func.getmodel(testcaseind, dur_pre, dur_post)
    
        suite = unittest.TestLoader().loadTestsFromTestCase(TestA1)
        result.append(unittest.TextTestRunner(verbosity=2).run(suite) )
        runs += result[-1].testsRun
        if not result[-1].wasSuccessful():
            success = False
            errors += 1
            failures += 1
            failedTests.append(testcaseind)
        
    print("Run ", runs, " tests with ", errors, "errors and ", failures, "failures.")
    if success:
        print("Test OK")
    else:
        print("Test FAILED: ", failedTests)
        print(result)