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
algorithm_tolerance = 1e-32
max_iteration = int(1e4)
start_step = 20.
test_step = 1e-6

duration = 0.8
dur_pre = 0.5
dur_post = 0.5

#tests = ["fhn1", "aln1", "fhn2", "aln2", "fhn2delay", "aln1delay", "aln2delay"]
tests = ["aln1"]#, "aln-control"]

class TestA1A2Conv(unittest.TestCase):
      
# set init vars zero everywhere or nowhere       
    def test_A1A2ConvergeForRandomTarget(self):
        print("test_A1inputControlForPrecisionCostOnly for model ", testcaseind)
        
        print("speeds = ", model.params.signalV, model.params.de, model.params.di)
        
        target_vars, output_vars, init_vars = model.target_output_vars, model.output_vars, model.init_vars
        c_scheme, u_mat, u_scheme = func.getSchemes(model)
        
        incl_steps = int(1. + duration/model.params.dt)
            
        func.setInitVarsZero(model, init_vars)
            
        model.params.duration = duration + dur_pre
        
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
        
        control1 = func.getRandomControl(model, cntrl_zeros_pre, controlmin, controlmax) 
        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
            
        func.setInitVarsZero(model, init_vars)
            
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
            
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, controlmin, controlmax)
        
        testip, testie, testis = random.uniform(0., 1.), random.uniform(0., 1.), random.uniform(0., 1.)
        #cost.setParams(1., 0., 0.)
        cost.setParams(testip, testie, 0.)
        
        func.setInitVarsZero(model, init_vars)
        
        A1_bestControl, A1_bestState, A1_cost, A1_runtime = model.A1(control2, target, c_scheme, u_mat, u_scheme, max_iteration,
                            algorithm_tolerance, start_step, 1e5 * controlmax, duration, dur_pre, dur_post, CGVar = None)
        
        func.setInitVarsZero(model, init_vars)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                            algorithm_tolerance, incl_steps, start_step, test_step, 1e5 * controlmax, duration, dur_pre, dur_post)
        
        self.assertEqual(A1_bestControl.shape[2], cntrl_len)
        self.assertEqual(A2_bestControl.shape[2], cntrl_len)
        
        for t in range(len(A1_runtime)-1):
            if (A1_runtime[t+1] == 0.):
                break
                self.assertLessEqual(A1_runtime[t], A1_runtime[t+1])
                
        for t in range(len(A2_runtime)-1):
            if (A2_runtime[t+1] == 0.):
                break
                self.assertLessEqual(A2_runtime[t], A2_runtime[t+1])
        
        for n in range(A2_bestControl.shape[0]):
            for v in range(A2_bestControl.shape[1]):
                for t in range(cntrl_zeros_pre, A2_bestControl.shape[2] - 1 - cntrl_zeros_post):
                    print(n,v,t)
                    print("difference = ", A2_bestControl[n, v, t] - A1_bestControl[n, v, t])
                    self.assertAlmostEqual(A2_bestControl[n, v, t], A1_bestControl[n, v, t], assertion_tolerance)    
    

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
        #model = getmodel(testcaseind)
    
        suite = unittest.TestLoader().loadTestsFromTestCase(TestA1A2Conv)
        result.append(unittest.TextTestRunner(verbosity=2).run(suite) )
        runs += result[-1].testsRun
        if not result[-1].wasSuccessful():
            success = False
            errors += 1
            failures += 1
            failedTests.append(testcaseind)
        
    print("Run ", runs, " tests with ", errors, " errors and ", failures, "failures.")
    if success:
        print("Test OK")
    else:
        print("Test FAILED: ", failedTests)
        print(result)