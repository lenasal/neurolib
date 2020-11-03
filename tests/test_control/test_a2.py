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
max_iteration = int(1e4)
start_step = 20.
test_step = 1e-12

duration = 0.9
dur_pre = 0.5
dur_post = 0.5

#tests = ["fhn1", "aln1", "fhn2", "aln2", "fhn2delay", "aln1delay", "aln2delay"]
tests = ["aln1"]#, "aln-control"]

def getmodel(i):
    if i == "fhn1":
        model_ = FHNModel()
        
    elif i == "aln1":
        model_ = ALNModel()
        dt = model_.params.dt
        
        model_.params.signalV = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        model_.params.de = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        model_.params.di = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        
        #func.setParametersALN(model_)   
        
    elif i == "aln-control":
        model_ = Model_ALN_control()
        dt = model_.params.dt
        
        model_.params.signalV = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        model_.params.de = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        model_.params.di = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        
        #func.setParametersALN(model_)
        
    elif i == "fhn2":
        coupling12 = random.uniform(0, 1)
        coupling21 = random.uniform(0, 1)
        c_mat = np.array( [[0, coupling21], [coupling12, 0]] )
        
        fiber_matrix = np.zeros(( len(c_mat), len(c_mat) ))
        model_ = FHNModel(Cmat = c_mat, Dmat = fiber_matrix)
        
    elif i == "aln2":
        coupling12 = random.uniform(0, 1)
        coupling21 = random.uniform(0, 1)
        c_mat = np.array( [[0, coupling21], [coupling12, 0]] )
        
        fiber_matrix = np.zeros(( len(c_mat), len(c_mat) ))
        model_ = ALNModel(Cmat = c_mat, Dmat = fiber_matrix)
        
        model_.params.signalV = 0.
        model_.params.de = 0.
        model_.params.di = 0.
        
    elif i == "fhn2delay":
        coupling12 = random.uniform(0, 1)
        coupling21 = random.uniform(0, 1)
        c_mat = np.array( [[0, coupling21], [coupling12, 0]] )
        
        delay12 = random.uniform(0, 1)
        delay21 = random.uniform(0, 1)
        fiber_matrix = np.array( [[0, delay21], [delay12, 0]] )
        model_ = FHNModel(Cmat = c_mat, Dmat = fiber_matrix)
        
    elif i == "aln1delay":
        model_ = ALNModel()
        dt = model_.params.dt
        
        model_.params.signalV = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        model_.params.de = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        model_.params.di = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        
    elif i == "aln2delay":
        coupling12 = random.uniform(0, 1)
        coupling21 = random.uniform(0, 1)
        c_mat = np.array( [[0, coupling21], [coupling12, 0]] )
        
        delay12 = random.uniform(0, 1)
        delay21 = random.uniform(0, 1)
        fiber_matrix = np.array( [[0, delay21], [delay12, 0]] )
        
        model_ = ALNModel(Cmat = c_mat, Dmat = fiber_matrix)
        dt = model_.params.dt
        
        model_.params.signalV = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        model_.params.de = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        model_.params.di = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0, 1), 1)
        
    return model_

class TestA2(unittest.TestCase):
 
    
    def test_A2inputControlForPrecisionCostOnly(self):
        print("test_A2inputControlForPrecisionCostOnly for model ", testcaseind)
        
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
            
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
            
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, controlmin, controlmax)
                        
        testip, testie, testis = 1., 0., 0.
        cost.setParams(testip, testie, testis)
        
        func.setInitVarsZero(model, init_vars)
        
        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                algorithm_tolerance, incl_steps, start_step, test_step, 1e5 * controlmax, duration, dur_pre, dur_post)
        
        
        self.assertEqual(A2_bestControl.shape[2], cntrl_len)
        
        for n in range(A2_bestControl.shape[0]):
            for v in range(A2_bestControl.shape[1]):
                for t in range(1, control1.shape[2]-1):
                    self.assertAlmostEqual(A2_bestControl[n, v, t], control1[n, v, t], assertion_tolerance)  
                    
        for t in range(len(A2_runtime)-1):
            if (A2_runtime[t+1] == 0.):
                break
                self.assertLessEqual(A2_runtime[t], A2_runtime[t+1])
    
    
    def test_A2zeroControlForEnergyAndSparsityCostOnly(self):
        print("test_A2inputControlForPrecisionCostOnly for model ", testcaseind)
        
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
            
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
            
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, controlmin, controlmax)
        
        testip, testie, testis = 0., random.uniform(0., 1.), random.uniform(0., 1.)
        cost.setParams(testip, testie, testis)
        
        func.setInitVarsZero(model, init_vars)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                            algorithm_tolerance, incl_steps, start_step, test_step, 1e5 * controlmax, duration, dur_pre, dur_post)
        
        
        self.assertEqual(A2_bestControl.shape[2], cntrl_len)
                
        for n in range(A2_bestControl.shape[0]):
            for v in range(A2_bestControl.shape[1]):
                for t in range(1, control2.shape[2] - 1):
                    self.assertAlmostEqual(A2_bestControl[n, v, t], 0., assertion_tolerance)
                    
        for t in range(len(A2_runtime)-1):
            if (A2_runtime[t+1] == 0.):
                break
            self.assertLessEqual(A2_runtime[t], A2_runtime[t+1])
    


if __name__ == '__main__':
    
    runs = 0
    errors = 0
    failures = 0
    success = True
    result = []
    failedTests = []
    
    for testcaseind in tests:
        print(testcaseind)
        model = getmodel(testcaseind)
    
        suite = unittest.TestLoader().loadTestsFromTestCase(TestA2)
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
    
    