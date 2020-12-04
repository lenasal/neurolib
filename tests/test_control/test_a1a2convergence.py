import unittest
import numpy as np
import random

#from neurolib.models.fhn import FHNModel
#from neurolib.models.aln import ALNModel
#from neurolib.models.rate_control import RateModel
#from neurolib.models.aln_control import Model_ALN_control

from neurolib.utils import costFunctions as cost
import test_control_functions as func

assertion_tolerance = 2
assertion_tolerance_grad = 5
        
c_controlmin, c_controlmax = -2., 2.
r_controlmin, r_controlmax = 0., 0.1
algorithm_tolerance = 1e-32
max_iteration = int(1e4)
start_step = 10.
test_step = 1e-6

duration = 1.
dur_pre = 0.5
dur_post = 0.5

#tests = ["fhn1", "aln1", "fhn2", "aln2", "fhn2delay", "aln1delay", "aln2delay"]
tests = ["rate_control"]#, "aln1", "aln-control", "rate_control"
cg_var = [None]#, "HS", "FR", "PR", "HZ"]
cntrl_var = [ 2 ] #, [ [0,1], [2,3] ]

class TestA1A2Conv(unittest.TestCase):
    
    def test_A1A2ConvergeForRandomTarget_PSE(self):
        print("test_A1A2ConvergeForRandomTarget_PE for model", testcaseind,
              "\n with conjugated gradient descent variant", cgv,
              "\n for control variables", c_var)
                
        target_vars, output_vars, init_vars = model.target_output_vars, model.output_vars, model.init_vars
        c_scheme, u_mat, u_scheme = func.getSchemes(model)
        
        incl_steps = int(1. + duration/model.params.dt)
            
        func.setInitVarsZero(model, init_vars)
            
        model.params.duration = duration + dur_pre
        
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
                
        control1 = func.getRandomControl(model, cntrl_zeros_pre, c_controlmin, c_controlmax, r_controlmin, r_controlmax, control_variables_ = cntrl_var) 
        
        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
            
        func.setInitVarsZero(model, init_vars)
            
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
            
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, c_controlmin, c_controlmax, r_controlmin, r_controlmax, control_variables_ = cntrl_var) 
        #control2 = model.getZeroControl()
        
        testip, testie, testis = random.uniform(1., 10.), random.uniform(0., 1.), random.uniform(0., 1.)
        cost.setParams(10.*testip, testie, testis)
        
        func.setInitVarsZero(model, init_vars)
        
        if c_var in [0,1,[0,1]]:
            c_max = 1e4 * c_controlmax
            c_min = 1e4 * c_controlmin
        else:
            c_max = 2. * r_controlmax
            c_min = 2. * r_controlmin
        
        A1_bestControl, A1_bestState, A1_cost, A1_runtime, A1_grad = model.A1(control2, target, c_scheme, u_mat, u_scheme, max_iteration,
                           algorithm_tolerance, start_step, c_max, c_min, duration, dur_pre, dur_post, CGVar = None, control_variables_ = cntrl_var)
        
        func.setInitVarsZero(model, init_vars)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                            algorithm_tolerance, incl_steps, start_step, test_step, c_max, c_min, duration, dur_pre, dur_post,
                            control_variables_ = cntrl_var)
        
        self.assertEqual(A1_bestControl.shape[2], cntrl_len)
        self.assertEqual(A2_bestControl.shape[2], cntrl_len)
        
        print("control1 = ", control1[0,cntrl_var,:])
        print("best control a1 = ", A1_bestControl[0,cntrl_var,:])
        print("best control a2 = ", A2_bestControl[0,cntrl_var,:])
        print("grad = ", A1_grad[0,cntrl_var,:])
        print("test weights ", testip, testie, testis)
        
        
        for t in range(len(A1_runtime)-1):
            if (A1_runtime[t+1] == 0.):
                break
                self.assertLessEqual(A1_runtime[t], A1_runtime[t+1])
                
        for t in range(len(A2_runtime)-1):
            if (A2_runtime[t+1] == 0.):
                break
                self.assertLessEqual(A2_runtime[t], A2_runtime[t+1])
        
        for n in range(A2_bestControl.shape[0]):
            for v in cntrl_var:
                for t in range(cntrl_zeros_pre, A2_bestControl.shape[2] - 1 - cntrl_zeros_post):
                    self.assertAlmostEqual(A2_bestControl[n, v, t], A1_bestControl[n, v, t], assertion_tolerance)   
                    
        for n in range(A2_bestControl.shape[0]):
            for v in cntrl_var:
                for t in range(0, A1_grad.shape[2]):
                    print(n, v, t, A1_grad[n, v, t])
                    if ( np.abs(A1_bestControl[n,v,t+cntrl_zeros_pre]) < 1e-10
                        or np.abs(np.amax(A1_bestControl[n,v,t+cntrl_zeros_pre]) - c_max) < 1e-4
                        or np.abs(np.amin(A1_bestControl[n,v,t+cntrl_zeros_pre]) - c_min) < 1e-4):
                        print("gradient could be nonvanishing because of absolute value, or because operating at boundary.")
                    else:
                        self.assertAlmostEqual(A1_grad[n, v, t], 0., assertion_tolerance_grad) 
      
                    
    """                
    def test_A1A2ConvergeForRandomTarget_PES_exc(self):
        print("test_A1A2ConvergeForRandomTarget_PES_exc for model", testcaseind,
              "\n with conjugated gradient descent variant", cgv,
              "\n for control variables", c_var)
                
        target_vars, output_vars, init_vars = model.target_output_vars, model.output_vars, model.init_vars
        c_scheme, u_mat, u_scheme = func.getSchemes(model)
        
        incl_steps = int(1. + duration/model.params.dt)
            
        func.setInitVarsZero(model, init_vars)
            
        model.params.duration = duration + dur_pre
        
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
        
        control1 = func.getRandomControl(model, cntrl_zeros_pre, c_controlmin, c_controlmax, r_controlmin, r_controlmax, control_variables_ = cntrl_var) 
        control1[:,1,:] = 0.
        
        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
            
        func.setInitVarsZero(model, init_vars)
            
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
            
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, c_controlmin, c_controlmax, r_controlmin, r_controlmax, control_variables_ = cntrl_var) 
        
        testip, testie, testis = random.uniform(0., 1.), random.uniform(0., 1.), random.uniform(0., 1.)
        cost.setParams(testip, testie, testis)
        
        func.setInitVarsZero(model, init_vars)
        
        A1_bestControl, A1_bestState, A1_cost, A1_runtime, A1_grad = model.A1(control2, target, c_scheme, u_mat, u_scheme, max_iteration,
                            algorithm_tolerance, start_step, 1e5 * c_controlmax, duration, dur_pre, dur_post, CGVar = None, control_variables_ = cntrl_var)
        
        func.setInitVarsZero(model, init_vars)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                            algorithm_tolerance, incl_steps, start_step, test_step, 1e5 * c_controlmax, duration, dur_pre, dur_post, control_variables_ = cntrl_var)
                
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
                    self.assertAlmostEqual(A2_bestControl[n, v, t], A1_bestControl[n, v, t], assertion_tolerance)    
                    
    def test_A1A2ConvergeForRandomTarget_PES_inh(self):
        print("test_A1A2ConvergeForRandomTarget_PES_inh for model", testcaseind,
              "\n with conjugated gradient descent variant", cgv,
              "\n for control variables", c_var)
                
        target_vars, output_vars, init_vars = model.target_output_vars, model.output_vars, model.init_vars
        c_scheme, u_mat, u_scheme = func.getSchemes(model)
        
        incl_steps = int(1. + duration/model.params.dt)
            
        func.setInitVarsZero(model, init_vars)
            
        model.params.duration = duration + dur_pre
        
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
        
        control1 = func.getRandomControl(model, cntrl_zeros_pre, c_controlmin, c_controlmax, r_controlmin, r_controlmax, control_variables_ = cntrl_var) 
        control1[:,0,:] = 0.
        
        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
            
        func.setInitVarsZero(model, init_vars)
            
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
            
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, c_controlmin, c_controlmax, r_controlmin, r_controlmax, control_variables_ = cntrl_var) 
        
        testip, testie, testis = random.uniform(0., 1.), random.uniform(0., 1.), random.uniform(0., 1.)
        cost.setParams(testip, testie, testis)
        
        func.setInitVarsZero(model, init_vars)
        
        A1_bestControl, A1_bestState, A1_cost, A1_runtime, A1_grad = model.A1(control2, target, c_scheme, u_mat, u_scheme, max_iteration,
                            algorithm_tolerance, start_step, 1e5 * c_controlmax, duration, dur_pre, dur_post, CGVar = None, control_variables_ = cntrl_var)
        
        func.setInitVarsZero(model, init_vars)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                            algorithm_tolerance, incl_steps, start_step, test_step, 1e5 * c_controlmax, duration, dur_pre, dur_post, control_variables_ = cntrl_var)
                
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
                    self.assertAlmostEqual(A2_bestControl[n, v, t], A1_bestControl[n, v, t], assertion_tolerance)   
    """
         

    """
    # if A1 runs into local minima for sparsity cost, make sure gradient vanishes       
    def test_A1ZeroGradient_PES(self):
        print("test_A1ZeroGradient_PES for model", testcaseind,
              "\n with conjugated gradient descent variant", cgv,
              "\n for control variables", c_var)
                
        target_vars, output_vars, init_vars = model.target_output_vars, model.output_vars, model.init_vars
        c_scheme, u_mat, u_scheme = func.getSchemes(model)
        
        incl_steps = int(1. + duration/model.params.dt)
            
        func.setInitVarsZero(model, init_vars)
            
        model.params.duration = duration + dur_pre
        
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
        
        control1 = func.getRandomControl(model, cntrl_zeros_pre, c_controlmin, c_controlmax, r_controlmin, r_controlmax, control_variables_ = cntrl_var) 
        
        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
            
        func.setInitVarsZero(model, init_vars)
            
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
            
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, c_controlmin, c_controlmax, r_controlmin, r_controlmax, control_variables_ = cntrl_var) 
        
        testip, testie, testis = random.uniform(0., 1.), random.uniform(0., 1.), random.uniform(0., 1.)
        cost.setParams(testip, testie, testis)
        
        func.setInitVarsZero(model, init_vars)
        
        A1_bestControl, A1_bestState, A1_cost, A1_runtime, A1_grad = model.A1(control2, target, c_scheme, u_mat, u_scheme, max_iteration,
                            algorithm_tolerance, start_step, 1e5 * c_controlmax, duration, dur_pre, dur_post, CGVar = None, control_variables_ = cntrl_var)
        
        func.setInitVarsZero(model, init_vars)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                            algorithm_tolerance, incl_steps, start_step, test_step, 1e5 * c_controlmax, duration, dur_pre, dur_post, control_variables_ = cntrl_var)
        
        self.assertEqual(A1_bestControl.shape[2], cntrl_len)
        self.assertEqual(A2_bestControl.shape[2], cntrl_len)
        
        print("control1 = ", control1)
        print("control2 = ", control2)
        
        print("grad = ", A1_grad)
        
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
                for t in range(0, A1_grad.shape[2]):
                    print(n, v, t)
                    self.assertAlmostEqual(A1_grad[n, v, t], 0., assertion_tolerance_grad) 
                    
                    
                    
    """
    

if __name__ == '__main__':
    
    runs = 0
    errors = 0
    failures = 0
    success = True
    result = []
    failedTests = []
    
    for testcaseind in tests:
        model = func.getmodel(testcaseind, dur_pre, dur_post)
    
        for cgv in cg_var:
            for c_var in cntrl_var:
                suite = unittest.TestLoader().loadTestsFromTestCase(TestA1A2Conv)
                result.append(unittest.TextTestRunner(verbosity=2).run(suite) )
                runs += result[-1].testsRun
                if not result[-1].wasSuccessful():
                    success = False
                    errors += 1
                    failures += 1
                    failedTests.append(testcaseind)
        
    print("Run", runs, "tests with", errors, "errors and", failures, "failures.")
    if success:
        print("Test OK")
    else:
        print("Test FAILED: ", failedTests)
        print(result)