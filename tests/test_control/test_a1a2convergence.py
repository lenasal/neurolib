import unittest
import numpy as np
import random

#from neurolib.models.fhn import FHNModel
#from neurolib.models.aln import ALNModel
#from neurolib.models.rate_control import RateModel
#from neurolib.models.aln_control import Model_ALN_control

from neurolib.utils import costFunctions as cost
import test_control_functions as func

np.set_printoptions(precision=4)
        
c_controlmin, c_controlmax = -5., 5.
r_controlmin, r_controlmax = 0., 0.1
algorithm_tolerance = 1e-24
max_iteration = 2. * int(1e3)
max_iteration_A2 = 200
start_step = 10.
test_step = 1e-6

dur_pre = 0.8
dur_post = 0.8

#tests = ["fhn1", "aln1", "fhn2", "aln2", "fhn2delay", "aln1delay", "aln2delay"]
tests = ["rate_control"]#, "aln1", "aln-control", "rate_control"
cg_var = [None]#, "HS", "FR", "PR", "HZ"]

"""
cntrl_var = [ 0 ] #, [ [0,1], [2,3] ]
prec_var = [ 1 ]
ind_timeshift = 4   # for c=0 and p=1, c=1 and p=0, c=2 and p=1
#ind_timeshift = 1   # for c=0 and p=0, c=1 and p=1, c=2 and p=0
"""

variation = [ 
              [0,0,1,False,4, 1.8],
              [0,0,1,True,5, 3.6], 
              #[0,1,4,False,5, 2.6],
              [0,1,4,True,11, 4.4],
              #[1,0,4,False,5, 2.6],
              [1,0,4,True,10, 4.4], 
              #[1,1,1,False,1, 1.8],
              [1,1,1,True,2, 3.6],
              #[2,0,1,False,2, 1.8],
              [2,0,1,True,4, 3.6], 
              #[2,1,4,False,6, 2.6],
              [2,1,4,True,8, 4.4] 
              ]

class TestA1A2Conv(unittest.TestCase):
    
    def test_A1A2ConvergeForRandomTarget_PSE(self):
        
        ###############################################
        assertion_tolerance = 1
        assertion_tolerance_grad = 5 + exponent_cost
        
        testip = round(random.uniform(1., 10.),1)
        testie = round(random.uniform(0., 10.**(-exponent_cost)),exponent_cost+1)
        testis = round(random.uniform(0., 10.**(-exponent_cost)),exponent_cost+1)
        ###############################################
        
        delay_ndt = func.getDelay_ndt(model)

        print("test_A1A2ConvergeForRandomTarget_PE for model", testcaseind,
              "\n with conjugated gradient descent variant", cgv,
              "\n for control variable", cntrl_var[0],
              "\n for precision measurement variable", prec_var[0],
              "\n with delay due to signal speeds", model.params.signalV, model.params.de, model.params.di)
                
        target_vars, output_vars, init_vars = model.target_output_vars, model.output_vars, model.init_vars
        c_scheme, u_mat, u_scheme = func.getSchemes(model)
        
        incl_steps = int(1. + duration/model.params.dt)
            
        func.setInitVarsZero(model, init_vars)
            
        model.params.duration = duration + dur_pre
        
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
                        
        control1 = func.getRandomControl(model, cntrl_zeros_pre, c_controlmin, c_controlmax, r_controlmin, r_controlmax,
                                         control_variables_ = cntrl_var) 
        
        # cannot be reconstructed reasonably as information is missing due to delay not within simulation duration
        control1[:,:,-2*(delay_ndt+ind_timeshift+1):] = 0.
        control1[:,:,:cntrl_zeros_pre+2*(delay_ndt+ind_timeshift+1)] = 0.
        #control1 = model.getZeroControl()
        #control1[0,0,cntrl_zeros_pre + 3] = 1.
        
        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
            
        func.setInitVarsZero(model, init_vars)
                    
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
                    
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, c_controlmin, c_controlmax, r_controlmin, r_controlmax,
                                         control_variables_ = cntrl_var) 
        
        control2 = control1[:,:,cntrl_zeros_pre:] * random.uniform(0.9,1.1)
        
        cost.setParams(testip, testie, testis)
        
        func.setInitVarsZero(model, init_vars)
        
        if cntrl_var[0] == 0 or cntrl_var[0] == 1:
            c_max = c_controlmax
            c_min = c_controlmin
        else:
            c_max = 2. * r_controlmax
            c_min = 2. * r_controlmin
        
        A1_bestControl, A1_bestState, A1_cost, A1_runtime, A1_grad = model.A1(control2, target, c_scheme, u_mat, u_scheme,
                        max_iteration, algorithm_tolerance, start_step, c_max, c_min, duration, dur_pre, dur_post,
                        CGVar = None, control_variables_ = cntrl_var, prec_variables_ = prec_var)
        
        if A1_cost[-1] > 0.:
            testAtMaxIt[ind_v] = 1
        if np.amax(np.abs(A1_bestControl)) < 1e-10:
            testAtZeroControl[ind_v] = 1
        
        func.setInitVarsZero(model, init_vars)
        
        control2 = A1_bestControl[:,:,cntrl_zeros_pre:-cntrl_zeros_post] * random.uniform(0.99,1.01)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration_A2,
                            algorithm_tolerance, incl_steps, start_step, test_step, c_max, c_min, duration, dur_pre, dur_post,
                            control_variables_ = cntrl_var, prec_variables_ = prec_var)
        
        self.assertEqual(A1_bestControl.shape[2], cntrl_len)
        self.assertEqual(A2_bestControl.shape[2], cntrl_len)
        
        print("control1 = ", control1[0,cntrl_var,cntrl_zeros_pre:])
        print("best control a1 = ", A1_bestControl[0,cntrl_var,cntrl_zeros_pre:-cntrl_zeros_post])
        print("best control a2 = ", A2_bestControl[0,cntrl_var,cntrl_zeros_pre:-cntrl_zeros_post])
        print("grad = ", A1_grad[0,cntrl_var,:])
        print("test weights ", testip, testie, testis)
        
        # make sure cost is decreasing monotonously
        A1lastind = max_iteration-1
        for t in range(len(A1_cost)-1):
            if (A1_cost[t+1] == 0.):
                A1lastind = t
                break
            self.assertLessEqual(A1_cost[t+1], A1_cost[t])
            
        A2lastind = max_iteration_A2-1
        for t in range(len(A2_cost)-1):
            if (A2_cost[t+1] == 0.):
                A2lastind = t
                break
            self.assertLessEqual(A2_cost[t+1], A2_cost[t])
            
        # make sure a1 performs better than a2
        #self.assertLessEqual(A1_cost[A1lastind], A2_cost[A2lastind])
        
        if A1_cost[A1lastind] > A2_cost[A2lastind] :
            testwithA2better[ind_v] = 1
        
                
        for t in range(len(A2_runtime)-1):
            if (A2_runtime[t+1] == 0.):
                break
                self.assertLessEqual(A2_runtime[t], A2_runtime[t+1])
                
        for n in range(A2_bestControl.shape[0]):
            for v in cntrl_var:
                for t in range(0, A1_grad.shape[2]):
                    #print(n, v, t, A1_grad[n, v, t])
                    if not ( np.abs(A1_bestControl[n,v,t+cntrl_zeros_pre]) < 1e-10
                        or np.abs(np.amax(A1_bestControl[n,v,t+cntrl_zeros_pre]) - c_max) < 1e-4
                        or np.abs(np.amin(A1_bestControl[n,v,t+cntrl_zeros_pre]) - c_min) < 1e-4):
                        self.assertAlmostEqual(A1_grad[n, v, t], 0., assertion_tolerance_grad) 
                    #else:
                        #print("gradient could be nonvanishing because of absolute value, or because operating at boundary.")
                        
        
        for n in range(A2_bestControl.shape[0]):
            for v in cntrl_var:
                for t in range(cntrl_zeros_pre, A2_bestControl.shape[2] - ind_timeshift - cntrl_zeros_post - delay_ndt):
                    self.assertAlmostEqual(A2_bestControl[n, v, t], A1_bestControl[n, v, t], assertion_tolerance)          
    

if __name__ == '__main__':
    
    runs = 0
    errors = 0
    failures = 0
    success = True
    result = []
    failedTests = []
    
    testAtMaxIt = np.zeros(( len(variation) ))
    testAtZeroControl = np.zeros(( len(variation) ))
    testwithA2better = np.zeros(( len(variation) ))
    
    for testcaseind in tests:
        model = func.getmodel(testcaseind, dur_pre, dur_post)
    
        for cgv in cg_var:
            for ind_v in range(len(variation)):
                model = func.getmodel(testcaseind, dur_pre, dur_post)
                
                cntrl_var = [ variation[ind_v][0] ]#, [ [0,1], [2,3] ]
                prec_var = [ variation[ind_v][1] ]
                ind_timeshift = variation[ind_v][2]
                exponent_cost = variation[ind_v][4]
                duration = variation[ind_v][5]
                
                if not variation[ind_v][3]:
                    model.params.de = 0.
                    model.params.di = 0.
                    
                print("-------------------------------------------------------------------------")
                print("-------------------------------", cntrl_var, prec_var, ind_timeshift, variation[ind_v][3], exponent_cost)
                    
                suite = unittest.TestLoader().loadTestsFromTestCase(TestA1A2Conv)
                result.append(unittest.TextTestRunner(verbosity=2).run(suite) )
                runs += result[-1].testsRun
                
                if not result[-1].wasSuccessful():
                    success = False
                    errors += 1
                    failures += 1
                    failedTests.append(testcaseind)
        
    print("Run", runs, "tests with", errors, "errors and", failures, "failures.")
    
    print("-------------------------------------------------------------------------")
    print("Not sufficiently many iterations for test cases ", testAtMaxIt)
    print("Zero control as result, check cost weights or simulation duration ", testAtZeroControl)
    print("A2 performs better: ", testwithA2better)
        
    if success:
        print("Test OK")
    else:
        print("Test FAILED: ", failedTests)
        print(result)