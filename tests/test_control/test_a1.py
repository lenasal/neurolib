import unittest
import numpy as np
import os, sys

path = os.getcwd().split(os.sep +'neurolib')[0] + os.sep + 'neurolib'
if path not in sys.path:
    sys.path.append(path)

from neurolib.utils import costFunctions as cost
import test_control_functions as func

np.set_printoptions(precision=4)
        
c_controlmin, c_controlmax = -2., 2.
r_controlmin, r_controlmax = 0., 0.1
algorithm_tolerance_prec = 1e-20
algorithm_tolerance_energy = 1e-32
max_iteration = int(1e4)
start_step = 10.

dur_pre = 0.2# 0.8
dur_post = 0.2# 0.8

#tests = ["fhn1", "aln1", "fhn2", "aln2", "fhn2delay", "aln1delay", "aln2delay"]
tests = ["rate_control"]#, "aln1", "aln-control"], "rate_control"
cg_var = [None]#, "HS", "FR", "PR", "HZ"]

"""
cntrl_var = [ 0 ]#, [ [0,1], [2,3] ]
prec_var = [ 0 ]
ind_timeshift = 4   # for c=0 and p=1, c=1 and p=0, c=2 and p=1
ind_timeshift = 1   # for c=0 and p=0, c=1 and p=1, c=2 and p=0
"""

# c var, p var, ind_Timeshift, delay
variation = [ 
              #############################
              # 1- dimensional output
              # 1-dimensional input
              [[0],[0],1,False,0,1.8],
              [[0],[0],1,True,-1,3.6], 
              [[0],[1],4,False,1,2.6],
              #[[0],[1],4,True,1,4.4],
              #[[1],[0],4,False,3,2.6],
              #[[1],[0],4,True,3,4.4], 
              #[[1],[1],1,False,0,1.8],
              #[[1],[1],1,True,-2,3.6],
              #[[2],[0],1,False,0,1.8],   # 2 : ee
              #[[2],[0],1,True,0,3.6], 
              #[[2],[1],4,False,2,2.6],
              #[[2],[1],4,True,0,4.4],
              #[[3],[0],1,False,0,0.8],   # 3 : ei
              #[[3],[0],1,True,0,3.6], 
              #[[3],[1],4,False,2,2.6],
              #[[3],[1],4,True,0,4.4],
              #[[4],[0],1,False,0,1.8],   # 4 : ie
              #[[4],[0],1,True,0,3.6], 
              #[[4],[1],4,False,2,2.6],
              #[[4],[1],4,True,0,4.4],
              #[[5],[0],1,False,0,0.8],   # 5 : ii
              #[[5],[0],1,True,0,3.6], 
              #[[5],[1],4,False,2,2.6],
              #[[5],[1],4,True,0,4.4],
              # 2-dimensional input
              #[[0,1],[0],4,False,0, 2.6],
              #[[0,1],[0],4,True,-1, 4.4],
              #[[0,1],[1],4,False,0, 2.6],
              #[[0,1],[1],4,True,-1, 4.4],
              #[[0,2],[0],4,False,0, 2.6],
              #[[0,2],[0],4,True,-1, 4.4],
              #[[0,2],[1],4,False,1, 2.6],
              #[[0,2],[1],4,True,1, 4.4],
              #[[1,2],[0],4,False,0, 2.6],
              #[[1,2],[0],4,True,0, 4.4],
              #[[1,2],[1],4,False,0, 2.6],
              #[[1,2],[1],4,True,-1, 4.4],
              # 3-dimensional input
              #[[0,1,2],[0],4,False,0, 2.6],
              #[[0,1,2],[0],4,True,0, 4.4],
              #[[0,1,2],[1],4,False,0, 2.6],
              #[[0,1,2],[1],4,True,0, 4.4],
              #############################
              # 2- dimensional output
              # 1-dimensional input
              #[[0],[0,1],4,False,0, 2.6],
              #[[0],[0,1],4,True,0, 4.4],
              #[[1],[0,1],4,False,0, 2.6],
              #[[1],[0,1],4,True,-1, 4.4],
              #[[2],[0,1],4,False,0, 2.6],
              #[[2],[0,1],4,True,0, 4.4],
              # 2-dimensional input
              #[[0,1],[0,1],4,False,0, 2.6],
              #[[0,1],[0,1],4,True,-1, 4.4],
              #[[0,2],[0,1],4,False,-1, 2.6],
              #[[0,2],[0,1],4,True,0, 4.4],
              #[[1,2],[0,1],4,False,-1, 2.6],
              #[[1,2],[0,1],4,True,-1, 4.4],
              # 3-dimensional input
              #[[0,1,2],[0,1],4,False,-1, 2.6],
              #[[0,1,2],[0,1],4,True,-1, 4.4],
              ]

np.set_printoptions(precision=4)

class TestA1(unittest.TestCase): 
# set init vars zero everywhere or nowhere
    

    def test_A1inputControlForPrecisionCostOnly(self):
        
        ###############################################
        algorithm_tolerance = algorithm_tolerance_prec
        assertion_tolerance = 1
        assertion_tolerance_grad = int(assertion_tolerance-4)
        ###############################################
        
        delay_ndt = func.getDelay_ndt(model)
                        
        print("test_A1inputControlForPrecisionCostOnly for model", testcaseind,
              "\n with conjugated gradient descent variant", cgv,
              "\n for control variables", cntrl_var[0],
              "\n for precision measurement variable", prec_var[0],
              "\n with delay by", delay_ndt, "timesteps.")
        
        target_vars, output_vars, init_vars = model.target_output_vars, model.output_vars, model.init_vars
        c_scheme, u_mat, u_scheme = func.getSchemes(model)
            
        func.setInitVarsZero(model, init_vars)
            
        model.params.duration = duration + dur_pre
            
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
            
        control1 = func.getRandomControl(model, cntrl_zeros_pre, c_controlmin, c_controlmax, r_controlmin, r_controlmax,
                                         control_variables_ = cntrl_var)  
        control1[:,:,-2*(delay_ndt+ind_timeshift+1):] = 0.
        control1[:,:,:cntrl_zeros_pre+2*(delay_ndt+ind_timeshift+1)] = 0.
        
        #control1 = model.getZeroControl()

        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
                
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
            
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, c_controlmin, c_controlmax, r_controlmin, r_controlmax,
                                         control_variables_ = cntrl_var)  
        #control2 = control1[:,:,cntrl_zeros_pre:].copy() #* random.uniform(0.97,1.03)
        #control2[0,cntrl_var,6] = 0.1
            
        testip, testie, testis = 1., 0., 0.
        cost.setParams(testip, testie, testis)
            
        func.setInitVarsZero(model, init_vars)
                
        c_max, c_min = func.setmaxmincontrol(cntrl_var, c_controlmax, c_controlmin, r_controlmax, r_controlmin)
                           
        A1_bestControl, A1_bestState, A1_cost, A1_runtime, A1_grad, A1_phi, A1_costnode = model.A1(control2, target, c_scheme, u_mat,
                            u_scheme, max_iteration_ = max_iteration, tolerance_ = algorithm_tolerance, startStep_ = start_step,
                            max_control_ = c_max, min_control_ = c_min, t_sim_ = duration, t_sim_pre_ = dur_pre, t_sim_post_ = dur_post,
                            CGVar = cgv, control_variables_ = cntrl_var, prec_variables_ = prec_var)        
            
        self.assertEqual(A1_bestControl.shape[2], cntrl_len)
        
        print("target = ", target)
        print("best state = ", A1_bestState[0,prec_var,:])
        print("control1 = ", control1[0,cntrl_var,:])
        print("best control a1 = ", A1_bestControl[0,cntrl_var,:])
        print("grad = ", A1_grad[0,cntrl_var,:])
                    
        for n in range(A1_bestControl.shape[0]):
            for v in cntrl_var:
                # last index could go wrong
                for t in range(1, control1.shape[2] - ind_timeshift - delay_ndt):
                    print(v, t, A1_bestControl[n, v, t], control1[n, v, t])
                    self.assertAlmostEqual(A1_bestControl[n, v, t], control1[n, v, t], assertion_tolerance) 
             
        for t in range(len(A1_cost)-1):
            if (A1_cost[t+1] == 0.):
                break
                self.assertLessEqual(A1_cost[t], A1_cost[t+1])
                                
        for n in range(A1_bestControl.shape[0]):
            for v in cntrl_var:
                for t in range(0, A1_grad.shape[2]):
                    #print(n, v, t, A1_grad[n, v, t])
                    if not ( np.abs(A1_bestControl[n,v,t+cntrl_zeros_pre]) < 1e-10
                        or np.abs(A1_bestControl[n,v,t+cntrl_zeros_pre] - c_max[v]) < 1e-4
                        or np.abs(A1_bestControl[n,v,t+cntrl_zeros_pre] - c_min[v]) < 1e-4):
                        print(v, t, A1_grad[n, v, t])
                        self.assertAlmostEqual(A1_grad[n, v, t], 0., assertion_tolerance_grad)
                    
"""
    def test_A1zeroControlForEnergyAndSparsityCostOnly(self):
        
        ###############################################
        algorithm_tolerance = algorithm_tolerance_energy
        assertion_tolerance = 2
        assertion_tolerance_grad = 10
        ###############################################
        
        delay_ndt = func.getDelay_ndt(model)
        
        print("test_A1zeroControlForEnergyCostOnly for model", testcaseind,
              "\n with conjugated gradient descent variant", cgv,
              "\n for control variables", cntrl_var[0],
              "\n for precision measurement variable", prec_var[0],
              "\n with delay by", delay_ndt, "timesteps.")
        
        target_vars, output_vars, init_vars = model.target_output_vars, model.output_vars, model.init_vars
        c_scheme, u_mat, u_scheme = func.getSchemes(model)
                
        func.setInitVarsZero(model, init_vars)
        
        model.params.duration = duration + dur_pre
        
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
        
        control1 = func.getRandomControl(model, cntrl_zeros_pre, c_controlmin, c_controlmax, r_controlmin, r_controlmax,
                                         control_variables_ = cntrl_var)  
        control1[:,:,-2*(delay_ndt+ind_timeshift+1):] = 0.
        control1[:,:,:cntrl_zeros_pre+2*(delay_ndt+ind_timeshift+1)] = 0.
        
        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
            
        target = func.setTargetFromControl(model, control1, output_vars, target_vars)[:,:, cntrl_zeros_pre:]
        
        model.params.duration = duration
        control2 = func.getRandomControl(model, 0, c_controlmin, c_controlmax, r_controlmin, r_controlmax, control_variables_ = cntrl_var)  
        
        testip, testie, testis = 0., random.uniform(0., 1.), random.uniform(0., 1.)
        cost.setParams(testip, testie, testis)
        
        func.setInitVarsZero(model, init_vars)
        
        if cntrl_var[0] == 0 or cntrl_var[0] == 1:
            c_max = 1e4 * c_controlmax
            c_min = 1e4 * c_controlmin
        else:
            c_max = 2. * r_controlmax
            c_min = 2. * r_controlmin
        
        A1_bestControl, A1_bestState, A1_cost, A1_runtime, A1_grad, A1_phi = model.A1(control2, target, c_scheme, u_mat,
                        u_scheme, max_iteration, algorithm_tolerance, start_step, c_max, c_min, duration,
                        dur_pre, dur_post, CGVar = cgv, control_variables_ = cntrl_var)
        
        print("control1 = ", control1[0,cntrl_var,cntrl_zeros_pre:])
        print("best control a1 = ", A1_bestControl[0,cntrl_var,cntrl_zeros_pre:-cntrl_zeros_post])
        print("grad = ", A1_grad[0,cntrl_var,:])
        
        self.assertEqual(A1_bestControl.shape[2], cntrl_len)
        
        for n in range(A1_bestControl.shape[0]):
            for v in cntrl_var:
                for t in range(1, A1_bestControl.shape[2] - 2):
                    #print(n, v, t)
                    self.assertAlmostEqual(A1_bestControl[n, v, t], 0., assertion_tolerance)  
                    
        for t in range(len(A1_cost)-1):
            if (A1_cost[t+1] == 0.):
                break
            self.assertLessEqual(A1_cost[t+1], A1_cost[t])
"""    
    
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
                
                cntrl_var = variation[ind_v][0] #, [ [0,1], [2,3] ]
                prec_var = variation[ind_v][1] 
                ind_timeshift = variation[ind_v][2]
                exponent_cost = variation[ind_v][4]
                duration = variation[ind_v][5]
                
                if not variation[ind_v][3]:
                    model.params.de = 0.
                    model.params.di = 0.
                    
                print("-------------------------------------------------------------------------")
                print("-------------------------------", cntrl_var, prec_var, ind_timeshift, variation[ind_v][3], exponent_cost)
                    
                suite = unittest.TestLoader().loadTestsFromTestCase(TestA1)
                result.append(unittest.TextTestRunner(verbosity=2).run(suite) )
                runs += result[-1].testsRun
                
                if not result[-1].wasSuccessful():
                    success = False
                    errors += 1
                    failures += 1
                    failedTests.append(str(testcaseind) + str("_") + str(ind_v))
        
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