import unittest
import numpy as np
import random

from neurolib.models.fhn import FHNModel
from neurolib.models.aln import ALNModel
from neurolib.utils import costFunctions as cost

assertion_tolerance = 2
        
controlmin, controlmax = -2., 2.
algorithm_tolerance = 1e-16
max_iteration = int(1e3)
start_step = 20.
test_step = 1e-12

duration = 0.6
dur_pre = 0.0
dur_post = 0.0


class TestStringMethods(unittest.TestCase):
        
    def getRandomControl(self, cntrl_zeros_pre):
        control_ = model.getZeroControl()
        for n in range(control_.shape[0]):
            for v in range(control_.shape[1]):
                for t in range(cntrl_zeros_pre, control_.shape[2]):
                    control_[n, v, t] = random.uniform(controlmin, controlmax)
        return control_
    
    def updateState(self, control_):
        state_ = model.getZeroState()
        model.run(control = control_)
        for o_ind in range(len(output_vars)):
            state_[:,o_ind,:] = model[output_vars[o_ind]][:,:]
        return state_
        
    
    def setTargetFromControl(self, control_):
        target_ = model.getZeroTarget()
        model.run(control = control_)
        for o_ind in range(len(output_vars)):
            for t_ind in range(len(target_vars)):
                if (target_vars[t_ind] == output_vars[o_ind]):
                    target_[:,t_ind,:] = model[output_vars[o_ind]][:,:]
        return target_
    
    def setInitVarsZero(self):
        for iv in range(len(init_vars)):
            model.params[init_vars[iv]] = np.zeros(( model.params[init_vars[iv]].shape ))
        
    
    def test_A2inputControlForPrecisionCostOnly(self):
        print("test_A2inputControlForPrecisionCostOnly")
        
        model.params.duration = duration + dur_pre
        
        cntrl_zeros_pre = int(dur_pre / model.params.dt)
        cntrl_zeros_post = int(dur_post / model.params.dt)
        control1 = self.getRandomControl(cntrl_zeros_pre)  
        
        cntrl_len = control1.shape[2] + cntrl_zeros_post
        if cntrl_zeros_post == 0:
            cntrl_zeros_post = 1
                
         
        target = self.setTargetFromControl(control1)[:,:, cntrl_zeros_pre:]    
        
        model.params.duration = duration
        control2 = self.getRandomControl(0)
        
        testip, testie, testis = 1., 0., 0.
        cost.setParams(testip, testie, testis)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                      algorithm_tolerance, incl_steps, start_step, test_step, 2.*controlmax, duration, dur_pre, dur_post)
        
        self.assertEqual(A2_bestControl.shape[2], cntrl_len)
        
        for n in range(A2_bestControl.shape[0]):
            for v in range(A2_bestControl.shape[1]):
                for t in range(control1.shape[2] - 2):
                    self.assertAlmostEqual(A2_bestControl[n, v, t], control1[n, v, t], assertion_tolerance)

"""
    def test_A1inputControlForPrecisionCostOnly(self):
        if (model.name == "aln"):
            return
        print("test_A1inputControlForPrecisionCostOnly")
        
        model.params.duration = duration
        
        c_scheme = np.zeros((len(output_vars), len(output_vars) ))
        c_scheme[0,0] = 1.

        u_mat = np.identity(model.params['N'])
        u_scheme = np.array([[1, 0], [0, 0]])
        
        control1 = self.getRandomControl()    
        target = self.setTargetFromControl(control1)    
        control2 = self.getRandomControl()
        state2 = self.updateState(control2)
        
        testip, testie, testis = 1., 0., 0.
        cost.setParams(testip, testie, testis)

        A1_bestControl, A1_bestState, A1_cost, A1_runtime = model.A1(state2, target, control2, c_scheme, u_mat, u_scheme, max_iteration,
                                    algorithm_tolerance, start_step, test_step, controlmax, CGVar = None)
        
        for n in range(A1_bestControl.shape[0]):
            for v in range(A1_bestControl.shape[1]):
                for t in range(A1_bestControl.shape[2] - 2):
                    self.assertAlmostEqual(A1_bestControl[n, v, t], control1[n, v, t], assertion_tolerance)

    def test_A2zeroControlForEnergyCostOnly(self):
        print("test_A2zeroControlForEnergyCostOnly")
        
        model.params.duration = duration
        
        control1 = self.getRandomControl()    
        target = self.setTargetFromControl(control1)    
        control2 = self.getRandomControl()
        
        testip, testie, testis = 0., 1., 0.
        cost.setParams(testip, testie, testis)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                                                                          algorithm_tolerance, incl_steps, start_step, test_step)
        
        for n in range(A2_bestControl.shape[0]):
            for v in range(A2_bestControl.shape[1]):
                for t in range(A2_bestControl.shape[2] - 2):
                    self.assertAlmostEqual(A2_bestControl[n, v, t], 0., assertion_tolerance)

    def test_A1zeroControlForEnergyCostOnly(self):
        if (model.name == "aln"):
            return
        print("test_A1zeroControlForEnergyCostOnly")
        
        model.params.duration = duration
        
        c_scheme = np.zeros((len(output_vars), len(output_vars) ))
        c_scheme[0,0] = 1.

        u_mat = np.identity(model.params['N'])
        u_scheme = np.array([[1, 0], [0, 0]])
        
        control1 = self.getRandomControl()    
        target = self.setTargetFromControl(control1)    
        control2 = self.getRandomControl()
        state2 = self.updateState(control2)  
        
        testip, testie, testis = 0., 1., 0.
        cost.setParams(testip, testie, testis)

        A1_bestControl, A1_bestState, A1_cost, A1_runtime = model.A1(state2, target, control2, c_scheme, u_mat, u_scheme, max_iteration,
                                    algorithm_tolerance, start_step, test_step, controlmax, CGVar = None)
        
        for n in range(A1_bestControl.shape[0]):
            for v in range(A1_bestControl.shape[1]):
                for t in range(A1_bestControl.shape[2] - 2):
                    self.assertAlmostEqual(A1_bestControl[n, v, t], 0., assertion_tolerance)
                    
    def test_A1A2ConvergeForRandomTarget(self):
        if (model.name == "aln"):
            return
        print("test_A1A2ConvergeForRandomTarget")
        
        model.params.duration = duration
        
        c_scheme = np.zeros((len(output_vars), len(output_vars) ))
        c_scheme[0,0] = 1.

        u_mat = np.identity(model.params['N'])
        u_scheme = np.array([[1, 0], [0, 0]])
        
        control1 = self.getRandomControl()    
        target = self.setTargetFromControl(control1)    
        control2 = self.getRandomControl()
        state2 = self.updateState(control2)  
        
        testip, testie, testis = random.uniform(0., 1.), random.uniform(0., 1.), random.uniform(0., 1.)
        cost.setParams(testip, testie, testis)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                                                                          algorithm_tolerance, incl_steps, start_step, test_step)
        
        A1_bestControl, A1_bestState, A1_cost, A1_runtime = model.A1(state2, target, control2, c_scheme, u_mat, u_scheme, max_iteration,
                                    algorithm_tolerance, start_step, test_step, controlmax, CGVar = None)
        
        for n in range(A2_bestControl.shape[0]):
            for v in range(A2_bestControl.shape[1]):
                for t in range(A2_bestControl.shape[2] - 2):
                    print("assert convergence for random input")
                    self.assertAlmostEqual(A2_bestControl[n, v, t], A1_bestControl[n, v, t], assertion_tolerance)
"""

if __name__ == '__main__':
    
    print("----------------------------------------------------")
    print("------------ SINGLE NODE TEST ---FHN ---------------")
    print("----------------------------------------------------")
    
    model = FHNModel()
    
    incl_steps = int(1. + duration/model.params.dt)
    model.params.duration = duration
    
    state0 = model.getZeroState()
    state1 = state0.copy()
    state2 = state0.copy()
    control0 = model.getZeroControl()
            
    target_vars = model.target_output_vars
    output_vars = model.output_vars
    init_vars = model.init_vars
    
    unittest.main()