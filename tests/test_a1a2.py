import unittest
import numpy as np
import random

from neurolib.models.fhn import FHNModel
from neurolib.models.aln import ALNModel
from neurolib.utils import costFunctions as cost

#model = FHNModel()
model = ALNModel()
assertion_tolerance = 2
        
controlmin, controlmax = -2., 2.
duration = 0.5
algorithm_tolerance = 1e-16
incl_steps = int(1. + duration/model.params.dt)
max_iteration = int(1e3)
start_step = 10.
test_step = 1e-12

model.params.duration = duration
if (model.name == "aln"):
    model.params.signalV = 0.
    model.params.de = 0.
    model.params.di = 0.

state0 = model.getZeroState()
state1 = state0.copy()
state2 = state0.copy()
control0 = model.getZeroControl()
        
target_vars = model.target_output_vars
output_vars = model.output_vars

class TestStringMethods(unittest.TestCase):
        
        
    def setup(self):
        print("setup")
        
    def getRandomControl(self):
        control_ = model.getZeroControl()
        for n in range(control_.shape[0]):
            for v in range(control_.shape[1]):
                for t in range(control_.shape[2]):
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
        
    def test_A2inputControlForPrecisionCostOnly(self):
        
        model.params.duration = duration
        
        control1 = self.getRandomControl()    
        target = self.setTargetFromControl(control1)    
        control2 = self.getRandomControl()
        
        testip, testie, testis = 1., 0., 0.
        cost.setParams(testip, testie, testis)

        A2_bestControl, A2_bestState, A2_cost, A2_runtime = model.A2(control2, target, max_iteration,
                                                                          algorithm_tolerance, incl_steps, start_step, test_step)
        
        for n in range(A2_bestControl.shape[0]):
            for v in range(A2_bestControl.shape[1]):
                for t in range(A2_bestControl.shape[2] - 2):
                    self.assertAlmostEqual(A2_bestControl[n, v, t], control1[n, v, t], assertion_tolerance)


    def test_A1inputControlForPrecisionCostOnly(self):
        if (model.name == "aln"):
            return
        
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
                    self.assertAlmostEquals(A1_bestControl[n, v, t], control1[n, v, t], assertion_tolerance)

    def test_A2zeroControlForEnergyCostOnly(self):
        
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


if __name__ == '__main__':
    unittest.main()