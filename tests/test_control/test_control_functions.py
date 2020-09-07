import numpy as np
import random

def setInitVarsZero(model, init_vars):
    for iv in range(len(init_vars)):
        model.params[init_vars[iv]] = np.zeros(( model.params[init_vars[iv]].shape ))
        
def getRandomControl(model, cntrl_zeros_pre, controlmin, controlmax):
    control_ = model.getZeroControl()
    for n in range(control_.shape[0]):
        for v in range(control_.shape[1]):
            for t in range(cntrl_zeros_pre, control_.shape[2]):
                control_[n, v, t] = random.uniform(controlmin, controlmax)
    return control_
    
def updateState(model, control_, output_vars):
    state_ = model.getZeroState()
    model.run(control = control_)
    for o_ind in range(len(output_vars)):
        state_[:,o_ind,:] = model[output_vars[o_ind]][:,:]
    return state_
        
    
def setTargetFromControl(model, control_, output_vars, target_vars):
    target_ = model.getZeroTarget()
    model.run(control = control_)
    for o_ind in range(len(output_vars)):
        for t_ind in range(len(target_vars)):
            if (target_vars[t_ind] == output_vars[o_ind]):
                target_[:,t_ind,:] = model[output_vars[o_ind]][:,:]
    return target_