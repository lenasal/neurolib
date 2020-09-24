import numpy as np
import random

def setInitVarsZero(model, init_vars):
    for iv in range(len(init_vars)):
        model.params[init_vars[iv]] = np.zeros(( model.params[init_vars[iv]].shape ))
    if model.name == "alnSimp":
        model.params.mufe_init = np.array( [4.] )
    if model.name == "aln":
        model.params.mufe_init = np.array( [2.] )
        model.params.mufi_init = np.array( [2.] )
        model.params.mue_ext_mean = 0.0
        model.params.mui_ext_mean = 0.0
        
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

def setParametersALN(model):
    model.params.ext_exc_current = 1.0
    model.params.ext_inh_current = 1.0
    
    model.params.cee = 0.
    model.params.cei = 0.
    model.params.cie = 0.
    model.params.cii = 0.
    model.params.Ke = 0.
    model.params.Ki = 0.
    model.params.Jee_max = 0.
    model.params.Jei_max = 1e-30
    model.params.Jie_max = 1e-30
    model.params.Jii_max = 1e-30
    
    model.params.cee = random.uniform(0.1, 0.5)
    #model.params.cie = random.uniform(0.1, 0.5)
    #model.params.cei = random.uniform(0.3, 0.7)
    #model.params.cii = random.uniform(0.3, 0.7)
    
    model.params.Ke = int(random.uniform(600, 1000) )
    #model.params.Ki = int(random.uniform(100, 300) )
    
    model.params.Jee_max = random.uniform(2., 3.)
    #model.params.Jie_max = random.uniform(2., 3.)
    #model.params.Jei_max = random.uniform(-4., -3.)
    #model.params.Jii_max = random.uniform(-2., -1.)