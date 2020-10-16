import numpy as np
import random

def setInitVarsZero(model, init_vars):
    if model.name == "fhn":
        for iv in range(len(init_vars)):
            model.params[init_vars[iv]] = np.zeros(( model.params[init_vars[iv]].shape ))
    else:
        setParametersALN(model)
        
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
    model.params.rates_exc_init = np.array( [[0.01 * 0.5 ]] )
    model.params.rates_inh_init = np.array( [[0.01 * 0.5 ]] )
    model.params.mufe_init = np.array( [[3. * 0.5 ]] )  # mV/ms
    model.params.mufi_init = np.array( [[3. * 0.5 ]] )  # mV/ms
    model.params.IA_init = np.array( [[200. * 0.5 ]] )  # pA
    model.params.seem_init = np.array( [[0.5 * 0.5 ]] )
    model.params.seim_init = np.array( [[0.5 * 0.5 ]] )   
    model.params.seev_init = np.array( [[0.01 * 0.5 ]] )
    model.params.seiv_init = np.array( [[0.01 * 0.5 ]] )
    model.params.siim_init = np.array( [[0.5 * 0.5 ]] )
    model.params.siem_init = np.array( [[0.5 * 0.5 ]] )
    model.params.siiv_init = np.array( [[0.01 * 0.5 ]] )
    model.params.siev_init = np.array( [[0.01 * 0.5 ]] )