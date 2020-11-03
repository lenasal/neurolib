import numpy as np
import random

from neurolib.models.fhn import FHNModel
from neurolib.models.aln import ALNModel
from neurolib.models.aln_control import Model_ALN_control

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
            for t in range(cntrl_zeros_pre+1, control_.shape[2]):
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
    model.params.mue_ou = np.array( [[0.4]] ) #* np.ones((model.params.N,))
    model.params.mui_ou = np.array( [[0.3]] ) #* np.ones((model.params.N,))
    
def getSchemes(model):
    c_scheme = np.zeros((len(model.output_vars), len(model.output_vars) ))
    c_scheme[0,0] = 1.
    u_mat = np.identity(model.params['N'])
    u_scheme = np.array([[1, 0], [0, 0]])
    return c_scheme, u_mat, u_scheme


def getmodel(i, dur_pre, dur_post):
    if i == "fhn1":
        model_ = FHNModel()
        
    elif i == "aln1":
        model_ = ALNModel()
        dt = model_.params.dt
        maxDelay = min(dur_pre - 2 * dt, dur_post - 2 * dt)
    
        model_.params.signalV = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.de = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.di = np.around( maxDelay * random.uniform(0., 1.), 1)
                        
        model_.params.ext_exc_current = 4. * random.uniform(0., 1.)
        model_.params.ext_inh_current = 4. * random.uniform(0., 1.)
        
        #setParametersALN(model_)
        
    elif i == "aln-control":
        model_ = Model_ALN_control()
        dt = model_.params.dt
        maxDelay = min(dur_pre - 2 * dt, dur_post - 2 * dt)
        
        model_.params.signalV = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.de = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.di = np.around( maxDelay * random.uniform(0., 1.), 1)
        
        model_.params.ext_exc_current = 4. * random.uniform(0., 1.)
        model_.params.ext_inh_current = 4. * random.uniform(0., 1.)
        
        #setParametersALN(model_)
        
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
        maxDelay = min(dur_pre - 2 * dt, dur_post - 2 * dt)
        
        model_.params.signalV = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.de = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.di = np.around( maxDelay * random.uniform(0., 1.), 1)
        
    elif i == "aln2delay":
        coupling12 = random.uniform(0, 1)
        coupling21 = random.uniform(0, 1)
        c_mat = np.array( [[0, coupling21], [coupling12, 0]] )
        
        delay12 = random.uniform(0, 1)
        delay21 = random.uniform(0, 1)
        fiber_matrix = np.array( [[0, delay21], [delay12, 0]] )
        
        model_ = ALNModel(Cmat = c_mat, Dmat = fiber_matrix)
        dt = model_.params.dt
        
        model_.params.signalV = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0., 1.), 1)
        model_.params.de = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0., 1.), 1)
        model_.params.di = round( min(dur_pre - dt, dur_post - dt) * random.uniform(0., 1.), 1)
        
    return model_