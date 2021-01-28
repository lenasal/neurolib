import numpy as np
import random

from neurolib.models.fhn import FHNModel
from neurolib.models.aln import ALNModel
from neurolib.models.rate_control import RateModel
from neurolib.models.aln_control import Model_ALN_control

def setInitVarsZero(model, init_vars):
    if model.name == "fhn":
        for iv in range(len(init_vars)):
            model.params[init_vars[iv]] = np.zeros(( model.params[init_vars[iv]].shape ))
    else:
       setParametersALN(model)
       
def setmaxmincontrol(cntrl_vars_, max_cntrl_c, min_cntrl_c, max_cntrl_r, min_cntrl_r):
    max_cntrl = np.zeros(( 6 ))
    min_cntrl = np.zeros(( 6 ))
    
    max_cntrl[0] = 1e2 * max_cntrl_c
    min_cntrl[0] = 1e2 * min_cntrl_c
    max_cntrl[1] = 1e2 * max_cntrl_c
    min_cntrl[1] = 1e2 * min_cntrl_c
    max_cntrl[2] = 2. * max_cntrl_r
    min_cntrl[2] = 2. * min_cntrl_r
    max_cntrl[3] = 2. * max_cntrl_r
    min_cntrl[3] = 2. * min_cntrl_r
    max_cntrl[4] = 2. * max_cntrl_r
    min_cntrl[4] = 2. * min_cntrl_r
    max_cntrl[5] = 2. * max_cntrl_r
    min_cntrl[5] = 2. * min_cntrl_r
    
    return max_cntrl, min_cntrl
    
    """
    if cntrl_vars_ == [0]:
        max_cntrl[0] = 1e2 * max_cntrl_c
        min_cntrl[0] = 1e2 * min_cntrl_c
    elif cntrl_vars_ == [1]:
        max_cntrl[1] = 1e2 * max_cntrl_c
        min_cntrl[1] = 1e2 * min_cntrl_c
    elif cntrl_vars_ == [2]:
        max_cntrl[2] = 2. * max_cntrl_r
        min_cntrl[2] = 2. * min_cntrl_r
    elif cntrl_vars_ == [3]:
        max_cntrl[2] = 2. * max_cntrl_r
        min_cntrl[2] = 2. * min_cntrl_r
    elif cntrl_vars_ == [0,1]:
        max_cntrl[0] = 1e2 * max_cntrl_c
        min_cntrl[0] = 1e2 * min_cntrl_c
        max_cntrl[1] = 1e2 * max_cntrl_c
        min_cntrl[1] = 1e2 * min_cntrl_c
    elif cntrl_vars_ == [0,2]:
        max_cntrl[0] = 1e2 * max_cntrl_c
        min_cntrl[0] = 1e2 * min_cntrl_c
        max_cntrl[2] = 2. * max_cntrl_r
        min_cntrl[2] = 2. * min_cntrl_r
    elif cntrl_vars_ == [0,3]:
        max_cntrl[0] = 1e2 * max_cntrl_c
        min_cntrl[0] = 1e2 * min_cntrl_c
        max_cntrl[3] = 2. * max_cntrl_r
        min_cntrl[3] = 2. * min_cntrl_r
    elif cntrl_vars_ == [1,2]:
        max_cntrl[1] = 1e2 * max_cntrl_c
        min_cntrl[1] = 1e2 * min_cntrl_c
        max_cntrl[2] = 2. * max_cntrl_r
        min_cntrl[2] = 2. * min_cntrl_r
    elif cntrl_vars_ == [0,1,2]:
        max_cntrl[0] = 1e2 * max_cntrl_c
        min_cntrl[0] = 1e2 * min_cntrl_c
        max_cntrl[1] = 1e2 * max_cntrl_c
        min_cntrl[1] = 1e2 * min_cntrl_c
        max_cntrl[2] = 2. * max_cntrl_r
        min_cntrl[2] = 2. * min_cntrl_r
    return max_cntrl, min_cntrl
    """
        
def getRandomControl(model, cntrl_zeros_pre, c_controlmin, c_controlmax, r_controlmin, r_controlmax, control_variables_ = [0,1]):
    control_ = model.getZeroControl()     
    maxDelay_ndt = getDelay_ndt(model)
        
    for n in range(control_.shape[0]):
        for v in control_variables_:
            for t in range(cntrl_zeros_pre+1, control_.shape[2]-1-maxDelay_ndt):
                if v == 0 or v == 1:
                    control_[n, v, t] = random.uniform(c_controlmin, c_controlmax)
                elif v in [2,3,4,5]:
                    control_[n, v, t] = random.uniform(r_controlmin, r_controlmax)
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
    model.params.rates_exc_init = np.array( [[1. * 0.01 * 0.5 ]] )
    model.params.rates_inh_init = np.array( [[1. * 0.01 * 0.5 ]] )
    model.params.mufe_init = np.array( [[1. * 3. * 0.5 ]] )  # mV/ms
    model.params.mufi_init = np.array( [[1. * 3. * 0.5 ]] )  # mV/ms
    model.params.IA_init = np.array( [[1. * 200. * 0.5 ]] )  # pA
    model.params.seem_init = np.array( [[1. * 0.5 * 0.5 ]] )
    model.params.seim_init = np.array( [[1. * 0.5 * 0.5 ]] )   
    model.params.seev_init = np.array( [[1. * 0.01 * 0.5 ]] )
    model.params.seiv_init = np.array( [[1. * 0.01 * 0.5 ]] )
    model.params.siim_init = np.array( [[1. * 0.5 * 0.5 ]] )
    model.params.siem_init = np.array( [[1. * 0.5 * 0.5 ]] )
    model.params.siiv_init = np.array( [[1. * 0.01 * 0.5 ]] )
    model.params.siev_init = np.array( [[1. * 0.01 * 0.5 ]] )
    model.params.mue_ou = np.array( [[1. * 0.4]] ) #* np.ones((model.params.N,))
    model.params.mui_ou = np.array( [[1. * 0.3]] ) #* np.ones((model.params.N,))
    
def getSchemes(model):
    c_scheme = np.zeros((len(model.output_vars), len(model.output_vars) ))
    c_scheme[0,0] = 1.
    u_mat = np.identity(model.params['N'])
    u_scheme = np.array([[1, 0], [0, 0]])
    return c_scheme, u_mat, u_scheme

def getDelay_ndt(model_):
    maxDelay = max(model_.params.di, model_.params.signalV, model_.params.de)
    maxDelay_ndt = int(maxDelay / model_.params.dt)
    return maxDelay_ndt

def getmodel(i, dur_pre, dur_post):
    if i == "fhn1":
        model_ = FHNModel()
        
    elif i == "aln1":
        model_ = ALNModel()
        dt = model_.params.dt
        maxDelay = min( max(0., dur_pre - 2 * dt), max(0., dur_post - 2 * dt) )
        model_.params.sigma_ou = 0.
    
        model_.params.signalV = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.de = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.di = np.around( maxDelay * random.uniform(0., 1.), 1)
                        
        # should not have too big impact
        model_.params.ext_exc_current = random.uniform(0., 1.2)
        model_.params.ext_inh_current = random.uniform(0., 1.2)
        
        model_.params.mue_ext_mean = random.uniform(0., 4.)
        model_.params.mui_ext_mean = random.uniform(0., 4.)
        
        model_.params.sigmae_ext = model_.params.mue_ext_mean * random.uniform(0.5, 1.)
        model_.params.sigmai_ext = model_.params.mui_ext_mean * random.uniform(0.5, 1.)
        
    elif i == "rate_control":
        #model_ = RateModel()
        model_ = ALNModel()
        
        dt = model_.params.dt
        maxDelay = 0.5 * min( max(0., dur_pre - 2 * dt), max(0., dur_post - 2 * dt) )
        model_.params.sigma_ou = 0.
    
        model_.params.signalV = 0.#np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.de = np.around( 0.1 + ( maxDelay - 0.1 ) * random.uniform(0., 1.), 1)
        model_.params.di = np.around( 0.1 + ( maxDelay - 0.1 ) * random.uniform(0., 1.), 1)
                        
        # derivative of interpolation function wrt mu close to zero for mu < 0.6
        model_.params.ext_exc_current = 0.3 * random.uniform(0.6, 1.2)
        model_.params.ext_inh_current = 0.3 * random.uniform(0.6, 1.2)
        
        model_.params.mue_ext_mean = random.uniform(0., 4.)
        model_.params.mui_ext_mean = random.uniform(0., 4.)
        
        model_.params.sigmae_ext = max(0.5, model_.params.mue_ext_mean * random.uniform(0.5, 1.) )
        model_.params.sigmai_ext = max(0.5, model_.params.mui_ext_mean * random.uniform(0.5, 1.) )
        
        model_.params.interpolate_rate = False
        model_.params.interpolate_V = False
        model_.params.interpolate_tau = False
        
        """
        model_.params.cee = 1.
        model_.params.cei = 1.
        model_.params.cie = 1.
        model_.params.cii = 1.
        
        model_.params.Jee_max = 1.
        model_.params.Jei_max = 1.
        model_.params.Jie_max = 1.
        model_.params.Jii_max = 1.
        
        model_.params.tau_se = 1.
        model_.params.tau_si = 1.
        
        model_.params.C = 1.
        model_.params.gL = 1.
        model_.params.Ke = 1000.
        model_.params.Ki = 1000.
        
        #model_.params.a = 0.
        #model_.params.b = 0.
        """
        
        #setParametersALN(model_)
        
    elif i == "aln-control":
        model_ = Model_ALN_control()
        dt = model_.params.dt
        maxDelay = min( max(0., dur_pre - 2 * dt), max(0., dur_post - 2 * dt) )
        
        model_.params.signalV = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.de = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.di = np.around( maxDelay * random.uniform(0., 1.), 1)
        
        model_.params.ext_exc_current = random.uniform(1., 4.)
        model_.params.ext_inh_current = random.uniform(1., 4.)
        
        model_.params.sigmae_ext = random.uniform(1., 4.)
        model_.params.sigmai_ext = random.uniform(1., 4.)
        
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
        maxDelay = min( max(0., dur_pre - 2 * dt), max(0., dur_post - 2 * dt) )
        
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
        maxDelay = min( max(0., dur_pre - 2 * dt), max(0., dur_post - 2 * dt) )
        
        model_.params.signalV = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.de = np.around( maxDelay * random.uniform(0., 1.), 1)
        model_.params.di = np.around( maxDelay * random.uniform(0., 1.), 1)
        
    return model_