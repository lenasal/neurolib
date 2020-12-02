import numpy as np
import logging
import numba
from . import costFunctions as cost

def updateState(model, control_):
    # set initial conditions once in other function
    state_ = model.getZeroState()
    output_vars = model.output_vars
    model.run(control = control_)    
    for i in range(len(output_vars)):
        state_[:,i,:] = model[output_vars[i]][:,:]
    #print("update state: ", state_[0,:,:3])
    return state_

def updateFullState(model, control_, state_vars_):
    state_ = model.getZeroFullState()
    model.run(control=control_)
    for sv in range(len(state_vars_)):
        state_[:,sv,:] = model.state[state_vars_[sv]][:,:]
    return state_

def get_output_from_full(model, state_):
    out_state_ = model.getZeroState()
    out_state_[:,0,:] = state_[:,0,:]
    out_state_[:,1,:] = state_[:,1,:]
    out_state_[:,2,:] = state_[:,4,:]
    return out_state_

def get_full_from_output_t(model, out_grad_):
    full_grad_ = np.zeros((model.params.N, len(model.state_vars) ))
    full_grad_[:,0] = out_grad_[:,0]
    full_grad_[:,1] = out_grad_[:,1]
    return full_grad_

def get_full_from_output(model, out_):
    full_ = model.getZeroFullState()
    full_[:,0,:] = out_[:,0,:]
    full_[:,1] = out_[:,1,:]
    return full_

def update_delayed_state(model, delay_state_vars_, state_vars_, init_vars_, startind_):
    for sv, iv in zip( range(len(state_vars_)), range(len(init_vars_)) ):
        if (state_vars_[sv] in init_vars_[iv]):
            if model.params[init_vars_[iv]].ndim == 2:
                if startind_ == 1:
                    model.params[init_vars_[iv]][:,0] = model.state[state_vars_[sv]][:,-2]
                else:
                    delay_state_vars_[:, sv, :-1] = delay_state_vars_[:, sv, 1:]
                    delay_state_vars_[:, sv, -1] = model.state[state_vars_[sv]][:,-2]
                    model.params[init_vars_[iv]][:,:] = delay_state_vars_[:, sv, :]
            else:
                if model.state[state_vars_[sv]].ndim == 2:
                    model.params[init_vars_[iv]] = model.state[state_vars_[sv]][:,-2]
                else:
                    model.params[init_vars_[iv]] = model.state[state_vars_[sv]]
        else:
            logging.error("Initial and state variable labelling does not agree.")

def adjust_shape_init_params(model, init_vars_, startind_):
    for iv in range(len(init_vars_)):
        if model.params[init_vars_[iv]].ndim == 2:
            if (model.params[init_vars_[iv]].shape[1] <= 1):
                model.params[init_vars_[iv]] = np.dot( model.params[init_vars_[iv]], np.ones((1, startind_)) )
        else:
            model.params[init_vars_[iv]] = model.params[init_vars_[iv]][0] * np.ones((1, startind_))

def get_init(model, init_vars_, state_vars_, startind_, delay_state_vars_):
    IC_ = np.zeros( (model.params.N, len(init_vars_), startind_) )
    
    for iv in range(len(init_vars_)):
        if ( type(model.params[init_vars_[iv]]) == np.float64 or type(model.params[init_vars_[iv]]) == float ):
            IC_[:, iv, 0] = model.params[init_vars_[iv]]
        elif len(model.params[init_vars_[iv]][:].shape) == 2:
            if startind_ == 1:
                IC_[:, iv, 0] = model.params[init_vars_[iv]][:,0]
            else:
                IC_[:, iv, :] = delay_state_vars_[:, iv, :]
        else:
            IC_[:, iv, 0] = model.params[init_vars_[iv]][:]

    return IC_

def set_init(model, IC_, init_vars_, state_vars_, startind_):
    for iv in range(len(init_vars_)):
        if model.params[init_vars_[iv]].ndim == 1:
            model.params[init_vars_[iv]][:] = IC_[:, iv, 0]
        else:
            if startind_ == 1:
                model.params[init_vars_[iv]][:,0] = IC_[:, iv, 0] 
            else: 
                model.params[init_vars_[iv]] = IC_[:, iv, :]

def update_init(model, init_vars_, state_vars_):
    for iv, sv in zip( range(len(init_vars_)), range(len(state_vars_)) ):
        if state_vars_[sv] in init_vars_[iv]:
            if ( type(model.params[init_vars_[iv]]) == np.float64 or type(model.params[init_vars_[iv]]) == float ):
                model.params[init_vars_[iv]] = model.state[state_vars_[sv]][:,-1]
            elif model.params[init_vars_[iv]].ndim == 2:
                model.params[init_vars_[iv]][:,0] = model.state[state_vars_[sv]][:,-1]
            else:
                model.params[init_vars_[iv]] = model.state[state_vars_[sv]][:,-1]
                    
def update_init_delayed(model, delay_state_vars_, init_vars_, state_vars_, t_pre_ndt_, startind_):
    for iv, sv in zip( range(len(init_vars_)), range(len(state_vars_)) ):
        if state_vars_[sv] in init_vars_[iv]:
            if model.params[init_vars_[iv]].ndim == 2:
                if (t_pre_ndt_ < startind_):
                    delay_state_vars_[:, sv, :-1] = model.params[init_vars_[iv]][:,1:]
                    delay_state_vars_[:, sv, -t_pre_ndt_:] = model.state[state_vars_[sv]][:,:]
                    model.params[init_vars_[iv]][:,:] = delay_state_vars_[:, sv, :]
                else:
                    model.params[init_vars_[iv]] = model.state[state_vars_[sv]][:,-startind_:]
                    delay_state_vars_[:, sv, :] = model.state[state_vars_[sv]][:,-startind_:]
            else:
                model.params[init_vars_[iv]] = model.state[state_vars_[sv]]

def test_step(model, N, V, T, state_, target_, control_, dir_, test_step_ = 1e-12, variables_ = [0,1]):
    dt = model.params['dt']
    cost0_int_ = cost.f_int(N, V, T, dt, state_, target_, control_, v_ = variables_)
    
    test_control_ = control_ + test_step_ * dir_
    state1_ = updateState(model, test_control_)
    cost1_int_ = cost.f_int(N, V, T, dt, state1_, target_, test_control_, v_ = variables_)
    #print("test step size computation : ------ step size, cost1, cost0 : ", test_step_, cost1_int_, cost0_int_)
        
    if (cost1_int_ < cost0_int_):
        return test_step_, cost1_int_
    else:
        return 0., cost0_int_
   
@numba.njit
def setmaxcontrol(n_control_vars, control_, max_control_):
    for j in range(len(control_[0,0,:])):
        for v in range(n_control_vars):
            if control_[0,v,j] > max_control_:
                control_[0,v,j] = max_control_
            elif control_[0,v,j] < - max_control_:
                control_[0,v,j] = - max_control_
    return control_
    
def step_size(model, N, V, T, dt, state_, target_, control_, dir_, start_step_ = 20., max_it_ = 1000,
              bisec_factor_ = 2., max_control_ = 20., tolerance_ = 1e-16, substep_ = 0.1, variables_ = [0,1], alg = "A1"):
    
    
    cost0_int_ = cost.f_int(N, V, T, dt, state_, target_, control_, v_ = variables_)
    cost_min_int_ = cost0_int_
    step_ = start_step_
    step_min_ = 0.
    
    factor = 2.**7
        
    start_step_out_ = start_step_
    
    for i in range(max_it_):
        test_control_ = control_ + step_ * dir_
        
        # include maximum control value to assure no divergence
        if ( np.amax(np.absolute(test_control_)) > max_control_):
            test_control_ = setmaxcontrol(V, test_control_, max_control_)
            
        state1_ = updateState(model, test_control_)
        
        
        cost1_int_ = cost.f_int(N, V, T, dt, state1_, target_, test_control_, v_ = variables_)
        
        #print("step, cost, initial cost = ", step_, cost1_int_, cost0_int_)
        
        if (step_ * np.amax(np.absolute(dir_)) < tolerance_ * 1e-3):
            #print("test control change smaller than tolerance, return zero step")
            return 0., cost0_int_, start_step_

        if (cost1_int_ < cost_min_int_):
            cost_min_int_ = cost1_int_
            step_min_ = step_
            
        # return smallest step size before cost is increasing again
        elif (cost1_int_ > cost_min_int_ and cost_min_int_ < cost0_int_):
            
            if (i == 1 and alg == "A1"):
                step_ = factor * start_step_
                #print("too small start step, increase to ", step_)
                return step_size(model, N, V, T, dt, state_, target_, control_, dir_, start_step_ = step_, max_it_ = max_it_,
                                 bisec_factor_ = bisec_factor_, max_control_ = max_control_, tolerance_ = tolerance_,
                                 substep_ = substep_, variables_ = variables_)
            elif (step_ < start_step_ / (2. * factor) and alg == "A1"):
                start_step_ /= factor
                #print("too large start step, decrease to ", start_step_)
            

            # iterate between step_range[0] and [2] more granularly
            substep = substep_
            
            step_min_up, cost_min_int_ = scan(model, N, V, T, dt, substep, control_, step_min_, dir_, target_,
                                                  cost_min_int_, max_control_, variables_)

            substep = - substep_
            step_min_down, cost_min_int_ = scan(model, N, V, T, dt, substep, control_, step_min_, dir_, target_,
                                                cost_min_int_, max_control_, variables_)

            
            #print("scan done")
            if (step_min_up > step_min_ ):
                if (step_min_down == step_min_):
                    result = step_min_up, cost_min_int_, start_step_
                elif (step_min_down < step_min_):
                    result = step_min_down, cost_min_int_, start_step_
            elif (step_min_down < step_min_):
                result = step_min_down, cost_min_int_, start_step_
            else:
                result = step_min_, cost_min_int_, start_step_
            
            return result
        
        if (i == max_it_-1):
            if (max_it_ != 1):
                print(" max iteration reached, step size = ", step_)
            return step_min_, cost_min_int_, start_step_
        
        
        step_ /= bisec_factor_
        
def scan(model_, N, V, T, dt_, substep_, control_, step_min_, dir_, target_, cost_min_int_, max_control_, variables_ = [0,1]):
    cntrl_ = control_ + ( 1. + substep_ ) * step_min_ * dir_
    cntrl_ = setmaxcontrol(V, cntrl_, max_control_)
    state_ = updateState(model_, cntrl_)
    cost_int = cost.f_int(N, V, T, dt_, state_, target_, cntrl_, v_ = variables_)
    step_min1_ = step_min_
    
    while (cost_int < cost_min_int_):
        cost_min_int_ = cost_int
        step_min1_ += substep_ * step_min_
        
        cntrl_ += substep_ * step_min_ * dir_
        cntrl_ = setmaxcontrol(V, cntrl_, max_control_)
        state_ = updateState(model_, cntrl_)
        cost_int = cost.f_int(N, V, T, dt_, state_, target_, cntrl_, v_ = variables_)
        
        
    return step_min1_, cost_min_int_

def set_pre_post(i1, i2, bc_, bs_, best_control_, state_pre_, state_, state_post_, state_vars, a, b):
    
    if (i2 != 0 and i1 != 0):   
        bc_[:,:,i1:-i2] = best_control_[:,:,:]
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        check_pre(i1, bs_, state_, state_vars, a, b)
        bs_[:,:,i1:-i2] = state_[:,:,:]
        bs_[:,:,-i2:] = state_post_[:,:,1:]
        check_post(i2, bs_, state_post_, state_vars, a, b)
        
    elif (i2 == 0 and i1 != 0):
        bc_[:,:,i1:] = best_control_[:,:,:]
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        check_pre(i1, bs_, state_, state_vars, a, b)
        bs_[:,:,i1:] = state_[:,:,:]
        
    elif (i2 != 0 and i1 == 0):
        bc_[:,:,:-i2] = best_control_[:,:,:]
        bs_[:,:,:-i2] = state_[:,:,:]
        bs_[:,:,-i2:] = state_post_[:,:,:]
        check_post(i2, bs_, state_post_, state_vars, a, b)
                    
    else:
        bc_[:,:,:] = best_control_[:,:,:]
        bs_[:,:,:] = state_[:,:,:]
        
    return bc_, bs_

def check_pre(i1, bs_, state_, state_vars, a, b):
    for n in range(bs_.shape[0]):
        for v in range(bs_.shape[1]):
            if ( state_vars[v] == "Vmean_exc" and (a == 0. or b == 0.) ):
                if np.abs(bs_[n,v,i1] - state_[n,v,0]) > 1e-1:
                    logging.error("Problem in initial value trasfer")
                    print("Problem in initial value trasfer: ", state_vars[v], bs_[n,v,i1], state_[n,v,0])
            elif np.abs(bs_[n,v,i1] - state_[n,v,0]) > 1e-8:
                logging.error("Problem in initial value trasfer")
                print("Problem in initial value trasfer: ", state_vars[v], bs_[n,v,i1], state_[n,v,0])
                
def check_post(i2, bs_, state_post_, state_vars, a, b):
    for n in range(bs_.shape[0]):
        for v in range(bs_.shape[1]):
            if state_vars[v] == "Vmean_exc" and (a == 0. or b == 0.):
                if np.abs(bs_[n,v,-i2-1] - state_post_[n,v,0]) > 1e-8:
                    logging.error("Problem in initial value trasfer")
                    print("Problem in initial value trasfer: ", state_vars[v], bs_[n,v,-i2-1], state_post_[n,v,0])
            elif np.abs(bs_[n,v,-i2-1] - state_post_[n,v,0]) > 1e-8:
                logging.error("Problem in initial value trasfer")
                print("Problem in initial value trasfer: ", state_vars[v], bs_[n,v,-i2-1], state_post_[n,v,0])

def adapt_step(control_, ind_node, ind_var, start_step_, dir_, max_control_):
    start_st_ = start_step_
    max_index = -1
    max_cntrl = max_control_
    for k in range(control_.shape[2]):
        if ( np.abs(control_[ind_node,ind_var,k] + start_step_ * dir_[ind_node,ind_var,k]) > max_cntrl ):
            max_index = k
            max_cntrl = np.abs(control_[ind_node, ind_var,k] + start_step_ * dir_[ind_node, ind_var,k])
    if max_index != -1:
        start_st_ = ( max_control_ - np.abs(control_[ind_node,ind_var,max_index]) ) / np.abs(dir_[ind_node,ind_var,max_index])
    
    return start_st_

# update rule for conjugate directions according to Hestenes-Stiefel
def betaHS(N, n_control_vars, grad0_, grad1_, dir0_):
    betaHS = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            denominator = np.dot( dir0_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            #print("numerator = ", numerator)
            #print("denominator = ", denominator)
            if np.abs(denominator) > 1e-6 :
                betaHS[n,v] = numerator / denominator
    return betaHS

# update rule for conjugate directions according to Fletcher-Reeves
def betaFR(N, n_control_vars, grad0_, grad1_):
    betaFR = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], grad1_[n,v,:] )
            denominator = np.dot( grad0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaFR[n,v] = numerator / denominator
    return betaFR

# update rule for conjugate directions according to Polak-Ribiere
def betaPR(N, n_control_vars, grad0_, grad1_):
    betaPR = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            denominator = np.dot( grad0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaPR[n,v] = numerator / denominator
    return betaPR

# update rule for conjugate directions according to Hager-Zhang
def betaHZ(N, n_control_vars, grad0_, grad1_, dir0_):
    betaHZ = np.zeros(( N, n_control_vars ))
    eta = 0.01
    for n in range(N):
        for v in range(n_control_vars):
            diff = grad1_[n,v,:] - grad0_[n,v,:]
            denominator = np.dot( dir0_[n,v,:], diff )
            if np.abs(denominator) > 1e-6 :
                beta0 = np.dot( diff - 2. * np.dot( diff, diff ) * dir0_[n,v,:] / denominator,
                           grad1_[n,v,:] / denominator )
            else:
                beta0 = - 1e10
            numerator = np.dot( grad0_[n,v,:], grad0_[n,v,:] )
            denominator = np.dot( dir0_[n,v,:], dir0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                eta0 = - min( eta, np.sqrt( numerator ) ) / np.sqrt( denominator )
            else:
                eta0 = 0.
            betaHZ[n,v] = max(beta0, eta0)
    return betaHZ