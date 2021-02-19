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

def test_step(model, N, V, T, dt, state_, target_, control_, dir_, cost0_, ip_, ie_, is_, test_step_ = 1e-12, prec_variables_ = [0,1]):
    cost0_int_ = cost0_
    
    test_control_ = control_ + test_step_ * dir_
    state1_ = updateState(model, test_control_)
    cost1_int_ = cost.f_int(N, V, T, dt, state1_, target_, test_control_, ip_, ie_, is_, v_ = prec_variables_)
    #print("test step size computation : ------ step size, cost1, cost0 : ", test_step_, cost1_int_, cost0_int_)
        
    if (cost1_int_ < cost0_int_):
        return test_step_, cost1_int_
    else:
        return 0., cost0_int_
   
@numba.njit
def setmaxcontrol(n_control_vars, control_, max_control_, min_control_):
    for j in range(len(control_[0,0,:])):
        for v in range(n_control_vars):
            #print("set control max ", v, max_control_[v], min_control_[v])
            if control_[0,v,j] > max_control_[v]:
                control_[0,v,j] = max_control_[v]
            elif control_[0,v,j] < min_control_[v]:
                control_[0,v,j] = min_control_[v]
    return control_

@numba.njit
def scalemaxcontrol(control_, max_control_, min_control_):
    max_val_ = np.amax(control_)
    if max_val_ != 0 and max_val_> max_control_:
        scale_factor_ = np.abs(max_control_ / max_val_)
        control_ *= scale_factor_
    min_val_ = np.amin(control_)
    if min_val_ != 0. and min_control_ != 0. and min_val_< min_control_:
        scale_factor_ = np.abs(min_control_ / min_val_)
        control_ *= scale_factor_
    return control_

def StrongWolfePowellLineSearch():
    # would require to compute new gradient for each step
    return None

# backtracking according to Armijo-Goldstein
def AG_line_search(model, N, V, T, dt, state_, target_, control_, dir_, ip_, ie_, is_, start_step_ = 20., max_it_ = 1000,
              bisec_factor_ = 0.5, max_control_ = [20., 20., 0.2, 0.2], min_control_ = [-20., -20., 0., 0.],
              tolerance_ = 1e-16, substep_ = 0.1, variables_ = [0,1], alg = "A1", control_parameter = 0.02, grad_ = None):
    
    m_ = np.zeros(( N, V ))
    for n_ in range(N):
        for v_ in range(V):
            m_[n_, v_] = np.dot(dir_[n_,v_,:], grad_[n_,v_,:])
    
    t_ = - control_parameter * np.sum(m_)
    cost0_ = cost.f_int(N, V, T, dt, state_, target_, control_, ip_, ie_, is_, v_ = variables_)
    step_ = start_step_
    
    for j in range(max_it_):
                
        test_control_ = control_ + step_ * dir_
        test_control_ = setmaxcontrol(V, test_control_, max_control_, min_control_)
        state1_ = updateState(model, test_control_)
        cost1_ = cost.f_int(N, V, T, dt, state1_, target_, test_control_, v_ = variables_)
        
        #print("cost0 = ", cost0_)
        #print("cost1 = ", cost1_)
        #print("step, t_ = ", step_, t_)
        
        if cost0_ - cost1_ - step_ * t_ >= 0.:
            return step_, cost1_, start_step_
        
        step_ *= bisec_factor_
        
    print("max iteration reached, condition not satisfied")
    return 0., cost0_, start_step_
    
    
    
def step_size(model, N, V, T, dt, state_, target_, control_, dir_, ip_, ie_, is_, start_step_ = 20., max_it_ = 1000,
              bisec_factor_ = 2., max_control_ = [20., 20., 0.2, 0.2], min_control_ = [-20., -20., 0., 0.],
              tolerance_ = 1e-16, substep_ = 0.1, variables_ = [0,1], alg = "A1", control_parameter = 0.2, grad_ = None):
    
    """
    print("into step size computation cost")
    print("exc rate = ", state_[0,0,:])
    print("target = ", target_[0,0,:])
    print("control = ", control_[0,1,:])
    print("direction = ", dir_[0,1,:])
    print("variables = ", variables_)
    """
    
    cost0_int_ = cost.f_int(N, V, T, dt, state_, target_, control_, ip_, ie_, is_, v_ = variables_)
    
    #print("first cost = ", cost0_int_)
    cost_min_int_ = cost0_int_
    step_ = start_step_
    step_min_ = 0.
    
    factor = 2.**7
        
    start_step_out_ = start_step_
    
    #print("try again")
    
    
    for i in range(max_it_):
        test_control_ = control_ + step_ * dir_
        #print("test control = ", test_control_[0,3,5:10], max_control_, min_control_)
        # include maximum control value to assure no divergence
        #if ( np.amax(test_control_) > max_control_ or np.amin(test_control_) < min_control_):
        #test_control_ = scalemaxcontrol(test_control_, max_control_, min_control_)
        test_control_ = setmaxcontrol(V, test_control_, max_control_, min_control_)
        
        state1_ = updateState(model, test_control_)
                
        
        cost1_int_ = cost.f_int(N, V, T, dt, state1_, target_, test_control_, ip_, ie_, is_, v_ = variables_)
        
        #print("test control = ", test_control_[0,3,5:10])
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
                return step_size(model, N, V, T, dt, state_, target_, control_, dir_, ip_, ie_, is_,
                                 start_step_ = step_, max_it_ = max_it_,
                                 bisec_factor_ = bisec_factor_, max_control_ = max_control_, min_control_ = min_control_,
                                 tolerance_ = tolerance_, substep_ = substep_, variables_ = variables_)
            elif (step_ < start_step_ / (2. * factor) and alg == "A1"):
                start_step_ /= factor
                #print("too large start step, decrease to ", start_step_)
            

            # iterate between step_range[0] and [2] more granularly
            substep = substep_
            
            step_min_up, cost_min_int_ = scan(model, N, V, T, dt, substep, control_, step_min_, dir_, target_,
                                                  cost_min_int_, max_control_, min_control_, ip_, ie_, is_, variables_)

            substep = - substep_
            step_min_down, cost_min_int_ = scan(model, N, V, T, dt, substep, control_, step_min_, dir_, target_,
                                                cost_min_int_, max_control_, min_control_, ip_, ie_, is_, variables_)

            
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
                
            ("result from scan : ", result)
            
            return result
        
        if (i == max_it_-1):
            if (max_it_ != 1):
                print(" max iteration reached, step size = ", step_)
            return step_min_, cost_min_int_, start_step_
        
        
        step_ /= bisec_factor_
        
def scan(model_, N, V, T, dt_, substep_, control_, step_min_, dir_, target_, cost_min_int_, max_control_,
         min_control_, ip_, ie_, is_, variables_ = [0,1]):
    #print("initial control = ", control_)
    #print("direction = ", dir_)
    i = 1.
    cntrl_ = control_ + ( 1. + i * substep_ ) * step_min_ * dir_
    #print("scan control = ", cntrl_[0,2,:])
    #cntrl_ = scalemaxcontrol(cntrl_, max_control_, min_control_)
    cntrl_ = setmaxcontrol(V, cntrl_, max_control_, min_control_)
    #print("scan control after setting = ", cntrl_[0,2,:])
    state_ = updateState(model_, cntrl_)
    cost_int = cost.f_int(N, V, T, dt_, state_, target_, cntrl_, ip_, ie_, is_, v_ = variables_)
    step_min1_ = step_min_
    #print("cost = ", cost_int)
    #print("previous step = ", step_min1_)

    
    while (i <= 10. and cost_int < cost_min_int_):
        #print("loop")
        i += 1.
        cost_min_int_ = cost_int
        step_min1_ += substep_ * step_min_
        #print("step = ", step_min1_)  
        
        # new control
        cntrl_ = control_ + ( 1. + i * substep_ ) * step_min_ * dir_
        #print("scan control = ", cntrl_[0,2,:])
        #cntrl_ = scalemaxcontrol(cntrl_, max_control_, min_control_)
        cntrl_ = setmaxcontrol(V, cntrl_, max_control_, min_control_)
        #print("scan control after setting = ", cntrl_[0,2,:])
        state_ = updateState(model_, cntrl_)
        cost_int = cost.f_int(N, V, T, dt_, state_, target_, cntrl_, ip_, ie_, is_, v_ = variables_)
        #print("cost = ", cost_int)  
        
        
    return step_min1_, cost_min_int_

def set_pre_post(i1, i2, bc_, bs_, best_control_, state_pre_, state_, state_post_, state_vars, a, b):
    
    if (i2 != 0 and i1 != 0):   
        bc_[:,:,i1:-i2] = best_control_[:,:,:]
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        check_pre(i1, bs_, state_, state_vars, a, b)
        bs_[:,:,i1:-i2] = state_[:,:,:]
        if i2 == 1:
            bs_[:,:,-i2] = state_post_[:,:,1]
        else:
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

def adapt_step(control_, ind_node, ind_var, start_step_, dir_, max_control_, min_control_):
    start_st_ = start_step_
    max_index = -1
    min_index = -1
    max_cntrl = max_control_
    min_cntrl = min_control_
    
    for k in range(control_.shape[2]):
        if ( control_[ind_node,ind_var,k] + start_step_ * dir_[ind_node,ind_var,k] > max_cntrl ):
            max_index = k
            max_cntrl = control_[ind_node, ind_var,k] + start_step_ * dir_[ind_node, ind_var,k]
        elif ( control_[ind_node,ind_var,k] + start_step_ * dir_[ind_node,ind_var,k] < min_cntrl ):
            min_index = k
            min_cntrl = control_[ind_node, ind_var,k] + start_step_ * dir_[ind_node, ind_var,k]
    if max_index != -1:
        start_st_ = ( max_control_ - control_[ind_node,ind_var,max_index] ) / dir_[ind_node,ind_var,max_index]
    elif min_index != -1:
        start_st_ = ( min_control_ - control_[ind_node,ind_var,min_index] ) / dir_[ind_node,ind_var,min_index]
    return start_st_

# update rule for direction according to Hestenes-Stiefel
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

# update rule for direction according to Fletcher-Reeves
def betaFR(N, n_control_vars, grad0_, grad1_):
    betaFR = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], grad1_[n,v,:] )
            denominator = np.dot( grad0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaFR[n,v] = numerator / denominator
    return betaFR

# update rule for direction according to Polak-Ribiere
def betaPR(N, n_control_vars, grad0_, grad1_):
    betaPR = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            denominator = np.dot( grad0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaPR[n,v] = numerator / denominator
    return betaPR

# update rule for direction "conjugate descent"
def betaCD(N, n_control_vars, grad0_, grad1_, dir0_):
    betaCD = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], grad1_[n,v,:] )
            denominator = np.dot( dir0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaCD[n,v] = - numerator / denominator
    return betaCD

# update rule for direction according to Liu-Storey
def betaLS(N, n_control_vars, grad0_, grad1_, dir0_):
    betaLS = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            denominator = np.dot( dir0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaLS[n,v] = - numerator / denominator
    return betaLS

# update rule for direction according to Dai-Yuan
def betaDY(N, n_control_vars, grad0_, grad1_, dir0_):
    betaDY = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], grad1_[n,v,:] )
            denominator = np.dot( dir0_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            if np.abs(denominator) > 1e-6 :
                betaDY[n,v] = numerator / denominator
    return betaDY

# update rule for direction according to Wei et al.
def betaWYL(N, n_control_vars, grad0_, grad1_):
    betaWYL = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            g0abs = np.sqrt( np.dot( grad0_[n,v,:], grad0_[n,v,:] ) )
            g1abs = np.sqrt( np.dot( grad1_[n,v,:], grad1_[n,v,:] ) )
            numerator = np.dot( grad1_[n,v,:], grad1_[n,v,:] - grad0_[n,v,:] * ( g1abs / g0abs ) )
            denominator = np.dot( grad0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaWYL[n,v] = numerator / denominator
    return betaWYL

# update rule for direction according to Hager-Zhang
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

def compute_gradient(N, n_control_vars, T, dt, best_control_, grad1_, phi1_, control_variables, ie_, is_):
    grad_cost_e_ = cost.cost_energy_gradient(best_control_, ie_)
    grad_cost_s_ = cost.cost_sparsity_gradient(N, n_control_vars, T, dt, best_control_, is_)
        
    for j in range(n_control_vars):
        if j in control_variables:
            #print("j, adjoint, energy, sparsity gradient = ", j)
            #print(phi1_[:,j,:20])
            #print(grad_cost_e_[:,j,:20])
            #print(grad_cost_s_[:,j,:20])
            
            grad1_[:,j,:] = grad_cost_e_[:,j,:] + grad_cost_s_[:,j,:] + phi1_[:,j,:]
    return grad1_

def set_direction(N, T, n_control_vars, grad0_, grad1_, dir0_, i, CGVar, tolerance_):
        
    beta = np.zeros(( N, n_control_vars ))
    
    if (i >= 2 and CGVar != None):
        if CGVar == "HS":        # Hestens-Stiefel
            beta = betaHS(N, n_control_vars, grad0_, grad1_, dir0_)
        elif CGVar == "FR":        # Fletcher-Reeves
            beta = betaFR(N, n_control_vars, grad0_, grad1_)
        elif CGVar == "PR":        # Polak-Ribiere
            beta = betaPR(N, n_control_vars, grad0_, grad1_)
        elif CGVar == "CD":        # conjugate descent
            beta = betaCD(N, n_control_vars, grad0_, grad1_, dir0_)
        elif CGVar == "LS":        # Liu-Storey
            beta = betaLS(N, n_control_vars, grad0_, grad1_, dir0_)
        elif CGVar == "DY":        # Dai-Yuan
            beta = betaDY(N, n_control_vars, grad0_, grad1_, dir0_)
        elif CGVar == "WYL":        # Wei et al.
            beta = betaWYL(N, n_control_vars, grad0_, grad1_)
        elif CGVar == "HZ":        # Hager-Zhang
            beta = betaHZ(N, n_control_vars, grad0_, grad1_, dir0_)
            
    dir1_ = np.zeros(( N, n_control_vars, T ))
    for n in range(N):
        for v in range(n_control_vars):
            dir1_[n,v,:] = beta[n,v] * dir0_[n,v,:]
    
    dir0_ = - grad1_.copy() + dir1_
    
    # if this is too close to zero, use beta = 0 instead
    if (CGVar != None and np.amax(np.absolute(dir0_)) < tolerance_ ):
        print("Descent direction vanishing, use standard gradient descent")
        dir0_ = - grad1_.copy()
        
    return dir0_