import numpy as np
import logging
from . import costFunctions as cost
from ..models import jacobian_aln as jac_aln

def printState(model, state_):
    state_vars = model.state_vars
    for sv in [0,1,2,3,4,15,16,17,18,19]:
        print(state_vars[sv], model.state[state_vars[sv]])

def updateState(model, control_):
    # set initial conditions once in other function
    state_ = model.getZeroState()
    output_vars = model.output_vars
    model.run(control = control_)    
    for i in range(len(output_vars)):
        state_[:,i,:] = model[output_vars[i]][:,:]
    return state_

def updateFullState(model, control_, duration_):
    state_vars = model.state_vars
    state_ = model.getZeroFullState()
    
    model.params.duration = duration_
    model.run(control=control_)
    for sv in range(len(state_vars)):
        state_[:,sv,:] = model.state[state_vars[sv]][:,:]
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

def get_init(model, init_vars_, state_vars_):
    IC_ = np.zeros(( model.params.N, len(init_vars_) ))
    for iv, sv in zip( range(len(init_vars_)), range(len(state_vars_)) ):
        if state_vars_[sv] in init_vars_[iv]:
            if model.params[init_vars_[iv]].ndim == 2:
                IC_[:,iv] = model.state[state_vars_[sv]][:,-1]
            else:
                print(init_vars_[iv], model.state[state_vars_[sv]].shape)
                IC_[:,iv] = model.state[state_vars_[sv]][:,-1]
    return IC_

def set_init(model, IC_, init_vars_, state_vars_):
    for iv in range(len(init_vars_)):
        if model.params[init_vars_[iv]].ndim == 2:
            model.params[init_vars_[iv]][:,0] = IC_[:,iv]
        else:
            model.params[init_vars_[iv]] = IC_[:,iv]

def update_init(model, init_vars_, state_vars_):
    for iv, sv in zip( range(len(init_vars_)), range(len(state_vars_)) ):
        if state_vars_[sv] in init_vars_[iv]:
            if model.params[init_vars_[iv]].ndim == 2:
                model.params[init_vars_[iv]][:,0] = model.state[state_vars_[sv]][:,-1]
            else:
                model.params[init_vars_[iv]] = model.state[state_vars_[sv]]
                    
def update_init_delayed(model, delay_state_vars_, init_vars_, state_vars_, t_pre_ndt_, startind_):
    for iv, sv in zip( range(len(init_vars_)), range(len(state_vars_)) ):
        if state_vars_[sv] in init_vars_[iv]:
            if model.params[init_vars_[iv]].ndim == 2:
                if (t_pre_ndt_ < startind_):
                    # delay larger than pre simulation time
                    delay_state_vars_[:, sv, :-1] = model.params[init_vars_[iv]][:,1:]
                    delay_state_vars_[:, sv, -t_pre_ndt_:] = model.state[state_vars_[sv]][:,:]
                    model.params[init_vars_[iv]][:,:] = delay_state_vars_[:, sv, :]
                else:
                    model.params[init_vars_[iv]] = model.state[state_vars_[sv]][:,-startind_:]
                    delay_state_vars_[:, sv, :] = model.state[state_vars_[sv]][:,-startind_:]
            else:
                model.params[init_vars_[iv]] = model.state[state_vars_[sv]]
        else:
            logging.error("Initial and state variable labelling does not agree.")

def test_step(model, state_, target_, control_, dir_, test_step_ = 1e-12):
    dt = model.params['dt']
    cost0_int_ = cost.f_int(dt, cost.f_cost(state_, target_, control_))
    
    test_control_ = control_ + test_step_ * dir_
    state1_ = updateState(model, test_control_)
    cost1_int_ = cost.f_int(dt, cost.f_cost(state1_, target_, test_control_))
    #print("test step size computation : ------ step size, cost1, cost0 : ", test_step_, cost1_int_, cost0_int_)
        
    if (cost1_int_ < cost0_int_):
        return test_step_, cost1_int_
    else:
        return 0., cost0_int_
    
    
def step_size(model, state_, target_, control_, dir_, start_step_ = 20., max_it_ = 10000,
              bisec_factor_ = 2., max_control_ = 20.):
    
    dt = model.params['dt']
    cost0_ = cost.f_cost(state_, target_, control_)
    cost0_int_ = cost.f_int(dt, cost0_)
    cost_min_int_ = cost0_int_
    step_ = start_step_
    step_min_ = step_
    
    for i in range(max_it_):
        #if (max_iteration_ == 1):
        test_control_ = control_ + step_ * dir_
        
        # include maximum control value to assure no divergence
        if ( np.amax(np.absolute(test_control_)) > max_control_):
            if (i < max_it_-1):
                #print("too big control")
                step_ /= bisec_factor_
                continue
            else:
                print("control too big, but no further iteration")
                return 0., cost0_int_
            
        state1_ = updateState(model, test_control_)
        cost1_ = cost.f_cost(state1_, target_, test_control_)
        cost1_int_ = cost.f_int(dt, cost1_)
        
        #print("step = ", step_, " , cost = ", cost1_int_, ", initial cost = ", cost0_int_)

        if (cost1_int_ < cost_min_int_):
            #print("found step = ", step_, " with cost1, cost0 : ", cost1_int_, cost0_int_)
            #print("with control = ", test_control_)
            cost_min_int_ = cost1_int_
            step_min_ = step_
        # return smallest step size before cost is increasing again
        elif (cost1_int_ > cost_min_int_ and cost_min_int_ < cost0_int_):
            #print("step size for minimal cost: ", step_min_)
            #print("state = ", state1_)
            if (step_min_ == start_step_):
                print("Using initial step.")
            return step_min_, cost_min_int_
        
        if (i == max_it_-1):
            if (max_it_ != 1):
                print(" max iteration reached, step size = ", step_)
            #else:
                #plt.plot(state1_[0,0,:], state1_[0,1,:])
               # plt.show()
            return 0., cost0_int_
        
        
        # decrease bisection factor once we approach the minimum value, such that we don't miss it
        if ( cost1_int_/cost0_int_ < 1.5 and bisec_factor_ > 1.2 ): 
            bisec_factor_ = 1.2
            #print("change bisection factor to ", bisec_factor_)
        if ( cost1_int_/cost0_int_ < 1.2 and bisec_factor_ > 1.1 ): 
            bisec_factor_ = 1.1
            #print("change bisection factor to ", bisec_factor_)
        if ( cost1_int_/cost0_int_ < 1.1 and bisec_factor_ > 1.05 ): 
            bisec_factor_ = 1.05
        #    print("change bisection factor to ", bisec_factor_)
        if ( cost1_int_/cost0_int_ < 1.05 and bisec_factor_ > 1.04 ): 
            bisec_factor_ = 1.04
            #print("change bisection factor to ", bisec_factor_)
        if ( cost1_int_/cost0_int_ < 1.04 and bisec_factor_ > 1.03 ): 
            bisec_factor_ = 1.03
        if ( cost1_int_/cost0_int_ < 1.03 and bisec_factor_ > 1.02 ): 
            bisec_factor_ = 1.02
        #    print("change bisection factor to ", bisec_factor_)
        #if ( cost1_int_/cost0_int_ < 1.02 and bisec_factor_ > 1.01 ): 
        #    bisec_factor_ = 1.01
        
        
        step_ /= bisec_factor_
    
