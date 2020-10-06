import numpy as np
import logging
import numba

from timeit import default_timer as timer
from . import costFunctions as cost
from . import func_optimize as fo
from ..models import jacobian_aln as jac_aln


VALID_VAR = {None, "FR", "HS"}

def A1(model, control_, target_state_, max_iteration_, tolerance_, startStep_, cntrl_max_, t_sim_):
        
    dt = model.params['dt']
    state_vars = model.state_vars
    init_vars = model.init_vars
    
    model.params['duration'] = t_sim_
    i=0
        
    mu_ = fo.updateState(model, control_)
    state0_ = model.getZeroFullState()
    state0_[:,0,:] = mu_[:,0,:]
    

    total_cost_ = np.zeros((max_iteration_+1))
    total_cost_[i] = cost.f_int(model.params['dt'], cost.f_cost(state0_, target_state_, control_) )
    print("RUN ", i, ", total integrated cost = ", total_cost_[i])

    
    state1_ = state0_.copy()
    u_opt0_ = control_.copy()
    best_control_ = control_.copy()
    
    phi0_ = model.getZeroFullState()
    
    while( i < max_iteration_):
        i += 1   
        
        phi1_ = phi(model, state0_, target_state_, best_control_, phi0_)
        
        outstate_ = model.getZeroState()
        outstate_[:,:,:] = state1_[:,0,:]
    
        g0_min_ = g(model, phi1_, state1_, best_control_)
        g1_min_ = g0_min_.copy()
        #print("phi = ", phi1_)
        print("g = ", g0_min_)

        dir0_ = - g0_min_.copy()
        dir1_ = dir0_.copy()

        
        step_, total_cost_[i] = fo.step_size(model, outstate_[:,:,:], target_state_,
                     best_control_, dir1_, start_step_ = startStep_, max_control_ = cntrl_max_)
        
        #print("step = ", step_, total_cost_[i])
        #if (step_ == 0.):
            #print("try other direction")
            #dir1_ *= -1.
            #step_, total_cost_[i] = fo.step_size(model, outstate_[:,:,:], target_state_,
            #         best_control_, dir1_, start_step_ = startStep_, max_control_ = cntrl_max_)
            #print("step = ", step_, total_cost_[i])
        
        print("RUN ", i, ", total integrated cost = ", total_cost_[i])
        best_control_ = u_opt0_ + step_ * dir1_
        
        #print("control = ", best_control_)
        
        u_diff_ = ( np.absolute(best_control_ - u_opt0_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i
            break
        u_opt0_ = best_control_.copy()
        
        mu_ = fo.updateState(model, best_control_)
        state1_[:,0,:] = mu_[:,0,:]
        
        
        s_diff_ = ( np.absolute(state1_ - state0_) < tolerance_ )
        if ( s_diff_.all() ):
            print("State only changes marginally.")
            max_iteration_ = i
            break
        
        g0_min_ = g1_min_.copy()
        if ( np.amax(np.absolute(g0_min_[:,:,1:])) < tolerance_ ):
            print( np.amax( np.absolute(g0_min_[:,:,1:]) ) )
            print("Gradient negligibly small.")
            max_iteration_ = i
            break
        
        state0_ = state1_.copy() 
        dir0_ = dir1_.copy()  
        phi0_ = phi1_.copy()         
    
    improvement = 100.
    if total_cost_[0] != 0.:
        improvement = improvement = 100. - (100.*(total_cost_[max_iteration_]/total_cost_[0]))
        
    print("Improved over ", max_iteration_, " iterations by ", improvement, " percent.")
    
    return best_control_, state1_, total_cost_, 0.

def phi(model, state_, target_state_, control_, phi_prev_, start_ind_ = 0):
    #print("ALN phi2 computation")
    phi_ = model.getZeroFullState()
    dt = model.params['dt']
    out_state = model.getZeroState()
    out_state[:,:,:] = state_[:,0,:]
            
    for ind_time in range(phi_.shape[2]-1, start_ind_, -1):
        
        if (ind_time == 1):
            break
        jac = jacobian(model, state_[:,:,:], control_[:,:,:], ind_time)
                
        f_p_grad_t = cost.cost_precision_gradient_t(out_state[:,:,ind_time], target_state_[:,:,ind_time])
        phi_[0,0,ind_time-1] = phi_[0,0,ind_time] - dt * (f_p_grad_t + phi_[0,0,ind_time] * jac)
   
    return phi_

# computation of g
# does not work with numba because takes model as parameter
def g(model, phi_, state_, control_):
    g_ = model.getZeroControl()
    
    grad_cost_e_ = cost.cost_energy_gradient(control_)
    grad_cost_s_ = cost.cost_sparsity_gradient1(model, control_)
    
    phi_shift = np.zeros(( phi_.shape ))
    phi_shift[:,:,1:] = phi_[:,:,0:-1]
    
    phi1_ = np.zeros(( grad_cost_e_.shape ))
    for t in range(state_.shape[2]):
        jac_u_ = D_u_h(model, state_[:,:,:],t)
        phi1_[0,0,t] = np.dot(phi_[0,:,t], jac_u_)
    
    g_[:,0,:] = grad_cost_e_[0,0,:] + grad_cost_s_[0,0,:] + phi1_
    
    print("energy contribution = ", grad_cost_e_[0,0,:])
    print("phi contribution = ", phi1_)

    return g_

def jacobian(model, state_, control_, t_):
    jacobian_ = np.zeros((state_.shape[1], state_.shape[1]))
    jacobian_[0,0] = control_[0,0,t_] / state_[0,0,t_]**2
    
    return jacobian_


def D_u_h(model, state_, t_):
    duh_ = np.zeros(( state_.shape[1], state_.shape[1] ))
    duh_[0,0] = -1. / state_[0,0,t_]
    return duh_