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
        
    rate_ = fo.updateState(model, control_)
    state0_ = model.getZeroFullState()
    state0_[:,0,:] = rate_[:,0,:]
    state0_[:,1,:] = model.state["mufe"][:,:]
    state0_[:,2,:] = model.state["tau_exc"][:,:]
    

    total_cost_ = np.zeros((max_iteration_+1))
    total_cost_[i] = cost.f_int(model.params['dt'], state0_, target_state_, control_ )
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

        dir0_ = - g0_min_.copy()
        dir1_ = dir0_.copy()
        
        
        step_, total_cost_[i] = fo.step_size(model, outstate_[:,:,:], target_state_,
                     best_control_, dir1_, start_step_ = startStep_, max_control_ = cntrl_max_)
        
        print("RUN ", i, ", total integrated cost = ", total_cost_[i])
        best_control_ = u_opt0_ + step_ * dir1_
        
        u_diff_ = ( np.absolute(best_control_ - u_opt0_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i
            break
        u_opt0_ = best_control_.copy()
        
        rate_ = fo.updateState(model, best_control_)
        state1_[:,0,:] = rate_[:,0,:]
        state1_[:,1,:] = model.state["mufe"][:,:]
        state1_[:,2,:] = model.state["tau_exc"][:,:]
        
        
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
    
    print("final grad = ", g0_min_)
    
    return best_control_, state1_, total_cost_, 0.

def phi(model, state_, target_state_, control_, phi_prev_, start_ind_ = 0):
    dt = model.params.dt
    phi_ = model.getZeroFullState()
    out_state = model.getZeroState()
    out_state[:,:,:] = state_[:,0,:]
            
    for ind_time in range(phi_.shape[2]-1, start_ind_-1, -1):
        jac = jacobian(model, state_[:,:,:], control_[:,:,:], ind_time)
        
        f_p_grad_t_ = cost.cost_precision_gradient_t(out_state[:,:,ind_time], target_state_[:,:,ind_time])
        full_cost_grad = np.zeros(( state_[0,:,ind_time].shape ))
        full_cost_grad[0] = f_p_grad_t_[0,0]
        
        jac1 = np.delete(jac, (1), axis=0)
        jac1 = np.delete(jac1, (1), axis=1)
        jac2 = np.delete(jac, (0,2), axis=0)
        jac2 = np.delete(jac2, (1), axis=1)
        res = np.dot( - np.array( [full_cost_grad[0], full_cost_grad[2]] ) - np.dot( phi_[0,1,ind_time],jac2 ) , np.linalg.inv(jac1))
        #print("res = ", res)
        phi_[0,0,ind_time] = res[0,0]
        phi_[0,2,ind_time] = res[0,1]
                
        #if (ind_time != phi_.shape[2]-1 ):  
            
        if (ind_time == 0):
            break
            
        der = phi_[0,0,ind_time] * jac[0,1] + phi_[0,2,ind_time] * jac[2,1]
        phi_[0,1,ind_time-1] = phi_[0,1,ind_time] - dt * der
                
    return phi_

# computation of g
# does not work with numba because takes model as parameter
def g(model, phi_, state_, control_):
    g_ = model.getZeroControl()
    
    grad_cost_e_ = cost.cost_energy_gradient(control_)
    grad_cost_s_ = cost.cost_sparsity_gradient1(model, control_)
    
    phi_shift = np.zeros(( phi_.shape ))
    phi_shift[:,:,1:] = phi_[:,:,0:-1]
    phi_shift[:,:,0] = phi_shift[:,:,1]
        
    phi1_ = np.zeros(( grad_cost_e_.shape ))
    for t in range(state_.shape[2]):
        jac_u_ = D_u_h(model, state_[:,:,t])
        phi1_[0,0,t] = np.dot(phi_[0,:,t], jac_u_)[1]
            
    g_[:,0,:] = grad_cost_e_[0,0,:] + grad_cost_s_[0,0,:] + phi1_[0,0,:]
    
    #print("grad e = ", grad_cost_e_)
    #print("grad phi = ", phi1_)

    return g_

def jacobian(model, state_, control_, t_):
    jacobian_ = np.zeros((state_.shape[1], state_.shape[1]))
    jacobian_[0,0] = 1.
    jacobian_[0,1] = - d_r_func_mu(state_[0,1,t_], 1.5) * 1e3
    #jacobian_[0,1] = - 1.
    
    #jacobian_[1,2] = control_t_[0,0] / state_t_[0,2]**2
    
    #jacobian_[2,1] = -1.
    jacobian_[2,2] = 1.
    
    return jacobian_

def D_xdot(model, state_t_):
    dxdot_ = np.zeros((state_t_.shape[1], state_t_.shape[1]))
    return dxdot_

def D_u_h(model, state_t_):
    duh_ = np.zeros(( state_t_.shape[1], state_t_.shape[1] ))
    duh_[1,1] = -1. #/ state_t_[0,2]
    return duh_

def dh_dmu(model, sigma, mu, table):
    return jac_aln.der_mu_up(model, sigma, mu, 0., table)

def dh_dsigma(model, sigma, mu, table):
    return jac_aln.der_sigma(model, sigma, mu, 0., table)

def d_r_func_mu(mu, sigma):
    x_shift = - 2.
    x_scale = 0.6
    y_scale = 0.1
    return y_scale * x_scale / np.cosh(x_scale * mu + x_shift)**2