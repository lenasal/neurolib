import numpy as np
import logging
import numba

from timeit import default_timer as timer
from . import costFunctions as cost
from . import func_optimize as fo
from ..models import jacobian_aln as jac_aln


VALID_VAR = {None, "FR", "HS"}

def A1(model, control_, target_state_, c_scheme_, u_mat_, u_scheme_, max_iteration_, tolerance_, startStep_,
       cntrl_max_, t_sim_, t_sim_pre_, t_sim_post_, CGVar):
        
    dt = model.params['dt']
    state_vars = model.state_vars
    init_vars = model.init_vars
    
    model.params['duration'] = t_sim_
    i=0
        
    rate_ = fo.updateState(model, control_)
    state0_ = model.getZeroFullState()
    state0_[:,0,:] = rate_[:,0,:]
    state0_[:,1,:] = model.state["mufe"][:,:]
    state0_[:,2,:] = model.state["IA"][:,:]
    state0_[:,3,:] = model.state["seem"][:,:]
    state0_[:,4,:] = model.state["seev"][:,:]
    state0_[:,5,:] = model.state["sigmae_f"][:,:]
    state0_[:,6,:] = model.state["Vmean_exc"][:,:]
    state0_[:,7,:] = model.state["tau_exc"][:,:]
    

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
        #print("phi = ", phi1_)
        
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
        state1_[:,2,:] = model.state["IA"][:,:]
        state1_[:,3,:] = model.state["seem"][:,:]
        state1_[:,4,:] = model.state["seev"][:,:]
        state1_[:,5,:] = model.state["sigmae_f"][:,:]
        state1_[:,6,:] = model.state["Vmean_exc"][:,:]
        state1_[:,7,:] = model.state["tau_exc"][:,:]
        
        
        s_diff_ = ( np.absolute(state1_ - state0_) < tolerance_)
        if ( s_diff_.all() ):
            print("State only changes marginally.")
        #    max_iteration_ = i
        #    break
        
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
    if max_iteration_ != 0:
        print("final gradient = ", g0_min_)
    
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
        
        #res = np.dot( - np.array( [full_cost_grad[0], full_cost_grad[1], full_cost_grad[2]] ), np.linalg.inv(jac))   
        #phi_[0,0,ind_time] = res[0]
        #phi_[0,1,ind_time-1] = res[1]
        #phi_[0,2,ind_time-1] = res[2]
        
        if (ind_time == 0):
            break
        
        phi_[0,0,ind_time] = - full_cost_grad[0] - np.dot( np.array( [phi_[0,1,ind_time], phi_[0,2,ind_time], phi_[0,3,ind_time],
                            phi_[0,4,ind_time], phi_[0,5,ind_time] ] ), np.array( [jac[1,0], jac[2,0], jac[3,0], jac[4,0], jac[5,0]] ) )
        
        if (ind_time == 0):
            break
        
        if (ind_time != phi_.shape[2]-1):
            der = phi_[0,0,ind_time+1] * jac[0,1] + phi_[0,1,ind_time] * jac[1,1] + phi_[0,6,ind_time] * jac[6,1] + phi_[0,7,ind_time] * jac[7,1]
            phi_[0,1,ind_time-1] = phi_[0,1,ind_time] - dt * der
            
            der = phi_[0,1,ind_time] * jac[1,3] + phi_[0,3,ind_time] * jac[3,3] + phi_[0,4,ind_time] * jac[4,3]
            phi_[0,3,ind_time-1] = phi_[0,3,ind_time] - dt * der
            
            der = phi_[0,1,ind_time] * jac[1,4] + phi_[0,4,ind_time] * jac[4,4] + phi_[0,5,ind_time] * jac[5,4]
            phi_[0,4,ind_time-1] = phi_[0,4,ind_time] - dt * der
            
                
        res = - phi_[0,1,ind_time-1] * jac[1,7]
        phi_[0,7,ind_time-1] = res
        
        res = - phi_[0,2,ind_time] * jac[2,6]
        phi_[0,6,ind_time-1] = res
        
        if (ind_time <= phi_.shape[2]-1):
            der = phi_[0,0,ind_time] * jac[0,2] + phi_[0,2,ind_time] * jac[2,2] + phi_[0,6,ind_time-1] * jac[6,2] + phi_[0,7,ind_time-1] * jac[7,2] 
            phi_[0,2,ind_time-1] = phi_[0,2,ind_time] - dt * der
        
        

        res = - phi_[0,0,ind_time] * jac[0,5] - phi_[0,7,ind_time-1] * jac[7,5] - phi_[0,6,ind_time-1] * jac[6,5] 
        phi_[0,5,ind_time-1] = res
                
    return phi_

# computation of g
# does not work with numba because takes model as parameter
def g(model, phi_, state_, control_):
    g_ = model.getZeroControl()
    
    grad_cost_e_ = cost.cost_energy_gradient(control_)
    grad_cost_s_ = cost.cost_sparsity_gradient1(model, control_)
    
    # shift if control is applied shifted wrt mu
    phi_shift = np.zeros(( phi_.shape ))
    phi_shift[:,:,1:] = phi_[:,:,0:-1] 
    
    phi1_ = np.zeros(( grad_cost_e_.shape ))
    for t in range(1,state_.shape[2]):
        jac_u_ = D_u_h(model, state_[:,:,:], t)
        phi1_[0,0,t] = np.dot(phi_shift[0,:,t], jac_u_)[1]
        
    #print("phi = ", phi_[0,1,:])
    #print("phi shift = ", phi_shift[0,1,:])
    #print("phi 1 = ", phi1_[0,0,:])
    
    g_[:,0,:] = grad_cost_e_[0,0,:] + grad_cost_s_[0,0,:] + phi1_

    return g_

def jacobian(model, state_, control_, t_):
    
    # TODO: time dependent exc current
    ext_exc_current = model.params.ext_exc_current
    sigmae_ext = model.params.sigmae_ext
    
    a = model.params["a"]
    b = model.params["b"]
    EA = model.params["EA"]
    tauA = model.params["tauA"]
    
    C = model.params["C"]
    
    Ke = model.params["Ke"]
    tau_se = model.params["tau_se"] 
    cee = model.params["cee"]
    Jee_max = model.params["Jee_max"]
    taum = model.params.C / model.params.gL
    
    rd_exc = np.zeros(( model.params.N,model.params.N ))
    rd_exc[0,0] = state_[0,0,t_] * 1e-3
    
    factor_ee1 = ( cee * Ke * tau_se / Jee_max )
    factor_ee2 = ( cee**2 * Ke * tau_se**2 / Jee_max**2 )
            
    z1ee = factor_ee1 * rd_exc[0,0]
    z2ee = factor_ee2 * rd_exc[0,0]
    
    jacobian_ = np.zeros((state_.shape[1], state_.shape[1]))
    jacobian_[0,0] = 1.
    jacobian_[0,1] = - d_r_func_mu(state_[0,1,t_] - state_[0,2,t_] / C, state_[0,5,t_]) * 1e3 # - state_[0,2,t_] / C
    jacobian_[0,2] = - d_r_func_mu(state_[0,1,t_-1] - state_[0,2,t_-1] / C, state_[0,5,t_-1]) * 1e3 * ( - 1. / C ) #  - state_[0,2,t_] / C
    jacobian_[0,5] = - d_r_func_sigma(state_[0,1,t_-1] - state_[0,2,t_] / C, state_[0,5,t_-1]) * 1e3 #  - state_[0,2,t_-1] / C
    
    jacobian_[1,1] = 1. / state_[0,7,t_]
    jacobian_[1,3] = - Jee_max / state_[0,7,t_]
    jacobian_[1,7] = ( Jee_max * state_[0,3,t_-1] + control_[0,0,t_] + ext_exc_current - state_[0,1,t_-1] ) / state_[0,7,t_-1]**2
    
    jacobian_[2,0] = b * 1e-3
    jacobian_[2,2] = 1. / tauA
    jacobian_[2,6] = - a / tauA
    
    jacobian_[3,0] = - (1. - state_[0,3,t_]) * factor_ee1 * 1e-3 / tau_se
    jacobian_[3,3] = ( 1. + z1ee ) / tau_se
    
    jacobian_[4,0] = - ( (1. - state_[0,3,t_])**2 * factor_ee2 + state_[0,4,t_] * ( factor_ee2 - 2. * tau_se *  factor_ee1 ) ) * 1e-3 / tau_se**2
    jacobian_[4,3] = 2. * (1. - state_[0,3,t_]) * z2ee / tau_se**2
    jacobian_[4,4] = - (z2ee - 2. * tau_se * ( z1ee + 1.) ) / tau_se**2
    
    sigma_sqrt = ( state_[0,4,t_] * ( 2. * Jee_max**2 * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1) + sigmae_ext**2 )**(-1./2.)
    
    jacobian_[5,0] = 0.5 * (1e-3) * factor_ee1 * taum * ( (1 + z1ee) * taum + tau_se )**(-2) * state_[0,4,t_] * ( 2. * Jee_max**2 * tau_se * taum ) * sigma_sqrt
    jacobian_[5,4] = - 0.5 * ( (1 + z1ee) * taum + tau_se )**(-1) * ( 2. * Jee_max**2 * tau_se * taum ) * sigma_sqrt
    jacobian_[5,5] = 1.
    
    jacobian_[6,1] = - d_V_func_mu(state_[0,1,t_] - state_[0,2,t_] / C, state_[0,5,t_])
    jacobian_[6,2] = - d_tau_func_mu(state_[0,1,t_-1] - state_[0,2,t_-1] / C, state_[0,5,t_-1]) * ( - 1. / C )
    jacobian_[6,5] = - d_V_func_sigma(state_[0,1,t_-1] - state_[0,2,t_-1] / C, state_[0,5,t_-1])
    
    jacobian_[7,1] = - d_tau_func_mu(state_[0,1,t_] - state_[0,2,t_] / C, state_[0,5,t_])
    jacobian_[7,2] = - d_tau_func_mu(state_[0,1,t_-1] - state_[0,2,t_-1] / C, state_[0,5,t_-1]) * ( - 1. / C )
    #jacobian_[7,2] = - 2. * state_[0,2,t_-1] / C
    jacobian_[7,5] = - d_tau_func_sigma(state_[0,1,t_-1] - state_[0,2,t_-1] / C, state_[0,5,t_-1])
    jacobian_[7,7] = 1.
    
    return jacobian_

def D_xdot(model, state_t_):
    dxdot_ = np.zeros((state_t_.shape[1], state_t_.shape[1]))
    return dxdot_

def D_u_h(model, state_, t_):
    duh_ = np.zeros(( state_.shape[1], state_.shape[1] ))
    duh_[1,1] = - 1. / state_[0,7,t_-1]
    return duh_

def d_r_func_mu(mu, sigma):
    x_shift_mu = - 2.
    x_scale_mu = 0.6
    y_scale_mu = 0.1
    return y_scale_mu * x_scale_mu / np.cosh(x_scale_mu * mu + x_shift_mu)**2

def d_r_func_sigma(mu, sigma):
    x_shift_sigma = -1.
    x_scale_sigma = 0.6
    y_scale_sigma = 1./2500.
    return np.sinh(x_scale_sigma * sigma + x_shift_sigma) * y_scale_sigma * x_scale_sigma

def d_tau_func_mu(mu, sigma):
    mu_shift = - 1.1
    mu_scale = - 10.
    sigma_shift = 1.4
    #return 1.
    return sigma + mu_scale + ( mu_scale / (sigma + sigma_shift) ) * np.exp( mu_scale * (mu_shift + mu) / (sigma + sigma_shift) )

def d_tau_func_sigma(mu, sigma):
    mu_shift = - 1.1
    mu_scale = - 10.
    sigma_shift = 1.4
    #return 1.
    return (mu_shift + mu) - (mu_scale * (mu_shift + mu) / (sigma + sigma_shift)**2) * np.exp( mu_scale * (mu_shift + mu) / (sigma + sigma_shift) )

def d_V_func_mu(mu, sigma):
    y_scale1 = 30.
    mu_shift1 = 1.
    y_scale2 = 2.
    mu_shift2 = 0.5
    return 1.
    return y_scale1 / np.cosh( mu + mu_shift1 )**2 - y_scale2 * 2. * ( mu - mu_shift2 ) * np.exp( - ( mu - mu_shift2 )**2 ) / sigma

def d_V_func_sigma(mu, sigma):
    y_scale2 = 2.
    mu_shift2 = 0.5
    return 1.
    return - y_scale2 * np.exp( - ( mu - mu_shift2 )**2 ) / sigma**2