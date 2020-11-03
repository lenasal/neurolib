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
    
    delay = model.params["delay"]
        
    rate_ = fo.updateState(model, control_)
    state0_ = model.getZeroFullState()
    state0_ = np.zeros(( state0_.shape[0], state0_.shape[1], int(state0_.shape[2]+delay) ))
    state0_[:,0,delay:] = rate_[:,0,:]
    state0_[:,1,delay:] = model.state["mufe"][:,:]
    state0_[:,2,delay:] = model.state["seem"][:,:]
    state0_[:,3,delay:] = model.state["seev"][:,:]
    state0_[:,4,delay:] = model.state["sigmae_f"][:,:]
    state0_[:,5,delay:] = model.state["tau_exc"][:,:]
        

    total_cost_ = np.zeros((max_iteration_+1))
    total_cost_[i] = cost.f_int(model.params['dt'], state0_[:,:,delay:], target_state_, control_ )
    print("RUN ", i, ", total integrated cost = ", total_cost_[i])

    
    state1_ = state0_.copy()
    u_opt0_ = control_.copy()
    best_control_ = control_.copy()
    
    phi0_ = model.getZeroFullState()
    
    while( i < max_iteration_):
        i += 1   
        
        phi1_ = phi(model, state0_, target_state_, best_control_, phi0_)
        
        outstate_ = model.getZeroState()
        outstate_[:,:,:] = state1_[:,0,delay:]
    
        g0_min_ = g(model, phi1_, state1_, best_control_)
        g1_min_ = g0_min_.copy()
        #print("phi = ", phi1_)
        

        dir0_ = - g0_min_.copy()
        dir1_ = dir0_.copy()
        
        #print("phi = ", phi1_)

        
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
        
        rate_ = fo.updateState(model, best_control_)
        state1_[:,0,delay:] = rate_[:,0,:]
        state1_[:,1,delay:] = model.state["mufe"][:,:]
        state1_[:,2,delay:] = model.state["seem"][:,:]
        state1_[:,3,delay:] = model.state["seev"][:,:]
        state1_[:,4,delay:] = model.state["sigmae_f"][:,:]
        state1_[:,5,delay:] = model.state["tau_exc"][:,:]
        
        
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
    #print("ALN phi2 computation")
    delay = model.params["delay"]
    phi_ = model.getZeroFullState()
    phi_ = np.zeros((phi_.shape[0], phi_.shape[1], int(phi_.shape[2] + delay) ))
    dt = model.params['dt']
    out_state = model.getZeroState()
    out_state[:,0,:] = state_[:,0,delay:]
            
    for ind_time in range(phi_.shape[2]-1, delay, -1):
        
        jac = jacobian(model, state_[:,:,:], control_[:,:,:], ind_time)
        
        f_p_grad_t_ = cost.cost_precision_gradient_t(out_state[:,:,ind_time-delay], target_state_[:,:,ind_time-delay])
        full_cost_grad = np.zeros(( state_[0,:,ind_time].shape ))
        full_cost_grad[0] = f_p_grad_t_[0,0]
        
        jac1 = np.delete(jac, (1,2,3), axis=0)
        jac1 = np.delete(jac1, (1,2,3), axis=1)
        jac2 = np.delete(jac, (0,4,5), axis=0) # remove all lines that do not have time derivative
        jac2 = np.delete(jac2, (1,2,3), axis=1)
        res = np.dot( - np.array( [ full_cost_grad[0], full_cost_grad[4], full_cost_grad[5] ] )
                     - np.dot( np.array( [ phi_[0,1,ind_time], phi_[0,2,ind_time], phi_[0,3,ind_time] ] ) ,jac2 ) , np.linalg.inv(jac1))
        
        phi_[0,0,ind_time] = res[0]
        phi_[0,4,ind_time] = res[1]
        phi_[0,5,ind_time] = res[2]
        
        if (ind_time == delay):
            break
        
        # if rates are computed from previous mufe
        #if (ind_time != phi_.shape[2]-1 ):  
            #phi_[0,0,ind_time] = phi_[0,0,ind_time+1]
            #phi_[0,4,ind_time] = phi_[0,4,ind_time+1]
            #phi_[0,5,ind_time] = phi_[0,5,ind_time+1]
            
        for j in [1,2,3]:
            der = full_cost_grad[j]
            der += np.dot(phi_[0,:,ind_time], jac[:,j])
            phi_[0,j,ind_time-1] = phi_[0,j,ind_time] - dt * der
        
        #der = full_cost_grad[1] + np.dot(phi_[0,:,ind_time], jac[:,1])
        #phi_[0,1,ind_time-1] = phi_[0,1,ind_time] - dt * der
        
        #phi_[0,0,ind_time] = res[0]
   
    return phi_

# computation of g
# does not work with numba because takes model as parameter
def g(model, phi_, state_, control_):
    delay = model.params["delay"]
    g_ = model.getZeroControl()
    
    grad_cost_e_ = cost.cost_energy_gradient(control_)
    grad_cost_s_ = cost.cost_sparsity_gradient1(model, control_)
    
    # shift if control is applied shifted
    #phi_shift = np.zeros(( phi_.shape ))
    #phi_shift[:,:,1:] = phi_[:,:,0:-1] 
    
    phi1_ = np.zeros(( grad_cost_e_.shape ))
    for t in range(grad_cost_e_.shape[2]):
        jac_u_ = D_u_h(model, state_[:,:,:],t+delay)
        phi1_[0,0,t] = np.dot(phi_[0,:,t+delay], jac_u_)[1]
    
    g_[:,0,:] = grad_cost_e_[0,0,:] + grad_cost_s_[0,0,:] + phi1_
    
    #print("energy contribution = ", grad_cost_e_[0,0,:])
    #print("phi contribution = ", phi1_)

    return g_

def jacobian(model, state_, control_, t_):
    
    Ke = model.params["Ke"]
    tau_se = model.params["tau_se"] 
    cee = model.params["cee"]
    Jee_max = model.params["Jee_max"]
    delay = model.params["delay"]
    
    rd_exc = np.zeros(( model.params.N,model.params.N ))
    delay = 0
    rd_exc[0,0] = state_[0,0,t_-delay] * 1e-3
    delay = model.params["delay"]
    
    factor_ee1 = ( cee * Ke * tau_se / Jee_max )
    factor_ee2 = ( cee**2 * Ke * tau_se**2 / Jee_max**2 )
            
    z1ee = factor_ee1 * rd_exc[0, 0]
    z2ee = factor_ee2 * rd_exc[0, 0]
    
    jacobian_ = np.zeros((state_.shape[1], state_.shape[1]))
    
    jacobian_[0,0] = 1.
    #jacobian_[0,1] = - dh_dmu_down(model, state_[0,4,t_], state_[0,1,t_], model.params.precalc_r) * 1e3
    
    jacobian_[0,1] = - d_r_func_mu(state_[0,1,t_], state_[0,4,t_]) * 1e3
    
    #maybe shift ext exc current by one?
    jacobian_[1,1] = 1. / state_[0,5,t_]
    jacobian_[1,2] = - 1. / state_[0,5,t_]
    jacobian_[1,5] = (state_[0,2,t_] + control_[0,0,t_-delay] + model.params.ext_exc_current - state_[0,1,t_]) / state_[0,5,t_]**2
    
    delay = 0
    
    jacobian_[2,0] = - (1. - state_[0,2,t_]) * factor_ee1 * ( ( state_[0,0,t_-delay] - state_[0,0,t_-delay-1] )
                        / (state_[0,0,t_] - state_[0,0,t_-1]) ) * 1e-3
    jacobian_[2,2] = ( 1. / tau_se + z1ee )
    
    jacobian_[4,4] = 1.
    
    #jacobian_[5,1] = - dh_dmu_up(model, state_[0,4,t_], state_[0,1,t_], model.params.precalc_tau_mu)
    #print("jacobian = ", jacobian_[5,1])
    jacobian_[5,1] = - d_tau_func(state_[0,1,t_])
    jacobian_[5,5] = 1.
    
    return jacobian_


def D_u_h(model, state_, t_):
    duh_ = np.zeros(( state_.shape[1], state_.shape[1] ))
    duh_[1,1] = -1. / state_[0,5,t_]
    return duh_

def dh_dmu_up(model, sigma, mu, table):
    result_ = jac_aln.der_mu_up(model, sigma, mu, 0., table)
    if np.abs(result_) < 1e-2:
        tabname = var_name(model, table)
        print("Derivative of transfer function wrt mu small. Table = ", tabname, ". Mu = ", mu, ". Sigma = ", sigma)
    return result_

def dh_dmu_down(model, sigma, mu, table):
    result_ = jac_aln.der_mu_down(model, sigma, mu, 0., table)
    if np.abs(result_) < 1e-2:
        tabname = var_name(model, table)
        print("Derivative of transfer function wrt mu small. Table = ", tabname, ". Mu = ", mu, ". Sigma = ", sigma)
    return result_

def var_name(model, table):
    rate_name = (table == model.params.precalc_r)
    tau_name = (table == model.params.precalc_tau_mu)
    if rate_name.all():
        return "rate"
    elif tau_name.all():
        return "tau"
    
def d_r_func_mu(mu, sigma):
    x_shift = - 2.
    x_scale = 0.6
    y_scale = 0.1
    return y_scale * x_scale / np.cosh(x_scale * mu + x_shift)**2

def d_tau_func(mu):
    x_shift = -1.
    x_scale = 1.
    y_scale = -10.
    return y_scale * x_scale / np.cosh(x_scale * mu + x_shift)**2