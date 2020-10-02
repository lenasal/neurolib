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
    #state0_[:,2,:] = model.state["tau_exc"][:,:]
    
    #print("state = ", state0_)

    total_cost_ = np.zeros((max_iteration_+1))
    total_cost_[i] = cost.f_int(model.params['dt'], cost.f_cost(state0_, target_state_, control_) )
    print("RUN ", i, ", total integrated cost = ", total_cost_[i])

    
    state1_ = state0_.copy()
    u_opt0_ = control_.copy()
    best_control_ = control_.copy()
    
    #best_control_[:,:,0] = best_control_[:,:,1]
    #best_control_[:,:,-2] = best_control_[:,:,-1]
    
    phi0_ = model.getZeroFullState()
    
    while( i < max_iteration_):
        i += 1   
        
        phi1_ = phi(model, state0_, target_state_, best_control_, phi0_)
        #print("phi = ", phi1_)
        
        outstate_ = model.getZeroState()
        outstate_[:,:,:] = state1_[:,0,:]
        
        #d0_precision_ = - g(model, phi_, state1_, zeroControl_)
        #step_precision_ = fo.step_size(model, outstate_[:,:,:], target_state_,
        #             best_control_, d0_precision_, start_step_ = startStep_, max_control_ = cntrl_max_)
        
        #control_precision_ = u_opt0_ + step_precision_[0] * d0_precision_
    
        #g0_min_ = g(model, phi_, state1_, control_precision_)
        g0_min_ = g(model, phi1_, state1_, best_control_)
        g1_min_ = g0_min_.copy()

        dir0_ = - g0_min_.copy()
        #dir0_[:,:,0] = dir0_[:,:,1].copy()
        #dir0_[:,:,-1] = dir0_[:,:,-2].copy()
        dir1_ = dir0_.copy()
        
        #print("dir = ", dir1_)
        
        step_, total_cost_[i] = fo.step_size(model, outstate_[:,:,:], target_state_,
                     best_control_, dir1_, start_step_ = startStep_, max_control_ = cntrl_max_)
        #print("step size = ", step_, total_cost_[i])
        
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
        #state1_[:,2,:] = model.state["tau_exc"][:,:]
        
        
        s_diff_ = ( np.absolute(state1_ - state0_) < tolerance_ )
        #if ( s_diff_.all() ):
        #    print("State only changes marginally.")
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
    
    return best_control_, state1_, total_cost_, 0.

def phi(model, state_, target_state_, control_, phi_prev_, start_ind_ = 0):
    #print("ALN phi2 computation")
    phi_ = model.getZeroFullState()
    dt = model.params['dt']
    out_state = model.getZeroState()
    out_state[:,:,:] = state_[:,0,:]
            
    for ind_time in range(phi_.shape[2]-1, start_ind_-1, -1):
        Dxdot = D_xdot(model, state_[:,:,ind_time])
        jac = jacobian(model, state_[:,:,ind_time], control_[:,:,ind_time])
        f_p_grad_t_ = cost.cost_precision_gradient_t(out_state[:,:,ind_time], target_state_[:,:,ind_time])
        full_cost_grad = np.zeros(( state_[0,:,ind_time].shape ))
        full_cost_grad[0] = f_p_grad_t_[0,0]
        
        
        res = - np.dot( full_cost_grad, np.linalg.inv(jac) )
        phi_[0,0,ind_time] = res[0]
        phi_[0,1,ind_time] = res[1]
        #phi_[0,2,ind_time] = res[2]
                
        """
        # delete rows and columns where Dxdot has entries
        jac1 = np.delete(jac, (1), axis=0)
        jac1 = np.delete(jac1, (1), axis=1)
        # delete rows and columns where 
        jac2 = np.delete(jac, (0), axis=0)
        jac2 = np.delete(jac2, (1), axis=1)
        res = np.dot( - np.array( [full_cost_grad[0]] ) - np.dot( phi_[0,1,ind_time],jac2 ) , np.linalg.inv(jac1))
        phi_[0,0,ind_time] = res[0]
        """
        
        #print("res = ", res)
        
        
                
        #phi_[0,0,ind_time] = - f_p_grad_t_[0,0]

        # mu = u
        #if (ind_time != phi_.shape[2]-1 ):
            #phi_[0,1,ind_time] = - phi_[0,0,ind_time] * jac[0,1]
            #phi_[0,1,ind_time] = - phi_[0,0,ind_time+1] * jac[0,1]
        
        
        if (ind_time == 0):
            break
        
        # mu dot = u or mu dot = u - mu, change jacobian
        #phi_[0,1,ind_time-1] = phi_[0,1,ind_time] - dt * ( phi_[0,1,ind_time] * jac[1,1] + phi_[0,0,ind_time] * jac[0,1] )
        
        
        #mu dot = u - mu, change jacobian
        if (ind_time != phi_.shape[2]-1 ):
            #phi_[0,1,ind_time-1] = phi_[0,1,ind_time] - dt * ( phi_[0,1,ind_time] * jac[1,1] + phi_[0,0,ind_time+1] * jac[0,1] )
            
            #print("res 0 = ", phi_[0,1,ind_time-1])
            
            lambda00 = phi_[0,0,ind_time+1]
            lambda01 = phi_[0,0,ind_time]
            #lambda20 = phi_[0,2,ind_time+1]
            #lambda21 = phi_[0,2,ind_time]
            phi_[0,0,ind_time] = lambda00
            #phi_[0,2,ind_time] = lambda20
            
            phi_[0,1,ind_time-1] = np.dot(phi_[0,:,ind_time],Dxdot[:,1]) - dt * ( full_cost_grad[1] + np.dot(phi_[0,:,ind_time],jac[:,1]))
            #print("phi = ", phi_[0,1,ind_time-1])
            
            #phi_dot_ = full_cost_grad + np.dot(phi_[0,:,ind_time],jac[:,:])
            #for i in range(Dxdot.shape[0]):
            #    if Dxdot[i,i] != 0.:
            #        # assume diagonal matrix
            #        phi_[0,i,ind_time-1] = phi_[0,i,ind_time] - dt * ( phi_dot_[i] / Dxdot[i,i] )
            
            phi_[0,0,ind_time] = lambda01
            #phi_[0,2,ind_time] = lambda21
        
            
        
        
        # runge kutta
       #k1 =  ( phi_[0,1,ind_time] * jac[1,1] + phi_[0,0,ind_time] * jac[0,1] )
        #k2 =  ( (phi_[0,1,ind_time] + 0.5 * (-dt) * k1) * jac[1,1] + phi_[0,0,ind_time] * jac[0,1] )
        #k3 =  ( (phi_[0,1,ind_time] + 0.5 * (-dt) * k2) * jac[1,1] + phi_[0,0,ind_time] * jac[0,1] )
        #k4 =  ( (phi_[0,1,ind_time] + (-dt) * k3) * jac[1,1] + phi_[0,0,ind_time] * jac[0,1] )
        
        #phi_[0,1, ind_time-1] = phi_[0,1,ind_time] + ((-dt) / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
        
        
    return phi_

# computation of g
# does not work with numba because takes model as parameter
def g(model, phi_, state_, control_):
    g_ = model.getZeroControl()
    
    grad_cost_e_ = cost.cost_energy_gradient(control_)
    grad_cost_s_ = cost.cost_sparsity_gradient1(model, control_)
    
    # shift, if control impacts same muf
    phi_shift = np.zeros(( phi_.shape ))
    phi_shift[:,:,0:-1] = phi_[:,:,1:]
    ########phi_shift[:,:,1:] = phi_[:,:,0:-1]
    #print("phi, phi shift: ", phi_, phi_shift)
    
    phi1_ = np.zeros(( grad_cost_e_.shape ))
    for t in range(state_.shape[2]):
        jac_u_ = D_u_h(model, state_[:,:,t])
        phi1_[0,0,t] = np.dot(phi_shift[0,:,t], jac_u_)[1]
    
    g_[:,0,:] = grad_cost_e_[0,0,:] + grad_cost_s_[0,0,:] - phi_[0,1,:]
    #g_[:,0,:] = grad_cost_e_[0,0,:] + grad_cost_s_[0,0,:] - phi_shift[0,1,:]
    #g_[:,0,:] = grad_cost_e_[0,0,:] + grad_cost_s_[0,0,:] + phi1_
    #print("contribution from energy : ", grad_cost_e_[0,0,:])
    #print("contribution from phi :", - phi_[0,1,:])

    return g_

def jacobian(model, state_t_, control_t_):
    jacobian_ = np.zeros((state_t_.shape[1], state_t_.shape[1]))
    jacobian_[0,0] = 1.
    jacobian_[0,1] = - dh_dmu(model, 1.5, state_t_[0,1], model.params.precalc_r) *1e3
    jacobian_[0,1] = - 1.
    
    #jacobian_[1,1] = 1. / state_t_[0,2]
    jacobian_[1,1] = 1.
    
    #jacobian_[2,1] = - dh_dmu(model, 1.5, state_t_[0,1], model.params.precalc_tau_mu)
    #print("jacobian = ", jacobian_[2,1])
    #jacobian_[2,1] = -1.
    #jacobian_[2,1] = 0.
    #jacobian_[2,2] = 1.
    
    return jacobian_

def D_xdot(model, state_t_):
    dxdot_ = np.zeros((state_t_.shape[1], state_t_.shape[1]))
    dxdot_[1,1] = 1.
    return dxdot_

def D_u_h(model, state_t_):
    duh_ = np.zeros(( state_t_.shape[1], state_t_.shape[1] ))
    duh_[1,1] = -1. #/ state_t_[0,2]
    return duh_
    

def dh_dmu(model, sigma, mu, table):
    #return 25.
    #print("parameters : ", model.params.sigmae, mu)
    return jac_aln.der_mu(model, sigma, mu, 0., table)

def dh_dsigma(model, sigma, mu, table):
    #return 25.
    #print("parameters : ", model.params.sigmae, mu)
    return jac_aln.der_sigma(model, sigma, mu, 0., table)

def h_inv(r):
    return r/25.