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
    state0_[:,1,:] = model.state["mufe"]
    state0_[:,2,:] = model.state["seev"]
    state0_[:,3,:] = model.state["sigmae_f"]
    state0_[:,4,:] = model.state["tau_exc"]
    #print("state 0 = ", state0_[:,:,:])
    #print("target = ", target_state_[:,:,:])
    #print("control = ", control_)
    total_cost_ = np.zeros((max_iteration_+1))
    total_cost_[i] = cost.f_int(model.params['dt'], cost.f_cost(state0_, target_state_, control_) )
    print("RUN ", i, ", total integrated cost = ", total_cost_[i])

    
    state1_ = state0_.copy()
    u_opt0_ = control_.copy()
    best_control_ = control_.copy()
    
    #best_control_[:,:,0] = best_control_[:,:,1]
    #best_control_[:,:,-2] = best_control_[:,:,-1]
    
    while( i < max_iteration_):
        i += 1   
        
        #phi_ = phi(model, state0_, target_state_, best_control_, c_scheme_)
        #print("phi = ", phi_)
        phi_ = phi(model, state0_, target_state_, best_control_)
        #print("phi = ", phi_)
        #diff = np.abs(phi_-phi2_)
        #print("equal results : ", diff.all() < 1e-16)
        #print("phi 0, 15, 2 : ", phi_[0,15,2], phi2_[0,15,2])
    
        g0_min_ = g(model, phi_, state1_, best_control_)
        g1_min_ = g0_min_.copy()
        #print("g_min = ", g0_min_)

        dir0_ = - g0_min_.copy()
        #dir0_[:,:,0] = dir0_[:,:,1].copy()
        #dir0_[:,:,-1] = dir0_[:,:,-2].copy()
        dir1_ = dir0_.copy()
        
        outstate_ = model.getZeroState()
        outstate_[:,:,:] = state1_[:,0,:]
        
        step_, total_cost_[i] = fo.step_size(model, outstate_[:,:,:], target_state_,
                     best_control_, dir1_, start_step_ = startStep_, max_control_ = cntrl_max_)
        print("step size = ", step_, total_cost_[i])
        
        print("RUN ", i, ", total integrated cost = ", total_cost_[i])
        best_control_ = u_opt0_ + step_ * dir1_
        #best_control_[:,:,0] = best_control_[:,:,1]
        #print("shape best control = ", best_control_.shape)
        #best_control_[:,:,-1] = best_control_[:,:,-2]
        #print("shape best control = ", best_control_.shape)
        
        u_diff_ = ( np.absolute(best_control_ - u_opt0_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i
            break
        u_opt0_ = best_control_.copy()
        
        rate_ = fo.updateState(model, best_control_)
    
        state1_[:,0,:] = rate_[:,0,:]
        state1_[:,1,:] = model.state["mufe"]
        state1_[:,2,:] = model.state["seev"]
        state1_[:,3,:] = model.state["sigmae_f"]
        state1_[:,4,:] = model.state["tau_exc"]
        
        #print("state 1 = ", state1_[:,:,:])
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
    
    improvement = 100.
    if total_cost_[0] != 0.:
        improvement = improvement = 100. - (100.*(total_cost_[max_iteration_]/total_cost_[0]))
        
    print("Improved over ", max_iteration_, " iterations by ", improvement, " percent.")
    
    return best_control_, state1_, total_cost_, 0.

def phi(model, state_, target_state_, control_, start_ind_ = 0):
    #print("ALN phi2 computation")
    phi_ = model.getZeroFullState()
    dt = model.params['dt']
    out_state = model.getZeroState()
    out_state[:,:,:] = state_[:,0,:]
    
    phi2 = phi_.copy()
        
    for ind_time in range(phi_.shape[2]-1, start_ind_-1, -1):
        f_p_grad_t_ = cost.cost_precision_gradient_t(out_state[:,:,ind_time], target_state_[:,:,ind_time])
        Dxdot = D_xdot(model, state_[:,:,ind_time])
        jac = jacobian(model, state_[:,:,ind_time], control_[:,:,ind_time])
        full_cost_grad = np.zeros(( state_[0,:,ind_time].shape ))
        full_cost_grad[0] = f_p_grad_t_[0,0]
        
        phi_[0,0,ind_time] = - f_p_grad_t_[0,0]
        #print("grad = ", phi_[0,0,ind_time])
        
        jac1 = np.delete(jac, (1,2), axis=0)
        jac1 = np.delete(jac1, (1,2), axis=1)
        jac2 = np.delete(jac, (0,3,4), axis=0)
        jac2 = np.delete(jac2, (1,2), axis=1)
        res = np.dot( - np.array( [full_cost_grad[0],full_cost_grad[3],full_cost_grad[4]] ) - np.dot( phi2[0,1:3,ind_time],jac2 ) , np.linalg.inv(jac1))
        phi2[0,0,ind_time] = res[0]
        phi2[0,3,ind_time] = res[1]
        phi2[0,4,ind_time] = res[2]
        
        if (ind_time == 0):
            break
        
        d1 = ( model.params.ext_exc_current - state_[0,1,ind_time] + control_[0,0, ind_time] ) / state_[0,4,ind_time]**2
        phi_[0,1,ind_time-1] = phi_[0,1,ind_time] - dt * ( - phi_[0,0,ind_time] * dh_dmu(model, state_[0,3,ind_time], state_[0,1,ind_time], model.params.precalc_r)
                                + phi_[0,1,ind_time] / state_[0,4,ind_time] 
                                #+ phi_[0,1,ind_time] * d1 * dh_dmu(model, state_[0,1,ind_time], model.params.precalc_tau_mu) 
                                )
        #print("summands : ", f_p_grad_t_[0,0] * dh_dmu(model, state_[0,2,ind_time], model.params.precalc_r), 
        #      + phi_[0,2,ind_time] / state_[0,4,ind_time],
        #      + phi_[0,2,ind_time] * d1 * dh_dmu(model, state_[0,2,ind_time], model.params.precalc_tau_mu))
        
                
        phi2[0,1:3,ind_time-1] = np.dot(phi_[0,:,ind_time],Dxdot[:,1:3]) - dt * ( full_cost_grad[1:3] + np.dot(phi_[0,:,ind_time],jac[:,1:3]))
     
    #print("phi as calc from matrix : ", phi2)
    return phi2

# computation of g
# does not work with numba because takes model as parameter
def g(model, phi_, state_, control_):
    g_ = model.getZeroControl()
    
    grad_cost_e_ = cost.cost_energy_gradient(control_)
    grad_cost_s_ = cost.cost_sparsity_gradient1(model, control_)
    
    g_[:,0,:] = grad_cost_e_[0,0,:] + grad_cost_s_[0,0,:] - phi_[0,1,:] / state_[0,4,:]
    #print("control = ", control_)
    #print("energy cost contribution exc: ", grad_cost_e_[0,0,:])
    #print("energy cost contribution inh: ", grad_cost_e_[0,1,:])
    #print("adjoint contribution exc: ", - phi_[0,2,:] / state_[0,4,:])
    #print("adjoint contribution inh: ", - phi_[0,3,:] / state_[0,5,:])
    return g_

def jacobian(model, state_t_, control_t_):
    jacobian_ = np.zeros((state_t_.shape[1], state_t_.shape[1]))
    
    cee = model.params.cee
    Ke = model.params.Ke
    Jee = model.params.Jee
    tau_se = model.params.tau_se
    taum = model.params.C / model.params.gL
    sigmae_ext = model.params.sigmae_ext
    
    # rates derivatives
    jacobian_[0,0] = 1.
    jacobian_[0,1] = - dh_dmu(model, state_t_[0,3], state_t_[0,1], model.params.precalc_r)
    jacobian_[0,3] = - dh_dsigma(model, state_t_[0,3], state_t_[0,1], model.params.precalc_r)
    # mu derivatives
    jacobian_[1,1] = 1. / state_t_[0,4]
    jacobian_[1,4] = (model.params.ext_exc_current - state_t_[0,1] + control_t_[0,0]) / state_t_[0,4]**2
    # sigma bar derivatives
    jacobian_[2,0] = - state_t_[0,2] * (cee**2 * Ke / Jee**2 - 2. * cee * Ke / Jee)
    jacobian_[2,2] = - (cee**2 * Ke * tau_se**2 * state_t_[0,0] / Jee**2
                        - 2. * tau_se * ( cee * Ke * tau_se * state_t_[0,0]/ Jee + 1)) / state_t_[0,4]**2
    #sigma derivatives
    jacobian_[3,0] = (2. * Jee**2 * state_t_[0,2] * tau_se * taum / (( 1. + cee * Ke * tau_se * state_t_[0,0] / Jee ) * taum + tau_se ) 
                      + sigmae_ext**2 )**(-1./2.) * ( Jee * state_t_[0,2] * tau_se**2 * taum**2 * cee * Ke
                                                     / (( 1. + cee * Ke * tau_se * state_t_[0,0] / Jee ) * taum + tau_se )**2 )
    jacobian_[3,2] = 0.5 * (2. * Jee**2 * state_t_[0,2] * tau_se * taum / (( 1. + cee * Ke * tau_se * state_t_[0,0] / Jee ) * taum + tau_se ) 
                      + sigmae_ext**2 )**(-1./2.) * (2. * Jee**2 * tau_se * taum
                                                     / (( 1. + cee * Ke * tau_se * state_t_[0,0] / Jee ) * taum + tau_se ) )
    jacobian_[3,3] = 1.
    # tau derivatives
    jacobian_[4,1] = - dh_dmu(model, state_t_[0,3], state_t_[0,1], model.params.precalc_tau_mu)
    jacobian_[4,3] = - dh_dsigma(model, state_t_[0,3], state_t_[0,1], model.params.precalc_tau_mu)
    jacobian_[4,4] = 1.
    
    return jacobian_

def D_xdot(model, state_t_):
    dxdot_ = np.zeros((state_t_.shape[1], state_t_.shape[1]))
    dxdot_[1,1] = 1.
    dxdot_[2,2,] = 2.
    return dxdot_
    

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