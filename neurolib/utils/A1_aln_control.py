import numpy as np
import logging
import numba

from timeit import default_timer as timer
from . import costFunctions as cost
from . import func_optimize as fo
from ..models import jacobian_aln as jac_aln

np.set_printoptions(precision=8)


VALID_VAR = {None, "FR", "HS"}

def A1(model, control_, target_state_, c_scheme_, u_mat_, u_scheme_, max_iteration_, tolerance_, startStep_,
       cntrl_max_, t_sim_, t_sim_pre_, t_sim_post_, CGVar):
        
    dt = model.params['dt']

    
    """    
    # run model with dt duration once to set delay matrix
    model.params['duration'] = dt
    model.run(control=model.getZeroControl())
    
    Dmat = model.params.Dmat
    Dmat_ndt = np.around(Dmat / dt).astype(int)
        
    Dmat_ndt = np.around(Dmat / dt).astype(int)
        
    if model.name == "aln":
        ndt_de = np.around(model.params.de / dt).astype(int)
        ndt_di = np.around(model.params.di / dt).astype(int)
        max_global_delay = max(np.max(Dmat_ndt), ndt_de, ndt_di)
    else:
        max_global_delay = np.max(Dmat_ndt)
        
    print(max_global_delay == model.getMaxDelay() )
        
    startind_ = int(max_global_delay + 1)
    """
    startind_ = 1
    
    state_vars = model.state_vars
    init_vars = model.init_vars
    output_vars = model.output_vars
    
    if (startind_ > 1):
        fo.adjust_shape_init_params(model, init_vars, startind_)    
    
    t_pre_ndt = np.around(t_sim_pre_ / dt).astype(int)
    delay_state_vars_ = np.zeros(( model.params.N, len(state_vars), startind_ ))
    
    # simulate with duration t_sim_pre before start
    if (t_sim_pre_ >= dt):
        model.params['duration'] = t_sim_pre_
        control_pre_ = model.getZeroControl()
        state_pre_ = fo.updateFullState(model, control_pre_)
        
        if startind_ == 1:
            fo.update_init(model, init_vars, state_vars)
        else:
            fo.update_init_delayed(model, delay_state_vars_, init_vars, state_vars, t_pre_ndt, startind_)
    
    model.params['duration'] = t_sim_
    i=0
        
    #rate_ = fo.updateState(model, control_)
    #state0_ = model.getZeroFullState()
    state0_ = fo.updateFullState(model, control_)
    
    """
    state0_[:,0,:] = rate_[:,0,:]
    state0_[:,1,:] = model.state["rates_inh"][:,:]
    state0_[:,2,:] = model.state["mufe"][:,:]
    state0_[:,3,:] = model.state["mufi"][:,:]
    state0_[:,4,:] = model.state["IA"][:,:]
    
    state0_[:,5,:] = model.state["seem"][:,:]
    state0_[:,6,:] = model.state["seim"][:,:]
    state0_[:,7,:] = model.state["siem"][:,:]
    state0_[:,8,:] = model.state["siim"][:,:]
    state0_[:,9,:] = model.state["seev"][:,:]
    state0_[:,10,:] = model.state["seiv"][:,:]
    state0_[:,11,:] = model.state["siev"][:,:]
    state0_[:,12,:] = model.state["siiv"][:,:]
    
    state0_[:,13,:] = model.state["mue_ou"][:,:]
    state0_[:,14,:] = model.state["mui_ou"][:,:]
    
    state0_[:,15,:] = model.state["sigmae_f"][:,:]
    state0_[:,16,:] = model.state["sigmai_f"][:,:]
    state0_[:,17,:] = model.state["Vmean_exc"][:,:]
    state0_[:,18,:] = model.state["tau_exc"][:,:]
    state0_[:,19,:] = model.state["tau_inh"][:,:]
    """    

    total_cost_ = np.zeros((max_iteration_+1))
    total_cost_[i] = cost.f_int(model.params['dt'], cost.f_cost(state0_, target_state_, control_) )
    runtime_ = np.zeros(( int(max_iteration_+1) ))
    runtime_start_ = timer()
    
    print("RUN ", i, ", total integrated cost = ", total_cost_[i])

    state1_ = state0_.copy()
    u_opt0_ = control_.copy()
    best_control_ = control_.copy()
    
    phi0_ = model.getZeroFullState()
    
    while( i < max_iteration_):
        i += 1   
        
        phi1_ = phi(model, state0_, target_state_, best_control_, phi0_)
        #print("phi = ")
        #print(phi1_[0,:,3:])
        
        outstate_ = model.getZeroState()
        outstate_[:,0,:] = state1_[:,0,:]
        outstate_[:,1,:] = state1_[:,1,:]
    
        g0_min_ = g(model, phi1_, state1_, best_control_)
        g1_min_ = g0_min_.copy()

        dir0_ = - g0_min_.copy()
        dir1_ = dir0_.copy()

        
        step_, total_cost_[i] = fo.step_size(model, outstate_[:,:,:], target_state_,
                     best_control_, dir1_, start_step_ = startStep_, max_control_ = cntrl_max_)
        
        runtime_[i] = timer() - runtime_start_
        
        print("RUN ", i, ", total integrated cost = ", total_cost_[i])
        best_control_ = u_opt0_ + step_ * dir1_
        
        u_diff_ = ( np.absolute(best_control_ - u_opt0_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i
            break
        u_opt0_ = best_control_.copy()
        
        rate_ = fo.updateState(model, best_control_)
        
        state1_ = fo.updateFullState(model, best_control_)
        
        a1 = state1_[:,0,:] == rate_[:,0,:]
        a2 = state1_[:,1,:] == model.state["rates_inh"][:,:]
        a3 = state1_[:,2,:] == model.state["mufe"][:,:]
        a4 = state1_[:,3,:] == model.state["mufi"][:,:]
        a5 = state1_[:,4,:] == model.state["IA"][:,:]
        
        a6 = state1_[:,5,:] == model.state["seem"][:,:]
        a7 = state1_[:,6,:] == model.state["seim"][:,:]
        a8 = state1_[:,7,:] == model.state["siem"][:,:]
        a9 = state1_[:,8,:] == model.state["siim"][:,:]
        a10 = state1_[:,9,:] == model.state["seev"][:,:]
        a11 = state1_[:,10,:] == model.state["seiv"][:,:]
        a12 = state1_[:,11,:] == model.state["siev"][:,:]
        a13 = state1_[:,12,:] == model.state["siiv"][:,:]
        
        a14 = state1_[:,13,:] == model.state["mue_ou"][:,:]
        a15 = state1_[:,14,:] == model.state["mui_ou"][:,:]
        
        a16 = state1_[:,15,:] == model.state["sigmae_f"][:,:]
        a17 = state1_[:,16,:] == model.state["sigmai_f"][:,:]
        a18 = state1_[:,17,:] == model.state["Vmean_exc"][:,:]
        a19 = state1_[:,18,:] == model.state["tau_exc"][:,:]
        a20 = state1_[:,19,:] == model.state["tau_inh"][:,:]
        
        if np.array( [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20]).any() == False:
            print("something wrong")
        
        
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
        print("max abs value of final gradient = ", np.amax(np.abs(g0_min_)) )
        
    if (t_sim_pre_ < dt and t_sim_post_ < dt):
        return best_control_, state1_, total_cost_, runtime_
    
    if (t_sim_post_ > dt):
        
        for iv, sv in zip( range(len(init_vars)), range(len(state_vars)) ):
            if state_vars[sv] in init_vars[iv]:
                #print("variable = ", state_vars[sv])
                #print(" state = ", model.state[state_vars[sv]])
                #print(" init param = ", model.params[init_vars[iv]])
                if model.params[init_vars[iv]].ndim == 2:
                    if startind_ == 1:
                        model.params[init_vars[iv]][:,0] = model.state[state_vars[sv]][:,-1]
                    else:
                        model.params[init_vars[iv]][:,:] = delay_state_vars_[:, sv, :]              
                else:
                    model.params[init_vars[iv]][:] = model.state[state_vars[sv]][:,-1]
            else:
                logging.error("Initial and state variable labelling does not agree.") 

    
        model.params.duration = t_sim_post_ - dt
        control_post_ = model.getZeroControl()
        state_post_ = fo.updateFullState(model, control_post_)
 
    model.params.duration = t_sim_ + t_sim_pre_ + t_sim_post_
    bc_ = model.getZeroControl()
    bs_ = model.getZeroFullState()
        
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    
    if (i2 != 0 and i1 != 0):   
        bc_[:,:,i1:-i2] = best_control_[:,:,:]
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        for n in range(bs_.shape[0]):
            for v in range(bs_.shape[1]):
                if state_vars[v] is not "Vmean_exc":
                    if bs_[n,v,i1] != state1_[n,v,0]:
                        logging.error("Problem in initial value trasfer")
                        print("Problem in initial value trasfer for ", state_vars[v])
                        print(bs_[n,v,:])
                        print(state1_[n,v,:])
        bs_[:,:,i1:-i2] = state1_[:,:,:]
        bs_[:,:,-i2:] = state_post_[:,:,:]
    elif (i2 == 0 and i1 != 0):
        bc_[:,:,i1:] = best_control_[:,:,:]
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        for n in range(bs_.shape[0]):
            for v in range(bs_.shape[1]):
                if state_vars[v] is not "Vmean_exc":
                    if bs_[n,v,i1] != state1_[n,v,0]:
                        logging.error("Problem in initial value trasfer")
                        print("Problem in initial value trasfer for ", state_vars[v])
                        print(bs_[n,v,:])
                        print(state1_[n,v,:])
        bs_[:,:,i1:] = state1_[:,:,:]
    elif (i2 != 0 and i1 == 0):
        bc_[:,:,:-i2] = best_control_[:,:,:]
        bs_[:,:,:-i2] = state1_[:,:,:]
        bs_[:,:,-i2:] = state_post_[:,:,:]
    else:
        bc_[:,:,:] = best_control_[:,:,:]
        bs_[:,:,:] = state1_[:,:,:]
            
    return bc_, bs_, total_cost_, runtime_


def phi(model, state_, target_state_, control_, phi_prev_, start_ind_ = 0):
    dt = model.params.dt
    phi_ = model.getZeroFullState()
    out_state = model.getZeroState()
    out_state[:,0,:] = state_[:,0,:]
    out_state[:,1,:] = state_[:,1,:]
            
    for ind_time in range(phi_.shape[2]-1, start_ind_-1, -1):
        jac = jacobian(model, state_[:,:,:], control_[:,:,:], ind_time)
        
        f_p_grad_t_ = cost.cost_precision_gradient_t(out_state[:,:,ind_time], target_state_[:,:,ind_time])
        full_cost_grad = np.zeros(( state_[0,:,ind_time].shape ))
        full_cost_grad[0] = f_p_grad_t_[0,0]
        full_cost_grad[1] = f_p_grad_t_[0,1]
        
        #res = np.dot( - np.array( [full_cost_grad[0], full_cost_grad[1], full_cost_grad[2]] ), np.linalg.inv(jac))   
        #phi_[0,0,ind_time] = res[0]
        #phi_[0,1,ind_time-1] = res[1]
        #phi_[0,2,ind_time-1] = res[2]
        
        if (ind_time == 0):
            break
        
        phi_[0,0,ind_time] = - full_cost_grad[0] - np.dot( np.array( [phi_[0,2,ind_time], phi_[0,4,ind_time], phi_[0,5,ind_time],
                                                                      phi_[0,7,ind_time], phi_[0,9,ind_time], phi_[0,11,ind_time],
                                                                      phi_[0,15,ind_time], phi_[0,16,ind_time] ] ),
                                                          np.array( [jac[2,0], jac[4,0], jac[5,0], jac[7,0], jac[9,0], jac[11,0], jac[15,0], jac[16,0]] ) )
        
        phi_[0,1,ind_time] = - full_cost_grad[1] - np.dot( np.array( [phi_[0,3,ind_time], phi_[0,6,ind_time], phi_[0,8,ind_time],
                                                                      phi_[0,10,ind_time], phi_[0,12,ind_time], phi_[0,15,ind_time],
                                                                      phi_[0,16,ind_time] ] ),
                                                          np.array( [jac[3,1], jac[6,1], jac[8,1], jac[10,1], jac[12,1], jac[15,1], jac[16,1]] ) )
    
        
        if (ind_time != phi_.shape[2]-1):
            der = phi_[0,0,ind_time+1] * jac[0,2] + phi_[0,2,ind_time] * jac[2,2] + phi_[0,17,ind_time] * jac[17,2] + phi_[0,18,ind_time] * jac[18,2]
            phi_[0,2,ind_time-1] = phi_[0,2,ind_time] - dt * der
            
            der = phi_[0,1,ind_time+1] * jac[1,3] + phi_[0,3,ind_time] * jac[3,3] + phi_[0,19,ind_time] * jac[19,3]
            phi_[0,3,ind_time-1] = phi_[0,3,ind_time] - dt * der
            
            der = phi_[0,2,ind_time] * jac[2,5] + phi_[0,5,ind_time] * jac[5,5] + phi_[0,9,ind_time] * jac[9,5]
            phi_[0,5,ind_time-1] = phi_[0,5,ind_time] - dt * der
            
            der = phi_[0,2,ind_time] * jac[2,6] + phi_[0,6,ind_time] * jac[6,6] + phi_[0,10,ind_time] * jac[10,6]
            phi_[0,6,ind_time-1] = phi_[0,6,ind_time] - dt * der
            
            der = phi_[0,3,ind_time] * jac[3,7] + phi_[0,7,ind_time] * jac[7,7] + phi_[0,11,ind_time] * jac[11,7]
            phi_[0,7,ind_time-1] = phi_[0,7,ind_time] - dt * der
            
            der = phi_[0,3,ind_time] * jac[3,8] + phi_[0,8,ind_time] * jac[8,8] + phi_[0,12,ind_time] * jac[12,8]
            phi_[0,8,ind_time-1] = phi_[0,8,ind_time] - dt * der
            
            der = phi_[0,3,ind_time] * jac[3,8] + phi_[0,8,ind_time] * jac[8,8] + phi_[0,12,ind_time] * jac[12,8]
            phi_[0,8,ind_time-1] = phi_[0,8,ind_time] - dt * der
            
            der = phi_[0,9,ind_time] * jac[9,9] + phi_[0,15,ind_time] * jac[15,9]
            phi_[0,9,ind_time-1] = phi_[0,9,ind_time] - dt * der
            
            der = phi_[0,10,ind_time] * jac[10,10] + phi_[0,15,ind_time] * jac[15,10]
            phi_[0,10,ind_time-1] = phi_[0,10,ind_time] - dt * der
            
            der = phi_[0,11,ind_time] * jac[11,11] + phi_[0,16,ind_time] * jac[16,11]
            phi_[0,11,ind_time-1] = phi_[0,11,ind_time] - dt * der
            
            der = phi_[0,12,ind_time] * jac[12,12] + phi_[0,16,ind_time] * jac[16,12]
            phi_[0,12,ind_time-1] = phi_[0,12,ind_time] - dt * der
            
            # do not impact anything else, could be left out
            der = phi_[0,2,ind_time] * jac[2,13] + phi_[0,13,ind_time] * jac[13,13]
            phi_[0,13,ind_time-1] = phi_[0,13,ind_time] - dt * der
            
            der = phi_[0,3,ind_time] * jac[3,14] + phi_[0,14,ind_time] * jac[14,14]
            phi_[0,14,ind_time-1] = phi_[0,14,ind_time] - dt * der
            
            
                
        res = - phi_[0,4,ind_time] * jac[4,17]
        phi_[0,17,ind_time-1] = res
        
        res = - phi_[0,2,ind_time-1] * jac[2,18]
        phi_[0,18,ind_time-1] = res
        
        res = - phi_[0,3,ind_time-1] * jac[3,19]
        phi_[0,19,ind_time-1] = res
        
        der = phi_[0,0,ind_time] * jac[0,4] + phi_[0,4,ind_time] * jac[4,4] + phi_[0,17,ind_time-1] * jac[17,4] + phi_[0,18,ind_time-1] * jac[18,4] 
        phi_[0,4,ind_time-1] = phi_[0,4,ind_time] - dt * der

        res = - phi_[0,0,ind_time] * jac[0,15] - phi_[0,18,ind_time-1] * jac[18,15] - phi_[0,17,ind_time-1] * jac[17,15] 
        phi_[0,15,ind_time-1] = res
        
        res = - phi_[0,1,ind_time] * jac[1,16] - phi_[0,19,ind_time-1] * jac[19,16]
        phi_[0,16,ind_time-1] = res
                
    return phi_

def jacobian(model, state_, control_, t_):
    
    # TODO: time dependent exc current
    ext_exc_current = model.params.ext_exc_current
    ext_inh_current = model.params.ext_inh_current
    sigmae_ext = model.params.sigmae_ext
    sigmai_ext = model.params.sigmai_ext
    
    a = model.params["a"]
    b = model.params["b"]
    tauA = model.params["tauA"]
    
    C = model.params["C"]
    
    Ke = model.params["Ke"]
    Ki = model.params["Ki"]
    tau_se = model.params["tau_se"] 
    tau_si = model.params["tau_si"] 
    cee = model.params["cee"]
    cei = model.params["cei"]
    cie = model.params["cie"]
    cii = model.params["cii"]
    Jee_max = model.params["Jee_max"]
    Jei_max = model.params["Jei_max"]
    Jie_max = model.params["Jie_max"]
    Jii_max = model.params["Jii_max"]
    taum = model.params.C / model.params.gL
    
    tau_ou = model.params.tau_ou
    
    rd_exc = np.zeros(( model.params.N,model.params.N ))
    rd_inh = np.zeros(( model.params.N ))
    rd_exc[0,0] = state_[0,0,t_] * 1e-3
    rd_inh[0] = state_[0,1,t_] * 1e-3
    
    factor_ee1 = ( cee * Ke * tau_se / np.abs(Jee_max) )
    factor_ee2 = ( cee**2 * Ke * tau_se**2 / Jee_max**2 )
    
    z1ee = factor_ee1 * rd_exc[0,0]
    z2ee = factor_ee2 * rd_exc[0,0]
    
    factor_ei1 = ( cei * Ki * tau_si / np.abs(Jei_max) )
    factor_ei2 = ( cei**2 * Ki * tau_si**2 / Jei_max**2 )
            
    z1ei = factor_ei1 * rd_inh[0]
    z2ei = factor_ei2 * rd_inh[0]
    
    factor_ie1 = ( cie * Ke * tau_se / np.abs(Jie_max) )
    factor_ie2 = ( cie**2 * Ke * tau_se**2 / Jie_max**2 )
    
    z1ie = factor_ie1 * rd_exc[0,0]
    z2ie = factor_ie2 * rd_exc[0,0]
    
    factor_ii1 = ( cii * Ki * tau_si / np.abs(Jii_max) )
    factor_ii2 = ( cii**2 * Ki * tau_si**2 / Jii_max**2 )
            
    z1ii = factor_ii1 * rd_inh[0]
    z2ii = factor_ii2 * rd_inh[0]
    
    jacobian_ = np.zeros((state_.shape[1], state_.shape[1]))
    jacobian_[0,0] = 1.
    jacobian_[0,2] = - d_r_func_mu(model, state_[0,2,t_] - state_[0,4,t_] / C, state_[0,15,t_]) * 1e3 # - state_[0,2,t_] / C
    jacobian_[0,4] = - d_r_func_mu(model, state_[0,2,t_-1] - state_[0,4,t_-1] / C, state_[0,15,t_-1]) * 1e3 * ( - 1. / C ) #  - state_[0,2,t_] / C
    jacobian_[0,15] = - d_r_func_sigma(model, state_[0,2,t_-1] - state_[0,4,t_] / C, state_[0,15,t_-1]) * 1e3 #  - state_[0,2,t_-1] / C
    
    jacobian_[1,1] = 1.
    jacobian_[1,3] = - d_r_func_mu(model, state_[0,3,t_], state_[0,16,t_]) * 1e3
    jacobian_[1,16] = - d_r_func_sigma(model, state_[0,3,t_-1], state_[0,16,t_-1]) * 1e3
    
    jacobian_[2,2] = 1. / state_[0,18,t_]
    jacobian_[2,5] = - Jee_max / state_[0,18,t_]
    jacobian_[2,6] = - Jei_max / state_[0,18,t_]
    jacobian_[2,13] = - 1. / state_[0,18,t_]
    jacobian_[2,18] = ( Jee_max * state_[0,5,t_-1] + Jei_max * state_[0,6,t_-1] + control_[0,0,t_] + ext_exc_current
                       + state_[0,13,t_-1] - state_[0,2,t_-1] ) / state_[0,18,t_-1]**2
    
    jacobian_[3,3] = 1. / state_[0,19,t_]
    jacobian_[3,7] = - Jie_max / state_[0,19,t_]
    jacobian_[3,8] = - Jii_max / state_[0,19,t_]
    jacobian_[3,14] = - 1. / state_[0,19,t_]
    jacobian_[3,19] = ( Jii_max * state_[0,8,t_-1] + Jie_max * state_[0,7,t_-1] + control_[0,1,t_] + ext_inh_current
                       + state_[0,14,t_-1] - state_[0,3,t_-1] ) / state_[0,19,t_-1]**2
    
    jacobian_[4,0] = - b * 1e-3
    jacobian_[4,4] = 1. / tauA
    jacobian_[4,17] = - a / tauA
    
    jacobian_[5,0] = - (1. - state_[0,5,t_]) * factor_ee1 * 1e-3 / tau_se
    jacobian_[5,5] = ( 1. + z1ee ) / tau_se
    
    jacobian_[6,1] = - (1. - state_[0,6,t_]) * factor_ei1 * 1e-3 / tau_si
    jacobian_[6,6] = ( 1. + z1ei ) / tau_si
    
    jacobian_[7,0] = - (1. - state_[0,7,t_]) * factor_ie1 * 1e-3 / tau_se
    jacobian_[7,7] = ( 1. + z1ie ) / tau_se
    
    jacobian_[8,1] = - (1. - state_[0,8,t_]) * factor_ii1 * 1e-3 / tau_si
    jacobian_[8,8] = ( 1. + z1ii ) / tau_si
    
    jacobian_[9,0] = - ( (1. - state_[0,5,t_])**2 * factor_ee2 + state_[0,9,t_] * ( factor_ee2 - 2. * tau_se *  factor_ee1 ) ) * 1e-3 / tau_se**2
    jacobian_[9,5] = 2. * (1. - state_[0,5,t_]) * z2ee / tau_se**2
    jacobian_[9,9] = - (z2ee - 2. * tau_se * ( z1ee + 1.) ) / tau_se**2
    
    jacobian_[10,1] = - ( (1. - state_[0,6,t_])**2 * factor_ei2 + state_[0,10,t_] * ( factor_ei2 - 2. * tau_si *  factor_ei1 ) ) * 1e-3 / tau_si**2
    jacobian_[10,6] = 2. * (1. - state_[0,6,t_]) * z2ei / tau_si**2
    jacobian_[10,10] = - (z2ei - 2. * tau_si * ( z1ei + 1.) ) / tau_si**2
    #jacobian_[10,10] = - 1. / tau_si**2
    
    jacobian_[11,0] = - ( (1. - state_[0,7,t_])**2 * factor_ie2 + state_[0,11,t_] * ( factor_ie2 - 2. * tau_se *  factor_ie1 ) ) * 1e-3 / tau_se**2
    jacobian_[11,7] = 2. * (1. - state_[0,7,t_]) * z2ie / tau_se**2
    jacobian_[11,11] = - (z2ie - 2. * tau_se * ( z1ie + 1.) ) / tau_se**2
    
    jacobian_[12,1] = - ( (1. - state_[0,8,t_])**2 * factor_ii2 + state_[0,12,t_] * ( factor_ii2 - 2. * tau_si *  factor_ii1 ) ) * 1e-3 / tau_si**2
    jacobian_[12,8] = 2. * (1. - state_[0,8,t_]) * z2ii / tau_si**2
    jacobian_[12,12] = - (z2ii - 2. * tau_si * ( z1ii + 1.) ) / tau_si**2
    
    jacobian_[13,13] = 1. / tau_ou
    
    jacobian_[14,14] = 1. / tau_ou
    
    sig_ee = state_[0,9,t_] * ( 2. * Jee_max**2 * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)
    sig_ei = state_[0,10,t_] * ( 2. * Jei_max**2 * tau_si * taum ) * ( (1 + z1ei) * taum + tau_si )**(-1)
    
    sigma_sqrt_e = ( sig_ee + sig_ei + sigmae_ext**2 )**(-1./2.)
    
    jacobian_[15,0] = 0.5 * (1e-3) * factor_ee1 * taum * ( (1 + z1ee) * taum + tau_se )**(-2) * state_[0,9,t_] * ( 2. * Jee_max**2 * tau_se * taum ) * sigma_sqrt_e
    jacobian_[15,1] = 0.5 * (1e-3) * factor_ei1 * taum * ( (1 + z1ei) * taum + tau_si )**(-2) * state_[0,10,t_] * ( 2. * Jei_max**2 * tau_si * taum ) * sigma_sqrt_e
    jacobian_[15,9] = - 0.5 * ( (1 + z1ee) * taum + tau_se )**(-1) * ( 2. * Jee_max**2 * tau_se * taum ) * sigma_sqrt_e
    jacobian_[15,10] = - 0.5 * ( (1 + z1ei) * taum + tau_si )**(-1) * ( 2. * Jei_max**2 * tau_si * taum ) * sigma_sqrt_e
    jacobian_[15,15] = 1.
    
    sig_ii = state_[0,12,t_] * ( 2. * Jii_max**2 * tau_si * taum ) * ( (1 + z1ii) * taum + tau_si )**(-1)
    sig_ie = state_[0,11,t_] * ( 2. * Jie_max**2 * tau_se * taum ) * ( (1 + z1ie) * taum + tau_se )**(-1)
    
    sigma_sqrt_i = ( sig_ii + sig_ie + sigmai_ext**2 )**(-1./2.)
    
    jacobian_[16,0] = 0.5 * (1e-3) * factor_ie1 * taum * ( (1 + z1ie) * taum + tau_se )**(-2) * state_[0,11,t_] * ( 2. * Jie_max**2 * tau_se * taum ) * sigma_sqrt_i
    jacobian_[16,1] = 0.5 * (1e-3) * factor_ii1 * taum * ( (1 + z1ii) * taum + tau_si )**(-2) * state_[0,12,t_] * ( 2. * Jii_max**2 * tau_si * taum ) * sigma_sqrt_i
    jacobian_[16,11] = - 0.5 * ( (1 + z1ie) * taum + tau_se )**(-1) * ( 2. * Jie_max**2 * tau_se * taum ) * sigma_sqrt_i
    jacobian_[16,12] = - 0.5 * ( (1 + z1ii) * taum + tau_si )**(-1) * ( 2. * Jii_max**2 * tau_si * taum ) * sigma_sqrt_i
    jacobian_[16,16] = 1.
    
    jacobian_[17,2] = - d_V_func_mu(model, state_[0,2,t_] - state_[0,4,t_] / C, state_[0,15,t_])
    jacobian_[17,4] = - d_V_func_mu(model, state_[0,2,t_-1] - state_[0,4,t_-1] / C, state_[0,15,t_-1]) * ( - 1. / C )
    jacobian_[17,15] = - d_V_func_sigma(model, state_[0,2,t_-1] - state_[0,4,t_-1] / C, state_[0,15,t_-1])
    
    jacobian_[18,2] = - d_tau_func_mu(model, state_[0,2,t_] - state_[0,4,t_] / C, state_[0,15,t_])
    jacobian_[18,4] = - d_tau_func_mu(model, state_[0,2,t_-1] - state_[0,4,t_-1] / C, state_[0,15,t_-1]) * ( - 1. / C )
    jacobian_[18,15] = - d_tau_func_sigma(model, state_[0,2,t_-1] - state_[0,4,t_-1] / C, state_[0,15,t_-1])
    
    jacobian_[19,3] = - d_tau_func_mu(model, state_[0,3,t_], state_[0,16,t_])
    jacobian_[19,16] = - d_tau_func_sigma(model, state_[0,3,t_-1], state_[0,16,t_-1])
    
    return jacobian_

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
        phi1_[0,0,t] = np.dot(phi_shift[0,:,t], jac_u_)[2]
        phi1_[0,1,t] = np.dot(phi_shift[0,:,t], jac_u_)[3]
    
    g_[:,0,:] = grad_cost_e_[0,0,:] + grad_cost_s_[0,0,:] + phi1_[0,0,:]
    g_[:,1,:] = grad_cost_e_[0,1,:] + grad_cost_s_[0,1,:] + phi1_[0,1,:]
    
    #print("grad cost e = ", grad_cost_e_)
    #print("g = ", g_)

    return g_

def D_xdot(model, state_t_):
    dxdot_ = np.zeros((state_t_.shape[1], state_t_.shape[1]))
    return dxdot_

def D_u_h(model, state_, t_):
    duh_ = np.zeros(( state_.shape[1], state_.shape[1] ))
    duh_[2,2] = - 1. / state_[0,18,t_-1]
    duh_[3,3] = - 1. / state_[0,19,t_-1]
    return duh_

def d_r_func_mu(model, mu, sigma):
    result = 0.
    if model.name == "aln-control":# or model.name == "aln":
        x_shift_mu = - 2.
        x_scale_mu = 0.6
        y_scale_mu = 0.1
        result = y_scale_mu * x_scale_mu / np.cosh(x_scale_mu * mu + x_shift_mu)**2
    elif model.name == "aln":
        result = jac_aln.der_mu_up(model, sigma, mu, model.params.precalc_r)
    else:
        print("no drivative of rate implemented")
    if np.abs(result) < 1e-7 and np.abs(result) > 1e-10:
        print("WARNING: small derivative of r wrt mu")
    return result

def d_r_func_sigma(model, mu, sigma):
    result = 0.
    if model.name == "aln-control":# or model.name == "aln":
        x_shift_sigma = -1.
        x_scale_sigma = 0.6
        y_scale_sigma = 1./2500.
        result = np.sinh(x_scale_sigma * sigma + x_shift_sigma) * y_scale_sigma * x_scale_sigma
    elif model.name == "aln":
        result = jac_aln.der_sigma(model, sigma, mu, model.params.precalc_r)
    else:
        print("no drivative of rate implemented")
    if np.abs(result) < 1e-7:
        print("WARNING: small derivative of r wrt sigma")
    return result

def d_tau_func_mu(model, mu, sigma):
    result = 0.
    if model.name == "aln-control":# or model.name == "aln":
        mu_shift = - 1.1
        sigma_scale = 0.5
        mu_scale = - 10
        mu_scale1 = - 3
        sigma_shift = 1.4
        result = sigma_scale * sigma + mu_scale1 + ( mu_scale / (sigma + sigma_shift) ) * np.exp( mu_scale * ( mu_shift + mu ) / ( sigma + sigma_shift ) )
    elif model.name == "aln":
        result = jac_aln.der_mu_up(model, sigma, mu, model.params.precalc_tau_mu)
    else:
        print("no drivative of tau implemented")
    if np.abs(result) < 1e-4:
        print("WARNING: small derivative of tau wrt mu")
    return result

def d_tau_func_sigma(model, mu, sigma):
    result = 0.
    if model.name == "aln-control":# or model.name == "aln":
        mu_shift = - 1.1
        sigma_scale = 0.5
        mu_scale = - 10
        sigma_shift = 1.4
        result = sigma_scale * ( mu_shift + mu ) - (mu_scale * (mu_shift + mu) / (sigma + sigma_shift)**2) * np.exp(
            mu_scale * ( mu_shift + mu ) / ( sigma + sigma_shift ) )  
    elif model.name == "aln":
        result = jac_aln.der_sigma(model, sigma, mu, model.params.precalc_tau_mu)
    else:
        print("no drivative of tau implemented")
    if np.abs(result) < 1e-7:
        print("WARNING: small derivative of tau wrt sigma")
    return result

def d_V_func_mu(model, mu, sigma):
    result = 0.
    if model.name == "aln-control":# or model.name == "aln":
        y_scale1 = 30.
        mu_shift1 = 1.
        y_scale2 = 2.
        mu_shift2 = 0.5
        result = y_scale1 / np.cosh( mu + mu_shift1 )**2 - y_scale2 * 2. * ( mu - mu_shift2 ) * np.exp( - ( mu - mu_shift2 )**2 ) / sigma
    elif model.name == "aln":
        result = jac_aln.der_mu_up(model, sigma, mu, model.params.precalc_V)
    else:
        print("no drivative of V implemented")
    if np.abs(result) < 1e-3:
        print("WARNING: small derivative of V wrt mu")
    return result

def d_V_func_sigma(model, mu, sigma):
    result = 0.
    if model.name == "aln-control":# or model.name == "aln":
        y_scale2 = 2.
        mu_shift2 = 0.5
        result = - y_scale2 * np.exp( - ( mu - mu_shift2 )**2 ) / sigma**2
    elif model.name == "aln":
        result = jac_aln.der_sigma(model, sigma, mu, model.params.precalc_V)
    else:
        print("no drivative of V implemented")
    if np.abs(result) < 1e-6:
        print("WARNING: small derivative of V wrt sigma")
    return result