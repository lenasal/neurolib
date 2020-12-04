import numpy as np
import logging
import numba
from numba.typed import List

from timeit import default_timer as timer
from . import costFunctions as cost
from . import func_optimize as fo
from ..models import jacobian_aln as jac_aln

np.set_printoptions(precision=8)


VALID_VAR = {None, "HS", "FR", "PR", "HZ"}

def A1(model, control_, target_state_, c_scheme_, u_mat_, u_scheme_, max_iteration_, tolerance_, startStep_,
       cntrl_max_, cntrl_min_, t_sim_, t_sim_pre_, t_sim_post_, CGVar = None, control_variables_ = [0,1], prec_variables_ = [0,1]):
        
    dt = model.params['dt']
    max_iteration_ = int(max_iteration_)
        
    prec_variables = List()
    for v in prec_variables_:
        prec_variables.append(v)
        
    control_variables = List()
    for v in control_variables_:
        control_variables.append(v)
       
    ##############################################
    # PARAMETERS FOR JACOBIAN
    # TODO: time dependent exc current
    ext_exc_current = model.params.ext_exc_current
    ext_inh_current = model.params.ext_inh_current
    
    ext_exc_rate = model.params.ext_exc_rate
    ext_inh_rate = model.params.ext_inh_rate
    
    
    sigmae_ext = model.params.sigmae_ext
    sigmai_ext = model.params.sigmai_ext
    
    a = model.params["a"]
    b = model.params["b"]
    tauA = model.params["tauA"]
    
    C = model.params["C"]
    c_gl = model.params["c_gl"]
    Ke_gl = model.params["Ke_gl"]
    
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
    
    tau_se_sq = tau_se**2
    tau_si_sq = tau_si**2
    Jee_sq = Jee_max**2
    Jei_sq = Jei_max**2
    Jie_sq = Jie_max**2
    Jii_sq = Jii_max**2
    
    tau_ou = model.params.tau_ou
    
    N = model.params.N
    
    ndt_de = np.around(model.params.de / dt).astype(int)
    ndt_di = np.around(model.params.di / dt).astype(int)
    
    factor_ee1 = ( cee * Ke * tau_se / np.abs(Jee_max) )
    factor_ee2 = ( cee**2 * Ke * tau_se_sq / Jee_sq )

    factor_ei1 = ( cei * Ki * tau_si / np.abs(Jei_max) )
    factor_ei2 = ( cei**2 * Ki * tau_si_sq / Jei_sq )
    
    factor_ie1 = ( cie * Ke * tau_se / np.abs(Jie_max) )
    factor_ie2 = ( cie**2 * Ke * tau_se_sq / Jie_sq )
   
    factor_ii1 = ( cii * Ki * tau_si / np.abs(Jii_max) )
    factor_ii2 = ( cii**2 * Ki * tau_si_sq / Jii_sq )
    
    factor_eec1 = c_gl * Ke_gl * tau_se / np.abs(Jee_max)
    factor_eec2 = c_gl**2 * Ke_gl * tau_se_sq / Jee_sq 
    
    rd_exc = np.zeros(( N,N ))
    rd_inh = np.zeros(( N ))
    
    sigmarange = model.params["sigmarange"]
    ds = model.params["ds"]
    Irange = model.params["Irange"]
    dI = model.params["dI"]
    precalc_r = model.params["precalc_r"]
    precalc_tau_mu = model.params["precalc_tau_mu"]
    precalc_V = model.params["precalc_V"]
    ##############################################
        
    startind_ = int(model.getMaxDelay() + 1)
            
    state_vars = model.state_vars
    init_vars = model.init_vars
    n_control_vars = len(model.control_input_vars)
    
    if (startind_ > 1):
        fo.adjust_shape_init_params(model, init_vars, startind_)
    
    t_pre_ndt = np.around(t_sim_pre_ / dt).astype(int)
    delay_state_vars_ = np.zeros(( model.params.N, len(state_vars), startind_ ))
    
    if ( startind_ > 1 and t_pre_ndt <= startind_ ):
        logging.error("Not possible to set up initial conditions without sufficient simulation time before control")
        #return
    
    # simulate with duration t_sim_pre before start
    if (t_sim_pre_ >= dt):
        model.params['duration'] = t_sim_pre_
        control_pre_ = model.getZeroControl()
        state_pre_ = fo.updateFullState(model, control_pre_, state_vars)
                
        if startind_ == 1:
            fo.update_init(model, init_vars, state_vars)
        else:
            fo.update_init_delayed(model, delay_state_vars_, init_vars, state_vars, t_pre_ndt, startind_)
    
    model.params['duration'] = t_sim_
    #control_[:,2:,-2:] = 0.
    state0_ = fo.updateFullState(model, control_, state_vars)
    #print("state0 = ", state0_)
    
    T = int( 1 + np.around(t_sim_ / dt, 1) )
    V = state0_.shape[1]
    V_target = target_state_.shape[1]
    i=0
    
    total_cost_ = np.zeros((max_iteration_+1))
    total_cost_[i] = cost.f_int(N, n_control_vars, T, dt, state0_, target_state_, control_, v_ = prec_variables )
    runtime_ = np.zeros(( int(max_iteration_+1) ))
    runtime_start_ = timer()
    
    print("RUN ", i, ", total integrated cost = ", total_cost_[i])
    
    if CGVar not in VALID_VAR:
        print("No valid variant of conjugate gradient descent selected, use none instead.")
        CGVar = None

    state1_ = state0_.copy()
    u_opt0_ = control_.copy()
    best_control_ = control_.copy()
    
    full_cost_grad = np.zeros(( N, 2, T ))   
    
    startstep_exc_ = startStep_
    startstep_inh_ = startStep_
    startstep_joint_ = startStep_
    
    grad0_ = np.zeros(( N, n_control_vars, T ))
    grad1_ = grad0_.copy()
    dir0_ = grad0_.copy()
        
    while( i < max_iteration_ ):
        
        for ind_time in range(T):
            f_p_grad_t_ = cost.cost_precision_gradient_t(N, V_target, state0_[:,:2,ind_time], target_state_[:,:,ind_time])
            for v in prec_variables:
                full_cost_grad[0,v,ind_time] = f_p_grad_t_[0,v] 
                
        #print("1")
        
        phi0_ = phi(N, V, T, dt, state0_, target_state_, best_control_, full_cost_grad,
                    ext_exc_current,
                    ext_inh_current,
                    ext_exc_rate,
                    ext_inh_rate,
                    sigmae_ext,
                    sigmai_ext,
                    a,
                    b,
                    tauA,
                    C,
                    Ke,
                    Ki,
                    tau_se,
                    tau_si,
                    Jee_max,
                    Jei_max,
                    Jie_max,
                    Jii_max,
                    tau_se_sq,
                    tau_si_sq,
                    Jee_sq,
                    Jei_sq,
                    Jie_sq,
                    Jii_sq,
                    taum,
                    tau_ou,
                    ndt_de,
                    ndt_di,
                    factor_ee1,
                    factor_ee2,
                    factor_ei1,
                    factor_ei2,
                    factor_ie1,
                    factor_ie2,
                    factor_ii1,
                    factor_ii2,
                    factor_eec1,
                    factor_eec2,
                    rd_exc,
                    rd_inh,
                    sigmarange, ds, Irange, dI, 
                    precalc_r, precalc_tau_mu, precalc_V,
                    )
        
        if ( total_cost_[i] < tolerance_ ):
            print("Cost negligibly small.")
            max_iteration_ = i
            break
            
        i += 1   
        
        grad0_ = grad1_.copy()
        
        phi1_ = phi1(N, V, T, n_control_vars, phi0_, state1_, best_control_,
                          sigmae_ext,
                          ext_exc_rate,
                          tau_se,
                          tau_si,
                          tau_se_sq,
                          Jee_sq,
                          Jei_sq,
                          taum,
                          factor_ee1,
                          factor_ee2,
                          factor_ei1,
                          factor_ei2,
                          factor_eec1,
                          factor_eec2,
                          rd_exc,
                          rd_inh,
                          ndt_de,
                          ndt_di,
                     )
        
            
        grad1_ = fo.compute_gradient(N, n_control_vars, T, dt, best_control_, grad1_, phi1_, control_variables)
        
        dir0_ = fo.set_direction(N, T, n_control_vars, grad0_, grad1_, dir0_, i, CGVar, tolerance_)
        
        #dir0_[:,2:,-2] = 0. #pre-last rate control does not impact anything
        
        minCost = []
        tc_exc = -1
        tc_inh = -1
        joint_cost = -1
        
        #print("5")
                
        # compute stepsize separately and then put together
        if 0 in control_variables:
            d_exc = dir0_.copy()
            d_exc[:,1:,:] = 0.
            
            s_exc, tc_exc, startstep_exc_ = fo.step_size(model, N, n_control_vars, T, dt, state1_[:,:2,:], target_state_,
                         best_control_, d_exc, start_step_ = startstep_exc_, max_it_ = 1000, max_control_ = cntrl_max_,
                         min_control_ = cntrl_min_, variables_ = prec_variables)
            minCost.append(tc_exc)
        
        #print("step size exc = ", s_exc)
        
        if 1 in control_variables:
            d_inh = dir0_.copy()
            d_inh[:,0,:] = 0.
            d_inh[:,2:,:] = 0.
            
            s_inh, tc_inh, startstep_inh_ = fo.step_size(model, N, n_control_vars, T, dt, state1_[:,:2,:], target_state_,
                         best_control_, d_inh, start_step_ = startstep_inh_, max_it_ = 1000, max_control_ = cntrl_max_,
                         min_control_ = cntrl_min_, variables_ = prec_variables)
            minCost.append(tc_inh)
            
            #print("step size inh = ", s_inh)
        
        if 0 in control_variables and 1 in control_variables:
            joint_dir = dir0_.copy()
            joint_dir[:,0,:] = s_exc * dir0_[:,0,:] #/ (s_exc + s_inh)
            joint_dir[:,1,:] = s_inh * dir0_[:,1,:] #/ (s_exc + s_inh)
            
            joint_step_, joint_cost, startstep_joint_ = fo.step_size(model, N, n_control_vars, T, dt, state1_[:,:2,:], target_state_,
                         best_control_, joint_dir, start_step_ = startstep_joint_, max_it_ = 1000, max_control_ = cntrl_max_,
                         min_control_ = cntrl_min_, variables_ = prec_variables)
            minCost.append(joint_cost)
            
        #print(state1_[:,:2,:])
        
        step_, total_cost_[i], startStep_ = fo.step_size(model, N, n_control_vars, T, dt, state1_[:,:2,:], target_state_,
                     best_control_, dir0_, start_step_ = startStep_, max_it_ = 1000, max_control_ = cntrl_max_,
                         min_control_ = cntrl_min_, variables_ = prec_variables)
        
        minCost.append(total_cost_[i])
        
        #print("step size = ", step_, total_cost_[i])
        
        costMin = np.amin( minCost )
            
        if (tc_exc ==  costMin):
            #print("choose exc only")
            step_ = s_exc
            total_cost_[i] = tc_exc
            dir0_ = d_exc.copy()
            #startStep_ = startstep_exc_
            
        elif (tc_inh ==  costMin):
            #print("choose inh only")
            step_ = s_inh
            total_cost_[i] = tc_inh
            dir0_ = d_inh.copy()
            #startStep_ = startstep_inh_
        
        elif (joint_cost ==  costMin):
            #print("choose exc, inh combination")
            step_ = joint_step_
            total_cost_[i] = joint_cost
            dir0_ = joint_dir.copy()
            #startStep_ = startstep_joint_ 
        #else:
            #print("choose adjoint")
            #startStep_ = startstep_adj_
            
        #print("found step ", step_)
        #print("continue with start steps ", startstep_exc_, startstep_inh_, startstep_joint_, startStep_)        
        
        runtime_[i] = timer() - runtime_start_
        
        print("RUN ", i, ", total integrated cost = ", total_cost_[i])
        best_control_ = u_opt0_ + step_ * dir0_
        
        # why is this needed?  
        best_control_ = fo.setmaxcontrol(n_control_vars, best_control_, cntrl_max_, cntrl_min_)
        
        u_diff_ = ( np.absolute(best_control_ - u_opt0_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i
            break
        
        if ( np.amax(np.absolute(grad1_)) < tolerance_ ):
            print("Gradient negligibly small.")
            max_iteration_ = i
            break
        
        u_opt0_ = best_control_.copy()   
        state1_ = fo.updateFullState(model, best_control_, state_vars)
                
        s_diff_ = ( np.absolute(state1_ - state0_) < tolerance_)
        if ( s_diff_.all() ):
            print("State only changes marginally.")
        #    max_iteration_ = i
        #    break
        
        state0_ = state1_.copy() 
                
    state1_ = fo.updateFullState(model, best_control_, state_vars)
    
    for j in [5,6,7,8]:
        if np.amin(state1_[0,j,:]) < 0. or np.amax(state1_[0,j,:]) > 1.:
            print("WARNING: s-parameter not in proper range")
    
    improvement = 100.
    if total_cost_[0] != 0.:
        improvement = 100. - (100.*(total_cost_[max_iteration_]/total_cost_[0]))
        
    print("Improved over ", max_iteration_, " iterations by ", improvement, " percent.")
    
    """
    if max_iteration_ != 0:
        max_g, min_g = np.amax(g0_min_), np.amin(g0_min_)
        print("max value of final gradient at index = ", np.where(g0_min_ == max_g), max_g )
        print("min value of final gradient at index = ", np.where(g0_min_ == min_g), min_g )
    """
        
    if (t_sim_pre_ < dt and t_sim_post_ < dt):
        return best_control_, state1_, total_cost_, runtime_, grad1_
    
    t_post_ndt = np.around(t_sim_post_ / dt).astype(int)
    
    state_post_ = 0.
    
    if (t_sim_post_ > dt):
        
        if startind_ == 1:
            fo.update_init(model, init_vars, state_vars)
        else:
            fo.update_init_delayed(model, delay_state_vars_, init_vars, state_vars, t_post_ndt, startind_)
    
        model.params.duration = t_sim_post_ #- dt
        control_post_ = model.getZeroControl()
        state_post_ = fo.updateFullState(model, control_post_, state_vars)
 
    model.params.duration = t_sim_ + t_sim_pre_ + t_sim_post_
    bc_ = model.getZeroControl()
    bs_ = model.getZeroFullState()
        
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    
    fo.set_pre_post(i1, i2, bc_, bs_, best_control_, state_pre_, state1_, state_post_, state_vars, model.params.a, model.params.b)
            
    return bc_, bs_, total_cost_, runtime_, grad1_

#@numba.njit
def phi(N, V, T, dt, state_, target_state_, control_, full_cost_grad,
                    ext_exc_current,
                    ext_inh_current,
                    ext_exc_rate,
                    ext_inh_rate,
                    sigmae_ext,
                    sigmai_ext,
                    a,
                    b,
                    tauA,
                    C,
                    Ke,
                    Ki,
                    tau_se,
                    tau_si,
                    Jee_max,
                    Jei_max,
                    Jie_max,
                    Jii_max,
                    tau_se_sq,
                    tau_si_sq,
                    Jee_sq,
                    Jei_sq,
                    Jie_sq,
                    Jii_sq,
                    taum,
                    tau_ou,
                    ndt_de,
                    ndt_di,
                    factor_ee1,
                    factor_ee2,
                    factor_ei1,
                    factor_ei2,
                    factor_ie1,
                    factor_ie2,
                    factor_ii1,
                    factor_ii2,
                    factor_eec1,
                    factor_eec2,
                    rd_exc,
                    rd_inh,
                    sigmarange, ds, Irange, dI, 
                    precalc_r, precalc_tau_mu, precalc_V,
                    ):
    
    phi_ = np.zeros(( N, V, T ))
    
            
    for ind_time in range(T-1, -1, -1):
                    
        if ind_time + ndt_de < T:
            shift_e = ndt_de
        else:
            shift_e = 0
    
        if ind_time + ndt_di < T:
            shift_i = ndt_di
        else:
            shift_i = 0
            
        # could leave shift at zero in jacobian and algorithm would do almost equally well
        
        #shift_e = 0
        #shift_i = 0
        
        rd_exc[0,0] = state_[0,0,ind_time+shift_e] * 1e-3        
        rd_inh[0] = state_[0,1,ind_time+shift_i] * 1e-3
    
        jac = jacobian(V, state_[:,:,:], control_[:,:,:], ind_time,
                       ext_exc_current,
                       ext_inh_current,
                       ext_exc_rate,
                       ext_inh_rate,
                       sigmae_ext,
                       sigmai_ext,
                       a,
                       b,
                       tauA,
                       tau_se,
                       tau_si,
                       Jee_max,
                       Jei_max,
                       Jie_max,
                       Jii_max,
                       tau_se_sq,
                       tau_si_sq,
                       Jee_sq,
                       Jei_sq,
                       Jie_sq,
                       Jii_sq,
                       taum,
                       tau_ou,
                       factor_ee1,
                       factor_ee2,
                       factor_ei1,
                       factor_ei2,
                       factor_ie1,
                       factor_ie2,
                       factor_ii1,
                       factor_ii2,
                       factor_eec1,
                       factor_eec2,
                       rd_exc,
                       rd_inh,
                       sigmarange, ds, Irange, dI,
                       C,
                       precalc_r, precalc_tau_mu, precalc_V,
                       )
                            
        phi_[0,0,ind_time] = - full_cost_grad[0,0,ind_time] - np.dot( np.array( [phi_[0,2,ind_time], phi_[0,4,ind_time],
                                phi_[0,5,ind_time+shift_e], phi_[0,7,ind_time+shift_e], phi_[0,9,ind_time+shift_e],
                                phi_[0,11,ind_time+shift_e], phi_[0,15,ind_time+shift_e], phi_[0,16,ind_time+shift_e] ] ),
                                np.array( [jac[2,0], jac[4,0], jac[5,0], jac[7,0], jac[9,0], jac[11,0], jac[15,0], jac[16,0]] ) )
                                
        phi_[0,1,ind_time] = - full_cost_grad[0,1,ind_time] - np.dot( np.array( [phi_[0,3,ind_time], phi_[0,6,ind_time+shift_i],
                                phi_[0,8,ind_time+shift_i], phi_[0,10,ind_time+shift_i], phi_[0,12,ind_time+shift_i], 
                                phi_[0,15,ind_time+shift_i], phi_[0,16,ind_time+shift_i] ] ), 
                                np.array( [jac[3,1], jac[6,1], jac[8,1], jac[10,1], jac[12,1], jac[15,1], jac[16,1]] ) )
        
        if (ind_time == 0):
                break
        
        if (ind_time != T-1):
            der = phi_[0,0,ind_time+1] * jac[0,2] + phi_[0,2,ind_time] * jac[2,2] + phi_[0,17,ind_time] * jac[17,2] + phi_[0,18,ind_time] * jac[18,2]
            phi_[0,2,ind_time-1] = phi_[0,2,ind_time] - dt * der
            #print("der = ", der)
            #print(phi_[0,0,ind_time+1] * jac[0,2] , phi_[0,2,ind_time] * jac[2,2] , phi_[0,17,ind_time] * jac[17,2] , phi_[0,18,ind_time] * jac[18,2])
            
            der = phi_[0,1,ind_time+1] * jac[1,3] + phi_[0,3,ind_time] * jac[3,3] + phi_[0,19,ind_time] * jac[19,3]
            phi_[0,3,ind_time-1] = phi_[0,3,ind_time] - dt * der
            
            der = phi_[0,2,ind_time] * jac[2,5] + phi_[0,5,ind_time] * jac[5,5] + phi_[0,9,ind_time] * jac[9,5]
            phi_[0,5,ind_time-1] = phi_[0,5,ind_time] - dt * der
            
            der = phi_[0,2,ind_time] * jac[2,6] + phi_[0,6,ind_time] * jac[6,6] + phi_[0,10,ind_time] * jac[10,6]
            #phi_[0,6,ind_time-1] = phi_[0,6,ind_time] - dt * der
            
            der = phi_[0,3,ind_time] * jac[3,7] + phi_[0,7,ind_time] * jac[7,7] + phi_[0,11,ind_time] * jac[11,7]
            #phi_[0,7,ind_time-1] = phi_[0,7,ind_time] - dt * der
            
            der = phi_[0,3,ind_time] * jac[3,8] + phi_[0,8,ind_time] * jac[8,8] + phi_[0,12,ind_time] * jac[12,8]
            #phi_[0,8,ind_time-1] = phi_[0,8,ind_time] - dt * der
            
            der = phi_[0,9,ind_time] * jac[9,9] + phi_[0,15,ind_time] * jac[15,9]
            phi_[0,9,ind_time-1] = phi_[0,9,ind_time] - dt * der
            
            der = phi_[0,10,ind_time] * jac[10,10] + phi_[0,15,ind_time] * jac[15,10]
            #phi_[0,10,ind_time-1] = phi_[0,10,ind_time] - dt * der
            
            der = phi_[0,11,ind_time] * jac[11,11] + phi_[0,16,ind_time] * jac[16,11]
            #phi_[0,11,ind_time-1] = phi_[0,11,ind_time] - dt * der
            
            der = phi_[0,12,ind_time] * jac[12,12] + phi_[0,16,ind_time] * jac[16,12]
            #phi_[0,12,ind_time-1] = phi_[0,12,ind_time] - dt * der
            
            # do not impact anything else, could be left out
            #der = phi_[0,2,ind_time] * jac[2,13] + phi_[0,13,ind_time] * jac[13,13]
            #phi_[0,13,ind_time-1] = phi_[0,13,ind_time] - dt * der
            
            #der = phi_[0,3,ind_time] * jac[3,14] + phi_[0,14,ind_time] * jac[14,14]
            #phi_[0,14,ind_time-1] = phi_[0,14,ind_time] - dt * der
                
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

@numba.njit
def phi1(N, V, T, n_control_vars, phi_, state_, control_,
                          sigmae_ext,
                          ext_exc_rate,
                          tau_se,
                          tau_si,
                          tau_se_sq,
                          Jee_sq,
                          Jei_sq,
                          taum,
                          factor_ee1,
                          factor_ee2,
                          factor_ei1,
                          factor_ei2,
                          factor_eec1,
                          factor_eec2,
                          rd_exc,
                          rd_inh,
                          ndt_de,
                          ndt_di,
                         ):  
    
    phi1_ = np.zeros(( N, n_control_vars, T ))
    
    ind_t = 0
            
        # could leave shift at zero in jacobian and algorithm would do almost equally well
        
    shift_e = 0
    shift_i = 0
        
    rd_exc[0,0] = state_[0,0,ind_t-shift_e] * 1e-3        
    rd_inh[0] = state_[0,1,ind_t-shift_i] * 1e-3
    

    jac_u_ = D_u_h(V, state_, control_, ind_t,
                          sigmae_ext,
                          ext_exc_rate,
                          tau_se,
                          tau_si,
                          tau_se_sq,
                          Jee_sq,
                          Jei_sq,
                          taum,
                          factor_ee1,
                          factor_ee2,
                          factor_ei1,
                          factor_ei2,
                          factor_eec1,
                          factor_eec2,
                          rd_exc,
                          rd_inh,
                           )
    phi = np.ascontiguousarray(phi_[0,:,0])

    y2 = np.ascontiguousarray(jac_u_[2,:])
    y3 = np.ascontiguousarray(jac_u_[3,:])
    
    #print("phi = ", phi)
    #print("y2 = ", y2)

    phi1_[0,2,0] = np.dot(phi, y2)
    phi1_[0,3,0] = np.dot(phi, y3)
    
    for ind_t in range(1, T):
        
        if ind_t - ndt_de > 0:
            shift_e = ndt_de
        else:
            shift_e = 0
    
        if ind_t - ndt_di > 0:
            shift_i = ndt_di
        else:
            shift_i = 0
            
        # could leave shift at zero in jacobian and algorithm would do almost equally well
        
        #shift_e = 0
        #shift_i = 0
        
        rd_exc[0,0] = state_[0,0,ind_t-shift_e] * 1e-3        
        rd_inh[0] = state_[0,1,ind_t-shift_i] * 1e-3
        
        jac_u_ = D_u_h(V, state_, control_, ind_t,
                          sigmae_ext,
                          ext_exc_rate,
                          tau_se,
                          tau_si,
                          tau_se_sq,
                          Jee_sq,
                          Jei_sq,
                          taum,
                          factor_ee1,
                          factor_ee2,
                          factor_ei1,
                          factor_ei2,
                          factor_eec1,
                          factor_eec2,
                          rd_exc,
                          rd_inh,
                           )
        phi = np.ascontiguousarray(phi_[0,:,ind_t])
        phi_shift = np.ascontiguousarray(phi_[0,:,ind_t-1])#, dtype=np.float64)
        
        y0 = np.ascontiguousarray(jac_u_[0,:])
        y1 = np.ascontiguousarray(jac_u_[1,:])
        y2 = np.ascontiguousarray(jac_u_[2,:])
        y3 = np.ascontiguousarray(jac_u_[3,:])
        
        #print("phi = ", phi)
        #print("y2 = ", y2)

        #res = np.dot(phi_[0,:,ind_t-1], jac_u_) # shift if control is applied shifted wrt mu
        phi1_[0,0,ind_t] = np.dot(phi_shift, y0)
        phi1_[0,1,ind_t] = np.dot(phi_shift, y1)
        phi1_[0,2,ind_t] = np.dot(phi, y2)
        phi1_[0,3,ind_t] = np.dot(phi, y3)
        
        #print(phi1_[0,0,ind_t] == res1, phi1_[0,1,ind_t] == res2)

    return phi1_

@numba.njit
def D_xdot(V, state_t_):
    dxdot_ = np.zeros(( V, V ))
    return dxdot_

@numba.njit
def D_u_h(V, state_, control_, t_, 
          sigmae_ext,
          ext_exc_rate,
          tau_se,
          tau_si,
          tau_se_sq,
          Jee_sq,
          Jei_sq,
          taum,
          factor_ee1,
          factor_ee2,
          factor_ei1,
          factor_ei2,
          factor_eec1,
          factor_eec2,
          rd_exc,
          rd_inh,
          ):
    duh_ = np.zeros(( 4, V ))
    duh_[0,2] = - 1. / state_[0,18,t_-1]
    duh_[1,3] = - 1. / state_[0,19,t_-1]
    
    #duh_[2,5] = - (1. - state_[0,5,t_]) * factor_eec1 / tau_se
    duh_[2,5] = - 0.1 * (1. - state_[0,5,t_]) * factor_eec1 / tau_se
        
    z1ee = factor_ee1 * rd_exc[0,0] + factor_eec1 * ( ext_exc_rate + control_[0,2,t_] )
    z2ee = factor_ee2 * rd_exc[0,0] + factor_eec2 * ( ext_exc_rate + control_[0,2,t_] )
    
    z1ei = factor_ei1 * rd_inh[0]
    z2ei = factor_ei2 * rd_inh[0]     

    #duh_[2,9] = - ( (1. - state_[0,5,t_])**2 * factor_eec2 + state_[0,9,t_] * ( factor_eec2
    #            - ( tau_se + tau_se ) *  factor_eec1 ) ) / tau_se_sq  
    # (z2ee - 2 * tau_se * (z1ee + 1))
    duh_[2,9] = (- 2. * tau_se * factor_eec1 ) * state_[0,9,t_] / tau_se_sq 
        
    sig_ee = state_[0,9,t_] * ( 2. * Jee_sq * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)
    #sig_ee = ( 2. * Jee_sq * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)
    sig_ei = state_[0,10,t_] * ( 2. * Jei_sq * tau_si * taum ) * ( (1 + z1ei) * taum + tau_si )**(-1)
    sig_ei = 0.
    
    if sig_ee + sig_ei + sigmae_ext**2 > 0.:
        sigma_sqrt_e = ( sig_ee + sig_ei + sigmae_ext**2 )**(-1./2.)
    else:
       # print("WARNING: sigma sqrt e not positive")
        sigma_sqrt_e = 0.
    
    #duh_[2,15] = 0.5 * factor_eec1 * taum * ( (1 + z1ee) * taum + tau_se )**(-2.) * sigma_sqrt_e * state_[0,9,t_]
    duh_[2,15] = 0.5 * factor_eec1 * taum * ( (1 + z1ee) * taum + tau_se )**(-2) * state_[0,9,t_] * ( 2. * Jee_sq * tau_se * taum ) * sigma_sqrt_e
    
    return duh_

@numba.njit
def jacobian(V, state_, control_, t_,
              ext_exc_current,
              ext_inh_current,
              ext_exc_rate,
              ext_inh_rate,
              sigmae_ext,
              sigmai_ext,
              a,
              b,
              tauA,
              tau_se,
              tau_si,
              Jee_max,
              Jei_max,
              Jie_max,
              Jii_max,
              tau_se_sq,
              tau_si_sq,
              Jee_sq,
              Jei_sq,
              Jie_sq,
              Jii_sq,
              taum,
              tau_ou,
              factor_ee1,
              factor_ee2,
              factor_ei1,
              factor_ei2,
              factor_ie1,
              factor_ie2,
              factor_ii1,
              factor_ii2,
              factor_eec1,
              factor_eec2,
              rd_exc,
              rd_inh,
              sigmarange, ds, Irange, dI,
              C,
              precalc_r, precalc_tau_mu, precalc_V,
              ):
    
    z1ee = factor_ee1 * rd_exc[0,0] + factor_eec1 * ( ext_exc_rate + control_[0,2,t_] )
    z2ee = factor_ee2 * rd_exc[0,0] + factor_eec2 * ( ext_exc_rate + control_[0,2,t_] )
    
    z1ei = factor_ei1 * rd_inh[0]
    z2ei = factor_ei2 * rd_inh[0]
    
    z1ie = factor_ie1 * rd_exc[0,0]
    z2ie = factor_ie2 * rd_exc[0,0]
            
    z1ii = factor_ii1 * rd_inh[0]
    z2ii = factor_ii2 * rd_inh[0]
    
    jacobian_ = np.zeros(( V, V ))
    jacobian_[0,0] = 1.
    jacobian_[0,2] = - d_r_func_mu(state_[0,2,t_] - state_[0,4,t_] / C, sigmarange, ds, state_[0,15,t_], Irange, dI, C, precalc_r) * 1e3
    jacobian_[0,4] = - d_r_func_mu(state_[0,2,t_-1] - state_[0,4,t_-1] / C, sigmarange, ds, state_[0,15,t_-1], Irange, dI, C, precalc_r) * 1e3 * ( - 1. / C ) 
    jacobian_[0,15] = - d_r_func_sigma(state_[0,2,t_-1] - state_[0,4,t_] / C, sigmarange, ds, state_[0,15,t_-1], Irange, dI, C, precalc_r) * 1e3
    
    jacobian_[1,1] = 1.
    jacobian_[1,3] = - d_r_func_mu(state_[0,3,t_], sigmarange, ds, state_[0,16,t_], Irange, dI, C, precalc_r) * 1e3
    jacobian_[1,16] = - d_r_func_sigma(state_[0,3,t_-1],sigmarange, ds, state_[0,16,t_-1], Irange, dI, C, precalc_r) * 1e3
    
    #jacobian_[2,2] = 1. / state_[0,18,t_]
    jacobian_[2,5] = - Jee_max / state_[0,18,t_]
    #jacobian_[2,6] = - Jei_max / state_[0,18,t_]
    #jacobian_[2,13] = - 1. / state_[0,18,t_]
    #jacobian_[2,18] = ( Jee_max * state_[0,5,t_-1] + Jei_max * state_[0,6,t_-1] + control_[0,0,t_] + ext_exc_current
    #                   + state_[0,13,t_-1] - state_[0,2,t_-1] ) / state_[0,18,t_-1]**2
    #jacobian_[2,2] = 1. 
    
    jacobian_[3,3] = 1. / state_[0,19,t_]
    jacobian_[3,7] = - Jie_max / state_[0,19,t_]
    jacobian_[3,8] = - Jii_max / state_[0,19,t_]
    jacobian_[3,14] = - 1. / state_[0,19,t_]
    jacobian_[3,19] = ( Jii_max * state_[0,8,t_-1] + Jie_max * state_[0,7,t_-1] + control_[0,1,t_] + ext_inh_current
                       + state_[0,14,t_-1] - state_[0,3,t_-1] ) / state_[0,19,t_-1]**2
    
    jacobian_[4,0] = - b * 1e-3
    jacobian_[4,4] = 1. / tauA
    jacobian_[4,17] = - a / tauA
    
    jacobian_[5,0] = - 0.1 * (1. - state_[0,5,t_]) * factor_ee1 * 1e-3 / tau_se
    jacobian_[5,5] = 0.1 * ( 1. + ( factor_ee1 * rd_exc[0,0] + factor_eec1 * control_[0,2,t_]) ) / tau_se
    #jacobian_[5,5] = 0.1 * ( 1. + z1ee ) / tau_se
    #jacobian_[5,5] = ( 1. + factor_eec1 * control_[0,2,t_] ) / tau_se
    
    #jacobian_[6,1] = - (1. - state_[0,6,t_]) * factor_ei1 * 1e-3 / tau_si
    #jacobian_[6,6] = ( 1. + z1ei ) / tau_si
    
    #jacobian_[7,0] = - (1. - state_[0,7,t_]) * factor_ie1 * 1e-3 / tau_se
    #jacobian_[7,7] = ( 1. + z1ie ) / tau_se
    
    #jacobian_[8,1] = - (1. - state_[0,8,t_]) * factor_ii1 * 1e-3 / tau_si
    #jacobian_[8,8] = ( 1. + z1ii ) / tau_si
    
    #jacobian_[9,0] = - ( (1. - state_[0,5,t_])**2 * factor_ee2 + state_[0,9,t_] * ( factor_ee2 - ( tau_se + tau_se ) *  factor_ee1 ) ) * 1e-3 / tau_se_sq
    jacobian_[9,0] = (- 2. * tau_se * factor_ee1 * 1e-3 ) * state_[0,9,t_] / tau_se_sq
    #jacobian_[9,5] = 2. * (1. - state_[0,5,t_]) * z2ee / tau_se_sq
    jacobian_[9,9] = ( - 2. * tau_se * ( z1ee + 1.) ) / tau_se_sq
    
    jacobian_[10,1] = - ( (1. - state_[0,6,t_])**2 * factor_ei2 + state_[0,10,t_] * ( factor_ei2 - ( tau_si + tau_si ) *  factor_ei1 ) ) * 1e-3 / tau_si_sq
    jacobian_[10,6] = 2. * (1. - state_[0,6,t_]) * z2ei / tau_si_sq
    jacobian_[10,10] = - (z2ei - ( tau_si + tau_si ) * ( z1ei + 1.) ) / tau_si_sq
    
    jacobian_[11,0] = - ( (1. - state_[0,7,t_])**2 * factor_ie2 + state_[0,11,t_] * ( factor_ie2 - ( tau_se + tau_se ) *  factor_ie1 ) ) * 1e-3 / tau_se_sq
    jacobian_[11,7] = 2. * (1. - state_[0,7,t_]) * z2ie / tau_se_sq
    jacobian_[11,11] = - (z2ie - ( tau_se + tau_se ) * ( z1ie + 1.) ) / tau_se_sq
    
    jacobian_[12,1] = - ( (1. - state_[0,8,t_])**2 * factor_ii2 + state_[0,12,t_] * ( factor_ii2 - ( tau_si + tau_si ) *  factor_ii1 ) ) * 1e-3 / tau_si_sq
    jacobian_[12,8] = 2. * (1. - state_[0,8,t_]) * z2ii / tau_si_sq
    jacobian_[12,12] = - (z2ii - ( tau_si + tau_si ) * ( z1ii + 1.) ) / tau_si_sq
    
    jacobian_[13,13] = 1. / tau_ou
    
    jacobian_[14,14] = 1. / tau_ou
    
    sig_ee = state_[0,9,t_] * ( 2. * Jee_sq * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)
    #sig_ee = ( 2. * Jee_sq * tau_se * taum ) * ( (1 + z1ee) * taum + tau_se )**(-1)
    sig_ei = state_[0,10,t_] * ( 2. * Jei_sq * tau_si * taum ) * ( (1 + z1ei) * taum + tau_si )**(-1)
    sig_ei = 0.
    
    if sig_ee + sig_ei + sigmae_ext**2 > 0.:
        sigma_sqrt_e = ( sig_ee + sig_ei + sigmae_ext**2 )**(-1./2.)
    else:
        #print("WARNING: sigma sqrt e not positive")
        sigma_sqrt_e = 0.
    
    jacobian_[15,0] = 0.5 * (1e-3) * factor_ee1 * taum * ( (1 + z1ee) * taum + tau_se )**(-2) * state_[0,9,t_] * ( 2. * Jee_sq * tau_se * taum ) * sigma_sqrt_e
    #jacobian_[15,0] = 0.5 * (1e-3) * factor_ee1 * taum * ( (1 + z1ee) * taum + tau_se )**(-2.) * ( 2. * Jee_sq * tau_se * taum ) * sigma_sqrt_e
    #jacobian_[15,1] = 0.5 * (1e-3) * factor_ei1 * taum * ( (1 + z1ei) * taum + tau_si )**(-2) * state_[0,10,t_] * ( 2. * Jei_sq * tau_si * taum ) * sigma_sqrt_e
    jacobian_[15,9] = - 0.5 * ( (1 + z1ee) * taum + tau_se )**(-1) * ( 2. * Jee_sq * tau_se * taum ) * sigma_sqrt_e
    #jacobian_[15,10] = - 0.5 * ( (1 + z1ei) * taum + tau_si )**(-1) * ( 2. * Jei_sq * tau_si * taum ) * sigma_sqrt_e
    #jacobian_[15,15] = 1.
    
    sig_ii = state_[0,12,t_] * ( 2. * Jii_sq * tau_si * taum ) * ( (1 + z1ii) * taum + tau_si )**(-1)
    sig_ie = state_[0,11,t_] * ( 2. * Jie_sq * tau_se * taum ) * ( (1 + z1ie) * taum + tau_se )**(-1)
    
    if sig_ii + sig_ie + sigmai_ext**2 > 0.:
        sigma_sqrt_i = ( sig_ii + sig_ie + sigmai_ext**2 )**(-1./2.)
    else:
        #print("WARNING: sigma sqrt i not positive")
        sigma_sqrt_i = 0.
    
    jacobian_[16,0] = 0.5 * (1e-3) * factor_ie1 * taum * ( (1 + z1ie) * taum + tau_se )**(-2) * state_[0,11,t_] * ( 2. * Jie_sq * tau_se * taum ) * sigma_sqrt_i
    jacobian_[16,1] = 0.5 * (1e-3) * factor_ii1 * taum * ( (1 + z1ii) * taum + tau_si )**(-2) * state_[0,12,t_] * ( 2. * Jii_sq * tau_si * taum ) * sigma_sqrt_i
    jacobian_[16,11] = - 0.5 * ( (1 + z1ie) * taum + tau_se )**(-1) * ( 2. * Jie_sq * tau_se * taum ) * sigma_sqrt_i
    jacobian_[16,12] = - 0.5 * ( (1 + z1ii) * taum + tau_si )**(-1) * ( 2. * Jii_sq * tau_si * taum ) * sigma_sqrt_i
    jacobian_[16,16] = 1.
    
    jacobian_[17,2] = - d_V_func_mu(state_[0,2,t_] - state_[0,4,t_] / C, sigmarange, ds, state_[0,15,t_], Irange, dI, C, precalc_V)
    jacobian_[17,4] = - d_V_func_mu(state_[0,2,t_-1] - state_[0,4,t_-1] / C, sigmarange, ds, state_[0,15,t_-1], Irange, dI, C, precalc_V) * ( - 1. / C )
    jacobian_[17,15] = - d_V_func_sigma(state_[0,2,t_-1] - state_[0,4,t_-1] / C, sigmarange, ds, state_[0,15,t_-1], Irange, dI, C, precalc_V)
    
    jacobian_[18,2] = - d_tau_func_mu(state_[0,2,t_] - state_[0,4,t_] / C, sigmarange, ds, state_[0,15,t_], Irange, dI, C, precalc_tau_mu)
    jacobian_[18,4] = - d_tau_func_mu(state_[0,2,t_-1] - state_[0,4,t_-1] / C, sigmarange, ds, state_[0,15,t_-1], Irange, dI, C, precalc_tau_mu) * ( - 1. / C )
    jacobian_[18,15] = - d_tau_func_sigma(state_[0,2,t_-1] - state_[0,4,t_-1] / C, sigmarange, ds, state_[0,15,t_-1], Irange, dI, C, precalc_tau_mu)
    
    jacobian_[19,3] = - d_tau_func_mu(state_[0,3,t_], sigmarange, ds, state_[0,16,t_], Irange, dI, C, precalc_tau_mu)
    jacobian_[19,16] = - d_tau_func_sigma(state_[0,3,t_-1], sigmarange, ds, state_[0,16,t_-1], Irange, dI, C, precalc_tau_mu)
    
    return jacobian_

@numba.njit
def d_r_func_mu(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_r):
    return 1e-3
    #result = jac_aln.der_mu(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_r)
    x_shift_mu = - 2.
    x_scale_mu = 0.6
    y_scale_mu = 0.1
    result = y_scale_mu * x_scale_mu / np.cosh(x_scale_mu * mu + x_shift_mu)**2
    #result = 1e-3
    return result

@numba.njit
def d_r_func_sigma(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_r):
    return 1e-3
    #result = jac_aln.der_sigma(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_r)
    x_shift_sigma = -1.
    x_scale_sigma = 0.6
    y_scale_sigma = 1./2500.
    result = np.sinh(x_scale_sigma * sigma + x_shift_sigma) * y_scale_sigma * x_scale_sigma
    #result = 1e-3
    return result

@numba.njit
def d_tau_func_mu(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_tau_mu):
    return 0.
    #result = jac_aln.der_mu(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_tau_mu)
    mu_shift = - 1.1
    sigma_scale = 0.5
    mu_scale = - 10
    mu_scale1 = - 3
    sigma_shift = 1.4
    result = sigma_scale * sigma + mu_scale1 + ( mu_scale / (sigma + sigma_shift) ) * np.exp( mu_scale * ( mu_shift + mu ) / ( sigma + sigma_shift ) )
    #result = 1e-3
    return result

@numba.njit
def d_tau_func_sigma(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_tau_mu):
    return 0.
    #result = jac_aln.der_sigma(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_tau_mu)
    mu_shift = - 1.1
    sigma_scale = 0.5
    mu_scale = - 10
    sigma_shift = 1.4
    result = sigma_scale * ( mu_shift + mu ) - (mu_scale * (mu_shift + mu) / (sigma + sigma_shift)**2) * np.exp(
        mu_scale * ( mu_shift + mu ) / ( sigma + sigma_shift ) )  
    #result = 1e-3
    return result

@numba.njit
def d_V_func_mu(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_V):
    return 0.
    #result = jac_aln.der_mu(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_V)
    y_scale1 = 30.
    mu_shift1 = 1.
    y_scale2 = 2.
    mu_shift2 = 0.5
    result = y_scale1 / np.cosh( mu + mu_shift1 )**2 - y_scale2 * 2. * ( mu - mu_shift2 ) * np.exp( - ( mu - mu_shift2 )**2 ) / sigma
    #result = 1e-3
    return result

@numba.njit
def d_V_func_sigma(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_V):
    return 0.
    #result = jac_aln.der_sigma(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_V)
    y_scale2 = 2.
    mu_shift2 = 0.5
    result = - y_scale2 * np.exp( - ( mu - mu_shift2 )**2 ) / sigma**2
    #result = 1e-3
    return result