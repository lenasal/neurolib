import numpy as np
import logging
import numba
from numba.typed import List

from timeit import default_timer as timer
from . import costFunctions as cost
from . import func_optimize as fo
from ..models import jacobian_aln as jac_aln

np.set_printoptions(precision=8)

model_name = "-aln"


VALID_VAR = {None, "HS", "FR", "PR", "CD", "LS", "DY", "WYL", "HZ"}
VALID_LS = {None, "AG"}

def A1(model, control_, target_state, c_scheme_, u_mat_, u_scheme_, max_iteration_, tolerance_, startStep_,
       cntrl_max_, cntrl_min_, t_sim_, t_sim_pre_, t_sim_post_,
       CGVar = None, line_search_ = None, control_variables_ = [0,1], prec_variables_ = [0,1], separate_comp = True, transition_time_ = 0.):
        
    dt = model.params['dt']
    max_iteration_ = int(max_iteration_)
        
    prec_variables = List()
    for v in prec_variables_:
        prec_variables.append(v)
                
    control_variables = List()
    for v in control_variables_:
        control_variables.append(v)
        
    n_maxDelay = model.getMaxDelay()
        
    startind_ = int(n_maxDelay + 1)
    target_state_ = target_state.copy()
            
    state_vars = model.state_vars
    init_vars = model.init_vars
    n_control_vars = len(model.control_input_vars)
        
    T = int( 1 + np.around(t_sim_ / dt, 1) )
    
    V = len(state_vars)
    V_target = target_state_.shape[1]
    i=0
       
    ##############################################
    # PARAMETERS FOR JACOBIAN
    # TODO: time dependent exc current
    ext_exc_current = model.params.ext_exc_current
    ext_inh_current = model.params.ext_inh_current
    
    # ee, ei, ie, ii
    ext_ee_rate = model.params.ext_ee_rate
    ext_ei_rate = model.params.ext_ei_rate
    ext_ie_rate = model.params.ext_ie_rate
    ext_ii_rate = model.params.ext_ii_rate
    
    sigmae_ext = model.params.sigmae_ext
    sigmai_ext = model.params.sigmai_ext
    
    a = model.params["a"]
    b = model.params["b"]
    tauA = model.params["tauA"]
    
    C = model.params["C"]
    c_gl = model.params["c_gl"]
    Ke_gl = model.params["Ke_gl"]
    Ki_gl = model.params["Ki_gl"]
    
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
    
    factor_eic1 = c_gl * Ki_gl * tau_si / np.abs(Jei_max)
    factor_eic2 = c_gl**2 * Ki_gl * tau_si_sq / Jei_sq 
    
    factor_iec1 = c_gl * Ke_gl * tau_se / np.abs(Jie_max)
    factor_iec2 = c_gl**2 * Ke_gl * tau_se_sq / Jie_sq 
    
    factor_iic1 = c_gl * Ki_gl * tau_si / np.abs(Jii_max)
    factor_iic2 = c_gl**2 * Ki_gl * tau_si_sq / Jii_sq 
    
    rd_exc = np.zeros(( N,N, T ))
    rd_inh = np.zeros(( N, T ))
    
    sigmarange = model.params["sigmarange"]
    ds = model.params["ds"]
    Irange = model.params["Irange"]
    dI = model.params["dI"]
    precalc_r = model.params["precalc_r"]
    precalc_tau_mu = model.params["precalc_tau_mu"]
    precalc_V = model.params["precalc_V"]
    
    interpolate_rate = model.params["interpolate_rate"]
    interpolate_V = model.params["interpolate_V"]
    interpolate_tau = model.params["interpolate_tau"]
    
    print("interpolate adjoint : ", interpolate_rate, interpolate_V, interpolate_tau)
    ##############################################
    
    if (startind_ > 1):
        fo.adjust_shape_init_params(model, init_vars, startind_)
    
    t_pre_ndt = np.around(t_sim_pre_ / dt).astype(int)
    delay_state_vars_ = np.zeros(( model.params.N, len(state_vars), startind_ ))
    
    if ( startind_ > 1 and t_pre_ndt <= startind_ ):
        logging.error("Not possible to set up initial conditions without sufficient simulation time before control")
        #return
        
    state_pre_ = np.zeros(( N,V,max(t_pre_ndt,1) ))
    
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
    
    state1_ = state0_.copy()
    u_opt0_ = control_.copy()
    best_control_ = control_.copy()
    #print("adj control = ", best_control_, cntrl_max_, cntrl_min_)
    
    # set max control
    #best_control_ = fo.scalemaxcontrol(best_control_, cntrl_max_, cntrl_min_)
    best_control_ = fo.setmaxcontrol(n_control_vars, best_control_, cntrl_max_, cntrl_min_)  
    #print("adj control = ", best_control_)
    
    ip_, ie_, is_ = cost.getParams()

    # set precision penalty to zero for transition time
    for t_ in range(T):
        if t_ < transition_time_ * T:
            for n_ in range(N):
                for v_ in range(2):
                        target_state_[n_,v_,t_] = - 1000.
                        
    total_cost_ = np.zeros((max_iteration_+1))
    #print("into initial cost")
    #print("exc rate = ", state0_[0,0,:])
    #print("target = ", target_state_[0,0,:])
    #print("control = ", control_[0,2,:])
    total_cost_[i] = cost.f_int(N, n_control_vars, T, dt, state0_, target_state_, best_control_,
                                ip_, ie_, is_, v_ = prec_variables )
    #print("initial cost = ", total_cost_[i])
    runtime_ = np.zeros(( int(max_iteration_+1) ))
    runtime_start_ = timer()
    
    print("RUN ", i, ", total integrated cost = ", total_cost_[i])
    
    if CGVar not in VALID_VAR:
        print("No valid variant of gradient descent selected, use None instead.")
        CGVar = None
    else:
        print("Gradient descend method: ", CGVar)
    
    full_cost_grad = np.zeros(( N, 2, T ))   
    
    startstep_exc_ = startStep_
    startstep_inh_ = startStep_
    startstep_ee_r_ = startStep_
    startstep_ei_r_ = startStep_
    startstep_ie_r_ = startStep_
    startstep_ii_r_ = startStep_
    startstep_joint_ = startStep_
    
    grad0_ = np.zeros(( N, n_control_vars, T ))
    grad1_ = grad0_.copy()
    dir0_ = grad0_.copy()
    phi0_ = np.zeros(( N, V, T ))
    
    state_maxDelay = np.zeros( (N, V, int( T + n_maxDelay ) ))
    control_maxDelay = np.zeros( (N, n_control_vars, int( T + n_maxDelay ) ))
    dur_maxDelay = t_sim_ + int( n_maxDelay ) * dt
    
    model.params['duration'] = t_sim_
    
    minCost = []
    tc_exc = -1
    tc_inh = -1
    joint_cost = -1
    tc_ee_r = -1
    tc_ei_r = -1
    tc_ie_r = -1
    tc_ii_r = -1    
    
    if line_search_ == None:
        line_search_func = fo.step_size
    elif line_search_ == "AG":
        line_search_func = fo.AG_line_search
    else:
        print("No valid line search selected use bisection instead")
        line_search_func = fo.step_size
        
    while( i < max_iteration_ ):
        
        for ind_time in range(T):
            f_p_grad_t_ = cost.cost_precision_gradient_t2(N, V_target, T, ind_time, state0_[:,:2,ind_time],
                                                         target_state_[:,:,ind_time], ip_)
            #f_p_grad_t_ = cost.cost_precision_gradient_t(N, V_target, state0_[:,:2,ind_time], target_state_[:,:,ind_time], ip_)
            for v in prec_variables:
                full_cost_grad[0,v,ind_time] = f_p_grad_t_[0,v] 
         
        for ind_time in range(T):
            rd_exc[0,0, ind_time] = state0_[0,0,ind_time] * 1e-3        
            rd_inh[0, ind_time] = state0_[0,1,ind_time] * 1e-3
                                                
        phi0_ = phi(N, V, T, dt, state0_, target_state_, best_control_, full_cost_grad, state_maxDelay, n_maxDelay, state_pre_, 
                    ext_exc_current,
                    ext_inh_current,
                    ext_ee_rate,
                    ext_ei_rate,
                    ext_ie_rate,
                    ext_ii_rate,
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
                    factor_eic1,
                    factor_eic2,
                    factor_iec1,
                    factor_iec2,
                    factor_iic1,
                    factor_iic2,
                    rd_exc,
                    rd_inh,
                    sigmarange, ds, Irange, dI, 
                    precalc_r, precalc_tau_mu, precalc_V,
                    interpolate_rate,
                    interpolate_V,
                    interpolate_tau,
                    )
                
        if ( total_cost_[i] < tolerance_ ):
            print("Cost negligibly small.")
            max_iteration_ = i
            break
        
        if False:
            from_ = 900
            to_ = -1
            print("adj exc rate =", phi0_[0,0,from_:to_])
            print("adj inh rate =", phi0_[0,1,from_:to_])
            print("adj mue =", phi0_[0,2,from_:to_])
            print("adj mui =", phi0_[0,3,from_:to_])
            print("adj sigma e =", phi0_[0,15,from_:to_])
            print("adj sigma i =", phi0_[0,16,from_:to_])
            print("adj s ie =", phi0_[0,7,from_:to_])
            
        i += 1   
        
        grad0_ = grad1_.copy()
                
                
        phi1_ = phi1(N, V, T, n_control_vars, phi0_, state1_, best_control_, state_pre_,
                          sigmae_ext,
                          sigmai_ext,
                          ext_ee_rate,
                          ext_ei_rate,
                          ext_ie_rate,
                          ext_ii_rate,
                          tau_se,
                          tau_si,
                          tau_se_sq,
                          tau_si_sq,
                          Jee_sq,
                          Jei_sq,
                          Jie_sq,
                          Jii_sq,
                          taum,
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
                          factor_eic1,
                          factor_eic2,
                          factor_iec1,
                          factor_iec2,
                          factor_iic1,
                          factor_iic2,
                          rd_exc,
                          rd_inh,
                          ndt_de,
                          ndt_di,
                     )
        
        #print(phi1_[0,3,:])
                                        
        grad1_ = fo.compute_gradient(N, n_control_vars, T, dt, best_control_, grad1_, phi1_, control_variables, ie_, is_)
        dir0_ = fo.set_direction(N, T, n_control_vars, grad0_, grad1_, dir0_, i, CGVar, tolerance_)
                
        #print(grad1_[0,3,:])
        
        minCost = []
            
        if separate_comp:
            # compute stepsize separately and then put together
            if 0 in control_variables and len(control_variables) > 1:
                d_exc = dir0_.copy()
                d_exc[:,1:,:] = 0.
                
                s_exc, tc_exc, startstep_exc_ = line_search_func(model, N, n_control_vars, T, dt, state1_[:,:2,:], target_state_,
                             best_control_, d_exc, ip_, ie_, is_, start_step_ = startstep_exc_, max_it_ = 500,
                             max_control_ = cntrl_max_,
                             min_control_ = cntrl_min_, variables_ = prec_variables, grad_ = grad1_)
                minCost.append(tc_exc)
            
                #print("1.1")
            
                #print("step size exc = ", s_exc)
            
            if 1 in control_variables and len(control_variables) > 1:
                d_inh = dir0_.copy()
                d_inh[:,0,:] = 0.
                d_inh[:,2:,:] = 0.
                
                s_inh, tc_inh, startstep_inh_ = line_search_func(model, N, n_control_vars, T, dt, state1_[:,:2,:], target_state_,
                             best_control_, d_inh, ip_, ie_, is_, start_step_ = startstep_inh_, max_it_ = 500,
                             max_control_ = cntrl_max_,
                             min_control_ = cntrl_min_, variables_ = prec_variables, grad_ = grad1_)
                minCost.append(tc_inh)
                
                #print("1.1")
                
                #print("step size inh = ", s_inh)
                
            if 2 in control_variables and len(control_variables) > 1:
                d_ee_r = dir0_.copy()
                d_ee_r[:,0,:] = 0.
                d_ee_r[:,1,:] = 0.
                d_ee_r[:,3:,:] = 0.
                
                s_ee_r, tc_ee_r, startstep_ee_r_ = line_search_func(model, N, n_control_vars, T, dt, state1_[:,:2,:],
                            target_state_, best_control_, d_ee_r, ip_, ie_, is_, start_step_ = startstep_ee_r_, max_it_ = 500,
                            max_control_ = cntrl_max_, min_control_ = cntrl_min_, variables_ = prec_variables, grad_ = grad1_)
                minCost.append(tc_ee_r)
                
                #print("1.1")
                
                #print("step size ee_r = ", s_ee_r)
                
            if 3 in control_variables and len(control_variables) > 1:
                d_ei_r = dir0_.copy()
                d_ei_r[:,0,:] = 0.
                d_ei_r[:,1,:] = 0.
                d_ei_r[:,2,:] = 0.
                d_ei_r[:,4:,:] = 0.
                
                s_ei_r, tc_ei_r, startstep_ei_r_ = line_search_func(model, N, n_control_vars, T, dt, state1_[:,:2,:],
                            target_state_, best_control_, d_ei_r, ip_, ie_, is_, start_step_ = startstep_ei_r_, max_it_ = 500,
                            max_control_ = cntrl_max_, min_control_ = cntrl_min_, variables_ = prec_variables, grad_ = grad1_)
                minCost.append(tc_ei_r)
                
                #print("1.1")
                
                #print("step size ei_r = ", s_ei_r)
                
            if 4 in control_variables and len(control_variables) > 1:
                d_ie_r = dir0_.copy()
                d_ie_r[:,0,:] = 0.
                d_ie_r[:,1,:] = 0.
                d_ie_r[:,2,:] = 0.
                d_ie_r[:,3,:] = 0.
                d_ie_r[:,5,:] = 0.
                
                s_ie_r, tc_ie_r, startstep_ie_r_ = line_search_func(model, N, n_control_vars, T, dt, state1_[:,:2,:],
                            target_state_, best_control_, d_ie_r, ip_, ie_, is_, start_step_ = startstep_ie_r_, max_it_ = 500,
                            max_control_ = cntrl_max_, min_control_ = cntrl_min_, variables_ = prec_variables, grad_ = grad1_)
                minCost.append(tc_ie_r)
                
                #print("1.1")
                
                #print("step size ie_r = ", s_ie_r)
                
            if 5 in control_variables and len(control_variables) > 1:
                d_ii_r = dir0_.copy()
                d_ii_r[:,0,:] = 0.
                d_ii_r[:,1,:] = 0.
                d_ii_r[:,2,:] = 0.
                d_ii_r[:,3,:] = 0.
                d_ii_r[:,4,:] = 0.
                
                s_ii_r, tc_ii_r, startstep_ii_r_ = line_search_func(model, N, n_control_vars, T, dt, state1_[:,:2,:],
                            target_state_, best_control_, d_ii_r, ip_, ie_, is_, start_step_ = startstep_ii_r_, max_it_ = 500,
                            max_control_ = cntrl_max_, min_control_ = cntrl_min_, variables_ = prec_variables, grad_ = grad1_)
                minCost.append(tc_ii_r)
                
                #print("1.1")
                
                #print("step size ii_r = ", s_ii_r)
            
            if 0 in control_variables and 1 in control_variables:
                joint_dir = dir0_.copy()
                joint_dir[:,0,:] = s_exc * dir0_[:,0,:] #/ (s_exc + s_inh)
                joint_dir[:,1,:] = s_inh * dir0_[:,1,:] #/ (s_exc + s_inh)
                
                joint_step_, joint_cost, startstep_joint_ = line_search_func(model, N, n_control_vars, T, dt, state1_[:,:2,:],
                             target_state_, best_control_, joint_dir, ip_, ie_, is_, start_step_ = startstep_joint_, max_it_ = 500,
                             max_control_ = cntrl_max_, min_control_ = cntrl_min_, variables_ = prec_variables, grad_ = grad1_)
                minCost.append(joint_cost)
         
                        
        step_, total_cost_[i], startStep_ = line_search_func(model, N, n_control_vars, T, dt, state1_[:,:2,:], target_state_,
                     best_control_, dir0_, ip_, ie_, is_, start_step_ = startStep_, max_it_ = 500, max_control_ = cntrl_max_,
                         min_control_ = cntrl_min_, variables_ = prec_variables, grad_ = grad1_)
                
        #print("step size = ", step_, total_cost_[i])
        
        minCost.append(total_cost_[i])
        
        #print("step size = ", step_, total_cost_[i])
                
        if separate_comp:
            costMin = np.amin( minCost )
              
            if (tc_exc ==  costMin):
                #print("choose exc only")
                step_ = s_exc
                total_cost_[i] = tc_exc
                dir0_ = d_exc.copy()
                startStep_ = startstep_exc_
                
            elif (tc_inh ==  costMin):
                #print("choose inh only")
                step_ = s_inh
                total_cost_[i] = tc_inh
                dir0_ = d_inh.copy()
                startStep_ = startstep_inh_
            
            elif (joint_cost ==  costMin):
                #print("choose exc, inh combination")
                step_ = joint_step_
                total_cost_[i] = joint_cost
                dir0_ = joint_dir.copy()
                startStep_ = startstep_joint_ 
            
            elif (tc_ee_r ==  costMin):
                #print("choose exc rate only")
                step_ = s_ee_r
                total_cost_[i] = tc_ee_r
                dir0_ = d_ee_r.copy()
                startStep_ = startstep_ee_r_
                
            elif (tc_ei_r ==  costMin):
                #print("choose exc rate only")
                step_ = s_ei_r
                total_cost_[i] = tc_ei_r
                dir0_ = d_ei_r.copy()
                startStep_ = startstep_ei_r_
                
            elif (tc_ie_r ==  costMin):
                #print("choose exc rate only")
                step_ = s_ie_r
                total_cost_[i] = tc_ie_r
                dir0_ = d_ie_r.copy()
                startStep_ = startstep_ie_r_
                
            elif (tc_ii_r ==  costMin):
                #print("choose exc rate only")
                step_ = s_ii_r
                total_cost_[i] = tc_ii_r
                dir0_ = d_ii_r.copy()
                startStep_ = startstep_ii_r_
                
            #else:
                #print("choose adjoint")
                #startStep_ = startstep_adj_
            
        #print("found step ", step_)
        #print("continue with start steps ", startstep_exc_, startstep_inh_, startstep_joint_, startStep_)        
        
        runtime_[i] = timer() - runtime_start_
        
        if (
            i <= 20
            or ( i <= 200 and i%10 == 0 ) 
            or ( i <= 2000 and i%100 == 0 ) 
            or ( i%1000 == 0 )
            ):
            print("RUN ", i, ", total integrated cost = ", total_cost_[i])
            
        best_control_ = u_opt0_ + step_ * dir0_
        
        # why is this needed?  
        #best_control_ = fo.scalemaxcontrol(best_control_, cntrl_max_, cntrl_min_)
        best_control_ = fo.setmaxcontrol(n_control_vars, best_control_, cntrl_max_, cntrl_min_)
        ########
                
        
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
        
        #print("update state")
        #print(state1_[0,0,:])
                
        s_diff_ = ( np.absolute(state1_ - state0_) < tolerance_)
        if ( s_diff_.all() ):
            print("State only changes marginally.")
        #    max_iteration_ = i
        #    break
        
        state0_ = state1_.copy() 
        
        control_maxDelay[:,:,:T] = best_control_
        model.params['duration'] = dur_maxDelay
        state_maxDelay = fo.updateFullState(model, control_maxDelay, state_vars)
        model.params['duration'] = t_sim_
                
    state1_ = fo.updateFullState(model, best_control_, state_vars)
    
    for j in [5,6,7,8]:
        if np.amin(state1_[0,j,:]) < 0. or np.amax(state1_[0,j,:]) > 1.:
            print("WARNING: s-parameter not in proper range")
    
    improvement = 100.
    if total_cost_[0] != 0.:
        improvement = 100. - (100.*(total_cost_[max_iteration_]/total_cost_[0]))
        
    print("RUN ", max_iteration_, ", total integrated cost = ", total_cost_[max_iteration_])
    print("Improved over ", max_iteration_, " iterations in ", runtime_[max_iteration_]," seconds by ", improvement, " percent.")
    
    # compute node-wise cost in precision, energy, sparsity for weight = 1
    cost_node = cost.cost_int_per_node(N, n_control_vars, T, dt, state0_, target_state_,
                                     best_control_, 1., 1., 1., v_ = prec_variables )
    
    """
    if max_iteration_ != 0:
        max_g, min_g = np.amax(g0_min_), np.amin(g0_min_)
        print("max value of final gradient at index = ", np.where(g0_min_ == max_g), max_g )
        print("min value of final gradient at index = ", np.where(g0_min_ == min_g), min_g )
    """
        
    if (t_sim_pre_ < dt and t_sim_post_ < dt):
        return best_control_, state1_, total_cost_, runtime_, grad1_, phi0_, cost_node
    
    t_post_ndt = np.around(t_sim_post_ / dt).astype(int)
    
    state_post_ = 0.
    
    if (t_sim_post_ >= dt):
        
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
            
    return bc_, bs_, total_cost_, runtime_, grad1_, phi0_, cost_node

@numba.njit
def phi(N, V, T, dt, state_, target_state_, control_, full_cost_grad, state_maxDelay, n_maxDelay, state_pre_,
                    ext_exc_current,
                    ext_inh_current,
                    ext_ee_rate,
                    ext_ei_rate,
                    ext_ie_rate,
                    ext_ii_rate,
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
                    factor_eic1,
                    factor_eic2,
                    factor_iec1,
                    factor_iec2,
                    factor_iic1,
                    factor_iic2,
                    rd_exc,
                    rd_inh,
                    sigmarange, ds, Irange, dI, 
                    precalc_r, precalc_tau_mu, precalc_V,
                    interpolate_rate,
                    interpolate_V,
                    interpolate_tau,
                    ):
    
    phi_ = np.zeros(( N, V, T ))
    
    jac = jacobian(V, state_[:,:,:], control_[:,:,:], T, state_pre_,
                       ext_exc_current,
                       ext_inh_current,
                       ext_ee_rate,
                       ext_ei_rate,
                       ext_ie_rate,
                       ext_ii_rate,
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
                       factor_eic1,
                       factor_eic2,
                       factor_iec1,
                       factor_iec2,
                       factor_iic1,
                       factor_iic2,
                       rd_exc,
                       rd_inh,
                       sigmarange, ds, Irange, dI,
                       C,
                       precalc_r, precalc_tau_mu, precalc_V,
                       ndt_de,
                       ndt_di,
                       interpolate_rate,
                       interpolate_V,
                       interpolate_tau,
                       ) 
    
    #print("n_dt de = ", ndt_de)
    #print("jac 7 = ", jac[7,0,900:])
    #print("jac 2 = ", jac[0,2,400:])
            
    for ind_time in range(T-1, -1, -1):
        
        if ind_time + ndt_de >= T-1:
            phi_[0,0,ind_time] = - ( full_cost_grad[0,0,ind_time]
                                    + np.dot( np.array( [ phi_[0,4,ind_time] ] ),
                                             np.array( [ jac[4,0,ind_time] ] ) ) )
        elif ind_time + ndt_de + 1 >= T-1:
            phi_[0,0,ind_time] = - ( full_cost_grad[0,0,ind_time]
                                    + np.dot( np.array( [ phi_[0,4,ind_time],
                                                         phi_[0,15,ind_time+ndt_de],
                                                         phi_[0,16,ind_time+ndt_de] ] ),
                                             np.array( [ jac[4,0,ind_time], 
                                                        jac[15,0,ind_time+ndt_de],
                                                        jac[16,0,ind_time+ndt_de] ] ) ) )
        else:
            phi_[0,0,ind_time] = - ( full_cost_grad[0,0,ind_time]
                                    + np.dot( np.array( [ phi_[0,4,ind_time],
                                                         phi_[0,5,ind_time+ndt_de+1],
                                                         phi_[0,7,ind_time+ndt_de+1],
                                                         phi_[0,9,ind_time+ndt_de+1],
                                                         phi_[0,11,ind_time+ndt_de+1],
                                                         phi_[0,15,ind_time+ndt_de],
                                                         phi_[0,16,ind_time+ndt_de] ] ),
                                             np.array( [ jac[4,0,ind_time], 
                                                        jac[5,0,ind_time+ndt_de],  # should not be plus 1
                                                        jac[7,0,ind_time+ndt_de],
                                                        jac[9,0,ind_time+ndt_de],
                                                        jac[11,0,ind_time+ndt_de],
                                                        jac[15,0,ind_time+ndt_de],
                                                        jac[16,0,ind_time+ndt_de] ] ) ) )
            
        if ind_time + ndt_di >= T-1:
            phi_[0,1,ind_time] = - ( full_cost_grad[0,1,ind_time] )
        elif ind_time + ndt_di + 1 >= T-1:
            phi_[0,1,ind_time] = - ( full_cost_grad[0,1,ind_time]
                                    + np.dot( np.array( [phi_[0,15,ind_time+ndt_di],
                                                         phi_[0,16,ind_time+ndt_di] ] ),
                                             np.array( [ jac[15,1,ind_time+ndt_di],
                                                        jac[16,1,ind_time+ndt_di] ] ) ) )
        else:
            phi_[0,1,ind_time] = - ( full_cost_grad[0,1,ind_time]
                                    + np.dot( np.array( [ phi_[0,6,ind_time+ndt_di+1],
                                                         phi_[0,8,ind_time+ndt_di+1],
                                                         phi_[0,10,ind_time+ndt_di+1],
                                                         phi_[0,12,ind_time+ndt_di+1],
                                                         phi_[0,15,ind_time+ndt_di],
                                                         phi_[0,16,ind_time+ndt_di] ] ),
                                             np.array( [ jac[6,1,ind_time+ndt_di],
                                                        jac[8,1,ind_time+ndt_di],
                                                        jac[10,1,ind_time+ndt_di],
                                                        jac[12,1,ind_time+ndt_di],
                                                        jac[15,1,ind_time+ndt_di],
                                                        jac[16,1,ind_time+ndt_di] ] ) ) )         
            
        
        
        #if ind_time == T-1:
        #    phi_[0,9,-1] = + dt * jac[15,9,-1] * phi_[0,0,-1] * jac[0,15,-1]
        #    print("last entry in phi 9 = ", phi_[0,9,-1])
        if ind_time == 0:
            break
        
        res = - phi_[0,2,ind_time] * jac[2,18, ind_time-1]
        phi_[0,18,ind_time-1] = res
        
        res = - phi_[0,3,ind_time] * jac[3,19, ind_time-1]
        phi_[0,19,ind_time-1] = res
            
        der = 0.
        der += phi_[0,0,ind_time] * jac[0,2,ind_time-1]
        
        # if not static state before simulation, should go back further in time
        
        # 2,2,-1 checked by trial and error
        der += phi_[0,2,ind_time] * jac[2,2,ind_time-1]
        der += phi_[0,17,ind_time] * jac[17,2,ind_time-1]
        der += phi_[0,18,ind_time-1] * jac[18,2,ind_time-1]
        phi_[0,2,ind_time-1] = phi_[0,2,ind_time] - dt * der

        der = 0.
        der += phi_[0,1,ind_time] * jac[1,3,ind_time-1]
        der += ( phi_[0,3,ind_time] * jac[3,3,ind_time-1] )
        der += phi_[0,19,ind_time-1] * jac[19,3,ind_time-1]
        phi_[0,3,ind_time-1] = phi_[0,3,ind_time] - dt * der
        
        der = ( phi_[0,0,ind_time] * jac[0,4,ind_time-1]
               + phi_[0,4,ind_time] * jac[4,4,ind_time-1]
               + phi_[0,17,ind_time] * jac[17,4,ind_time-1]
               + phi_[0,18,ind_time-1] * jac[18,4,ind_time-1] )
        phi_[0,4,ind_time-1] = phi_[0,4,ind_time] - dt * der
        
        der = ( phi_[0,2,ind_time] * jac[2,5,ind_time-1]
               + phi_[0,5,ind_time] * jac[5,5,ind_time-1]
               + phi_[0,9,ind_time] * jac[9,5,ind_time-1]
               )
        phi_[0,5,ind_time-1] = phi_[0,5,ind_time] - dt * der
        
        der = ( phi_[0,2,ind_time] * jac[2,6,ind_time-1]
               + phi_[0,6,ind_time] * jac[6,6,ind_time-1]
               + phi_[0,10,ind_time] * jac[10,6,ind_time-1]
               )
        phi_[0,6,ind_time-1] = phi_[0,6,ind_time] - dt * der
        
        der = ( phi_[0,3,ind_time] * jac[3,7,ind_time-1]
               + phi_[0,7,ind_time] * jac[7,7,ind_time-1]
               + phi_[0,11,ind_time] * jac[11,7,ind_time-1]
               )
        phi_[0,7,ind_time-1] = phi_[0,7,ind_time] - dt * der
        
        der = ( phi_[0,3,ind_time] * jac[3,8,ind_time-1]
               + phi_[0,8,ind_time] * jac[8,8,ind_time-1]
               + phi_[0,12,ind_time] * jac[12,8,ind_time-1]
               )
        phi_[0,8,ind_time-1] = phi_[0,8,ind_time] - dt * der
        
        
        #der = ( phi_[0,16,ind_time] * jac[16,11,ind_time] )
        #phi_[0,11,ind_time-1] = phi_[0,11,ind_time] - dt * der
        
        res = ( - phi_[0,0,ind_time] * jac[0,15, ind_time-1]
               - phi_[0,17,ind_time] * jac[17,15, ind_time-1]
               - phi_[0,18,ind_time-1] * jac[18,15, ind_time-1] )
        phi_[0,15,ind_time-1] = res
        
        res = - phi_[0,1,ind_time] * jac[1,16, ind_time-1] - phi_[0,19,ind_time-1] * jac[19,16, ind_time-1]
        phi_[0,16,ind_time-1] = res
        
        res = - phi_[0,4,ind_time-1] * jac[4,17, ind_time-1]
        phi_[0,17,ind_time-1] = res
        
        der = ( phi_[0,9,ind_time] * jac[9,9,ind_time-1] + phi_[0,15,ind_time-1] * jac[15,9,ind_time-1] )
        phi_[0,9,ind_time-1] = phi_[0,9,ind_time] - dt * der
        
        der = ( phi_[0,10,ind_time] * jac[10,10,ind_time-1] + phi_[0,15,ind_time-1] * jac[15,10,ind_time-1] )
        phi_[0,10,ind_time-1] = phi_[0,10,ind_time] - dt * der
        
        der = ( phi_[0,11,ind_time] * jac[11,11,ind_time-1] + phi_[0,16,ind_time-1] * jac[16,11,ind_time-1] )
        phi_[0,11,ind_time-1] = phi_[0,11,ind_time] - dt * der
        
        der = ( phi_[0,12,ind_time] * jac[12,12,ind_time-1] + phi_[0,16,ind_time-1] * jac[16,12,ind_time-1] )
        phi_[0,12,ind_time-1] = phi_[0,12,ind_time] - dt * der
    
    
                
    return phi_

@numba.njit
def phi1(N, V, T, n_control_vars, phi_, state_, control_, state_pre_,
                          sigmae_ext,
                          sigmai_ext,
                          ext_ee_rate,
                          ext_ei_rate,
                          ext_ie_rate,
                          ext_ii_rate,
                          tau_se,
                          tau_si,
                          tau_se_sq,
                          tau_si_sq,
                          Jee_sq,
                          Jei_sq,
                          Jie_sq,
                          Jii_sq,
                          taum,
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
                          factor_eic1,
                          factor_eic2,
                          factor_iec1,
                          factor_iec2,
                          factor_iic1,
                          factor_iic2,
                          rd_exc,
                          rd_inh,
                          ndt_de,
                          ndt_di,
                         ):  
    
    phi1_ = np.zeros(( N, n_control_vars, T ))
            
        # could leave shift at zero in jacobian and algorithm would do almost equally well
            
    for ind_t in range(0, T):
        
        jac_u_ = D_u_h(V, n_control_vars, state_, control_, ind_t, state_pre_,
                      sigmae_ext,
                      sigmai_ext,
                      ext_ee_rate,
                      ext_ei_rate,
                      ext_ie_rate,
                      ext_ii_rate,
                      tau_se,
                      tau_si,
                      tau_se_sq,
                      tau_si_sq,
                      Jee_sq,
                      Jei_sq,
                      Jie_sq,
                      Jii_sq,
                      taum,
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
                      factor_eic1,
                      factor_eic2,
                      factor_iec1,
                      factor_iec2,
                      factor_iic1,
                      factor_iic2,
                      rd_exc,
                      rd_inh,
                      ndt_de,
                      ndt_di,
                    )
                
        phi = np.ascontiguousarray(phi_[0,:,ind_t])
        #phi_shift = np.ascontiguousarray(phi_[0,:,ind_t-1])#, dtype=np.float64)
        #phi_shift = np.ascontiguousarray(phi_[0,:,ind_t])
        phi_shift = phi.copy()
        

        if ind_t == T-1:
            for i in [5,6,7,8,9,10,11,12]:
                phi_shift[i] = 0.
        else:
            for i in [5,6,7,8,9,10,11,12]:
                phi_shift[i] = phi_[0,i,ind_t+1]
           
        #print("t, phi 9, 15 = ", ind_t, phi_shift[9], phi_shift[15])
             
        y0 = np.ascontiguousarray(jac_u_[0,:])
        y1 = np.ascontiguousarray(jac_u_[1,:])
        y2 = np.ascontiguousarray(jac_u_[2,:])
        y3 = np.ascontiguousarray(jac_u_[3,:])
        y4 = np.ascontiguousarray(jac_u_[4,:])
        y5 = np.ascontiguousarray(jac_u_[5,:])
        
        phi1_[0,2,ind_t] = np.dot(phi_shift, y2)
        phi1_[0,3,ind_t] = np.dot(phi_shift, y3)
        phi1_[0,4,ind_t] = np.dot(phi_shift, y4)
        phi1_[0,5,ind_t] = np.dot(phi_shift, y5)
        
        #print("time, phi 6,10 15, y3 = ", ind_t, phi_shift[6], phi_shift[10], phi_shift[15], y3)

        if ind_t > 0:
            phi1_[0,0,ind_t] = np.dot(phi_shift, y0)
            phi1_[0,1,ind_t] = np.dot(phi_shift, y1)
        
        
        #print(phi1_[0,0,ind_t] == res1, phi1_[0,1,ind_t] == res2)

    return phi1_

@numba.njit
def D_xdot(V, state_t_):
    dxdot_ = np.zeros(( V, V ))
    return dxdot_

@numba.njit
def D_u_h(V, n_control_vars, state_, control_, t_, state_pre_,
          sigmae_ext,
          sigmai_ext,
          ext_ee_rate,
          ext_ei_rate,
          ext_ie_rate,
          ext_ii_rate,
          tau_se,
          tau_si,
          tau_se_sq,
          tau_si_sq,
          Jee_sq,
          Jei_sq,
          Jie_sq,
          Jii_sq,
          taum,
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
          factor_eic1,
          factor_eic2,
          factor_iec1,
          factor_iec2,
          factor_iic1,
          factor_iic2,
          rd_exc,
          rd_inh,
          shift_e,
          shift_i,
          ):
    
    if t_-shift_e >= 0:
        z1ee = factor_ee1 * rd_exc[0,0,t_-shift_e] + factor_eec1 * ( control_[0,2,t_] + ext_ee_rate )
        z1ie = factor_ie1 * rd_exc[0,0,t_-shift_e] + factor_iec1 * ( control_[0,4,t_] + ext_ie_rate )
    else:
        z1ee = factor_ee1 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_eec1 * ( control_[0,2,t_] + ext_ee_rate )
        z1ie = factor_ie1 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_iec1 * ( control_[0,4,t_] + ext_ie_rate )
        
    if t_-shift_i >= 0:
        z1ei = factor_ei1 * rd_inh[0,t_-shift_i] + factor_eic1 * ( control_[0,3,t_] + ext_ei_rate )
        z1ii = factor_ii1 * rd_inh[0,t_-shift_i] + factor_iic1 * ( control_[0,5,t_] + ext_ii_rate )
    else:
        z1ei = factor_ei1 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_eic1 * ( control_[0,3,t_] + ext_ei_rate )
        z1ii = factor_ii1 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_iic1 * ( control_[0,5,t_] + ext_ii_rate )

    #z1ee = max(z1ee,0.)
    #z2ee = max(z2ee,0.)
    
    duh_ = np.zeros(( n_control_vars, V ))
    
    # -1 checked by trial and error
    duh_[0,2] = - 1. / state_[0,18,t_-1]
    duh_[1,3] = - 1. / state_[0,19,t_-1]
    
    # ee, ei, ie, ii
    duh_[2,5] = - factor_eec1 * (1. - state_[0,5,t_]) / tau_se
    duh_[2,9] = ( - ( 1. - state_[0,5,t_] )**2 * factor_eec2
                 - ( factor_eec2 - 2. * tau_se * factor_eec1 ) * state_[0,9,t_] ) / tau_se_sq 
    
    duh_[3,6] = - factor_eic1 * (1. - state_[0,6,t_]) / tau_si
    duh_[3,10] = ( - ( 1. - state_[0,6,t_] )**2 * factor_eic2
                 - ( factor_eic2 - 2. * tau_si * factor_eic1 ) * state_[0,10,t_] ) / tau_si_sq 
    
    duh_[4,7] = - factor_iec1 * (1. - state_[0,7,t_]) / tau_se
    duh_[4,11] = ( - ( 1. - state_[0,7,t_] )**2 * factor_iec2
                 - ( factor_iec2 - 2. * tau_se * factor_iec1 ) * state_[0,11,t_] ) / tau_se_sq
    
    duh_[5,8] = - factor_iic1 * (1. - state_[0,8,t_]) / tau_si
    duh_[5,12] = ( - ( 1. - state_[0,8,t_] )**2 * factor_iic2
                 - ( factor_iic2 - 2. * tau_si * factor_iic1 ) * state_[0,12,t_] ) / tau_si_sq
    
    sigma_ee = 2. * Jee_sq * state_[0,9,t_] * tau_se * taum / ((1. + z1ee) * taum + tau_se)
    sigma_ei = 2. * Jei_sq * state_[0,10,t_] * tau_si * taum * ( (1. + z1ei ) * taum + tau_si )**(-1)
        
    arg = ( sigma_ee + sigma_ei + sigmae_ext**2 )
    if arg > 0:
        sigma_sqrt_e = arg**(-1./2.)
    else:
        sigma_sqrt_e = 0.
        #print("WARNING: sigma e smaller zero")
    
    duh_[2,15] = ( 0.5 * ( (1. + z1ee) * taum + tau_se )**(-2.) * factor_eec1 * taum *
                  ( 2. * Jee_sq * tau_se * taum * state_[0,9,t_] ) * sigma_sqrt_e )
    
    duh_[3,15] = ( 0.5 * ( (1. + z1ei) * taum + tau_si )**(-2.) * factor_eic1 * taum *
                  ( 2. * Jei_sq * tau_si * taum * state_[0,10,t_] ) * sigma_sqrt_e )
    
    sigma_ie = 2. * Jie_sq * state_[0,11,t_] * tau_se * taum / ((1. + z1ie) * taum + tau_se)
    sigma_ii = 2. * Jii_sq * state_[0,12,t_] * tau_si * taum * ( (1. + z1ii ) * taum + tau_si )**(-1)
        
    arg = ( sigma_ie + sigma_ii + sigmai_ext**2 )
    if arg > 0:
        sigma_sqrt_i = arg**(-1./2.)
    else:
        sigma_sqrt_i = 0.
        #print("WARNING: sigma e smaller zero")
    
    duh_[4,16] = ( 0.5 * ( (1. + z1ie) * taum + tau_se )**(-2.) * factor_iec1 * taum *
                  ( 2. * Jie_sq * tau_se * taum * state_[0,11,t_] ) * sigma_sqrt_i )
    
    duh_[5,16] = ( 0.5 * ( (1. + z1ii) * taum + tau_si )**(-2.) * factor_iic1 * taum *
                  ( 2. * Jii_sq * tau_si * taum * state_[0,12,t_] ) * sigma_sqrt_i )
    
    return duh_

@numba.njit
def jacobian(V, state_, control_, T, state_pre_,
              ext_exc_current,
              ext_inh_current,
              ext_ee_rate,
              ext_ei_rate,
              ext_ie_rate,
              ext_ii_rate,
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
              factor_eic1,
              factor_eic2,
              factor_iec1,
              factor_iec2,
              factor_iic1,
              factor_iic2,
              rd_exc,
              rd_inh,
              sigmarange, ds, Irange, dI,
              C,
              precalc_r, precalc_tau_mu, precalc_V,
              shift_e,
              shift_i,  
              interpolate_rate,
              interpolate_V,
              interpolate_tau,
              ):
    
    jacobian_ = np.zeros(( V, V, T))
    
    for t_ in range(0,T):
           
    # ee, ei, ie, ii
        if t_-shift_e >= 0:
            z1ee = factor_ee1 * rd_exc[0,0,t_-shift_e] + factor_eec1 * ( control_[0,2,t_] + ext_ee_rate )
            z2ee = factor_ee2 * rd_exc[0,0,t_-shift_e] + factor_eec2 * ( control_[0,2,t_] + ext_ee_rate ) 
            z1ie = factor_ie1 * rd_exc[0,0,t_-shift_e] + factor_iec1 * ( control_[0,4,t_] + ext_ie_rate )
            z2ie = factor_ie2 * rd_exc[0,0,t_-shift_e] + factor_iec2 * ( control_[0,4,t_] + ext_ie_rate )
        else:
            z1ee = factor_ee1 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_eec1 * ( control_[0,2,t_] + ext_ee_rate )
            z2ee = factor_ee2 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_eec2 * ( control_[0,2,t_] + ext_ee_rate )
            z1ie = factor_ie1 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_iec1 * ( control_[0,4,t_] + ext_ie_rate )
            z2ie = factor_ie2 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_iec2 * ( control_[0,4,t_] + ext_ie_rate )
            
        if t_-shift_i >= 0:
            z1ei = factor_ei1 * rd_inh[0,t_-shift_i] + factor_eic1 * ( control_[0,3,t_] + ext_ei_rate )
            z2ei = factor_ei2 * rd_inh[0,t_-shift_i] + factor_eic2 * ( control_[0,3,t_] + ext_ei_rate )       
            z1ii = factor_ii1 * rd_inh[0,t_-shift_i] + factor_iic1 * ( control_[0,5,t_] + ext_ii_rate )
            z2ii = factor_ii2 * rd_inh[0,t_-shift_i] + factor_iic2 * ( control_[0,5,t_] + ext_ii_rate )
        else:
            z1ei = factor_ei1 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_eic1 * ( control_[0,3,t_] + ext_ei_rate )
            z2ei = factor_ei2 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_eic2 * ( control_[0,3,t_] + ext_ei_rate )         
            z1ii = factor_ii1 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_iic1 * ( control_[0,5,t_] + ext_ii_rate )
            z2ii = factor_ii2 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_iic2 * ( control_[0,5,t_] + ext_ii_rate )
         
        """
        z1ee = max(z1ee,0.)
        z2ee = max(z2ee,0.)
        z1ei = max(z1ei,0.)
        z2ei = max(z2ei,0.)
        z1ie = max(z1ie,0.)
        z2ie = max(z2ie,0.)
        z1ii = max(z1ii,0.)
        z2ii = max(z2ii,0.)
        """
        
        jacobian_[0,2,t_] = - d_r_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_r, interpolate_rate) * 1e3
        jacobian_[0,4,t_] = d_r_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_r, interpolate_rate) * 1e3 / C
        jacobian_[0,15,t_] = - d_r_func_sigma(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                              state_[0,15,t_], Irange, dI, C, precalc_r, interpolate_rate) * 1e3
        
        jacobian_[1,3,t_] = - d_r_func_mu(state_[0,3,t_], sigmarange, ds,
                                          state_[0,16,t_], Irange, dI, C, precalc_r, interpolate_rate) * 1e3
        jacobian_[1,16,t_] = - d_r_func_sigma(state_[0,3,t_], sigmarange, ds,
                                              state_[0,16,t_], Irange, dI, C, precalc_r, interpolate_rate) * 1e3
        
        jacobian_[2,2,t_] = 1. / state_[0,18,t_]
        jacobian_[2,5,t_] = - Jee_max / state_[0,18,t_]
        jacobian_[2,6,t_] = - Jei_max / state_[0,18,t_]
        if t_ < T - 1:
            jacobian_[2,18,t_] = ( Jee_max * state_[0,5,t_] + Jei_max * state_[0,6,t_] + control_[0,0,t_+1] + ext_exc_current
                           + state_[0,13,t_] - state_[0,2,t_] ) / state_[0,18,t_]**2
        else:
            jacobian_[2,18,t_] = ( Jee_max * state_[0,5,t_] + Jei_max * state_[0,6,t_] + ext_exc_current
                           + state_[0,13,t_] - state_[0,2,t_] ) / state_[0,18,t_]**2
        
        jacobian_[3,3,t_] = 1. / state_[0,19,t_]
        jacobian_[3,7,t_] = - Jie_max / state_[0,19,t_]
        jacobian_[3,8,t_] = - Jii_max / state_[0,19,t_]
        if t_ < T - 1:
            jacobian_[3,19,t_] = ( Jie_max * state_[0,7,t_] + Jii_max * state_[0,8,t_] + control_[0,1,t_+1] + ext_inh_current
                           + state_[0,14,t_] - state_[0,3,t_] ) / state_[0,19,t_]**2
        else:
            jacobian_[3,19,t_] = ( Jie_max * state_[0,7,t_] + Jii_max * state_[0,8,t_] + ext_inh_current
                           + state_[0,14,t_] - state_[0,3,t_] ) / state_[0,19,t_]**2
        
        jacobian_[4,0,t_] = - b * 1e-3
        jacobian_[4,4,t_] = 1. / tauA
        jacobian_[4,17,t_] = - a / tauA
                
        jacobian_[5,0,t_] = - factor_ee1 * 1e-3 * ( 1. - state_[0,5,t_] ) / tau_se
        jacobian_[5,5,t_] = ( 1. + z1ee ) / tau_se
        
        jacobian_[6,1,t_] = - factor_ei1 * 1e-3 * ( 1. - state_[0,6,t_] ) / tau_si
        jacobian_[6,6,t_] = ( 1. + z1ei ) / tau_si
        
        jacobian_[7,0,t_] = - factor_ie1 * 1e-3 * ( 1. - state_[0,7,t_] ) / tau_se
        jacobian_[7,7,t_] = ( 1. + z1ie ) / tau_se
        
        jacobian_[8,1,t_] = - factor_ii1 * 1e-3 * ( 1. - state_[0,8,t_] ) / tau_si
        jacobian_[8,8,t_] = ( 1. + z1ii ) / tau_si
        
        jacobian_[9,0,t_] = ( - ( 1. - state_[0,5,t_] )**2 * factor_ee2 * 1e-3
                             - ( factor_ee2 - 2. * tau_se * factor_ee1 ) * 1e-3 * state_[0,9,t_] ) / tau_se_sq
        jacobian_[9,5,t_] = 2. * ( 1. - state_[0,5,t_] ) * z2ee / tau_se_sq
        jacobian_[9,9,t_] = - ( z2ee - 2. * tau_se * ( z1ee + 1 ) ) / tau_se_sq
        
        jacobian_[10,1,t_] = ( - ( 1. - state_[0,6,t_] )**2 * factor_ei2 * 1e-3
                             - ( factor_ei2 - 2. * tau_si * factor_ei1 ) * 1e-3 * state_[0,10,t_] ) / tau_si_sq
        jacobian_[10,6,t_] = 2. * ( 1. - state_[0,6,t_] ) * z2ei / tau_si_sq
        jacobian_[10,10,t_] = - ( z2ei - 2. * tau_si * ( z1ei + 1 ) ) / tau_si_sq
        
        jacobian_[11,0,t_] = ( - ( 1. - state_[0,7,t_] )**2 * factor_ie2 * 1e-3
                              - ( factor_ie2 - 2. * tau_se * factor_ie1 ) * 1e-3 * state_[0,11,t_] ) / tau_se_sq
        jacobian_[11,7,t_] = 2. * ( 1. - state_[0,7,t_] ) * z2ie / tau_se_sq
        jacobian_[11,11,t_] = - ( z2ie - 2. * tau_se * ( z1ie + 1 ) ) / tau_se_sq
        
        jacobian_[12,1,t_] = ( - ( 1. - state_[0,8,t_] )**2 * factor_ii2 * 1e-3
                             - ( factor_ii2 - 2. * tau_si * factor_ii1 ) * 1e-3 * state_[0,12,t_]) / tau_si_sq
        jacobian_[12,8,t_] = 2. * ( 1. - state_[0,8,t_] ) * z2ii / tau_si_sq
        jacobian_[12,12,t_] = - ( z2ii - 2. * tau_si * ( z1ii + 1 ) ) / tau_si_sq
        
        sigma_ee = 2. * Jee_sq * state_[0,9,t_] * tau_se * taum / ((1. + z1ee) * taum + tau_se)
        sigma_ei = 2. * Jei_sq * state_[0,10,t_] * tau_si * taum * ( (1. + z1ei ) * taum + tau_si )**(-1)
        
        arg = ( sigma_ee + sigma_ei + sigmae_ext**2 )
        if arg > 0:
            sigma_sqrt_e = arg**(-1./2.)
        else:
            sigma_sqrt_e = 0.
            #print("WARNING: sigma e smaller zero")
        
        jacobian_[15,0,t_] = ( 0.5 * ( (1. + z1ee) * taum + tau_se )**(-2.) * factor_ee1 * 1e-3 * taum
                              * ( 2. * Jee_sq * tau_se * taum * state_[0,9,t_] ) * sigma_sqrt_e )
        # should not be state[0,9,t-1] because also computing sigma t-1
        jacobian_[15,1,t_] = ( 0.5 * ( (1. + z1ei) * taum + tau_si )**(-2.) * factor_ei1 * 1e-3 * taum
                              * ( 2. * Jei_sq * tau_si * taum * state_[0,10,t_] ) * sigma_sqrt_e )
        jacobian_[15,9,t_] = - 0.5 * 2. * Jee_sq * tau_se * taum * ( (1. + z1ee) * taum + tau_se )**(-1.) * sigma_sqrt_e
        jacobian_[15,10,t_] = - 0.5 * 2. * Jei_sq * tau_si * taum * ( (1. + z1ei) * taum + tau_si )**(-1.) * sigma_sqrt_e
        
        sigma_ie = 2. * Jie_sq * state_[0,11,t_] * tau_se * taum / ((1. + z1ie) * taum + tau_se)
        sigma_ii = 2. * Jii_sq * state_[0,12,t_] * tau_si * taum * ( (1. + z1ii ) * taum + tau_si )**(-1)
        
        arg = ( sigma_ie + sigma_ii + sigmai_ext**2 )
        if arg > 0:
            sigma_sqrt_i = arg**(-1./2.)
        else:
            sigma_sqrt_i = 0.
            print("WARNING: sigma i smaller zero")
        
        jacobian_[16,0,t_] = ( 0.5 * ( (1. + z1ie) * taum + tau_se )**(-2.) * factor_ie1 * 1e-3 * taum
                              * ( 2. * Jie_sq * tau_se * taum * state_[0,11,t_] ) * sigma_sqrt_i )
        jacobian_[16,1,t_] = ( 0.5 * ( (1. + z1ii) * taum + tau_si )**(-2.) * factor_ii1 * 1e-3 * taum
                              * ( 2. * Jii_sq * tau_si * taum * state_[0,12,t_] ) * sigma_sqrt_i )
        jacobian_[16,11,t_] = - 0.5 * 2. * Jie_sq * tau_se * taum * ( (1. + z1ie) * taum + tau_se )**(-1.) * sigma_sqrt_i
        jacobian_[16,12,t_] = - 0.5 * 2. * Jii_sq * tau_si * taum * ( (1. + z1ii) * taum + tau_si )**(-1.) * sigma_sqrt_i
        
        jacobian_[17,2,t_] = - d_V_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_V, interpolate_V)
        jacobian_[17,4,t_] = d_V_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_V, interpolate_V) / C
        jacobian_[17,15,t_] = - d_V_func_sigma(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_V, interpolate_V)
        
        jacobian_[18,2,t_] = - d_tau_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_tau_mu, interpolate_tau)
        jacobian_[18,4,t_] = d_tau_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_tau_mu, interpolate_tau) / C
        jacobian_[18,15,t_] = - d_tau_func_sigma(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_tau_mu, interpolate_tau)
        
        jacobian_[19,3,t_] = - d_tau_func_mu(state_[0,3,t_], sigmarange, ds,
                                          state_[0,16,t_], Irange, dI, C, precalc_tau_mu, interpolate_tau)
        jacobian_[19,16,t_] = - d_tau_func_sigma(state_[0,3,t_], sigmarange, ds,
                                          state_[0,16,t_], Irange, dI, C, precalc_tau_mu, interpolate_tau)

    
    return jacobian_

@numba.njit
def d_r_func_mu(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_r, interpolate_r):
    if interpolate_r:
        result = jac_aln.der_mu(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_r)
        return result
    x_shift_mu = - 2.
    x_scale_mu = 0.6
    y_scale_mu = 0.1
    result = y_scale_mu * x_scale_mu / np.cosh(x_scale_mu * mu + x_shift_mu)**2
    #result = 1e-3
    return result

@numba.njit
def d_r_func_sigma(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_r, interpolate_r):
    #return 1. * 1e-3
    #return (1. + 3. * sigma**2) * 1e-3
    if interpolate_r:
        result = jac_aln.der_sigma(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_r)
        return result
    x_shift_sigma = -1.
    x_scale_sigma = 0.6
    y_scale_sigma = 1./2500.
    result = np.sinh(x_scale_sigma * sigma + x_shift_sigma) * y_scale_sigma * x_scale_sigma
    #result = 1e-3
    return result

@numba.njit
def d_tau_func_mu(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_tau_mu, interpolate_tau):
    #return 1. - ( 1. + mu )**(-2.) #+ sigma
    if interpolate_tau:
        result = jac_aln.der_mu(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_tau_mu)
        return result
    mu_shift = - 1.1
    sigma_scale = 0.5
    mu_scale = - 10
    mu_scale1 = - 3
    sigma_shift = 1.4
    result = sigma_scale * sigma + mu_scale1 + ( mu_scale / (sigma + sigma_shift) ) * np.exp( mu_scale * ( mu_shift + mu ) / ( sigma + sigma_shift ) )
    #result = 1e-3
    return result

@numba.njit
def d_tau_func_sigma(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_tau_mu, interpolate_tau):
    #return 1. - ( 1. + sigma )**(-2.) #+ mu
    if interpolate_tau:
        result = jac_aln.der_sigma(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_tau_mu)
        return result
    mu_shift = - 1.1
    sigma_scale = 0.5
    mu_scale = - 10
    sigma_shift = 1.4
    result = sigma_scale * ( mu_shift + mu ) - (mu_scale * (mu_shift + mu) / (sigma + sigma_shift)**2) * np.exp(
        mu_scale * ( mu_shift + mu ) / ( sigma + sigma_shift ) )  
    #result = 1e-3
    return result

@numba.njit
def d_V_func_mu(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_V, interpolate_V):
    if interpolate_V:
        result = jac_aln.der_mu(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_V)
        return result
    y_scale1 = 30.
    mu_shift1 = 1.
    y_scale2 = 2.
    mu_shift2 = 0.5
    sigma_shift = 0.1
    result = y_scale1 / np.cosh( mu + mu_shift1 )**2 - y_scale2 * 2. * ( mu - mu_shift2 ) * np.exp( - ( mu - mu_shift2 )**2 ) / ( sigma + sigma_shift )
    #result = 1e-3
    return result

@numba.njit
def d_V_func_sigma(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_V, interpolate_V):
    if interpolate_V:
        result = jac_aln.der_sigma(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_V)
        return result
    y_scale2 = 2.
    mu_shift2 = 0.5
    sigma_shift = 0.1
    result = - y_scale2 * np.exp( - ( mu - mu_shift2 )**2 ) / ( sigma + sigma_shift )**2
    #result = 1e-3
    return result