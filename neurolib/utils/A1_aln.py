import numpy as np
import logging
import numba

from timeit import default_timer as timer
from . import costFunctions as cost
from . import func_optimize as fo
from ..models import jacobian_aln as jac_aln

VALID_VAR = {None, "FR", "HS"}

def A1(model, control_, target_state_, c_scheme_, u_mat_, u_scheme_, max_iteration_, tolerance_, startStep_, cntrl_max_, t_sim_, t_sim_pre_, t_sim_post_, CGVar):
    
    if CGVar not in VALID_VAR:
        CGVar = None
        raise ValueError("u_opt control optimization: conjugate gradient variant must be one of %r." % VALID_VAR)
        
    dt = model.params['dt']
    state_vars = model.state_vars
    init_vars = model.init_vars
    
    # run model with dt duration once to set delay matrix
    model.params['duration'] = dt

    model.run(control=model.getZeroControl())
    
    Dmat = model.params.Dmat
    Dmat_ndt = np.around(Dmat / dt).astype(int)
        
    Dmat_ndt = np.around(Dmat / dt).astype(int)
        
    ndt_de = np.around(model.params.de / dt).astype(int)
    ndt_di = np.around(model.params.di / dt).astype(int)
    max_global_delay = max(np.max(Dmat_ndt), ndt_de, ndt_di)
        
    startind_ = int(max_global_delay + 1)
    
    state_vars = model.state_vars
    init_vars = model.init_vars
    
    if (startind_ > 1):
        for iv in range(len(init_vars)):
            if model.params[init_vars[iv]].ndim == 2:
                if (model.params[init_vars[iv]].shape[1] <= 1):
                    model.params[init_vars[iv]] = np.dot( model.params[init_vars[iv]], np.ones((1, startind_)) )
                    
    t_pre_ndt = np.around(t_sim_pre_ / dt).astype(int)
    delay_state_vars_ = np.zeros(( model.params.N, len(state_vars), startind_ ))
    
    # simulate with duration t_sim_pre before start
    if (t_sim_pre_ >= dt):
        model.params['duration'] = t_sim_pre_
        control_pre_ = model.getZeroControl()
        state_pre_ = fo.updateFullState(model, control_pre_, t_sim_pre_)
        
        if startind_ == 1:
            fo.update_init(model, init_vars, state_vars)
        else:
            fo.update_init_delayed(model, delay_state_vars_, init_vars, state_vars, t_pre_ndt, startind_)
    
    model.params['duration'] = t_sim_   
        
    runtime_ = np.zeros((max_iteration_+1))
    runtime_start_ = timer()
    beta_ = 0.
    
    #control_[:,:,0] = control_[:,:,1]
    #control_[:,:,-2] = control_[:,:,-1]
    
    state0_ = fo.updateFullState(model, control_, t_sim_)
    i=0
    
    total_cost_ = np.zeros((max_iteration_+1))
    total_cost_[i] = cost.f_int(model.params['dt'], cost.f_cost(state0_, target_state_, control_) )
    print("RUN ", i, ", total integrated cost = ", total_cost_[i])

    runtime_[i] = timer() - runtime_start_
    
    state1_ = state0_.copy()
    u_opt0_ = control_.copy()
    best_control_ = control_.copy()
        
    while( i < max_iteration_):
        i += 1   
        
        #phi_ = phi(model, state0_, target_state_, best_control_, c_scheme_)
        #print("phi = ", phi_)
        phi_ = phi2(model, state0_, target_state_, best_control_, c_scheme_)
    
        g0_min_ = g(model, phi_, state1_, target_state_, best_control_, u_mat_, u_scheme_)
        g1_min_ = g0_min_.copy()
        #print("control = ", best_control_)
        #print("state = ", state0_[:,:2,:])
        #print("phi = ", phi_)
        print("g_min = ", g0_min_)
        
        
        if i == 1:
            dir0_ = - g0_min_.copy()
            dir1_ = dir0_.copy()
            
        else:
            CGVar1 = CGVar
            
            if (CGVar1 == None):
                # do not apply special variation of conjugate gradient descend
                beta_ = 0.
            elif (CGVar1 == "FR"):
                beta_ = beta_FR_t(model, g1_min_, g0_min_)
            elif (CGVar1 == "HS"):
                beta_ = beta_HS(model, g1_min_, g0_min_, dir0_)
                
            dir1_ = - g1_min_ + beta_ * dir0_
            
        #dir1_[:,:,0] = dir1_[:,:,1].copy()
        #dir1_[:,:,-1] = dir1_[:,:,-2].copy()
            
        #dc1 = np.zeros(( state0_.shape[2] ))
        
        #check descent condition   
        #for i_time in range(control_.shape[2]):
            #for i_node in range(control_.shape[0]):
                #for i_var in range(control_.shape[1]):
                    #dc1[i_time] += (g1_min_[i_node, i_var, i_time] * dir1_[i_node, i_var, i_time])
            #if (dc1[i_time] > 0):
                #print("descent condition 1 not satisfied for time index ", i_time)
                #dir1_[:,:,i_time] = - g1_min_[:,:,i_time]
        
        step_, total_cost_[i] = fo.step_size(model, fo.get_output_from_full(model, state1_), target_state_,
                     best_control_, dir1_, start_step_ = startStep_, max_control_ = cntrl_max_)
        
        print("RUN ", i, ", total integrated cost = ", total_cost_[i])
        best_control_ = u_opt0_ + step_ * dir1_
        
        u_diff_ = ( np.absolute(best_control_ - u_opt0_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i
            break
        u_opt0_ = best_control_.copy()
        
        state1_ = fo.updateFullState(model, best_control_, t_sim_)
        #print("step size = ", step_, total_cost_[i])
        #print("control after = ", best_control_)
        #print("state after = ", state0_[:,:2,:])
        
        s_diff_ = ( np.absolute(state1_ - state0_) < tolerance_ )
        if ( s_diff_.all() ):
            print("State only changes marginally.")
            max_iteration_ = i
            break
        
        g0_min_ = g1_min_.copy()
        if ( np.amax(np.absolute(g0_min_[:,:,1:])) < tolerance_ ):
            print(np.amax(np.absolute(g0_min_[:,:,1:])))
            print("Gradient negligibly small.")
            max_iteration_ = i
            break
        
        state0_ = state1_.copy() 
        runtime_[i] = timer() - runtime_start_
        dir0_ = dir1_.copy()   
           
    improvement = 100.
    if total_cost_[0] != 0.:
        improvement = improvement = 100 - int(100.*(total_cost_[max_iteration_]/total_cost_[0]))
        
    print("Improved over ", max_iteration_, " iterations by ", improvement, " percent.")
    #fo.printState(model, state1_)
    
    
    if (t_sim_pre_ < dt and t_sim_post_ < dt):
        return best_control_, state1_, total_cost_, runtime_
    
    if (t_sim_post_ > dt):
        
        for iv, sv in zip( range(len(init_vars)), range(len(state_vars)) ):
            if state_vars[sv] in init_vars[iv]:
                #print("variable = ", state_vars[sv])
                #print(" state = ", model.state[state_vars[sv]])
                #print(" init param = ", model.params[init_vars[iv]])
                if model.params[init_vars[iv]].ndim == 2:
                    model.params[init_vars[iv]][0,0] = model.state[state_vars[sv]][0,-1]
                else:
                    model.params[init_vars[iv]][0] = model.state[state_vars[sv]][0]
            else:
                logging.error("Initial and state variable labelling does not agree.")
                
        model.params.duration = t_sim_post_
        control_post_ = model.getZeroControl()
        state_post_ = fo.updateFullState(model, control_post_, t_sim_post_)    
    
    model.params.duration = t_sim_ + t_sim_pre_ + t_sim_post_
    bc_ = model.getZeroControl()
    bs_ = model.getZeroFullState()
        
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    
    #print("state pre = ", state_pre_)
    
    if (i2 != 0 and i1 != 0):   
        bc_[:,:,i1:-i2] = best_control_[:,:,:]   
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        for n in range(bs_.shape[0]):
            for v in range(bs_.shape[1]):
                if (bs_[n,v,i1] != state0_[n,v,0] and v not in range(15,20)):
                    logging.error("Problem in initial value trasfer")
        bs_[:,:,i1+1:-i2] = state0_[:,:,1:]
        bs_[:,:,-i2:] = state_post_[:,:,1:]
    elif (i2 == 0 and i1 != 0):
        bc_[:,:,i1:] = best_control_[:,:,:]
        #print("best state = ", bs_)
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        #print("best state = ", bs_)
        for n in range(bs_.shape[0]):
            for v in range(bs_.shape[1]):
                if (bs_[n,v,i1] != state0_[n,v,0] and v not in range(15,20)):
                    print("state = ", v, bs_[n,v,i1], state0_[n,v,0])
                    logging.error("Problem in initial value trasfer")
        bs_[:,:,i1+1:] = state0_[:,:,1:]
        #print("best state = ", bs_)
    elif (i2 != 0 and i1 == 0):
        bc_[:,:,:-i2] = best_control_[:,:,:]
        bs_[:,:,:-i2] = state0_[:,:,:]
        bs_[:,:,-i2:] = state_post_[:,:,1:]
    else:
        bc_[:,:,:] = best_control_[:,:,:]
        bs_[:,:,:] = state0_[:,:,:]
        
    
    return bc_, bs_, total_cost_, runtime_ 


# computation of phi
def phi(model, state_, target_state_, control_, c_scheme_, start_ind_ = 0):
    #print("ALN phi computation")
    phi_ = model.getZeroFullState()
    dt = model.params['dt']
    out_state = fo.get_output_from_full(model, state_)
    zeros = [0,1,13,14,15,16,17,18,19]
    
    for ind_time in range(phi_.shape[2]-1, start_ind_-1, -1):
        f_p_grad_t_ = cost.cost_precision_gradient_t(out_state[:,:,ind_time], target_state_[:,:,ind_time])
        print("precision cost gradient : ", f_p_grad_t_)
        if ind_time != 0:
            D_x_t = D_x_h(model, state_[:,:,ind_time], control_[:,:,ind_time], state_[:,4,ind_time-1])
        else:
            # no IA for smaller zero
            D_x_t = D_x_h(model, state_[:,:,ind_time], control_[:,:,ind_time], state_[:,4,ind_time])
        
        if (ind_time != phi_.shape[2]-1):
            # result for index 1,2 
            phi_[:,0,ind_time] = - f_p_grad_t_[:,0]
            phi_[:,1,ind_time] = - f_p_grad_t_[:,1]
            
            # compute index 13 to 19 from algebraic equations
            if ind_time == 0:
                print("jacobian matrix = ", D_x_t[0,0,15:, 15:])
                print("inverse = ", np.linalg.inv( D_x_t[0,0,15:, 15:].T ))
                print("upper part T= ", D_x_t[0,0,0:5, 15:].T)
                print("phi = ", phi_[0,0:5,ind_time] )
            phi_[0,15:,ind_time] = - np.dot( np.dot( np.linalg.inv( D_x_t[0,0,15:, 15:].T ), D_x_t[0,0,0:5, 15:].T ), phi_[0,0:5,ind_time] )
            
        phi_dot_ = phi_dot(model, phi_[:,:,ind_time], D_x_t)
        
        # results for index 2 to 12 for next timestep
        for i in range(phi_dot_.shape[1]):
            if i not in zeros:
                phi_[:,:, ind_time-1] = phi_[:,:, ind_time] + phi_dot_[:,:] * (-dt)
        
    return phi_

def phi2(model, state_, target_state_, control_, c_scheme_, start_ind_ = 0):
    #print("ALN phi2 computation")
    phi_ = model.getZeroFullState()
    dt = model.params['dt']
    out_state = fo.get_output_from_full(model, state_)
    
    for ind_time in range(phi_.shape[2]-1, start_ind_-1, -1):
        
        if ind_time != 0:
            D_x_t = D_x_h(model, state_[:,:,ind_time], control_[:,:,ind_time], state_[:,4,ind_time-1])
        else:
            # no IA for smaller zero
            D_x_t = D_x_h(model, state_[:,:,ind_time], control_[:,:,ind_time], state_[:,4,ind_time])
        
        f_p_grad_t_ = cost.cost_precision_gradient_t(out_state[:,:,ind_time], target_state_[:,:,ind_time])
        # result for index 1,2 
        d0_ = ( ( - phi_[0,17,ind_time] * D_x_t[0,0,17,15] - phi_[0,18,ind_time] * D_x_t[0,0,18,15] ) * D_x_t[0,0,15,0]
               + ( f_p_grad_t_[:,0] + phi_[0,5,ind_time] * D_x_t[0,0,5,0] + phi_[0,9,ind_time] * D_x_t[0,0,9,0] ) )
        phi_[:,0,ind_time] = d0_ / ( - 1 + D_x_t[0,0,0,15] * D_x_t[0,0,15,0] )
        #print("grad = ", phi_[0,0,ind_time])
        
        phi_[:,0,ind_time] = - f_p_grad_t_[:,0]
        
        
        # compute index 13 to 19 from algebraic equations
        phi_[:,17,ind_time] = - phi_[0,4,ind_time] * D_x_t[0,0,4,17]
        #print("phi17 = ", phi_[:,17,ind_time])
        phi_[:,18,ind_time] = - phi_[0,2,ind_time] * D_x_t[0,0,2,18]
        phi_[:,19,ind_time] = - phi_[0,3,ind_time] * D_x_t[0,0,3,19]
        
        #if ind_time != phi_.shape[2]-1:
        phi_[:,15,ind_time] = - ( phi_[0,0,ind_time] * D_x_t[0,0,0,15] + phi_[0,17,ind_time] * D_x_t[0,0,17,15]
                                     + phi_[0,18,ind_time] * D_x_t[0,0,18,15] )
        phi_[:,16,ind_time] = - ( phi_[0,1,ind_time] * D_x_t[0,0,1,16] + phi_[0,19,ind_time] * D_x_t[0,0,19,16] )
        
        phi_[:,1,ind_time] = - f_p_grad_t_[:,1] - ( phi_[0,6,ind_time] * D_x_t[0,0,6,1] + phi_[0,10,ind_time] * D_x_t[0,0,10,1] 
                                                   + phi_[0,15,ind_time] * D_x_t[0,0,15,1] )

            
        if (ind_time == 0):
            #print("prefactors 15= ", D_x_t[0,0,0,15], D_x_t[0,0,17,15], D_x_t[0,0,18,15])
            #print("prefactors 16= ", D_x_t[0,0,1,16], D_x_t[0,0,19,16])
            #print("prefactors 17= ", D_x_t[0,0,4,17])
            break
        
        phi_[0,2,ind_time-1] = phi_[0,2,ind_time] - dt * ( phi_[0,0,ind_time] * D_x_t[0,0,0,2]
                                                          + phi_[0,2,ind_time] * D_x_t[0,0,2,2]
                                                          #+ phi_[0,17,ind_time] * D_x_t[0,0,17,2]
                                                          #+ phi_[0,18,ind_time] * D_x_t[0,0,18,2] 
                                                          )
        #print("phi2 : ", phi_[0,2,ind_time-1] )
        #print("summands : ", phi_[0,0,ind_time] * D_x_t[0,0,0,2],
        #                     + phi_[0,2,ind_time] * D_x_t[0,0,2,2],
        #                     + phi_[0,17,ind_time] * D_x_t[0,0,17,2],
        #                      + phi_[0,18,ind_time] * D_x_t[0,0,18,2])
        phi_[0,3,ind_time-1] = phi_[0,3,ind_time] - dt * ( phi_[0,1,ind_time] * D_x_t[0,0,1,3] + phi_[0,3,ind_time] * D_x_t[0,0,3,3] + phi_[0,19,ind_time] * D_x_t[0,0,19,3] )
        phi_[0,4,ind_time-1] = phi_[0,4,ind_time] - dt * ( phi_[0,4,ind_time] * D_x_t[0,0,4,4] )
        phi_[0,5,ind_time-1] = phi_[0,5,ind_time] - dt * ( phi_[0,5,ind_time] * D_x_t[0,0,5,5] + phi_[0,9,ind_time] * D_x_t[0,0,9,5])
        phi_[0,6,ind_time-1] = phi_[0,6,ind_time] - dt * ( phi_[0,6,ind_time] * D_x_t[0,0,6,6] + phi_[0,10,ind_time] * D_x_t[0,0,10,6])
        #print(phi_[0,9,ind_time], D_x_t[0,0,9,9], phi_[0,15,ind_time], D_x_t[0,0,15,9])
        phi_[0,9,ind_time-1] = phi_[0,9,ind_time] - dt * ( phi_[0,9,ind_time] * D_x_t[0,0,9,9] + phi_[0,15,ind_time] * D_x_t[0,0,15,9])
        phi_[0,10,ind_time-1] = phi_[0,10,ind_time] - dt * ( phi_[0,10,ind_time] * D_x_t[0,0,10,10] + phi_[0,15,ind_time] * D_x_t[0,0,15,10])
        
        #print("----1")
        
    return phi_

def phi_dot(model, phi_t_, D_x_t):
    phi_dot_ = np.zeros((phi_t_.shape))
    Dxhphi = np.dot( D_x_t[0,0,:,:].T, phi_t_.T)[2:13]
    D_xdot_t_inverse = np.linalg.inv( ((D_xdot_h(model)[0,0,:,:]).T)[2:13, 2:13] )
    phi_dot_sub = np.dot(D_xdot_t_inverse, Dxhphi)
    #print("phi dot computation : ")
    #print("inverse = ", D_xdot_t_inverse)
    #print("transposed : ", D_x_t[0,0,:,:].T)
    #print("phi : ", phi_t_.T)
    phi_dot_[0,2:13] = phi_dot_sub[:,0]
    print("phi dot = ", phi_dot_[0,2:13])
    return phi_dot_
    

def D_x_h(model, state_t_, control_t_, IA_min1):
    N = model.params.N
    Dh_ = np.zeros(( N, N, len(model.state_vars), len(model.state_vars) ))
    cee = model.params.cee
    Ke = model.params.Ke
    Jee = model.params.Jee_max
    tau_se = model.params.tau_se
    cei = model.params.cei
    Ki = model.params.Ki
    Jei = model.params.Jei_max
    tau_si = model.params.tau_si
    tau_m = model.params.C / model.params.gL
    sigmae_ext = model.params.sigmae_ext
    
    #print("params = ", cee, Ke, Jee)
    
    #convert Hz to kHz
    r_e = state_t_[:,0] * 1e-3
    r_i = state_t_[:,1] * 1e-3
    
    for no in range(N):
        # exc rate row
        Dh_[no,no,0,0] = 1.
        Dh_[no,no,0,2] = - jac_aln.der_mu(model, state_t_[no,15], state_t_[no,2], IA_min1, model.params.precalc_r)
        #print("parameters : ", state_t_[no,15], state_t_[no,2])
        Dh_[no,no,0,15] = - jac_aln.der_sigma(model, state_t_[no,15], state_t_[no,2], IA_min1, model.params.precalc_r)
        #print("jacobian : ", Dh_[:,:,0,2], Dh_[:,:,0,15])
        # inh rate row
        Dh_[no,no,1,1] = 1.
        Dh_[no,no,1,3] = - jac_aln.der_mu(model, state_t_[no,16], state_t_[no,3], IA_min1, model.params.precalc_r)
        Dh_[no,no,1,16] = - jac_aln.der_sigma(model, state_t_[no,16], state_t_[no,3], IA_min1, model.params.precalc_r)
        #print("jacobian : ", Dh_[:,:,1,3], Dh_[:,:,1,16])
        # mu_e row
        Dh_[no,no,2,2] = 1./state_t_[no,18]
        Dh_[no,no,2,18] = (model.params.ext_exc_current - state_t_[no,2] + control_t_[no,0]) /state_t_[no,18]**2
        #print("Dxh computation : ", model.params.ext_exc_current, state_t_[no,2], control_t_[no,0], state_t_[no,18]**2)
        # mu_i row
        Dh_[no,no,3,3] = 1./state_t_[no,19]
        Dh_[no,no,3,19] = (model.params.ext_inh_current - state_t_[no,3] + control_t_[no,1]) /state_t_[no,19]**2
        # IA row
        Dh_[no,no,4,4] = 1./model.params.tauA
        Dh_[no,no,4,17] = - model.params.a/model.params.tauA
        # s rows
        Dh_[no,no,5,0] = - ( 1 - state_t_[no,5] ) * cee * Ke / Jee
        Dh_[no,no,5,5] = 1. / tau_se + cee * Ke * r_e[no] / Jee
        Dh_[no,no,6,1] = - ( 1 - state_t_[no,6] ) * cei * Ki / Jei
        Dh_[no,no,6,6] = 1. / tau_si + cei * Ki * r_i[no] / Jei
        # sigma_s rows
        Dh_[no,no,9,0] = - ( 1 - state_t_[no,5] )**2 * cee**2 * Ke / Jee**2 - state_t_[no,9] * ( cee**2 * Ke / Jee**2
                        - 2 * ( cee * Ke / Jee ) )
        Dh_[no,no,9,5] = 2 * ( 1 - state_t_[no,5] ) * ( cee**2 * Ke * r_e[no] / Jee**2 )
        Dh_[no,no,9,9] = - cee**2 * Ke * r_e[no] / Jee**2 + 2. * ( cee * Ke * tau_se * r_e[no] / Jee + 1. ) / tau_se
        Dh_[no,no,10,1] = - ( 1 - state_t_[no,6] )**2 * cei**2 * Ki / Jei**2 + state_t_[no,10] * ( cei**2 * Ki / Jei**2
                        - 2 * ( cei * Ki / Jei ) )
        Dh_[no,no,10,6] = 2 * ( 1 - state_t_[no,6] ) * ( cei**2 * Ki * r_i[no] / Jei**2 )
        Dh_[no,no,10,10] = - cei**2 * Ki * r_i[no] / Jei**2 - 2. * ( cei * Ki * tau_si * r_i[no] / Jei + 1. ) / tau_si
        # sigma rows
        prefactor_e = ( f1(model, Jee, tau_se, cee, Ke, r_e[no]) * state_t_[no,9]
                       + f1(model, Jei, tau_si, cei, Ki, r_i[no]) * state_t_[no,10] + sigmae_ext**2 )**(-1./2.)
        Dh_[no,no,15,0] = prefactor_e * state_t_[no,9] * f2(model, Jee, tau_se, cee, Ke, r_e[no])
        Dh_[no,no,15,1] = prefactor_e * state_t_[no,10] * f2(model, Jei, tau_si, cei, Ki, r_i[no])
        Dh_[no,no,15,9] = prefactor_e * f1(model, Jee, tau_se, cee, Ke, r_e[no]) / 2.
        Dh_[no,no,15,10] = prefactor_e * f1(model, Jei, tau_si, cei, Ki, r_i[no]) / 2.
        Dh_[no,no,15,15] = 1.
        Dh_[no,no,16,16] = 1.
        # V row
        Dh_[no,no,17,2] = - jac_aln.der_mu(model, state_t_[no,15], state_t_[no,2], IA_min1, model.params.precalc_V)
        Dh_[no,no,17,15] = - jac_aln.der_sigma(model, state_t_[no,15], state_t_[no,2], IA_min1, model.params.precalc_V)
        Dh_[no,no,17,17] = 1.
        # tau_exc row
        Dh_[no,no,18,2] = - jac_aln.der_mu(model, state_t_[no,15], state_t_[no,2], IA_min1, model.params.precalc_tau_mu)
        Dh_[no,no,18,15] = - jac_aln.der_sigma(model, state_t_[no,15], state_t_[no,2], IA_min1, model.params.precalc_tau_mu)
        Dh_[no,no,18,18] = 1.
        # tau_inh row
        Dh_[no,no,19,3] = - jac_aln.der_mu(model, state_t_[no,16], state_t_[no,3], IA_min1, model.params.precalc_tau_mu)
        Dh_[no,no,19,16] = - jac_aln.der_sigma(model, state_t_[no,16], state_t_[no,3], IA_min1, model.params.precalc_tau_mu)
        Dh_[no,no,19,19] = 1.
        
    return Dh_
    
def f1(model, J_, tau_bar_, c_, K_, r_):
    tau_m_ = model.params.C / model.params.gL
    return 2. * J_**2 * tau_bar_ *tau_m_ / ( (1. + (c_ * K_ * r_ / J_) * tau_m_) + tau_bar_ )

def f2(model, J_, tau_bar_, c_, K_, r_):
    tau_m_ = model.params.C / model.params.gL
    return c_ * K_ * J_ * tau_bar_ *tau_m_**2 / ( (1. + (c_ * K_ * r_ / J_) * tau_m_) + tau_bar_ )**2

def D_xdot_h(model):
    Dh_ = np.zeros(( model.params.N, model.params.N, len(model.state_vars), len(model.state_vars) ))
    ones = [2,3,4,5,6,7,8,9,10,11,12]
    for i in ones:
        Dh_[:,:,i,i] = 1.
    return Dh_

def D_u_h(model, state_):
    N = model.params.N
    Duh_ = np.zeros(( model.params.N, model.params.N, len(model.state_vars), len(model.state_vars), state_.shape[2] ))
    for no in range(N):
        Duh_[no,no,2,2,:] = -1./state_[no,18,:]
        Duh_[no,no,3,3,:] = -1./state_[no,19,:]
    return Duh_

# computation of g
# does not work with numba because takes model as parameter
def g(model, phi_, state_, target_, control_, u_mat_, u_scheme_):
    g_ = model.getZeroFullState()
    grad_cost_e_ = cost.cost_energy_gradient(control_)
    grad_cost_s_ = cost.cost_sparsity_gradient1(model, control_)
    D_u = D_u_h(model, state_)
    
    for t in range(g_.shape[2]):
        g_[:,:,t] = np.dot(phi_[:,:,t], D_u[0,0,:,:,t])
    g_out_ = g_[:,2:4,:]
    #print("g, phi contribution = ", g_out_)
    #print("g, grad e contribution = ", grad_cost_e_, grad_cost_s_)
    g_out_ += grad_cost_e_ + grad_cost_s_
    return g_out_


def beta_FR(model, g_1_, g_0_):
    beta_ = 0.
    denominator_ = 0.
    numerator_ = 0.
    for ind_time in range(g_1_.shape[2]):
        for ind_node in range(g_1_.shape[0]):
            for ind_var in range(g_1_.shape[1]):
                denominator_ += g_1_[ind_node, ind_var, ind_time]**2.
                numerator_ += g_1_[ind_node, ind_var, ind_time] * (g_1_[ind_node, ind_var, ind_time]
                                                                   - g_0_[ind_node, ind_var, ind_time])
        if (denominator_ == 0.):
            print("zero denominator, numerator = ", numerator_)
            denominator_ = 1.
            
        #print("numerator, denominator = ", numerator_, denominator_)
        beta_ += numerator_/denominator_
    return beta_

def beta_FR_t(model, g_1_, g_0_):
    beta_ = np.zeros((int( round(model.params['duration']/model.params['dt'],1) +1)))
    denominator_ = 0.
    numerator_ = 0.
    for ind_time in range(g_1_.shape[2]):
        for ind_node in range(g_1_.shape[0]):
            for ind_var in range(g_1_.shape[1]):
                denominator_ += g_1_[ind_node, ind_var, ind_time]**2.
                numerator_ += g_1_[ind_node, ind_var, ind_time] * (g_1_[ind_node, ind_var, ind_time]
                                                                   - g_0_[ind_node, ind_var, ind_time])
        if (denominator_ == 0.):
            print("zero denominator, numerator = ", numerator_)
            denominator_ = 1.
        beta_[ind_time] = numerator_/denominator_
    return beta_

def beta_HS(model, g_1_, g_0_, dir0_):
    beta_ = np.zeros((int( round(model.params['duration']/model.params['dt'],1) +1)))
    denominator_ = 0.
    numerator_ = 0.
    for ind_time in range(g_1_.shape[2]):
        for ind_node in range(g_1_.shape[0]):
            for ind_var in range(g_1_.shape[1]):
                denominator_ += dir0_[ind_node, ind_var, ind_time] * (g_1_[ind_node, ind_var, ind_time] - g_0_[ind_node, ind_var, ind_time])
                numerator_ += g_1_[ind_node, ind_var, ind_time] * (g_1_[ind_node, ind_var, ind_time] - g_0_[ind_node, ind_var, ind_time])
        if (denominator_ == 0.):
            print("zero denominator, numerator = ", numerator_)
            denominator_ = 1.
        beta_[ind_time] = numerator_/denominator_
    return beta_