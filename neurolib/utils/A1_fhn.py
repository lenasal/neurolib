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
        raise ValueError("u_opt control optimization: conjugate gradient variant must be one of %r." % VALID_VAR)
        CGVar = None
        
    dt = model.params['dt']
    state_vars = model.state_vars
    init_vars = model.init_vars
    
    # simulate with duration t_sim_pre before start
    if (t_sim_pre_ > dt):
        model.params['duration'] = t_sim_pre_
        control_pre_ = model.getZeroControl()
        state_pre_ = fo.updateState(model, control_pre_)
        
        #print("state vars = ", state_vars)
        #print("init vars = ", init_vars)
        
        for iv, sv in zip( range(len(init_vars)), range(len(state_vars)) ):
            if state_vars[sv] in init_vars[iv]:
                #print("variable = ", state_vars[sv])
                #print(" state = ", model.state[state_vars[sv]])
                #print(" init param = ", model.params[init_vars[iv]])
                if model.params[init_vars[iv]].ndim == 2:
                    model.params[init_vars[iv]][:,0] = model.state[state_vars[sv]][:,-1]
                else:
                    model.params[init_vars[iv]][0] = model.state[state_vars[sv]][0]
            else:
                logging.error("Initial and state variable labelling does not agree.")
                        
                    #print("setting inital parameters for post-control run: ",init_vars[iv], " : ",  model.params[init_vars[iv]])
    
    model.params['duration'] = t_sim_    
        
    runtime_ = np.zeros((max_iteration_+1))
    runtime_start_ = timer()
    beta_ = 0.
    
    state0_ = fo.updateState(model, control_)
    state1_ = state0_.copy()
    u_opt0_ = control_.copy()
    best_control_ = control_.copy()
    phi_ = phi(model, state0_, target_state_, c_scheme_)
    #print("phi = ", phi_[0,0,-10:])
    g0_min_ = g(model, phi_, state1_, target_state_, best_control_, u_mat_, u_scheme_)
    g1_min_ = g0_min_.copy()
    #print("g_min = ", g_min_)
    dir0_ = - g0_min_.copy()
    dir1_ = dir0_.copy()
    
    dc1 = np.zeros(( state0_.shape[2] ))
    
    """
    #check descent condition   
    for i_time in range(state_.shape[2]):
        for i_node in range(state_.shape[0]):
            for i_var in range(state_.shape[1]):
                dc1[i_time] += (g1_min_[i_node, i_var, i_time] * dir0_[i_node, i_var, i_time])
        if (dc1[i_time] > 0):
            print("descent condition 1 not satisfied for time index ", i_time)
            """
    
    
    i=0
    
    total_cost_ = np.zeros((max_iteration_+1))
    total_cost_[i] = cost.f_int(model.params['dt'], state0_, target_state_, control_ )
    print("RUN ", i, ", total integrated cost = ", total_cost_[i])
    runtime_[i] = timer() - runtime_start_
    
    while( np.amax(np.absolute(g0_min_[:,:,1:])) > tolerance_ and i < max_iteration_ - 1 ):
        i += 1        
        step_, total_cost_[i] = fo.step_size(model, state1_, target_state_, best_control_, dir1_, start_step_ = startStep_,
                                          max_control_ = cntrl_max_)
        print("RUN ", i, ", total integrated cost = ", total_cost_[i])
        #print("step = ", step_)
        #print("dir = ", dir1_)
        best_control_ = u_opt0_ + step_ * dir1_
        #print("u opt = ", best_control_)
        u_diff_ = ( np.absolute(best_control_ - u_opt0_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i
            #print("improved over ", i, " iterations by ", 100 - int(100.*(total_cost_[i]/total_cost_[0])), " percent.")
            break
            #return u_opt0_, state0_, total_cost_[:i-1], runtime_[:i-1]
        u_opt0_ = best_control_.copy()
        
        state1_ = fo.updateState(model, best_control_)
        s_diff_ = ( np.absolute(state1_ - state0_) < tolerance_ )
        if ( s_diff_.all() ):
            print("State only changes marginally.")
            max_iteration_ = i
            #print("improved over ", i, " iterations by ", 100 - int(100.*(total_cost_[i]/total_cost_[0])), " percent.")
            break
            #return u_opt0_, state0_, total_cost_[:i-1], runtime_[:i-1]
        state0_ = state1_.copy()
        
        runtime_[i] = timer() - runtime_start_
        #print("new state = ", state_)
        #print('-----------')
        
        
        phi_ = phi(model, state1_, target_state_, c_scheme_)
        g1_min_ = g(model, phi_, state1_, target_state_, best_control_, u_mat_, u_scheme_)
        #print("phi = ", phi_[0,0,-10:])
        #print("g = ", g1_min_)
        
        """
        diffT = np.zeros(( state_.shape[2] ))
        
        for i_time in range(state_.shape[2]):
            for i_node in range(state_.shape[0]):
                for i_var in range(state_.shape[1]):
                    diffT[i_time] += (g1_min_[i_node, i_var, i_time] * g0_min_[i_node, i_var, i_time])

        print("gradient change min, max, mean : ", np.amin(diffT), np.amax(diffT), np.mean(diffT))
        """
        
        CGVar1 = CGVar
        #if (np.mean(diffT) <= 0.):
        #    CGVar1 = None
        
        #dir_test_step_ = 0.
        #test_step_size_ = test_step_
        
        #if (i == 5 or i == 6):
          #  print(" phi = ", phi_)
           # print(" g = ", g1_min_)
            #print(" dir = ", dir_)
            
        if (CGVar1 == None):
            # do not apply special variation of conjugate gradient descend
            beta_ = 0.
        elif (CGVar1 == "FR"):
           beta_ = beta_FR_t(model, g1_min_, g0_min_)
        elif (CGVar1 == "HS"):
            beta_ = beta_HS(model, g1_min_, g0_min_, dir0_)
           
        #print("beta min, max = ", np.amin(beta_), np.amax(beta_) )
 
        dir1_ = - g1_min_ + beta_ * dir0_
        
        dc1 = np.zeros(( state0_.shape[2] ))
        
        #check descent condition   
        for i_time in range(control_.shape[2]):
            for i_node in range(control_.shape[0]):
                for i_var in range(control_.shape[1]):
                    dc1[i_time] += (g1_min_[i_node, i_var, i_time] * dir1_[i_node, i_var, i_time])
            if (dc1[i_time] > 0):
                #print("descent condition 1 not satisfied for time index ", i_time)
                dir1_[:,:,i_time] = - g1_min_[:,:,i_time]
        
        #print("descent condition min, max, mean = ", np.amin(dc1), np.amax(dc1), np.mean(dc1) )
        
        g0_min_ = g1_min_.copy()
        dir0_ = dir1_.copy()
 
        """
        while(dir_test_step_ == 0.):
            #dir1_ = - g1_min_ + beta_*dir_
            #rint("1")
            dir_test_step_, dir_test_cost_ = test_step(model, state0_, target_state_, best_control_, dir1_, test_step_size_)
            print(dir_test_step_)
            if (dir_test_step_ == 0.):
                print("no descent direction")
                dir1_ = - g1_min_
                dir_test_step_, dir_test_cost_ = test_step(model, state0_, target_state_, best_control_, dir1_, test_step_size_)
                if (dir_test_step_ == 0.): # not descent direction
                    # can happen for unstable targets, when small changes cause overshooting
                    # decrease size of test step for all future iterations, because dynamical state will stay unstable
                    # change also initial step size because is likely to be much too large
                    test_step_size_ /= 10.
                    print("No descent direction found, try with smaller test step: ", test_step_size_)
                    #startStep_ = test_step_size_
                    if (test_step_size_ < 1e-30):
                        print("Cannot find descent direction")
                        return best_control_, state0_, total_cost_, runtime_
        """
        
                        
    i += 1
    step_, total_cost_[i] = fo.step_size(model, state1_, target_state_, best_control_, dir1_, start_step_ = startStep_, max_control_ = cntrl_max_)
    print("RUN ", i, ", total integrated cost = ", total_cost_[i])
    #print("step = ", step_)
    #print("dir = ", dir1_)
    best_control_ += step_ * dir1_
    #print("u optimal = ", u_opt_)
    state1_ = fo.updateState(model, best_control_)
    runtime_[i] = timer() - runtime_start_
    print("improved over ", i, " iterations by ", 100 - int(100.*(total_cost_[-1]/total_cost_[0])), " percent.")
    
    
    if (t_sim_pre_ <= dt and t_sim_post_ <= dt):
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
                
        model.params.duration = t_sim_post_ - dt
        control_post_ = model.getZeroControl()
        state_post_ = fo.updateState(model, control_post_)
    
    
    model.params.duration = t_sim_ + t_sim_pre_ + t_sim_post_
    bc_ = model.getZeroControl()
    bs_ = model.getZeroState()
        
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    
    if (i2 != 0 and i1 != 0):   
        bc_[:,:,i1:-i2] = best_control_[:,:,:]   
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        for n in range(bs_.shape[0]):
            for v in range(bs_.shape[1]):
                if bs_[n,v,i1+1] != state0_[n,v,0]:
                    logging.error("Problem in initial value trasfer")
        bs_[:,:,i1:-i2] = state0_[:,:,:]
        bs_[:,:,-i2:] = state_post_[:,:,:]
    elif (i2 == 0 and i1 != 0):
        bc_[:,:,i1:] = best_control_[:,:,:]
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        for n in range(bs_.shape[0]):
            for v in range(bs_.shape[1]):
                if bs_[n,v,i1+1] != state0_[n,v,0]:
                    logging.error("Problem in initial value trasfer")
        bs_[:,:,i1:] = state0_[:,:,:]
    elif (i2 != 0 and i1 == 0):
        bc_[:,:,:-i2] = best_control_[:,:,:]
        bs_[:,:,:-i2] = state0_[:,:,:]
        bs_[:,:,-i2:] = state_post_[:,:,:]
    else:
        bc_[:,:,:] = best_control_[:,:,:]
        bs_[:,:,:] = state0_[:,:,:]
    
    
    return bc_, bs_, total_cost_, runtime_ 


# computation of phi
def phi(model, state_, target_state_, c_scheme_, start_ind_ = 0, runge_kutta_ = False):
    phi_ = model.getZeroState()
    dt = model.params['dt']    
    
    for ind_time in range(phi_.shape[2]-1, start_ind_, -1):
        f_p_grad_t_ = cost.cost_precision_gradient_t(state_[:,:,ind_time], target_state_[:,:,ind_time])
        #print("f grad = ", f_p_grad_t_)
        if not runge_kutta_:
            phi_dot_ = phi_dot(model, state_[:,:,ind_time], phi_[:,:,ind_time], f_p_grad_t_, c_scheme_)
            phi_[:,:, ind_time-1] = phi_[:,:, ind_time] + phi_dot_[:,:] * (-dt)
        else:
            k1 = phi_dot(model, state_[:,:,ind_time], phi_[:,:,ind_time], f_p_grad_t_, c_scheme_)
            k2 = phi_dot(model, state_[:,:,ind_time], phi_[:,:,ind_time] + 0.5 * (-dt) * k1, f_p_grad_t_, c_scheme_)
            k3 = phi_dot(model, state_[:,:,ind_time], phi_[:,:,ind_time] + 0.5 * (-dt) * k2, f_p_grad_t_, c_scheme_)
            k4 = phi_dot(model, state_[:,:,ind_time], phi_[:,:,ind_time] + (-dt) * k3, f_p_grad_t_, c_scheme_)
            
            phi_[:,:, ind_time-1] = phi_[:,:,ind_time] + ((-dt) / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
    return phi_

def phi_dot(model, state_t_, phi_, f_p_grad_t_, c_scheme_):
    N = model.params['N']
    alpha = model.params['alpha']
    beta = model.params['beta']
    gamma = model.params['gamma']
    tau = model.params['tau']
    epsilon = model.params['epsilon']
    coupling_ = model.params['K_gl']
    c_mat_ = model.params['Cmat']
    no_output = len(model.output_vars)
    
    phi_dot_ = np.zeros(( N, no_output ))
    jac_h_t_ = np.zeros(( N, N, no_output, no_output ))
    jac_h_t_ = jac_h_t(jac_h_t_, state_t_, alpha, beta, gamma, tau, epsilon)
    
    for i_node in range(phi_dot_.shape[0]):
        for i_var in range(phi_dot_.shape[1]):
            coupling_prod = 0.
            jac_prod = 0.
                
            for j_node in range(phi_dot_.shape[0]):
                for j_var in range(phi_dot_.shape[1]):
                    jac_prod += jac_h_t_[j_node, i_node, j_var, i_var] * phi_[j_node, j_var]
                    coupling_prod += coupling_ * c_mat_[j_node, i_node] * c_scheme_[j_var, i_var] * phi_[j_node, j_var]   
            phi_dot_[i_node, i_var] += - jac_prod - coupling_prod - f_p_grad_t_[i_node, i_var]
            
    return phi_dot_
                

# block-diagonal jacobian matrix
@numba.njit
def jac_h_t(jac_h_t_, state_t_, alpha, beta, gamma, tau, epsilon):
    x1_ = state_t_[:,0]
    for ind_node in range(jac_h_t_.shape[0]):
        jac_h_t_[ind_node, ind_node, 0, 0] = -3. * alpha * x1_[ind_node]**2 + 2. * beta *x1_[ind_node] + gamma
        jac_h_t_[ind_node, ind_node, 0, 1] = -1.
        jac_h_t_[ind_node, ind_node, 1, 0] = 1. / tau
        jac_h_t_[ind_node, ind_node, 1, 1] = - epsilon / tau
    return jac_h_t_

# computation of g
def g(model, phi_, state_, target_, control_, u_mat_, u_scheme_):
    g_ = model.getZeroControl()
    grad_cost_e_ = cost.cost_energy_gradient(control_)
    grad_cost_s_ = cost.cost_sparsity_gradient1(model, control_)
    
    for i_time in range(g_.shape[2]):
        for i_node in range(g_.shape[0]):
            for i_var in range(g_.shape[1]):
                matProd = 0.
                for j_node in range(u_mat_.shape[0]):
                    for j_var in range(u_scheme_.shape[0]):
                        matProd += u_mat_[j_node, i_node] * u_scheme_[j_var, i_var] * phi_[j_node, j_var, i_time]
                        
                #print(g_.shape, grad_cost_e_.shape, grad_cost_s_.shape)
                g_[i_node, i_var, i_time] = matProd + grad_cost_e_[i_node, i_var, i_time] + grad_cost_s_[i_node, i_var, i_time]
    return g_


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