import numpy as np
import logging
from timeit import default_timer as timer
from . import costFunctions as cost
from . import func_optimize as fo
np.set_printoptions(precision=8)


# control optimization

def A2(model, cntrl_, target_, max_iteration_, tolerance_, include_timestep_, start_step_, test_step_, max_control_, min_control_,
       t_sim_ = 100, t_sim_pre_ = 50, t_sim_post_ = 50, control_variables_ = [0,1], prec_variables_ = [0,1]):
                
    dt = model.params['dt']
    max_iteration_ = int(max_iteration_)
        
    startind_ = int(model.getMaxDelay() + 1)
    
    state_vars = model.state_vars
    init_vars = model.init_vars
        
    if (startind_ > 1):
        fo.adjust_shape_init_params(model, init_vars, startind_)
            
    t_pre_ndt = np.around(t_sim_pre_ / dt).astype(int)
    delay_state_vars_ = np.zeros(( model.params.N, len(state_vars), startind_ ))
    
    if ( startind_ > 1 and t_pre_ndt <= startind_ ):
        logging.error("Not possible to set up initial conditions without sufficient simulation time before control")
        return
    
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
    
    len_time = int(round(t_sim_/dt,1)+1)
    
    if (len_time <= include_timestep_):
        include_timestep_ = len_time
    else:
        logging.error("not implemented for less than full timesteps")
    
    best_control_ = cntrl_.copy()
    total_cost_ = np.zeros(( int(max_iteration_+1) ))
    runtime_ = np.zeros(( int(max_iteration_+1) ))
    runtime_start_ = timer()
    
    state_ = fo.updateState(model, best_control_)
    state0_ = state_.copy()
    
    N = model.params.N
    T = int(1 + np.around(t_sim_ / dt, 1))
    V = state_.shape[1]
    n_control_vars = len(model.control_input_vars)
    
    for i in range( int(max_iteration_) ):
            
        #cost_ = cost.f_cost(state_, target_, best_control_)
        #print("into cost ")
        #print(state_, target_, best_control_)
        total_cost_[i] = cost.f_int(N, n_control_vars, T, dt, state_, target_, best_control_, v_ = prec_variables_)
        
        if (i < 5 or i%20 == 0):
            print('RUN ', i, ', total integrated cost: ', total_cost_[i])

        delta_ = gf_dc(model, N, n_control_vars, T, best_control_, target_, include_timestep_, start_step_, test_step_,
                       max_control_, min_control_, startind_, delay_state_vars_, control_variables_, prec_variables_)
        
        #step_ = fo.step_size(model, N, V, T, dt, state_, target_, best_control_, delta_, start_step_, max_it_ = 1000,
        #                                     bisec_factor_ = 2., max_control_ = max_control_, min_control_ = min_control_,
        #                                     variables_ = prec_variables_, alg = "A2")
        #print("step = ", step_)
        
        best_control_ += delta_
        
        best_control_ = fo.setmaxcontrol(n_control_vars, best_control_, max_control_, min_control_)
        
        #cannot compute last entry
        #best_control_[:,:,-1] = best_control_[:,:,-2]
        #best_control_[:,:,0] = best_control_[:,:,1]

        state0_ = state_
        state_ = fo.updateState(model, best_control_)
        
        #print("best control = ", best_control_[0,control_variables_[0],:])
        
        runtime_[i] = timer() - runtime_start_
        
        u_diff_ = ( np.absolute(delta_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i+1
            break
        
        s_diff_ = ( np.absolute(state_ - state0_) < tolerance_ )
        if ( s_diff_.all() ):
            print("State only changes marginally.")
            #max_iteration_ = i+1
            #break
        
    state_ = fo.updateFullState(model, best_control_, state_vars)
    
    #for i in range(len(output_vars)):
    #    state_[:,i,:] = model[output_vars[i]][:,:]
    #cost_ = cost.f_cost(state_, target_, best_control_)
    total_cost_[max_iteration_] = cost.f_int(N, n_control_vars, T, dt, state_, target_, best_control_, v_ = prec_variables_)
    print('RUN ', max_iteration_, ', total integrated cost: ', total_cost_[max_iteration_])
    runtime_[max_iteration_] = timer() - runtime_start_
    
    improvement = 100.
    if total_cost_[0] != 0.:
        improvement = improvement = 100. - 100.*(total_cost_[max_iteration_]/total_cost_[0])
        
    print("Improved over ", max_iteration_, " iterations by ", improvement, " percent.")
            
    if (t_sim_pre_ < dt and t_sim_post_ < dt):
        return best_control_, state_, total_cost_, runtime_
    
    t_post_ndt = np.around(t_sim_post_ / dt).astype(int)
    
    state_post_ = 0.
    
    if (t_sim_post_ >= dt):
        
        if startind_ == 1:
            fo.update_init(model, init_vars, state_vars)
        else:
            fo.update_init_delayed(model, delay_state_vars_, init_vars, state_vars, t_post_ndt, startind_)
    
        model.params.duration = t_sim_post_
        control_post_ = model.getZeroControl()
        state_post_ = fo.updateFullState(model, control_post_, state_vars)
    
    model.params.duration = t_sim_ + t_sim_pre_ + t_sim_post_
    bc_ = model.getZeroControl()
    bs_ = model.getZeroFullState()
        
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    
    #print(i1, i2)
        
    bc_, bs_ = fo.set_pre_post(i1, i2, bc_, bs_, best_control_, state_pre_, state_, state_post_, model.state_vars,
                               model.params.a, model.params.b)
            
    return bc_, bs_, total_cost_, runtime_


def get_dir(model, N, V, T, dt, ind_node, ind_var, ind_time, state0, target0_, control0_, test_step_,
            prec_variables_ = [0,1]):
    dir_ = model.getZeroControl()
    dir_up_ = dir_.copy()
    dir_down_ = dir_.copy()
    
    dir_up_[ind_node, ind_var, ind_time+1] += 1.
    
    dir_down_[ind_node, ind_var, ind_time+1] -= 1.
    
    counter = 0
    maxcounter = 5
    
    cost0 = cost.f_int(N, V, T, dt, state0, target0_, control0_, v_ = prec_variables_)
        
    while (np.all(dir_ == 0.) and counter < maxcounter):
        #print("step up with direction ", dir_up_)
        step_up_ = fo.test_step(model, N, V, T, dt, state0, target0_, control0_, dir_up_, cost0, test_step_,
                                prec_variables_ = prec_variables_)
        #print("step down")
        step_down_ = fo.test_step(model, N, V, T, dt, state0, target0_, control0_, dir_down_, cost0, test_step_,
                                  prec_variables_ = prec_variables_)
        
        #print("test steps = ", step_up_, step_down_)
        
        if (step_up_[0] != 0. or step_down_[0] != 0.):
            if (step_down_ == 0. or step_up_[1] < step_down_[1]):
                dir_ = dir_up_.copy()
            elif (step_up_ == 0. or step_up_[1] > step_down_[1]):
                dir_ = dir_down_.copy()
            # both directions improvement
            else:
                if (counter == maxcounter - 1):
                    dir_ = dir_up_.copy()         
                test_step_ *= 10.
                counter += 1
        # both directions no improvement
        else:
            test_step_ *= 10.
            counter += 1
        
    return dir_
    


# Gradient of the cost function with respect to the control
def gf_dc(model, N, V, T, control_, target_, include_timestep_, start_step_, test_step_, max_control_, min_control_,
          startind_, delay_state_vars_, control_variables_, prec_variables_):
    
    dt = model.params['dt']

    delta_c = np.zeros((control_.shape))
    
    control0_ = control_.copy()
    
    duration_init = model.params['duration']
    duration_sim = model.params['duration']
    
    state0 = model.getZeroState()
    target0_ = target_.copy()

    control_input = model.control_input_vars
    init_vars = model.init_vars
    state_vars = model.state_vars 
    
    delay_state_vars0_ = delay_state_vars_.copy()
    
    IC_init = fo.get_init(model, init_vars, state_vars, startind_, delay_state_vars_)
    
    change_dur_ = False
        
    ##!!!!! -1 ???
    i_p, i_e, i_s = cost.getParams()
    #print(i_p, i_e, i_s)
    
    T_ = T
    
    cntrl_vars = []
    for ind_var in range(len(control_input)):
        if ind_var in control_variables_:
            cntrl_vars.append(ind_var)
        
   # print("cntrl_vars = ", cntrl_vars)
    
    # t = 0 does matter for rate control, not for current control
    if 2 in cntrl_vars:
        for ind_node in range(N):
            for ind_var in cntrl_vars:
                state0 = fo.updateState(model, control0_)
                dir_ = get_dir(model, N, V, T_, dt, ind_node, ind_var, -1, state0, target0_, control0_,
                               test_step_, prec_variables_)
                            
                if (dir_.any() != 0.):
                    start_st_ = fo.adapt_step(control0_, ind_node, ind_var, start_step_, dir_,
                                              max_control_[ind_var], min_control_[ind_var]) 
                    if not start_st_ == 0.:
                        #print("get step")
                        step_ = fo.step_size(model, N, V, T_, dt, state0, target0_, control0_, dir_,
                                             start_st_, max_it_ = 1000, bisec_factor_ = 2.,
                                             max_control_ = max_control_, min_control_ = min_control_,
                                             variables_ = prec_variables_, alg = "A2")
                
                        control0_[ind_node, ind_var, 0] += step_[0] * dir_[ind_node, ind_var, 0]
                        delta_c[ind_node, ind_var, 0] = step_[0] * dir_[ind_node, ind_var, 0]                    
    
    #print("control = ", control0_)
    
    #model.params['duration'] = 2. * dt
    #model.run(control=control0_[:, :, :3])
    #fo.update_delayed_state(model, delay_state_vars0_, state_vars, init_vars, startind_)
                
    model.params['duration'] = duration_init    
                    
    if i_s < 1e-12:
        for ind_time in range(control_.shape[2]-2):
            for ind_node in range(N):
                for ind_var in cntrl_vars:
                    
                    #print("no sparsity, ", ind_time, ind_node, ind_var)
                    
                    if (change_dur_):
                        change_dur_ = False
                        control0_ = control0_[:,:,1:].copy()
                        target0_ = target0_[:,:,1:].copy()
                        
                    state0 = fo.updateState(model, control0_)
                    #print("get dir ")
                    dir_ = get_dir(model, N, V, T_, dt, ind_node, ind_var, 0, state0, target0_, control0_,
                                   test_step_, prec_variables_)
                    #print(" dir = ", dir_)
                    
                    if (dir_.any() != 0.):
                        start_st_ = fo.adapt_step(control0_, ind_node, ind_var, start_step_, dir_,
                                                  max_control_[ind_var], min_control_[ind_var]) 
                        if not start_st_ == 0.:
                            #print("get step")
                            step_ = fo.step_size(model, N, V, T_, dt, state0, target0_, control0_, dir_, start_st_,
                                                 max_it_ = 1000, bisec_factor_ = 2., max_control_ = max_control_,
                                                 min_control_ = min_control_, variables_ = prec_variables_, alg = "A2")
                            
                            #print("step size = ", step_)
                    
                            control0_[ind_node, ind_var, 1] += step_[0] * dir_[ind_node, ind_var, 1]
                            delta_c[ind_node, ind_var, ind_time+1] = step_[0] * dir_[ind_node, ind_var, 1]                    
                        
            model.params['duration'] = 2. * dt
            model.run(control=control0_[:, :, :3])
            fo.update_delayed_state(model, delay_state_vars0_, state_vars, init_vars, startind_)
            
            duration_sim -= dt
            T_ -= 1
            model.params['duration'] = duration_sim
            change_dur_ = True       
    else:   
        for ind_time in range(control_.shape[2]-2):
            for ind_node in range(N):
                for ind_var in cntrl_vars:
                                            
                    #print("update state with control = ", control0_)
                    state0 = fo.updateState(model, control0_)
                   # print("get dir --------------------------------------------- ")
                    dir_ = get_dir(model, N, V, T, dt, ind_node, ind_var, ind_time, state0, target0_, control0_,
                                   test_step_, prec_variables_)
                    #print("dir ", dir_)
                    
                    if (dir_.any() != 0.):
                        #print("2")
                        start_st_ = fo.adapt_step(control0_, ind_node, ind_var, start_step_, dir_, max_control_[ind_var],
                                                  min_control_[ind_var]) 
                        if not start_st_ == 0.:
                            #print("3")
                            step_ = fo.step_size(model, N, V, T, dt, state0, target0_, control0_, dir_,
                                                 start_st_, max_it_ = 1000, bisec_factor_ = 2.,
                                                 max_control_ = max_control_, min_control_ = min_control_,
                                                 variables_ = prec_variables_, alg = "A2")
                            
                            #print("step size = ", step_)
                    
                            control0_[ind_node, ind_var, ind_time+1] += step_[0] * dir_[ind_node, ind_var, ind_time+1]
                            delta_c[ind_node, ind_var, ind_time+1] = step_[0] * dir_[ind_node, ind_var, ind_time+1]                    
             
    model.params['duration'] = duration_init
    
    fo.set_init(model, IC_init, init_vars, state_vars, startind_)
                
    return delta_c

