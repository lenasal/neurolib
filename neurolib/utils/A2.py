import numpy as np
import logging
from timeit import default_timer as timer
from . import costFunctions as cost
from . import func_optimize as fo
np.set_printoptions(precision=8)


# control optimization

def A2(model, cntrl_, target_, max_iteration_, tolerance_, include_timestep_, start_step_, test_step_, max_control_,
       t_sim_ = 100, t_sim_pre_ = 50, t_sim_post_ = 50):
            
    dt = model.params['dt']
    
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
        
    startind_ = int(model.getMaxDelay() + 1)
    
    state_vars = model.state_vars
    init_vars = model.init_vars
    
    if (startind_ > 1):
        fo.adjust_shape_init_params(model, init_vars, startind_)    
    
    t_pre_ndt = np.around(t_sim_pre_ / dt).astype(int)
    delay_state_vars_ = np.zeros(( model.params.N, len(state_vars), startind_ ))
    
    # simulate with duration t_sim_pre before start
    if (t_sim_pre_ >= dt):
        model.params['duration'] = t_sim_pre_
        control_pre_ = model.getZeroControl()
        state_pre_ = fo.updateState(model, control_pre_)
        
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
    
    output_vars = model.output_vars
    
    best_control_ = cntrl_.copy()
    total_cost_ = np.zeros(( int(max_iteration_+1) ))
    runtime_ = np.zeros(( int(max_iteration_+1) ))
    runtime_start_ = timer()
    
    
    state_ = fo.updateState(model, best_control_)
    state0_ = state_.copy()
    
    for i in range( int(max_iteration_) ):
            
        cost_ = cost.f_cost(state_, target_, best_control_)
        total_cost_[i] = cost.f_int(dt, cost_)
        print('RUN ', i, ', total integrated cost: ', total_cost_[i])

        delta_ = gf_dc1(model, best_control_, target_, include_timestep_, start_step_, test_step_, max_control_,
                       startind_, delay_state_vars_)
        best_control_ += delta_
        #best_control_[:,:,-1] = best_control_[:,:,-2]
        #best_control_[:,:,0] = best_control_[:,:,1]

        state0_ = state_
        state_ = fo.updateState(model, best_control_)
        
        runtime_[i] = timer() - runtime_start_
        
        u_diff_ = ( np.absolute(delta_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i+1
            break
        
        s_diff_ = ( np.absolute(state_ - state0_) < tolerance_ )
        if ( s_diff_.all() ):
            print("State only changes marginally.")
            max_iteration_ = i+1
            break
        
    model.run(control = best_control_)
    for i in range(len(output_vars)):
        state_[:,i,:] = model[output_vars[i]][:,:]
    cost_ = cost.f_cost(state_, target_, best_control_)
    total_cost_[max_iteration_] = cost.f_int(dt, cost_)
    print('RUN ', max_iteration_, ', total integrated cost: ', total_cost_[max_iteration_])
    runtime_[max_iteration_] = timer() - runtime_start_
    
    improvement = 100.
    if total_cost_[0] != 0.:
        improvement = improvement = 100 - int(100.*(total_cost_[max_iteration_]/total_cost_[0]))
        
    print("Improved over ", max_iteration_, " iterations by ", improvement, " percent.")
    
    if (t_sim_pre_ < dt and t_sim_post_ < dt):
        return best_control_, state_, total_cost_, runtime_
    
    if (t_sim_post_ > dt):
        
        for iv, sv in zip( range(len(init_vars)), range(len(state_vars)) ):
            if state_vars[sv] in init_vars[iv]:
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
        state_post_ = fo.updateState(model, control_post_)
    
    
    model.params.duration = t_sim_ + t_sim_pre_ + t_sim_post_
    bc_ = model.getZeroControl()
    bs_ = model.getZeroState()
        
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
        
    bc_, bs_ = fo.set_pre_post(i1, i2, bc_, bs_, best_control_, state_pre_, state_, state_post_)
            
    return bc_, bs_, total_cost_, runtime_


def get_dir(model, ind_node, ind_var, ind_time, state0, target0_, control0_, test_step_):
    dir_ = model.getZeroControl()
    
    dir_up_ = model.getZeroControl()
    dir_up_[ind_node, ind_var, 1] += 1.
    
    dir_down_ = model.getZeroControl()
    dir_down_[ind_node, ind_var, 1] -= 1.
    
    counter = 0
    maxcounter = 5
    
    while (np.all(dir_ == 0.) and counter < maxcounter):
        step_up_ = fo.test_step(model, state0, target0_, control0_, dir_up_, test_step_)
        step_down_ = fo.test_step(model, state0, target0_, control0_, dir_down_, test_step_)
        
        if (step_up_[0] != 0. or step_down_[0] != 0.):
            if (step_down_ == 0. or step_up_[1] < step_down_[1]):
                dir_ = dir_up_.copy()
            elif (step_up_ == 0. or step_up_[1] > step_down_[1]):
                dir_ = dir_down_.copy()
            # both directions improvement
            else:
                if (counter == maxcounter - 1):
                    dir_ = dir_up_.copy()         
                test_step_ *= 4.
                counter += 1
        # both directions no improvement
        else:
            test_step_ *= 4.
            counter += 1
        
    return dir_
    


# Gradient of the cost function with respect to the control
def gf_dc1(model, control_, target_, include_timestep_, start_step_, test_step_, max_control_, startind_, delay_state_vars_):
    
    N = model.params['N']
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
    
    IC_init = np.zeros( (N, len(init_vars), startind_) )
    delay_state_vars0_ = delay_state_vars_.copy()
    
    for ind_var in range(len(init_vars)):
        if ( type(model.params[init_vars[ind_var]]) == np.float64 or type(model.params[init_vars[ind_var]]) == float ):
            IC_init[:, ind_var, 0] = model.params[init_vars[ind_var]]
        elif len(model.params[init_vars[ind_var]][:].shape) == 2:
            if startind_ == 1:
                IC_init[:, ind_var, 0] = model.params[init_vars[ind_var]][:,0]
            else:
                IC_init[:, ind_var, :] = delay_state_vars0_[:, ind_var, :]
        else:
            IC_init[:, ind_var, 0] = model.params[init_vars[ind_var]][:]    
    
    change_dur_ = False
        
    ##!!!!! -1 ???
    for ind_time in range(control_.shape[2]-2):#2):
        for ind_node in range(N):
            for ind_var in range(len(control_input)):
                
                if (change_dur_):
                    change_dur_ = False
                    control0_ = control0_[:,:,1:].copy()
                    target0_ = target0_[:,:,1:].copy()
                    
                state0 = fo.updateState(model, control0_)
                dir_ = get_dir(model, ind_node, ind_var, ind_time, state0, target0_, control0_, test_step_)
                
                if (dir_.any() != 0.):
                    start_st_ = fo.adapt_step(control0_, ind_node, ind_var, start_step_, dir_, max_control_) 
                    if not start_st_ == 0.:
                        step_ = fo.step_size(model, state0, target0_, control0_, dir_, start_st_, max_it_ = 10000,
                                             bisec_factor_ = 2., max_control_=max_control_)
                
                        control0_[ind_node, ind_var, 1] += step_[0] * dir_[ind_node, ind_var, 1]
                        delta_c[ind_node, ind_var, ind_time+1] = step_[0] * dir_[ind_node, ind_var, 1]                    
                    
        model.params['duration'] = 2. * dt
        model.run(control=control0_[:, :, :3])
        fo.update_delayed_state(model, delay_state_vars0_, state_vars, init_vars, startind_)
        
        
        duration_sim -= dt
        model.params['duration'] = duration_sim
        change_dur_ = True
    
    model.params['duration'] = duration_init
    
    fo.set_init(model, IC_init, init_vars, state_vars, startind_)
       
    """
    for j_var in range(len(init_vars)):
        if model.params[init_vars[j_var]].ndim == 1:
            model.params[init_vars[j_var]][:] = IC_init[:, j_var, 0]
            
        else:
            if startind_ == 1:
                model.params[init_vars[j_var]][:,0] = IC_init[:, j_var, 0] 
            else: 
                model.params[init_vars[j_var]] = IC_init[:, j_var, :]
    """
                
    return delta_c

