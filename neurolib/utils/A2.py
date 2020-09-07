import numpy as np
import logging
from timeit import default_timer as timer
from . import costFunctions as cost
from . import func_optimize as fo


# control optimization

def A2(model, cntrl_, target_, max_iteration_, tolerance_, include_timestep_, start_step_, test_step_, max_control_,
       t_sim_ = 100, t_sim_pre_ = 50, t_sim_post_ = 50):
            
    dt = model.params['dt']
    
    # run model with dt duration once to set delay matrix
    model.params['duration'] = dt
    model.run()
    
    Dmat = model.params.Dmat
    Dmat_ndt = np.around(Dmat / dt).astype(int)
        
    Dmat_ndt = np.around(Dmat / dt).astype(int)
        
    if model.name == "aln":
        ndt_de = np.around(model.params.de / dt).astype(int)
        ndt_di = np.around(model.params.di / dt).astype(int)
        max_global_delay = max(np.max(Dmat_ndt), ndt_de, ndt_di)
    else:
        max_global_delay = np.max(Dmat_ndt)
        
    startind_ = int(max_global_delay + 1)
    print("start ind = ", startind_)  
    
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
        state_pre_ = fo.updateState(model, control_pre_)
            
        for iv, sv in zip( range(len(init_vars)), range(len(state_vars)) ):
            if state_vars[sv] in init_vars[iv]:
                if model.params[init_vars[iv]].ndim == 2:
                    if startind_ == 1:
                        model.params[init_vars[iv]][:,0] = model.state[state_vars[sv]][:,-1]
                    else:
                        if (t_pre_ndt < startind_):
                            # delay larger than pre simulation time
                            #else:
                            delay_state_vars_[:, sv, :-1] = model.params[init_vars[iv]][:,1:]
                            delay_state_vars_[:, sv, -t_pre_ndt:] = model.state[state_vars[sv]][:,:]
                            model.params[init_vars[iv]][:,:] = delay_state_vars_[:, sv, :]
                        else:
                            model.params[init_vars[iv]] = model.state[state_vars[sv]][:,-startind_:]
                            delay_state_vars_[:, sv, :] = model.state[state_vars[sv]][:,-startind_:]
                else:
                    model.params[init_vars[iv]] = model.state[state_vars[sv]]
            else:
                logging.error("Initial and state variable labelling does not agree.")
    
    model.params['duration'] = t_sim_
    len_time = int(round(t_sim_/dt,1)+1)
    
    if (len_time <= include_timestep_):
        include_timestep_ = len_time
    else:
        logging.error("not implemented for less than full timesteps")
    
    output_vars = model.output_vars
    
    #print(fhn.params['duration']/fhn.params['dt'], state_.shape)
    best_control_ = cntrl_.copy()
    total_cost_ = np.zeros(( int(max_iteration_+1) ))
    runtime_ = np.zeros(( int(max_iteration_+1) ))
    runtime_start_ = timer()
    
    #print("try to update state with new initial values")
    
    state_ = fo.updateState(model, best_control_)
    state0_ = state_.copy()
    #print("state with initial guess = ", state0_)
    
    for i in range( int(max_iteration_) ):
            
        cost_ = cost.f_cost(state_, target_, best_control_)
        total_cost_[i] = cost.f_int(dt, cost_)
        print('RUN ', i, ', total integrated cost: ', total_cost_[i])
        #print("state = ", state_)
        #print("cost = ", len(cost_), cost_)
        #print("control : ", best_control_)

        delta_ = gf_dc1(model, best_control_, target_, include_timestep_, start_step_, test_step_, max_control_,
                       startind_, delay_state_vars_)
        best_control_ += delta_
        best_control_[:,:,-1] = best_control_[:,:,-2]  
        state0_ = state_
        state_ = fo.updateState(model, best_control_)
        
        runtime_[i] = timer() - runtime_start_
        
        u_diff_ = ( np.absolute(delta_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i+1
            #print("Improved over ", i, " iterations by ", 100 - int(100.*(total_cost_[i]/total_cost_[0])), " percent.")
            break
            #return best_control_, state_, total_cost_, runtime_
        
        s_diff_ = ( np.absolute(state_ - state0_) < tolerance_ )
        if ( s_diff_.all() ):
            print("State only changes marginally.")
            max_iteration_ = i+1
            #print("Improved over ", i, " iterations by ", 100 - int(100.*(total_cost_[i]/total_cost_[0])), " percent.")
            break
            #return best_control_, state_, total_cost_, runtime_
        
        
        
    model.run(control = best_control_)
    for i in range(len(output_vars)):
        state_[:,i,:] = model[output_vars[i]][:,:]
    cost_ = cost.f_cost(state_, target_, best_control_)
    #print(cost_)
    total_cost_[max_iteration_] = cost.f_int(dt, cost_)
    print('RUN ', max_iteration_, ', total integrated cost: ', total_cost_[max_iteration_])
    runtime_[max_iteration_] = timer() - runtime_start_
    
    if total_cost_[0] == 0.:
        improvement = 100
    else:
        improvement = 100 - int(100.*(total_cost_[max_iteration_]/total_cost_[0]))
        
    print("Improved over ", max_iteration_, " iterations by ", improvement, " percent.")
    
    if (t_sim_pre_ < dt and t_sim_post_ < dt):
        return best_control_, state_, total_cost_, runtime_
    
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
                    model.params[init_vars[iv]][:] = model.state[state_vars[sv]][:]
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
                if bs_[n,v,i1] != state_[n,v,0]:
                    logging.error("Problem in initial value trasfer")
        bs_[:,:,i1:-i2] = state_[:,:,:]
        bs_[:,:,-i2:] = state_post_[:,:,:]
    elif (i2 == 0 and i1 != 0):
        bc_[:,:,i1:] = best_control_[:,:,:]
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        for n in range(bs_.shape[0]):
            for v in range(bs_.shape[1]):
                if bs_[n,v,i1] != state_[n,v,0]:
                    logging.error("Problem in initial value trasfer for ", output_vars[v])
        bs_[:,:,i1:] = state_[:,:,:]
    elif (i2 != 0 and i1 == 0):
        bc_[:,:,:-i2] = best_control_[:,:,:]
        bs_[:,:,:-i2] = state_[:,:,:]
        bs_[:,:,-i2:] = state_post_[:,:,:]
    else:
        bc_[:,:,:] = best_control_[:,:,:]
        bs_[:,:,:] = state_[:,:,:]
            
    return bc_, bs_, total_cost_, runtime_



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
        if len(model.params[init_vars[ind_var]][:].shape) == 2:
            if startind_ == 1:
                IC_init[:, ind_var, 0] = model.params[init_vars[ind_var]][:,0]
            else:
                IC_init[:, ind_var, :] = delay_state_vars0_[:, ind_var, :]
        else:
            IC_init[:, ind_var, 0] = model.params[init_vars[ind_var]][:]    
    
    change_dur_ = False
    
    ##!!!!! -1 ???
    for ind_time in range(control_.shape[2]-1):
        for ind_node in range(N):
            for ind_var in range(len(control_input)):
                
                print_ = False
                if (ind_time in [-1]):
                    print_ = True
                    
                if (print_):
                    print('time, node, var: ', ind_time, ind_node, ind_var) 
                
                if (change_dur_):
                    change_dur_ = False
                    control0_ = control0_[:,:,1:].copy()
                    target0_ = target0_[:,:,1:].copy()
                    
                state0 = fo.updateState(model, control0_)
                #print("state = ", state0)
                #print("target = ", target0_)
                
                dir_ = model.getZeroControl()
                dir_up_ = model.getZeroControl()
                dir_up_[ind_node, ind_var, 0] += 1.
                
                step_up_ = fo.test_step(model, state0, target0_, control0_, dir_up_, test_step_)
                
                dir_down_ = model.getZeroControl()
                dir_down_[ind_node, ind_var, 0] -= 1.
                
                step_down_ = fo.test_step(model, state0, target0_, control0_, dir_down_, test_step_)
                
                if (print_):
                    print("step up = ", step_up_)
                    print("step down = ", step_down_)
                
                if (step_up_[0] != 0. or step_down_[0] != 0.):
                    if (step_down_ == 0. or step_up_[1] < step_down_[1]):
                        dir_ = dir_up_
                        #if (print_):
                        #    print("go up")
                    elif (step_up_ == 0. or step_up_[1] > step_down_[1]):
                        dir_ = dir_down_
                        #if (print_):
                        #    print("go down")
                    else:
                        print("something forgotten: step up, step down = ", step_up_, step_down_)
                        
                    step_ = fo.step_size(model, state0, target0_, control0_, dir_, start_step_, max_iteration_ = 100, bisec_factor_ = 2., max_control_=max_control_)
                    
                    #if (print_):
                    #    print("found stepsize = ", step_)
                
                    control0_[ind_node, ind_var, 0] += step_[0] * dir_[ind_node, ind_var, 0]
                    delta_c[ind_node, ind_var, ind_time] = step_[0] * dir_[ind_node, ind_var, 0]
        
        # simulate one time step for initial conditions
        model.params['duration'] = dt
        model.run(control=control0_[:, :, :2])
                
        for sv, iv in zip( range(len(state_vars)), range(len(init_vars)) ):
            if (state_vars[sv] in init_vars[iv]):
                if model.params[init_vars[iv]].ndim == 2:
                    if startind_ == 1:
                        model.params[init_vars[iv]][:,0] = model.state[state_vars[sv]][:,-1]
                    else:
                        delay_state_vars0_[:, sv, :-1] = delay_state_vars0_[:, sv, 1:]
                        delay_state_vars0_[:, sv, -1] = model.state[state_vars[sv]][:,-1]
                        model.params[init_vars[iv]][:,:] = delay_state_vars0_[:, sv, :]
                else:
                    model.params[init_vars[iv]] = model.state[state_vars[sv]]
            else:
                logging.error("Initial and state variable labelling does not agree.")
        duration_sim -= dt
        model.params['duration'] = duration_sim
        change_dur_ = True
    
    model.params['duration'] = duration_init
    
    for j_var in range(len(init_vars)):
        if model.params[init_vars[j_var]].ndim == 1:
            model.params[init_vars[j_var]][:] = IC_init[:, j_var, 0]
            
        else:
            if startind_ == 1:
                model.params[init_vars[j_var]][:,0] = IC_init[:, j_var, 0] 
            else: 
                model.params[init_vars[j_var]] = IC_init[:, j_var, :]
    
                
    return delta_c