import numpy as np
from timeit import default_timer as timer
from . import costFunctions as cost


def test_step(model, state_, target_, control_, dir_, test_step_ = 1e-12):
    dt = model.params['dt']
    cost0_int_ = cost.f_int(dt, cost.f_cost(state_, target_, control_))
    
    test_control_ = control_ + test_step_ * dir_
    state1_ = updateState(model, test_control_)
    cost1_int_ = cost.f_int(dt, cost.f_cost(state1_, target_, test_control_))
    #print("test step size computation : ------ step size, cost1, cost0 : ", test_step_, cost1_int_, cost0_int_)
        
    if (cost1_int_ < cost0_int_):
        return test_step_, cost1_int_
    else:
        return 0., cost0_int_

def step_size(model, state_, target_, control_, dir_, start_step_ = 20., max_iteration_ = 1000,
              bisec_factor_ = 2., max_control_ = 20.):
    #if (max_iteration_ == 1):
    #    print("2")
    dt = model.params['dt']
    cost0_ = cost.f_cost(state_, target_, control_)
    cost0_int_ = cost.f_int(dt, cost0_)
    cost_min_int_ = cost0_int_
    step_ = start_step_
    step_min_ = step_
          
    for i in range(max_iteration_):
        #if (max_iteration_ == 1):
        #    print("i = ", i)
        test_control_ = control_ + step_ * dir_
        
        # include maximum control value to assure no divergence
        if ( np.amax(np.absolute(test_control_)) > max_control_):
            if (i < max_iteration_-1):
                #print("too big control")
                step_ /= bisec_factor_
                continue
            else:
                print("control too big, but no further iteration")
                return 0., cost0_int_
            
        state1_ = updateState(model, test_control_)
        cost1_ = cost.f_cost(state1_, target_, test_control_)
        cost1_int_ = cost.f_int(dt, cost1_)
        
   
        if (cost1_int_ < cost_min_int_):
            #print("found step = ", step_, " with cost1, cost0 : ", cost1_int_, cost0_int_)
            cost_min_int_ = cost1_int_
            step_min_ = step_
        # return smallest step size before cost is increasing again
        elif (cost1_int_ >= cost_min_int_ and cost_min_int_ < cost0_int_):
            #print("step size for minimal cost: ", step_min_)
            return step_min_, cost_min_int_
        
        if (i == max_iteration_-1):
            if (max_iteration_ != 1):
                print(" max iteration reached, step size = ", step_)
            #else:
                #plt.plot(state1_[0,0,:], state1_[0,1,:])
               # plt.show()
            return 0., cost0_int_
        step_ /= bisec_factor_


def updateState(model, control_):
    # set initial conditions once in other function
    state1_ = model.getZeroState()
    output_vars = model.output_vars
    model.run(control = control_)    
    for i in range(len(output_vars)):
        state1_[:,i,:] = model[output_vars[i]][:,:]
    return state1_

# Gradient of the cost function with respect to the control
def gf_dc(model, control_, target_, include_timestep_, start_step_, test_step_, max_control_):
    
    N = model.params['N']
    dt = model.params['dt']

    delta_c = np.zeros((control_.shape))
    
    control0_ = control_.copy()
    #control1_ = control0_.copy()
    #control2_ = control0_.copy()
    
    duration_init = model.params['duration']
    duration_sim = include_timestep_ * dt
    model.params['duration'] = duration_sim
    
    state0 = model.getZeroState()
   # state1 = state0.copy()
    #state2 = state0.copy()
    
    #input_vars = model.input_vars
    control_input = model.control_input_vars
    init_vars = model.init_vars
    state_vars = model.state_vars
    #output_vars = model.output_vars
    
    IC_init = np.zeros( (N, len(init_vars)) )
    #IC = IC_init.copy()
    #print(init_vars, model.params[init_vars[0]])
    for ind_var in range(len(init_vars)):
        #print(ind_var)
        #print(model.params[init_vars[ind_var]][:])
        #IC_init.append(model.params[init_vars[ind_var]][:])
        IC_init[:, ind_var] = model.params[init_vars[ind_var]][:]
        
    
    for ind_time in range(control_.shape[2]-1):
        for ind_node in range(N):
            for ind_var in range(len(control_input)):
                
                print_ = False
                if (ind_time in range(0,-1,1)):
                    print_ = True
                    
                if (print_):
                    print('time, node, var: ', ind_time, ind_node, ind_var)
                    #print('incl timestep : ', include_timestep_)
                    #print("control0 : ", control0_[:, :, :ind_time+include_timestep_+1])
                
                if (ind_time + include_timestep_ + 1 > control_.shape[2] and ind_node == 0 and ind_var == 0):
                    include_timestep_-= 1
                    #if (print_):
                    #   print("lower duration for rest of simulation to ", include_timestep_*model.params['dt'])
                    #    print("lower included timestep for rest of simulation to ", include_timestep_)  
                    duration_sim = include_timestep_*dt
                    model.params['duration'] = duration_sim
                    state0 = model.getZeroState()
                    #state1 = state0.copy()
                    #state2 = state0.copy()

                #model.run(control = control0_[:, :, ind_time:ind_time+include_timestep_+1])
                
                #for i in range(len(output_vars)):
                    #state0[:,i,:] = model[output_vars[i]][:,:]  
                    
                state0 = updateState(model, control0_[:, :, ind_time:ind_time+include_timestep_+1])
                
                #if (print_):
                #    print("state0 = ", state0[:,:,:])
                #    cost0 = cost.f_cost(state0[:,:,:], target_[:,:,ind_time:ind_time+include_timestep_+1], control0_[:, :, ind_time:ind_time+include_timestep_+1])     
                #cost0_int = cost.f_int(dt, cost0, 0, include_timestep_+1)
                
                dir_ = model.getZeroControl()
                dir_up_ = model.getZeroControl()
                dir_up_[ind_node, ind_var, 0] += 1.
                
                step_up_ = test_step(model, state0, target_[:,:,ind_time:ind_time+include_timestep_+1],
                                     control0_[:, :, ind_time:ind_time+include_timestep_+1], dir_up_, test_step_)
                
                dir_down_ = model.getZeroControl()
                dir_down_[ind_node, ind_var, 0] -= 1.
                
                step_down_ = test_step(model, state0, target_[:,:,ind_time:ind_time+include_timestep_+1],
                                     control0_[:, :, ind_time:ind_time+include_timestep_+1], dir_down_, test_step_)
                
                if (print_):
                    print("state 0 = ", state0)
                    #control1_ = control0_[:, :, ind_time:ind_time+include_timestep_+1] + dir_up_
                    #state1_ = updateState(model, control1_)
                    #control2_ = control0_[:, :, ind_time:ind_time+include_timestep_+1] + dir_down_
                    #state2_ = updateState(model, control2_)
                    #cost0 = cost.f_cost(state0[:,:,:], target_[:,:,ind_time:ind_time+include_timestep_+1], control0_[:, :, ind_time:ind_time+include_timestep_+1])
                    #cost1 = cost.f_cost(state1_[:,:,:], target_[:,:,ind_time:ind_time+include_timestep_+1], control1_[:, :, ind_time:ind_time+include_timestep_+1])
                    #cost2 = cost.f_cost(state2_[:,:,:], target_[:,:,ind_time:ind_time+include_timestep_+1], control2_[:, :, ind_time:ind_time+include_timestep_+1])
                    #print("cost 0 ", cost0)
                    #print("cost 1 ", cost1)
                    #print("cost 2 ", cost2)
                
                if (print_):
                    print("step up = ", step_up_)
                    print("step down = ", step_down_)
                
                if (step_up_[0] != 0. or step_down_[0] != 0.):
                    if (step_down_ == 0. or step_up_[1] < step_down_[1]):
                        dir_ = dir_up_
                        if (print_):
                            print("go up")
                    elif (step_up_ == 0. or step_up_[1] > step_down_[1]):
                        dir_ = dir_down_
                        if (print_):
                            print("go down")
                    else:
                        print("something forgotten: step up, step down = ", step_up_, step_down_)
                        
                    step_ = step_size(model, state0, target_[:,:,ind_time:ind_time+include_timestep_+1], control0_[:, :, ind_time:ind_time+include_timestep_+1], 
                                      dir_, start_step_, max_iteration_ = 100, bisec_factor_ = 2., max_control_=max_control_)
                    
                    if (print_):
                        print("found stepsize = ", step_)
                
                    control0_[ind_node, ind_var, ind_time] += step_[0]*dir_[ind_node, ind_var, 0]
                    delta_c[ind_node, ind_var, ind_time] = step_[0]*dir_[ind_node, ind_var, 0]
                    
                    if (print_):
                        state0 = updateState(model, control0_[:, :, ind_time:ind_time+include_timestep_+1])
                        print("updated state 0 = ", state0)
                        #cost0 = cost.f_cost(state0[:,:,:], target_[:,:,ind_time:ind_time+include_timestep_+1], control0_[:, :, ind_time:ind_time+include_timestep_+1])
                        #print("updated cost = ", cost0)
                    
                    #control1_ = control0_.copy()
                    #control2_ = control0_.copy()
                
                
                """
                control1_[ind_node, ind_var, ind_time] += step_size_
                state1 = updateState(model, control1_[:, :, ind_time:ind_time+include_timestep_+1])
                #if (print_):
                #    print("state1 = ", state1[:,:,:])

                cost1 = cost.f_cost(state1[:,:,:], target_[:,:,ind_time:ind_time+include_timestep_+1], control1_[:, :, ind_time:ind_time+include_timestep_+1])     
                cost1_int = cost.f_int(dt, cost1, 0, include_timestep_+1)
                
                #up_ = step_size_ * (cost1_int - cost0_int)/cost0_int
                up_ = np.sign(cost1_int - cost0_int)*step_size_ #((cost1_int - cost0_int)) * step_size_

                control2_[ind_node, ind_var, ind_time] -= step_size_
                state2 = updateState(model, control2_[:, :, ind_time:ind_time+include_timestep_+1])
                #if (print_):
                    #print("state2 = ", state2[:,:,:])
                cost2 = cost.f_cost(state2[:,:,:], target_[:,:,ind_time:ind_time+include_timestep_+1], control2_[:, :, ind_time:ind_time+include_timestep_+1])     
                cost2_int = cost.f_int(dt, cost2, 0, include_timestep_+1)
                
                #down_ =  step_size_ * (cost2_int - cost0_int)/cost0_int
                down_ = np.sign(cost2_int - cost0_int)*step_size_ #((cost2_int - cost0_int)) * step_size_
                
                if (print_):
                    #print("cost 1 = ", len(cost1), cost1)
                    #print("cost 2 = ", len(cost2), cost2)
                    print("cost 0 int = ", cost0_int)
                    #print("cost 0 = ", len(cost0), cost0)
                    #print("control 0 = ", control0_)
                    print("cost 1 int = ", cost1_int)
                    #print("control 1 = ", control1_)
                    print("cost 2 int = ", cost2_int)
                    #print("control 2 = ", control2_)
                    
                dir_ = model.getZeroControl()
                    
                if (up_ < 0 and down_ >= 0):
                    # choose upward direction
                   # step_ = getstep()
                    if (print_):
                        print('upwards by ', - max(-max_increment_, up_))
                    delta_c[ind_node, ind_var, ind_time] = - max(-max_increment_, up_) #up_*step_size_
                    dir_[ind_node, ind_var, 0] = - up_/step_size_
                elif (up_ >= 0 and down_ < 0):
                    # choose downward direction
                    if (print_):
                        print('downwards by ', max(-max_increment_, down_))
                    delta_c[ind_node, ind_var, ind_time] = max(-max_increment_, down_)
                    dir_[ind_node, ind_var, 0] = down_/step_size_
                elif (up_ < 0 and down_ < 0):
                    # both good, choose better option
                    #if (print_):
                        #print('both good')
                    if (up_ < down_):
                        # upward
                        delta_c[ind_node, ind_var, ind_time] = - max(-max_increment_, up_)
                        dir_[ind_node, ind_var, 0] = - up_/step_size_
                    else:
                        # choose downward direction
                        delta_c[ind_node, ind_var, ind_time] = max(-max_increment_, down_)
                        dir_[ind_node, ind_var, ind_time] = down_/step_size_
                
                step_ = step_size(model, state0, target_[:,:,ind_time:ind_time+include_timestep_+1], control0_[:, :, ind_time:ind_time+include_timestep_+1], dir_, start_step_ = 20., max_iteration_ = 100,
              bisec_factor_ = 2., max_control_ = 20.)
                if (print_):
                    print("found stepsize = ", step_)
                
                #control0_[ind_node, ind_var, ind_time] += delta_c[ind_node, ind_var, ind_time]
                control0_[ind_node, ind_var, ind_time] += step_[0]*dir_[ind_node, ind_var, 0]
                delta_c[ind_node, ind_var, ind_time] = step_[0]*dir_[ind_node, ind_var, 0]
                
                
                control1_ = control0_.copy()
                control2_ = control0_.copy()
                """
                #if (print_):
                    #print("control 0 = ", control0_)
                    #print("control 1 = ", control1_)
                    #print("control 2 = ", control2_)
                
        # simulate one time step for initial conditions
        if (ind_time < control_.shape[2] - 1):
            #if(print_):
                #print("simulate for one timestep")
                #print("shape = ", control0_[:, :, ind_time:ind_time + 2].shape)
            model.params['duration'] = dt
            model.run(control=control0_[:, :, ind_time:ind_time + 2])
            
            for sv in state_vars:
                for iv in init_vars:
                    if (sv in iv):
                        #print(sv)
                        #if (print_):
                            #print("setting initial value ", iv, " from ", sv)
                        if (model.state[sv].ndim == 2):
                            #if (print_):
                            #    print("setting initial value ", iv, " from ", sv)
                            #    print(model.params[iv][0])
                            #    print(model.state[sv])
                            #print( model.params[iv][:], [model.state[sv][:, 0]])
                            model.params[iv][:] = model.state[sv][:, 0][0]         #np.array()
                        else:
                            model.params[iv] = model.state[sv]
                            #if (print_): 
                            #    print("setting initial value ", iv, model.params[iv], " from ", sv, model.state[sv])
                            #    print(model.params[iv])
                            #    print(model.state[sv])
            model.params.duration = duration_sim
                    
            
            #for j_var in range(len(state_vars)):
            #    print("state variable: ", state_vars[j_var], model.state[state_vars[j_var]][:])
            #    IC[:,j_var] = model.state[state_vars[j_var]][:]
                
            #model.params['duration'] = duration_sim
            #for j_var in range(len(init_vars)):
            #    model.params[init_vars[j_var]][:] = IC[:,j_var]  
        
            #if (print_):
            #    print(delta_c)
    
    model.params['duration'] = duration_init
    for j_var in range(len(init_vars)):
        model.params[init_vars[j_var]][:] = IC_init[:,j_var] 
                
    return delta_c


# control optimization

def A2(model, cntrl_, target_, max_iteration_, tolerance_, include_timestep_, start_step_, test_step_, max_control_):
        
    dt = model.params['dt']
    len_time = int(round(model.params['duration']/dt,1)+1)
    
    if (len_time < include_timestep_):
        include_timestep_ = len_time
    
    output_vars = model.output_vars
    
    #print(fhn.params['duration']/fhn.params['dt'], state_.shape)
    best_control_ = cntrl_.copy()
    total_cost_ = np.zeros(( int(max_iteration_+1) ))
    runtime_ = np.zeros(( int(max_iteration_+1) ))
    runtime_start_ = timer()
    
    state_ = updateState(model, best_control_)
    state0_ = state_.copy()
    
    for i in range( int(max_iteration_) ):
            
        cost_ = cost.f_cost(state_, target_, best_control_)
        total_cost_[i] = cost.f_int(dt, cost_)
        print('RUN ', i, ', total integrated cost: ', total_cost_[i])
        #print("state = ", state_)
        #print("cost = ", len(cost_), cost_)
        #print("shape control : ", best_control_.shape)

        delta_ = gf_dc(model, best_control_, target_, include_timestep_, start_step_, test_step_, max_control_)
        best_control_ += delta_
        best_control_[:,:,-1] = best_control_[:,:,-2]   
        state_ = updateState(model, best_control_)
        
        runtime_[i] = timer() - runtime_start_
        
        u_diff_ = ( np.absolute(delta_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            print("Improved over ", i, " iterations by ", 100 - int(100.*(total_cost_[i]/total_cost_[0])), " percent.")
            return best_control_, state_, total_cost_, runtime_
        
        s_diff_ = ( np.absolute(state_ - state0_) < tolerance_ )
        if ( s_diff_.all() ):
            print("State only changes marginally.")
            print("Improved over ", i, " iterations by ", 100 - int(100.*(total_cost_[i]/total_cost_[0])), " percent.")
            return best_control_, state_, total_cost_, runtime_
        
        
        
    model.run(control = best_control_)
    for i in range(len(output_vars)):
        state_[:,i,:] = model[output_vars[i]][:,:]
    cost_ = cost.f_cost(state_, target_, best_control_)
    #print(cost_)
    total_cost_[-1] = cost.f_int(dt, cost_)
    print('RUN ', max_iteration_, ', total integrated cost: ', total_cost_[-1])
    runtime_[-1] = timer() - runtime_start_
        
    print("Improved over ", max_iteration_, " iterations by ", 100 - int(100.*(total_cost_[-1]/total_cost_[0])), " percent.")
            
    return best_control_, state_, total_cost_, runtime_