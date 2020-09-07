import numpy as np
from . import costFunctions as cost



def updateState(model, control_):
    # set initial conditions once in other function
    state1_ = model.getZeroState()
    output_vars = model.output_vars
    model.run(control = control_)    
    for i in range(len(output_vars)):
        state1_[:,i,:] = model[output_vars[i]][:,:]
    return state1_


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
    
