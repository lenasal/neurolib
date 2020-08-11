import logging
import numpy as np
import numba

from ..utils.collections import dotdict

costparams = dotdict({})
costparamsdefault = np.array([1., 1., 1.])

def getParams():
    if (len(costparams) == 0):
        setDefaultParams()
    return costparams.I_p, costparams.I_e, costparams.I_s

def setParams(I_p, I_e, I_s):
    print("set cost params")
    if (I_p < 0):
        logging.error("Cost parameter I_p smaller 0 not allowed, use default instead")
        costparams.I_p = costparamsdefault[0]
    else:
        costparams.I_p = I_p
    if (I_e < 0):
        logging.error("Cost parameter I_e smaller 0 not allowed, use default instead")
        costparams.I_e = costparamsdefault[1]
    else:
        costparams.I_e = I_e
    if (I_s < 0):
        logging.error("Cost parameter I_s smaller 0 not allowed, use default instead")
        costparams.I_s = costparamsdefault[2]
    else:
        costparams.I_s = I_s
    
def setDefaultParams():
    print("set default params")
    #costparams = dotdict({})
    costparams.I_p = costparamsdefault[0]
    costparams.I_e = costparamsdefault[1]
    costparams.I_s = costparamsdefault[2]

# cost function for precision
#@numba.njit
def f_cost_precision(state_, target_state_):
    # input:
    # state_: [N,dim_Model,t] dimensional array containing all nodes, state variables at all times
    # target_state: [N, dim_Model, t] dimensional array containing target state for all nodes,
    # all variables (time-independent) at all times
    # return
    # cost: [t] dimensional array containing cost for all times
    i_p, i_e, i_s = getParams()
    #print(i_p, i_e, i_s)
    cost =  np.zeros((target_state_.shape[2]))
    for ind_time in range(target_state_.shape[2]):
        for ind_node in range(target_state_.shape[0]):
            for ind_var in range(target_state_.shape[1]):
                cost[ind_time] += 0.5 * i_p * (state_[ind_node, ind_var, ind_time] - 
                                               target_state_[ind_node, ind_var, ind_time])**2.
    return cost

# gradient of cost function for precision at time t
#@numba.njit
def cost_precision_gradient_t(state_t_, target_state_t_):
    # state_t: [N,dim_Model] dimensional array containing all nodes and state variables for one specific point of time
    i_p, i_e, i_s = getParams()
    cost_gradient_ = np.zeros(( target_state_t_.shape[0], target_state_t_.shape[1] ))
    for ind_node in range(target_state_t_.shape[0]):
        for ind_var in range(target_state_t_.shape[1]):
                cost_gradient_[ind_node, ind_var] += i_p * (state_t_[ind_node, ind_var] - 
                                               target_state_t_[ind_node, ind_var])
    return cost_gradient_
    
# cost function for energy
#@numba.njit
def f_cost_energy(control_):
    # control_: [N,dim_Model,t] dimensional array containing control for all nodes, all state variables at all times
    # return cost: [t] dimensional array containing cost for all times
    i_p, i_e, i_s = getParams()
    cost =  np.zeros((control_.shape[2]))
    for ind_time in range(control_.shape[2]):
        for ind_node in range(control_.shape[0]):
            for ind_var in range(control_.shape[1]):
                cost[ind_time] += 0.5 * i_e * control_[ind_node, ind_var, ind_time]**2
    return cost
   
"""     
# gradient of cost function for energy at time t
@numba.njit
def cost_energy_gradient(control_, I_e = 1e-3):
    # state_t: [N,dim_Model] dimensional array containing all nodes and state variables for all times
    cost_gradient_ = np.zeros(( control_.shape[0], control_.shape[1], control_.shape[2]))
    for ind_time in range(control_.shape[2]):
        for ind_node in range(control_.shape[0]):
            for ind_var in range(control_.shape[1]):
                    cost_gradient_[ind_node, ind_var] = I_e * np.absolute(control_[ind_node, ind_var, ind_time])
    return cost_gradient_
"""

# gradient of cost function for energy
#@numba.njit
def cost_energy_gradient(model, control_):
    # state_t: [N,dim_Model] dimensional array containing all nodes and state variables for all times
    i_p, i_e, i_s = getParams()
    cost_gradient_ = model.getZeroState()
    for ind_time in range(control_.shape[2]):
        for ind_node in range(control_.shape[0]):
            for state_index in range(len(model.output_vars)):
                for control_index in range(len(model.control_input_vars)):
                    if (model.output_vars[state_index] in model.control_input_vars[control_index]):
                        cost_gradient_[ind_node, control_index, ind_time] = i_e * control_[ind_node, control_index, ind_time]
    return cost_gradient_

# cost function for sparsity
#@numba.njit
def f_cost_sparsity1(control_):
    i_p, i_e, i_s = getParams()
    cost =  np.zeros((control_.shape[2]))
    for ind_node in range(control_.shape[0]):
        for ind_var in range(control_.shape[1]):
            for ind_time in range(control_.shape[2]):
                cost[ind_time] += i_s * np.absolute(control_[ind_node, ind_var, ind_time])
    return cost

def f_cost_sparsity1_int(dt, control_):
    i_p, i_e, i_s = getParams()
    cost =  0.
    for ind_time in range(control_.shape[2]):
        for ind_node in range(control_.shape[0]):
            for ind_var in range(control_.shape[1]):
                cost += i_s * np.absolute(control_[ind_node, ind_var, ind_time]) * dt
    return cost

def f_cost_sparsity2_int(dt, control_):
    i_p, i_e, i_s = getParams()
    cost_ =  0.
    cost = 0.
    for ind_time in range(control_.shape[2]):
        sum_ = 0.
        for ind_node in range(control_.shape[0]):
            for ind_var in range(control_.shape[1]):
                sum_ += np.abs(control_[ind_node, ind_var, ind_time])
        cost_ += sum_**2. * dt
    cost = i_s * np.sqrt(cost_)
    return cost

def f_cost_sparsity3_int(dt, control_):
    i_p, i_e, i_s = getParams()
    cost_ =  0.
    cost = 0.
    for ind_node in range(control_.shape[0]):
        for ind_var in range(control_.shape[1]):
            cost_ = 0.
            for ind_time in range(control_.shape[2]):
                cost_ += control_[ind_node, ind_var, ind_time]**2 * dt
            cost += i_s * np.sqrt(cost_)
    return cost

# acquisition cost
def f_cost_sparsity4_int(dt, control_):
    i_p, i_e, i_s = getParams()
    cost_node =  np.zeros((control_.shape[0]))
    cost = 0.
    for ind_node in range(control_.shape[0]):
        if np.any(control_[ind_node, :,:] != 0.):
            cost_node[ind_node] = 1.
            cost += i_s 
    return cost

"""
# gradient of cost function for energy at time t
@numba.njit
def cost_sparsity_gradient1(control_, I_s = 1.):
    # state_t: [N,dim_Model] dimensional array containing all nodes and state variables for one specific point of time
    cost_gradient_ = np.zeros(( control_.shape[0], control_.shape[1], control_.shape[2]))
    for ind_time in range(control_.shape[2]):
        for ind_node in range(control_.shape[0]):
            for ind_var in range(control_.shape[1]):
                    cost_gradient_[ind_node, ind_var, ind_time] += I_s * np.sign(control_[ind_node, ind_var, ind_time])
    return cost_gradient_
"""

# gradient of cost function for energy at time t
#@numba.njit
def cost_sparsity_gradient1(model, control_):
    # state_t: [N,dim_Model] dimensional array containing all nodes and state variables for one specific point of time
    i_p, i_e, i_s = getParams()
    cost_gradient_ = model.getZeroState()
    for ind_time in range(control_.shape[2]):
        for ind_node in range(control_.shape[0]):
            for state_index in range(len(model.output_vars)):
                for control_index in range(len(model.control_input_vars)):
                    if (model.output_vars[state_index] in model.control_input_vars[control_index]):
                        cost_gradient_[ind_node, control_index, ind_time] += i_s * np.sign(control_[ind_node, control_index, ind_time])
    return cost_gradient_

# total cost
#@numba.njit
def f_cost(state_, target_state_, control_):
    i_p, i_e, i_s = getParams()
    cost =  np.zeros((control_.shape[2]))
    if not (i_p == 0.):
        c_precision_ = f_cost_precision(state_, target_state_)
        for ind_time in range(control_.shape[2]):
            cost[ind_time] += c_precision_[ind_time]
    if not (i_e == 0.):
        c_energy_ = f_cost_energy(control_)
        for ind_time in range(control_.shape[2]):
            cost[ind_time] += c_energy_[ind_time]
    if not (i_s == 0.):
        c_sparsity_ = f_cost_sparsity1(control_)
        for ind_time in range(control_.shape[2]):
            cost[ind_time] += c_sparsity_[ind_time]

    return cost

# integrated cost
#@numba.njit
def f_int(dt, cost_, start_t_ = -1, stop_t_ = -1):
    # cost_: [t] dimensional array containing cost for all times
    # return cost_int: integrated (total) cost
    if start_t_ == -1:
        start_ind = 0
    else:
        start_ind = start_t_
    if stop_t_ == -1:
        stop_ind = len(cost_)
    else:
        stop_ind = stop_t_
    
    cost_int = 0.
    for ind_t in range(start_ind, stop_ind):
        cost_int += cost_[ind_t] * dt
    return cost_int