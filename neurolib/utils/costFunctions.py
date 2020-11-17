import logging
import numpy as np
import numba
from numba.typed import List

from ..utils.collections import dotdict

costparams = dotdict({})
costparamsdefault = np.array([1., 1., 1.])
tolerance = 1e-16

def makeList(list_):
    l = List()
    for l0 in list_:
        l.append(l0)
    return l

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

###########################################################
# cost functions for precision
###########################################################

# gradient of cost function for precision at time t
# time interval for transition can be set by defining respective target state to -1.
def cost_precision_gradient_t(state_t_, target_state_t_):
    i_p, i_e, i_s = getParams()
    cost_gradient_ = numba_precision_gradient_t(i_p, state_t_, target_state_t_)
    return cost_gradient_

#@numba.njit
def numba_precision_gradient_t(i_p, state_t_, target_state_t_):
    cost_gradient_ = np.zeros(( target_state_t_.shape[0], target_state_t_.shape[1] ))
    for ind_node in range(target_state_t_.shape[0]):
        for ind_var in range(target_state_t_.shape[1]):
            if target_state_t_[ind_node, ind_var] == -1000:
                cost_gradient_[ind_node, ind_var] += 0.
            else:
                cost_gradient_[ind_node, ind_var] += i_p * (state_t_[ind_node, ind_var] - 
                                               target_state_t_[ind_node, ind_var])
    return cost_gradient_

def cost_precision_int(dt, i_p, state_, target_, va_):
    cost_int = numba_cost_precision_int(dt, i_p, state_, target_, var_ = va_ )
    return cost_int

#@numba.njit
def numba_cost_precision_int(dt, i_p, state_, target_state_, var_):
    cost =  0.
    for ind_time in range(target_state_.shape[2]):
        for ind_node in range(target_state_.shape[0]):
            for ind_var in var_:
                diff = np.abs(state_[ind_node, ind_var, ind_time] - target_state_[ind_node, ind_var, ind_time])
                if target_state_[ind_node, ind_var, ind_time] == -1000:
                    cost += 0.
                elif ( diff < tolerance ):
                    cost += 0.
                else:
                    cost += dt * 0.5 * i_p * diff**2.
    return cost
        
    
###########################################################
# cost functions for energy
###########################################################    
    
# cost function for energy
def f_cost_energy(i_e, control_, var_ = [0,1]):
    # control_: [N,dim_Model,t] dimensional array containing control for all nodes, all state variables at all times
    # return cost: [t] dimensional array containing cost for all times
    #i_p, i_e, i_s = getParams()
    #print("ie = ", i_e)
    cost =  numba_cost_energy(i_e, control_, vari_ = var_)
    return cost

#@numba.njit
def numba_cost_energy(i_e, control_, vari_ = [0,1]):
    cost =  np.zeros((control_.shape[2]))
    for ind_time in range(control_.shape[2]):
        for ind_node in range(control_.shape[0]):
            for ind_var in vari_:
                cost[ind_time] += 0.5 * i_e * control_[ind_node, ind_var, ind_time]**2
    return cost

# gradient of cost function for energy
def cost_energy_gradient(control_):
    # state_t: [N,dim_Model] dimensional array containing all nodes and state variables for all times
    i_p, i_e, i_s = getParams()
    cost_gradient_ = numba_energy_gradient(i_e, control_)
    return cost_gradient_

@numba.njit
def numba_energy_gradient(i_e, control_):
    cost_gradient_ = i_e * control_
    return cost_gradient_

def cost_energy_int(dt, i_e, control_, va_ = [0,1]):
    cost_ = f_cost_energy(i_e, control_, var_ = va_)
    #print("cost energy = ", cost_)
    return sum(cost_) * dt

@numba.njit
def control_energy_components(dt, control_):
    control_energy = np.zeros(( control_.shape[0], control_.shape[1] ))
    for ind_node in range(control_.shape[0]):
        for ind_var in range(control_.shape[1]):
            energy = 0.
            for ind_time in range(control_.shape[2]):
                energy += dt * control_[ind_node, ind_var, ind_time]**2
            control_energy[ind_node, ind_var] = np.sqrt(energy)
    return control_energy

###########################################################
# cost functions for sparsity
########################################################### 

# cost function for sparsity: simple absolute value
def cost_sparsity_gradient(dt, control_):
    i_p, i_e, i_s = getParams()
    control_energy = control_energy_components(dt, control_)
    cost_grad =  numba_cost_sparsity_gradient(i_s, control_, control_energy)
    return cost_grad

@numba.njit
def numba_cost_sparsity_gradient(i_s, control_, control_energy):
    cost_grad =  np.zeros(( control_.shape ))
    for ind_node in range(control_.shape[0]):
        for ind_var in range(control_.shape[1]):
            if control_energy[ind_node, ind_var] == 0.:
                cost_grad[ind_node, ind_var, :] = 0.
            else:
                cost_grad[ind_node, ind_var, :] = i_s * control_[ind_node, ind_var,:] / control_energy[ind_node, ind_var]
    return cost_grad

def f_cost_sparsity_int(dt, i_s, control_, va_):
    cost =  numba_cost_sparsity_int(i_s, dt, control_, var_ = va_)
    return cost

@numba.njit
def numba_cost_sparsity_int(i_s, dt, control_, var_):
    int_ =  0.
    for ind_node in range(control_.shape[0]):
        for ind_var in var_:
            cost = 0.
            for ind_time in range(control_.shape[2]):
                cost += (control_[ind_node, ind_var, ind_time])**2 * dt
            int_ += i_s * np.sqrt(cost)
    return int_

###########################################################
# total cost
########################################################### 


def f_cost(state_, target_state_, control_):
    i_p, i_e, i_s = getParams()
    cost =  numba_cost(i_p, i_e, i_s, state_, target_state_, control_)
    return cost

@numba.njit
def numba_cost(i_p, i_e, i_s, state_, target_state_, control_):
    cost =  np.zeros((control_.shape[2]))
    logging.error("not implemented")
    return cost
"""
    if not (i_p == 0.):
        c_precision_ = numba_cost_precision(i_p, state_, target_state_)
        #print("precision cost = ", c_precision_)
        for ind_time in range(control_.shape[2]):
            cost[ind_time] += c_precision_[ind_time]
    if not (i_e == 0.):
        c_energy_ = numba_cost_energy(i_e, control_)
        #print("energy cost = ", c_energy_)
        for ind_time in range(control_.shape[2]):
            cost[ind_time] += c_energy_[ind_time]
    #if not (i_s == 0.):
    #    logging.error("Sparsity cost not defined as time series")
        #c_sparsity_ = numba_cost_sparsity1(i_s, control_)
        #print("sparsity cost = ", c_sparsity_)
        #for ind_time in range(control_.shape[2]):
         #   cost[ind_time] += c_sparsity_[ind_time]
    #print("total cost = ", cost)
    return cost
"""

# integrated cost
#@numba.njit
def f_int(dt, state_, target_, control_, v_ = [0,1], start_t_ = -1, stop_t_ = -1):
    # cost_: [t] dimensional array containing cost for all times
    # return cost_int: integrated (total) cost
        
    var = makeList(v_)
    
    i_p, i_e, i_s = getParams()
    cost_prec, cost_energy, cost_sparsity = 0., 0., 0.
            
    if not (i_p == 0.):
        cost_prec = cost_precision_int(dt, i_p, state_, target_, va_ = var)
    if not (i_e == 0.):
        cost_energy = cost_energy_int(dt, i_e, control_, va_ = var)
    if not (i_s == 0.):
        cost_sparsity = f_cost_sparsity_int(dt, i_s, control_, va_ = var)
    
    #print("cost precision = ", cost_prec)
    #print("cost energy = ", cost_energy)
    #("cost sparsity = ", cost_sparsity)
    
    cost_int = cost_prec + cost_energy + cost_sparsity
    
    return cost_int