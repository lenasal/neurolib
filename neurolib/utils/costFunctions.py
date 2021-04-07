import logging
import numpy as np
import numba
from numba.typed import List

from ..utils.collections import dotdict

costparams = dotdict({})
costparamsdefault = np.array([1., 1., 1.])
tolerance = 1e-16

#@numba.njit
def makeList(list_):
    l = List()
    for l0 in list_:
        l.append(l0)
    return l

def getParams():
    if (len(costparams) == 0):
        logging.warn("Cost parameters not found, set default values.")
        setDefaultParams()
    return costparams.I_p, costparams.I_e, costparams.I_s

def setParams(I_p, I_e, I_s):
    print("set cost params: ", I_p, I_e, I_s)
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
def cost_precision_gradient_t(N, V_target, state_t_, target_state_t_, i_p):
    cost_gradient_ = numba_precision_gradient_t(N, V_target, i_p, state_t_, target_state_t_)
    return cost_gradient_

@numba.njit
def numba_precision_gradient_t(N, V_target, i_p, state_t_, target_state_t_):
    cost_gradient_ = np.zeros(( N, V_target ))
    for ind_node in range(N):
        for ind_var in range(V_target):
            if target_state_t_[ind_node, ind_var] == -1000:
                cost_gradient_[ind_node, ind_var] += 0.
            else:
                cost_gradient_[ind_node, ind_var] += i_p * (state_t_[ind_node, ind_var] - 
                                               target_state_t_[ind_node, ind_var])
    return cost_gradient_

def cost_precision_int(N, T, dt, i_p, state_, target_, va_):
    cost_int = numba_cost_precision_int(N, T, dt, i_p, state_, target_, var_ = va_ )
    return cost_int

@numba.njit
def numba_cost_precision_int(N, T, dt, i_p, state_, target_state_, var_):
    cost =  0.
    for ind_time in range(T):
        for ind_node in range(N):
            for ind_var in var_:
                diff = np.abs(state_[ind_node, ind_var, ind_time] - target_state_[ind_node, ind_var, ind_time])
                if target_state_[ind_node, ind_var, ind_time] == -1000:
                    cost += 0.
                elif ( diff < tolerance ):
                    cost += 0.
                else:
                    cost += dt * 0.5 * i_p * diff**2.
    return cost

def cost_precision_node(N, T, dt, i_p, state_, target_, va_):
    var = makeList(va_)
    cost_int = numba_cost_precision_node(N, T, dt, i_p, state_, target_, var_ = var )
    return cost_int

@numba.njit
def numba_cost_precision_node(N, T, dt, i_p, state_, target_state_, var_):
    cost =  np.zeros(( N, 2 ))
    for ind_node in range(N):
        for ind_var in var_:
            for ind_time in range(T):
                diff = np.abs(state_[ind_node, ind_var, ind_time] - target_state_[ind_node, ind_var, ind_time])
                if target_state_[ind_node, ind_var, ind_time] == -1000:
                    cost[ind_node, ind_var] += 0.
                elif ( diff < tolerance ):
                    cost[ind_node, ind_var] += 0.
                else:
                    cost[ind_node, ind_var] += dt * 0.5 * i_p * diff**2.
    return cost

###########################################################
# cost functions for precision
###########################################################

# gradient of cost function for precision at time t
# time interval for transition can be set by defining respective target state to -1.
def cost_precision_gradient_t2(N, V_target, T, t_, state_t_, target_state_t_, i_p):
    cost_gradient_ = numba_precision_gradient_t2(N, V_target, T, t_, i_p, state_t_, target_state_t_)
    return cost_gradient_

@numba.njit
def numba_precision_gradient_t2(N, V_target, T, t_, i_p, state_t_, target_state_t_):
    cost_gradient_ = np.zeros(( N, V_target ))
    for ind_node in range(N):
        for ind_var in range(V_target):
            if target_state_t_[ind_node, ind_var] == -1000:
                cost_gradient_[ind_node, ind_var] += 0.
            else:
                cost_gradient_[ind_node, ind_var] += i_p * t_ * (state_t_[ind_node, ind_var] - 
                                               target_state_t_[ind_node, ind_var]) / T
    return cost_gradient_

def cost_precision_int2(N, T, dt, i_p, state_, target_, va_):
    cost_int = numba_cost_precision_int2(N, T, dt, i_p, state_, target_, var_ = va_ )
    return cost_int

@numba.njit
def numba_cost_precision_int2(N, T, dt, i_p, state_, target_state_, var_):
    cost =  0.
    for ind_time in range(T):
        for ind_node in range(N):
            for ind_var in var_:
                diff = np.abs(state_[ind_node, ind_var, ind_time] - target_state_[ind_node, ind_var, ind_time])
                if target_state_[ind_node, ind_var, ind_time] == -1000:
                    cost += 0.
                elif ( diff < tolerance ):
                    cost += 0.
                else:
                    cost += dt * 0.5 * i_p * ind_time * diff**2. / T
    return cost

def cost_precision_node2(N, T, t_, dt, i_p, state_, target_, va_):
    var = makeList(va_)
    cost_int = numba_cost_precision_node2(N, T, t_, dt, i_p, state_, target_, var_ = var )
    return cost_int

@numba.njit
def numba_cost_precision_node2(N, T, t_, dt, i_p, state_, target_state_, var_):
    cost =  np.zeros(( N, 2 ))
    for ind_node in range(N):
        for ind_var in var_:
            for ind_time in range(T):
                diff = np.abs(state_[ind_node, ind_var, ind_time] - target_state_[ind_node, ind_var, ind_time])
                if target_state_[ind_node, ind_var, ind_time] == -1000:
                    cost[ind_node, ind_var] += 0.
                elif ( diff < tolerance ):
                    cost[ind_node, ind_var] += 0.
                else:
                    cost[ind_node, ind_var] += dt * 0.5 * i_p * t_ * diff**2. / T
    return cost
        
    
###########################################################
# cost functions for energy
###########################################################    

# gradient of cost function for energy
def cost_energy_gradient(control_, i_e):
    # state_t: [N,dim_Model] dimensional array containing all nodes and state variables for all times
    cost_gradient_e_ = numba_energy_gradient(i_e, control_)
    return cost_gradient_e_

@numba.njit
def numba_energy_gradient(i_e, control_):
    cost_grad_ = i_e * control_.copy()
    return cost_grad_

def cost_energy_int(N, V, T, dt, i_e, control_, va_ = [0,1]):
    cost_ = numba_cost_energy_int(N, V, T, dt, i_e, control_)
    return cost_

@numba.njit
def numba_cost_energy_int(N, V, T, dt, i_e, control_):
    cost =  0.
    for ind_time in range(T):
        for ind_node in range(N):
            for ind_var in range(V):
                cost += dt * 0.5 * i_e * control_[ind_node, ind_var, ind_time]**2
    return cost

def cost_energy_node(N, V, T, dt, i_e, control_, va_ = [0,1]):
    cost_ = numba_cost_energy_node(N, V, T, dt, i_e, control_)
    return cost_

@numba.njit
def numba_cost_energy_node(N, V, T, dt, i_e, control_):
    cost =  np.zeros(( N, V ))
    for ind_node in range(N):
        for ind_var in range(V):
            for ind_time in range(T):
                cost[ind_node,ind_var] += dt * 0.5 * i_e * control_[ind_node, ind_var, ind_time]**2
    return cost

@numba.njit
def control_energy_components(N, V, T, dt, control_):
    control_energy = np.zeros(( N, V ))
    for ind_node in range(N):
        for ind_var in range(V):
            energy = 0.
            for ind_time in range(0,T):
                energy += dt * control_[ind_node, ind_var, ind_time]**2
            control_energy[ind_node, ind_var] = np.sqrt(energy)
    return control_energy

###########################################################
# cost functions for sparsity
########################################################### 

# cost function for sparsity: simple absolute value
def cost_sparsity_gradient(N, V, T, dt, control_, i_s):
    control_energy = control_energy_components(N, V, T, dt, control_)
    cost_gradient_s_ =  numba_cost_sparsity_gradient(N, V, T, i_s, control_, control_energy)
    return cost_gradient_s_

@numba.njit
def numba_cost_sparsity_gradient(N, V, T, i_s, control_, control_energy):
    cost_grad =  np.zeros(( N, V, T ))
    
    if i_s != 0.:
        for ind_node in range(N):
            for ind_var in range(V):
                if control_energy[ind_node, ind_var] == 0.:
                    cost_grad[ind_node, ind_var, :] = 0.
                else:
                    cost_grad[ind_node, ind_var, :] = i_s * control_[ind_node, ind_var,:] / control_energy[ind_node, ind_var]
        #cost_grad[ind_node, ind_var, 0] = 0.
        
    return cost_grad

def f_cost_sparsity_int(N, V, T, dt, i_s, control_):
    cost =  numba_cost_sparsity_int(N, V, T, i_s, dt, control_)
    return cost

@numba.njit
def numba_cost_sparsity_int(N, V, T, i_s, dt, control_):
    int_ =  0.
    for ind_node in range(N):
        for ind_var in range(V):
            cost = 0.
            for ind_time in range(0,T):
                cost += (control_[ind_node, ind_var, ind_time])**2 * dt
            int_ += i_s * np.sqrt(cost)
    return int_

def f_cost_sparsity_node(N, V, T, dt, i_s, control_):
    cost =  numba_cost_sparsity_node(N, V, T, i_s, dt, control_)
    return cost

@numba.njit
def numba_cost_sparsity_node(N, V, T, i_s, dt, control_):
    int_ =  np.zeros(( N, V ))
    for ind_var in range(V):
        for ind_node in range(N):
            cost = 0.
            for ind_time in range(0,T):
                cost += (control_[ind_node, ind_var, ind_time])**2 * dt
            int_[ind_node, ind_var] += i_s * np.sqrt(cost)
    return int_

###########################################################
# cost functions for sparsity
########################################################### 


def f_cost(state_, target_state_, control_, i_p, i_e, i_s):
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
def f_int(N, V, T, dt, state_, target_, control_, i_p, i_e, i_s, v_ = [0,1]):
    # cost_: [t] dimensional array containing cost for all times
    # return cost_int: integrated (total) cost
        
    var = makeList(v_)
            
    cost_prec, cost_energy, cost_sparsity = 0., 0., 0.
            
    if not i_p < 1e-12:
        cost_prec = cost_precision_int(N, T, dt, i_p, state_, target_, va_ = var)
    if not i_e < 1e-12:
        cost_energy = cost_energy_int(N, V, T, dt, i_e, control_)
    if not i_s < 1e-12:
        cost_sparsity = f_cost_sparsity_int(N, V, T, dt, i_s, control_)
    
    
    #if (cost_energy > 100.):
    #    print("control = ", control_)
    
    cost_int = cost_prec + cost_energy + cost_sparsity
    
    return cost_int

def f_int2(N, V, T, dt, state_, target_, control_, i_p, i_e, i_s, v_ = [0,1]):
    # cost_: [t] dimensional array containing cost for all times
    # return cost_int: integrated (total) cost
        
    var = makeList(v_)
        
    cost_prec, cost_energy, cost_sparsity = 0., 0., 0.
            
    if not i_p < 1e-12:
        cost_prec = cost_precision_int2(N, T, dt, i_p, state_, target_, va_ = var)
    if not i_e < 1e-12:
        cost_energy = cost_energy_int(N, V, T, dt, i_e, control_)
    if not i_s < 1e-12:
        cost_sparsity = f_cost_sparsity_int(N, V, T, dt, i_s, control_)
    
    """
    print("cost precision = ", cost_prec)
    print("cost energy = ", cost_energy)
    print("cost sparsity = ", cost_sparsity)
    """
    
    #if (cost_energy > 100.):
    #    print("control = ", control_)
    
    cost_int = cost_prec + cost_energy + cost_sparsity
    
    return cost_int

def cost_int_per_node(N, V, T, dt, state_, target_, control_, i_p, i_e, i_s, v_ = [0,1]):
    # cost_: [t] dimensional array containing cost for all times
    # return cost_int: integrated (total) cost
        
    var = makeList(v_)
        
    cost_prec_node = np.zeros(( N, 2 ))
    cost_e_node = np.zeros(( N, V ))
    cost_s_node = np.zeros(( N, V ))
            
    if not i_p < 1e-12:
        cost_prec_node = cost_precision_node(N, T, dt, i_p, state_, target_, va_ = var)
    if not i_e < 1e-12:
        cost_e_node = cost_energy_node(N, V, T, dt, i_e, control_)
    if not i_s < 1e-12:
        cost_s_node = f_cost_sparsity_node(N, V, T, dt, i_s, control_)
    
    """
    print("cost precision = ", cost_prec)
    print("cost energy = ", cost_energy)
    print("cost sparsity = ", cost_sparsity)
    """
    
    return [cost_prec_node, cost_e_node, cost_s_node]