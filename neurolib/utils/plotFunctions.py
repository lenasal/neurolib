import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from . import costFunctions as cost
from . import func_optimize as fo

plt.rcParams['axes.grid'] = True
plt.rcParams['hatch.linewidth'] = 1.

cmap=cm.get_cmap('viridis')
tolerance_cost_ = 1e-4

def plot_fullState(state_, dur, dt, state_vars, path_, filename_ = "full_state.png", plot_vars_ = np.arange(0,20,1)):
    time = np.arange(0, dur+dt, dt)
    
    fig, ax = plt.subplots(10, 2, figsize=(16, 30), linewidth=8, edgecolor='grey')
    
    for i in [0,2,5,7,9,11,13,15,18]:
        row_index = int( np.ceil(i/2) )
        ax[row_index,0].plot(time, state_[0,i,:], label = state_vars[i])
        ax[row_index,0].legend()
        ax[row_index,1].plot(time, state_[0,i+1,:], label = state_vars[i+1])
        ax[row_index,1].legend()
        
    ax[2,0].plot(time, state_[0,4,:], label = state_vars[4])
    ax[2,0].legend()
    ax[2,1].plot(time, state_[0,17,:], label = state_vars[17])
    ax[2,1].legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(path_, filename_))

def plot_fullState_log(state_, dur, dt, state_vars, path_, filename_ = "full_state_log.png", plot_vars_ = np.arange(0,20,1)):
    time = np.arange(0, dur+dt, dt)
    state_ = np.abs(state_)
        
    fig, ax = plt.subplots(10, 2, figsize=(16, 30), linewidth=8, edgecolor='grey')
    
    for i in [0,2,5,7,9,11,13,15,18]:
        row_index = int( np.ceil(i/2) )
        ax[row_index,0].plot(time, state_[0,i,:], label = state_vars[i])
        ax[row_index,0].legend()
        ax[row_index,1].plot(time, state_[0,i+1,:], label = state_vars[i+1])
        ax[row_index,1].legend()
        
        ax[row_index,0].set_yscale('log')
        ax[row_index,1].set_yscale('log')
        
    ax[2,0].plot(time, state_[0,4,:], label = state_vars[4])
    ax[2,0].legend()
    ax[2,1].plot(time, state_[0,17,:], label = state_vars[17])
    ax[2,1].legend()
    
    ax[2,0].set_yscale('log')
    ax[2,1].set_yscale('log')
    
    fig.tight_layout()
    plt.savefig(os.path.join(path_, filename_))

def plot_gradient(grad_, dur, dt, path_, filename_ = "gradient.png", plot_vars = [0,1,2,3,4,5]):
    
    time = np.arange(0, dur+dt, dt)
    grad_abs = np.abs(grad_)
    n_col = 2
    n_row = len(plot_vars)
    fig_height = n_row * 4
    
    fig, ax = plt.subplots(n_row, n_col, figsize=(16, fig_height), linewidth=8, edgecolor='grey')
    
    label_y = ['Cost grad exc current control', 'Cost grad inh current control',
               'Cost grad ee rate control', 'Cost grad ei rate control', 'Cost grad ie rate control', 'Cost grad ii rate control']
    
    if n_row > 1:
        for i in range(len(plot_vars)):
            ax[i,0].set_xlabel('Simulation time [ms]')
            ax[i,0].set_ylabel(label_y[plot_vars[i]])
            ax[i,0].plot(time, grad_[0,plot_vars[i],:])
        
            if np.amax(grad_abs[0,plot_vars[i],:]) > 0.:
                ax[i,1].set_xlabel('Simulation time [ms]')
                ax[i,1].set_ylabel(label_y[plot_vars[i]])
                ax[i,1].plot(time, grad_abs[0,plot_vars[i],:])
                ax[i,1].set_yscale('log')
                
    else:
        ax[0].set_xlabel('Simulation time [ms]')
        ax[0].set_ylabel(label_y[plot_vars[0]])
        ax[0].plot(time, grad_[0,plot_vars[0],:])
        
        if np.amax(grad_abs[0,plot_vars[0],:]) > 0.:
            ax[1].set_xlabel('Simulation time [ms]')
            ax[1].set_ylabel(label_y[plot_vars[0]])
            ax[1].plot(time, grad_abs[0,plot_vars[0],:])
            ax[1].set_yscale('log')


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.savefig(os.path.join(path_, filename_))
    plt.show()

def plot_conv_runtime(timeArray_, costArray_, labelArray_, path_, filename_ = "convergence_runtime.png"):
    
    fig, ax1 = plt.subplots(figsize=(12, 6), linewidth=8, edgecolor='grey')    
    ax1.set_xlabel('Runtime [s]')
    ax1.set_ylabel('Cost')
    
    # time and cost should be arrays containing all convergence numbers to plot
    for time_, cost_, label_ in zip(timeArray_, costArray_, labelArray_):
        iterations_ = cost_.shape[0]
        for i in range(iterations_-1, 0, -1):
            if (cost_[i] != 0.):
                iterations_ = i+1
                break
            
        ax1.plot(time_[1:iterations_], cost_[1:iterations_], label=str(label_) )
        
    ax1.legend()
        
    ax1.tick_params(axis='y')
    ax1.yaxis.get_major_formatter().set_useOffset(False)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.savefig(os.path.join(path_, filename_))
    plt.show()

def plot_convergence(cost_, path_, filename_ = "cost_convergence.png", ratio_ = 0.25):

    iterations_ = cost_.shape[0]
    for i in range(iterations_-1, 0, -1):
        if (cost_[i] != 0.):
            iterations_ = i+1
            break
    
    x1 = np.arange(1,iterations_,1)
    startind_ = int(ratio_ * iterations_)
    x2 = np.arange(startind_, iterations_,1)
    
    fig, ax1 = plt.subplots(figsize=(12, 6), linewidth=8, edgecolor='grey')

    ax1.set_title('cost of uncontrolled activity = {:.2f}'.format(cost_[0]))
    
    color = 'tab:blue'
    ax1.set_xlabel('Iteration #')
    ax1.set_ylabel('Total cost', color=color)
    ax1.plot(x1, cost_[1:iterations_], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.yaxis.get_major_formatter().set_useOffset(False)

    ax2 = ax1.twinx() 

    color = 'tab:orange'
    ax2.set_ylabel('Total cost', color=color)  # we already handled the x-label with ax1
    ax2.plot(x2, cost_[startind_:iterations_], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.yaxis.get_major_formatter().set_useOffset(False)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.savefig(os.path.join(path_, filename_))
    plt.show()
    

def plot_runtime(time_, path_, filename_ = "runtime.png"):

    iterations_ = time_.shape[0]
    for i in range(iterations_-1, 0, -1):
        if (time_[i] != 0.):
            iterations_ = i+1
            break
    
    x1 = np.arange(1,iterations_,1)
    
    fig, ax1 = plt.subplots(figsize=(12, 6), linewidth=8, edgecolor='grey')

    ax1.set_title('total runtime = {:.2f} seconds'.format(time_[-1]))
    
    color = 'tab:blue'
    ax1.set_xlabel('Iteration #')
    ax1.set_ylabel('Runtime [s]', color=color)
    ax1.plot(x1, time_[1:iterations_], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.yaxis.get_major_formatter().set_useOffset(False)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.savefig(os.path.join(path_, filename_))
    plt.show()
    
def plot_traces_control_current(model, control_, path_="", filename_=""):
    model.run(control=control_)
    
    y_labels = ["Activity [Hz]", "External current [nA]"]
    
    rows = 2
    columns = 2
    fig, ax = plt.subplots(rows, columns, figsize=(12, 8), linewidth=8, edgecolor='grey')
    fig.suptitle("System dynamics for external step current input", fontsize=20)
    
    ax[0,0].plot(model.t, model.rates_exc[0,:])
    ax[0,1].plot(model.t, model.rates_inh[0,:])
    ax[1,0].plot(model.t, control_[0,0,:]/5.)
    ax[1,1].plot(model.t, control_[0,1,:]/5.)
    
    for r in range(rows):
        for c in range(columns):
            ax[r,c].set_xlabel(xlabel='t [ms]', fontsize=14)
            ax[r,c].set_ylabel(ylabel=y_labels[r], fontsize=14)
            ax[r,c].tick_params(axis="x", labelsize=10)
            ax[r,c].tick_params(axis="y", labelsize=10)
            
    
    cols = ['Excitatory', 'Inhibitory']
            
    for a, col in zip(ax[0,:], cols):
        a.annotate(col, xy=(0.5, 1.05), xytext=(0,5), xycoords='axes fraction', textcoords='offset points',
                   size=20, ha='center', va='baseline', weight='bold')

    
    if path_ == "" or filename_ == "" :
        return
    
    plt.savefig(os.path.join(path_, filename_))
    #plt.show()
    
def plot_traces(model, control_, path_="", filename_=""):
    model.run(control=control_)
    
    rows = 4
    columns = 2
    fig, ax = plt.subplots(rows, columns, figsize=(16, 8), linewidth=8, edgecolor='grey')
    
    ax[0,0].plot(model.t, model.rates_exc[0,:], label ="exc rates")
    ax[0,1].plot(model.t, model.rates_inh[0,:], label ="inh rates")
    ax[1,0].plot(model.t, control_[0,0,:], label ="exc current control")
    ax[1,1].plot(model.t, control_[0,1,:], label ="inh current control")
    ax[2,0].plot(model.t, control_[0,2,:], label ="ee rate control")
    ax[2,1].plot(model.t, control_[0,3,:], label ="ei rate control")
    ax[3,0].plot(model.t, control_[0,4,:], label ="ie rate control")
    ax[3,1].plot(model.t, control_[0,5,:], label ="ii rate control")
    
    for r in range(rows):
        for c in range(columns):
            ax[r,c].legend()
    
    if path_ == "" or filename_ == "" :
        #plt.show()
        return
    
    plt.savefig(os.path.join(path_, filename_))
    plt.show()
    
    
# plot uncontrolled dynamics, controlled dynamics
def plot_control(model, control_array, cost_node_array, weights_array, t_sim_, t_sim_pre_, t_sim_post_, initial_params_, target_, path_, filename_ = '',
                 shading = False, transition_time_ = 0., labels_ = []):
    
    dt = model.params.dt
    if model.name == "aln" or model.name == "aln-control":
        control_factor = model.params.C/1000.
    else:
        control_factor = 1.
        
    control_ = control_array[0]
        
    model.params.duration = (control_.shape[2] - 1.) * dt
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    i3 = int(round( (t_sim_pre_ + transition_time_ * t_sim_) / dt, 1) + 1 )
    
    init_vars = model.init_vars
    for iv in range(len(init_vars)):
        if (type(model.params[init_vars[iv]]) == np.float64 or type(model.params[init_vars[iv]]) == float):
            model.params[init_vars[iv]] = initial_params_[iv]
        elif model.params[init_vars[iv]].ndim == 2:
            model.params[init_vars[iv]][0,:] = initial_params_[iv]
            #print("set initial vars = ", model.params[init_vars[iv]] )
        else:
            model.params[init_vars[iv]][0] = initial_params_[iv]
            

    # no control
    model.run(control=model.getZeroControl())

    output_vars = model.output_vars
    control_vars = model.control_input_vars

    
    control_time_exc = []
    control_time_inh = []
    cntrl_limit_scaled = 10 * 1e-3
    cntrl_limit = cntrl_limit_scaled * 5. # 1e-3 nA (factor 5 because capacitance)

    for t in range(len(model.t)):
        if (np.abs(control_[0,0,t]) > cntrl_limit):
            control_time_exc.append(dt * t)
        if model.name == "aln" or model.name == "aln-control":
            if (np.abs(control_[0,1,t]) > cntrl_limit):
                control_time_inh.append(dt * t)
    
    columns = len(model.output_vars)-1
    rows = 4
    n_vars = len(control_vars)
            
    fig, ax = plt.subplots(rows, columns, figsize=(16, 14) )#, linewidth=8, edgecolor='grey')
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    y_labels_rates = ['Rates exc. [Hz]', 'Rates inh. [Hz]', 'Adaptation current [pA]']
    y_labels_control = ['Control current [nA]', 'Control current [nA]', 'Control rate to E [kHz]', 'Control rate to I [kHz]']
    sim_legend = ['Uncontrolled rate', 'Control', 'Control current', 'Control rate']
    target_legend = ['Target']
    cntrl_time_legend = ['Control > {} pA'.format(cntrl_limit_scaled * 1000), 'Control active', 'Transition time']
    
    if labels_ == []:
        for control_ in control_array:
            labels_.append("")
            
    n_colors = len(control_array)
    color_array = np.zeros(( 2 + len(control_array) ))
    color_array[0] = 0.0
    color_array[1] = 0.95
    color_distance = 0.25
    color_array[2:] = np.linspace(color_array[0] + color_distance, color_array[1] - color_distance, n_colors)
    colors_ = cmap(color_array)
    ### 0: target, 1: uncontrolled rate, 2...: control inputs
    
    #### UNCONTROLLED ACTIVITY
    ax[0,0].plot(model.t, model[output_vars[0]][0,:], linewidth = 1., label=sim_legend[0], color=colors_[0])
    ax[0,1].plot(model.t, model[output_vars[1]][0,:], linewidth = 1., color=colors_[0])
    for i in range(columns):
        ax[0,i].set(xlabel='t [ms]', ylabel=y_labels_rates[i])
        ax[0,i].set_xlim([model.t[0],model.t[-1]])
        #ax[0,0].axvspan(control_time_exc[0], control_time_exc[1], facecolor='0.1', alpha=0.2, zorder=-100)
        #ax[0,0].axvspan(0, 50, facecolor='0.7', alpha=0.2, zorder=-100)
        
    ##### PLOT TARGET
    i3 = i1 # plot full target
    if (i2 == 0):
        ax[0,0].plot(model.t[i3:], target_[0,0,i3-i1:], '--', linewidth = 3, label=target_legend[0], color=colors_[1])
        ax[0,1].plot(model.t[i3:], target_[0,1,i3-i1:], '--', linewidth = 3, color=colors_[1])
    else:
        ax[0,0].plot(model.t[i3:-i2], target_[0,0,i3-i1:], '--', linewidth = 3, label=target_legend[0], color=colors_[1])
        ax[0,1].plot(model.t[i3:-i2], target_[0,1,i3-i1:], '--', linewidth = 3, color=colors_[1])
    
    ##################### go through all controls in control array 
    
    for c_ind in range(len(control_array)):
        
        weights_ = weights_array[c_ind]
        cp_ = [ [cost_node_array[c_ind][0][0,0] / weights_[0], cost_node_array[c_ind][0][0,0]],
               [cost_node_array[c_ind][0][0,1] / weights_[0], cost_node_array[c_ind][0][0,1]] ]
        str_cp_ = [ str(r'$C_p = {:.1f}$'.format(weights_[0]) + r' s $\times {:.1f}$'.format(cp_[0][0])
                        + r' $s^{-1}$' + r'$ = {:.2f}$'.format(cp_[0][1]) ),
                   str(r'$C_p = {:.1f}$'.format(weights_[0]) + r' s $\times {:.1f}$'.format(cp_[1][0])
                        + r' $s^{-1}$' + r'$ = {:.2f}$'.format(cp_[1][1]) )] #'= {:.2f}$'.format(cp_[0])
        
        control_ = control_array[c_ind]
        
        model.run(control=control_)
        
        ax[0,0].plot(model.t, model[output_vars[0]][0,:], linewidth = 1., color=colors_[c_ind+2],
                     label=labels_[c_ind] )
        ax[0,0].text(0.03, 0.95 - c_ind * 0.2, str_cp_[0],
                    transform=ax[0,0].transAxes, color = 'black', fontsize=14, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=colors_[c_ind+2], edgecolor = 'black', alpha=0.3))
        ax[0,1].plot(model.t, model[output_vars[1]][0,:], linewidth = 1., color=colors_[c_ind+2])
        ax[0,1].text(0.03, 0.95 - c_ind * 0.2, str_cp_[1],
                    transform=ax[0,1].transAxes, color = 'black', fontsize=14, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=colors_[c_ind+2], edgecolor = 'black', alpha=0.5))

        
        # control, # cost type, # network node, # variable
        ce_ = [ [], [] ]
        str_ce_ = [str(r'$C_e = 0$'), str(r'$C_e = 0$') ]
        
        if weights_[1] != 0.:
            ce_[0] = [cost_node_array[c_ind][1][0,0] / weights_[1], cost_node_array[c_ind][1][0,0]]
            str_ce_[0] = str(r'$C_e = {:.1f}$'.format(weights_[1]) + r' s V^{-2} $\times {:.1f}$'.format(ce_[0][0])
                        + r' $V^2 s^{-1}$' + r'$ = {:.1f}$'.format(ce_[0][1]) )
            ce_[1] = [cost_node_array[c_ind][1][0,1] / weights_[1], cost_node_array[c_ind][1][0,1]]
            str_ce_[1] = str(r'$C_e = {:.1f}$'.format(weights_[1]) + r' s V^{-2} $\times {:.1f}$'.format(ce_[1][0])
                       + r' $V^2 s^{-1}$' + r'$ = {:.1f}$'.format(ce_[1][1]) )
            
        cs_ = [ [], [] ]
        str_cs_ = [str(r'$C_s = 0$'), str(r'$C_s = 0$') ]
        
        if weights_[2] != 0.:
            cs_[0] = [cost_node_array[c_ind][2][0,0] / weights_[2], cost_node_array[c_ind][2][0,0]]
            str_cs_[0] = str(r'$C_s = {:.1f}$'.format(weights_[2]) + r' V s^{-1/2} $\times {:.1f}$'.format(cs_[0][0])
                        + r' $V^x sqrt s$' + r'$ = {:.1f}$'.format(cs_[0][1]) )
            cs_[1] = [cost_node_array[c_ind][2][0,1] / weights_[2], cost_node_array[c_ind][2][0,1]]
            str_cs_[1] = str(r'$C_s = {:.1f}$'.format(weights_[2]) + r' V s^{-1/2} $\times {:.1f}$'.format(cs_[1][0])
                        + r' $V^x sqrt s$' + r'$ = {:.1f}$'.format(cs_[1][1]) )
                               
        for i in range(columns):
            
            ax[1,i].plot(model.t, control_[0,i,:] * control_factor, linewidth = 1., color=colors_[c_ind+2] ) # divide by five to take into account capacitance
            ax[1,i].text(0.03, 0.95 - c_ind * 0.3, str(str_ce_[i] + '\n' + str_cs_[i]), 
                transform=ax[1,i].transAxes, color = 'black', fontsize=14, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=colors_[c_ind+2], edgecolor = 'black', alpha=0.4))
            ax[1,i].set(xlabel='t [ms]', ylabel=y_labels_control[1])
            ax[1,i].set_xlim([model.t[0],model.t[-1]])
            
            # ee, ei, ie, ii
            ax[2,i].plot(model.t, control_[0,i+2,:], linewidth = 1., color=colors_[c_ind+2] )
            ax[2,i].text(0.03, 0.95 - c_ind * 0.3,
                str(r'$C_e = {:.2f}$'.format(cost_node_array[0][1][0,i+1]) + r' $(kHz)^2 s$' + '\n' + r'$C_s = {:.2f}$'.format(
                    cost_node_array[0][2][0,i+1]) + r' $(kHz)^2 s$'),
                transform=ax[2,i].transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax[2,i].set(xlabel='t [ms]', ylabel=y_labels_control[2])
            ax[2,i].set_xlim([model.t[0],model.t[-1]])
            
            ax[3,i].plot(model.t, control_[0,i+4,:], linewidth = 1., color=colors_[c_ind+2] )
            ax[3,i].text(0.03, 0.95,
                str(r'$C_e = {:.2f}$'.format(cost_node_array[0][2][0,i+2]) + r' $(kHz)^2 s$' + '\n' + r'$C_s = {:.2f}$'.format(
                    cost_node_array[0][2][0,i+2]) + r' $(kHz)^2 s$'),
                transform=ax[3,i].transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax[3,i].set(xlabel='t [ms]', ylabel=y_labels_control[3])
            ax[3,i].set_xlim([model.t[0],model.t[-1]])
            
    #####################
    
    ax[0,0].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='grey', alpha=0.1, zorder=-2, label=cntrl_time_legend[1])
    ax[0,0].axvspan(t_sim_pre_, t_sim_pre_ + transition_time_ * t_sim_, facecolor='grey', alpha=0.1, zorder=-1, hatch='///',
                            edgecolor = 'g', label=cntrl_time_legend[2])
    ax[0,1].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='grey', alpha=0.1, zorder=-2)
    ax[0,1].axvspan(t_sim_pre_, t_sim_pre_ + transition_time_ * t_sim_, facecolor='grey', alpha=0.1, zorder=-1, hatch='///',
                            edgecolor = 'g')
        
    for i in range(1,rows):
        for j in range(columns):
            ax[i,j].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='grey', alpha=0.1, zorder=-2)
            ax[i,j].axvspan(t_sim_pre_, t_sim_pre_ + transition_time_ * t_sim_, facecolor='grey', alpha=0.1, zorder=-1, hatch='///',
                            edgecolor = 'grey')
    
    if shading:
        facecol = 'grey'
        al = 0.5
        for i in range(rows):
            for times in control_time_exc:
                if (times == control_time_exc[0]):
                    ax[i,0].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1, label=cntrl_time_legend[0])
                else:
                    ax[i,0].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1)
            for times in control_time_inh:
                if (times == control_time_inh[0]):
                    ax[i,1].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1, label=cntrl_time_legend[0])
                else:
                    ax[i,1].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1)
    
    """    
    for i in range(2):
        for j in range(columns):
            ax[i,j].legend(loc='upper right', bbox_to_anchor=(0.5, 0.5))
      
            
    rows_legend = ['Node activity', 'Control']
            
    for a, row in zip(ax[:,0], rows_legend):
        a.annotate(row, xy=(-0.05, 0.5), xytext=(-a.yaxis.labelpad - 15, 0), rotation = 90,
                xycoords=a.yaxis.label, textcoords='offset points', size=20, ha='right', va='center', weight='bold')
    """
        
    #ax[0,0]
    leg = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol = 3)
    leg.set_in_layout(True)
    #ax[1,0].legend(loc='upper left', bbox_to_anchor=(1, 1.05))
    #ax[2,0].legend(loc='upper left', bbox_to_anchor=(1, 1.05))

    cols = ['Excitatory', 'Inhibitory']
            
    for a, col in zip(ax[0,:], cols):
        a.annotate(col, xy=(0.5, 1.05), xytext=(0,5), xycoords='axes fraction', textcoords='offset points',
                   size=20, ha='center', va='baseline', weight='bold')
    
    fig.tight_layout()
        
    if not filename_ == '':
        plt.savefig(os.path.join(path_, filename_), bbox_inches='tight')
        
        
def plot_control_current(model, control_array, cost_node_array, weights_array, t_sim_, t_sim_pre_, t_sim_post_, initial_params_,
                         target_, path_, filename_ = '', shading = False, transition_time_ = 0., labels_ = [],
                         precision_variables_ = [0,1], print_cost_ = True):
    
    dt = model.params.dt
    if model.name == "aln" or model.name == "aln-control":
        control_factor = model.params.C/1000.
    else:
        control_factor = 1.
        
    control_ = control_array[0]
        
    model.params.duration = (control_.shape[2] - 1.) * dt
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    i3 = int(round( (t_sim_pre_ + transition_time_ * t_sim_) / dt, 1) + 1 )
    
    N = model.params.N
    T = control_.shape[2] - i1 - i2
    
    i_p = weights_array[0][0]
    for w in weights_array:
        if i_p != w[0]:
            print("WARNING: Precision cost weight differs, cannot consistently compute cost of uncontrolled system.")
    
    
    init_vars = model.init_vars
    for iv in range(len(init_vars)):
        if (type(model.params[init_vars[iv]]) == np.float64 or type(model.params[init_vars[iv]]) == float):
            model.params[init_vars[iv]] = initial_params_[iv]
        elif model.params[init_vars[iv]].ndim == 2:
            model.params[init_vars[iv]][0,:] = initial_params_[iv]
            #print("set initial vars = ", model.params[init_vars[iv]] )
        else:
            model.params[init_vars[iv]][0] = initial_params_[iv]
            

    # no control
    state_ = fo.updateState(model, model.getZeroControl())

    output_vars = model.output_vars
    
    control_time_exc = []
    control_time_inh = []
    cntrl_limit_scaled = 10 * 1e-3
    cntrl_limit = cntrl_limit_scaled * 5. # 1e-3 nA (factor 5 because capacitance)

    for t in range(len(model.t)):
        if (np.abs(control_[0,0,t]) > cntrl_limit):
            control_time_exc.append(dt * t)
        if model.name == "aln" or model.name == "aln-control":
            if (np.abs(control_[0,1,t]) > cntrl_limit):
                control_time_inh.append(dt * t)
    
    columns = len(model.output_vars)-1
    rows = 2
            
    fig, ax = plt.subplots(rows, columns, figsize=(8, 6) )#, linewidth=8, edgecolor='grey')
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    y_labels_rates = ['Rates exc. [Hz]', 'Rates inh. [Hz]', 'Adaptation current [pA]']
    y_labels_control = ['Control current [nA]', 'Control current [nA]', 'Control rate to E [kHz]', 'Control rate to I [kHz]']
    sim_legend = ['Uncontrolled rate', 'Control', 'Control current', 'Control rate']
    target_legend = ['Target']
    cntrl_time_legend = ['Control > {} pA'.format(cntrl_limit_scaled * 1000), 'Control active', 'Transition time']
    
    if labels_ == []:
        for control_ in control_array:
            labels_.append("")
            
    n_colors = len(control_array)
    color_array = np.zeros(( 2 + len(control_array) ))
    color_array[0] = 0.0
    color_array[1] = 1.
    color_distance = 0.2
    color_array[2:] = np.linspace(color_array[0] + color_distance, color_array[1] - color_distance, n_colors)
    colors_ = cmap(color_array)
    ### 0: target, 1: uncontrolled rate, 2...: control inputs
    
    target_trans = target_.copy()
    for t_ in range(T):
        if t_ < transition_time_ * T:
            for n_ in range(N):
                for v_ in range(2):
                        target_trans[n_,v_,t_] = - 1000.
                        
    cost_uncontrolled = cost.cost_precision_node(N, T, dt, 1., state_, target_trans, precision_variables_)
    
    #### UNCONTROLLED ACTIVITY
    ax[0,0].plot(model.t, model[output_vars[0]][0,:], linewidth = 1., label=sim_legend[0], color=colors_[0])
    ax[0,1].plot(model.t, model[output_vars[1]][0,:], linewidth = 1., color=colors_[0])
    
    if print_cost_:
        str_cp_uncontrolled = [str(r'$C_p = 0.0$'), str(r'$C_p = 0.0$')]
        if i_p != 0.:
            #if cost_uncontrolled[0][0] > tolerance_cost_:
            str_cp_uncontrolled[0] = str(r'$C_p = {:,.1f}$'.format(i_p) + r' s $\times {:,.0f}$'.format(cost_uncontrolled[0][0]) 
                            + r' $s^{-1}$' + r'$ = {:,.0f}$'.format(cost_uncontrolled[0][0] * i_p) )
            #if cost_uncontrolled[0][1] > tolerance_cost_:
            str_cp_uncontrolled[1] = str(r'$C_p = {:,.1f}$'.format(i_p) + r' s $\times {:,.0f}$'.format(cost_uncontrolled[0][1])
                            + r' $s^{-1}$' + r'$ = {:,.0f}$'.format(cost_uncontrolled[0][1] * i_p) )
        
        ax[0,0].text(1.05, 1., str_cp_uncontrolled[0],
                        transform=ax[0,0].transAxes, color = 'black', fontsize=14, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=colors_[0], edgecolor = 'black', alpha=0.3)) 
        ax[0,1].text(1.05, 1., str_cp_uncontrolled[1],
                        transform=ax[0,1].transAxes, color = 'black', fontsize=14, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=colors_[0], edgecolor = 'black', alpha=0.3))
    
    for i in range(columns):
        ax[0,i].set(xlabel='t [ms]', ylabel=y_labels_rates[i])
        ax[0,i].set_xlim([model.t[0],model.t[-1]])
        #ax[0,0].axvspan(control_time_exc[0], control_time_exc[1], facecolor='0.1', alpha=0.2, zorder=-100)
        #ax[0,0].axvspan(0, 50, facecolor='0.7', alpha=0.2, zorder=-100)
        
    ##### PLOT TARGET
    i3 = i1 # plot full target
    if (i2 == 0):
        ax[0,0].plot(model.t[i3:], target_[0,0,i3-i1:], '--', linewidth = 3, label=target_legend[0], color=colors_[1])
        ax[0,1].plot(model.t[i3:], target_[0,1,i3-i1:], '--', linewidth = 3, color=colors_[1])
    else:
        ax[0,0].plot(model.t[i3:-i2], target_[0,0,i3-i1:], '--', linewidth = 3, label=target_legend[0], color=colors_[1])
        ax[0,1].plot(model.t[i3:-i2], target_[0,1,i3-i1:], '--', linewidth = 3, color=colors_[1])
    
    ##################### go through all controls in control array 
    
    for c_ind in range(len(control_array)):
        
        control_ = control_array[c_ind]
        model.run(control=control_)
        
        ax[0,0].plot(model.t, model[output_vars[0]][0,:], linewidth = 1., color=colors_[c_ind+2],
                     label=labels_[c_ind] )
        ax[0,1].plot(model.t, model[output_vars[1]][0,:], linewidth = 1., color=colors_[c_ind+2])
        
        if print_cost_:
        
            weights_ = weights_array[c_ind]
            str_cp_ = [str(r'$C_p = 0.0$'), str(r'$C_p = 0.0$')]
            cp_ = [ [cost_node_array[c_ind][0][0,0], cost_node_array[c_ind][0][0,0]  * weights_[0]],
                   [cost_node_array[c_ind][0][0,1], cost_node_array[c_ind][0][0,1]  * weights_[0]] ]
            #if cp_[0][1] > tolerance_cost_:
            str_cp_[0] = str(r'$C_p = {:,.1f}$'.format(weights_[0]) + r' s $\times {:,.1f}$'.format(cp_[0][0])
                            + r' $s^{-1}$' + r'$ = {:.1f}$'.format(cp_[0][1]) )
            #if cp_[1][1] > tolerance_cost_:
            str_cp_[1] =  str(r'$C_p = {:,.1f}$'.format(weights_[0]) + r' s $\times {:,.1f}$'.format(cp_[1][0])
                            + r' $s^{-1}$' + r'$ = {:,.1f}$'.format(cp_[1][1]) ) 
        
        
            ax[0,0].text(1.05, 0.88 - c_ind * 0.13, str_cp_[0],
                        transform=ax[0,0].transAxes, color = 'black', fontsize=14, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=colors_[c_ind+2], edgecolor = 'black', alpha=0.3))
            ax[0,1].text(1.05, 0.88 - c_ind * 0.13, str_cp_[1],
                        transform=ax[0,1].transAxes, color = 'black', fontsize=14, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=colors_[c_ind+2], edgecolor = 'black', alpha=0.3))

        
            # control, # cost type, # network node, # variable
            ce_ = [ [], [] ]
            str_ce_ = [str(r'$C_e = 0.0$'), str(r'$C_e = 0.0$') ]
            
            #if weights_[1] != 0.:
            #if cost_node_array[c_ind][1][0,0] > tolerance_cost_:
            ce_[0] = [cost_node_array[c_ind][1][0,0], cost_node_array[c_ind][1][0,0] * weights_[1]]
            str_ce_[0] = str(r'$C_e = {:,.0f}$'.format(weights_[1]) + r' $s V^{-2}$' + r' $\times {:,.2f}$'.format(ce_[0][0])
                            + r' $V^2 s^{-1}$' + r'$ = {:,.0f}$'.format(ce_[0][1]) )
            #if cost_node_array[c_ind][1][0,1] > tolerance_cost_:
            ce_[1] = [cost_node_array[c_ind][1][0,1], cost_node_array[c_ind][1][0,1]  * weights_[1]]
            str_ce_[1] = str(r'$C_e = {:,.0f}$'.format(weights_[1]) + r' $s V^{-2}$' + r' $\times {:,.2f}$'.format(ce_[1][0])
                        + r' $V^2 s^{-1}$' + r'$ = {:,.0f}$'.format(ce_[1][1]) )
                
            cs_ = [ [], [] ]
            str_cs_ = [str(r'$C_s = 0.0$'), str(r'$C_s = 0.0$') ]
                    
            #if weights_[2] != 0.:
            #if cost_node_array[c_ind][2][0,0] > tolerance_cost_:
            cs_[0] = [cost_node_array[c_ind][2][0,0], cost_node_array[c_ind][2][0,0]  * weights_[2]]
            str_cs_[0] = str(r'$C_s = {:,.0f}$'.format(weights_[2]) + r' $V s^{-1/2}$' + r' $\times {:,.2f}$'.format(cs_[0][0])
                        + r' $V^{-1} \sqrt{s}$' + r'$ = {:,.0f}$'.format(cs_[0][1]) )
            #if cost_node_array[c_ind][2][0,1] > tolerance_cost_:
            cs_[1] = [cost_node_array[c_ind][2][0,1], cost_node_array[c_ind][2][0,1]  * weights_[2]]
            str_cs_[1] = str(r'$C_s = {:,.0f}$'.format(weights_[2]) + r' $V s^{-1/2}$' + r' $\times {:,.2f}$'.format(cs_[1][0])
                            + r' $V^{-1} \sqrt{s}$' + r'$ = {:,.0f}$'.format(cs_[1][1]) )
                               
        for i in range(columns):
                                    
            ax[1,i].plot(model.t, control_[0,i,:] * control_factor, linewidth = 1., color=colors_[c_ind+2] ) # divide by five to take into account capacitance
            ax[1,i].set(xlabel='t [ms]', ylabel=y_labels_control[1])
            ax[1,i].set_xlim([model.t[0],model.t[-1]])
            
            if print_cost_:
                ax[1,i].text(1.05, 1. - c_ind * 0.2, str(str_ce_[i] + '\n' + str_cs_[i]), 
                             transform=ax[1,i].transAxes, color = 'black', fontsize=14, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor=colors_[c_ind+2], edgecolor = 'black', alpha=0.3))
            
            
    #####################
    
    ax[0,0].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='grey', alpha=0.1, zorder=-2, label=cntrl_time_legend[1])
    ax[0,0].axvspan(t_sim_pre_, t_sim_pre_ + transition_time_ * t_sim_, facecolor='grey', alpha=0.1, zorder=-1, hatch='///',
                            edgecolor = 'g', label=cntrl_time_legend[2])
    ax[0,1].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='grey', alpha=0.1, zorder=-2)
    ax[0,1].axvspan(t_sim_pre_, t_sim_pre_ + transition_time_ * t_sim_, facecolor='grey', alpha=0.1, zorder=-1, hatch='///',
                            edgecolor = 'g')
        
    for i in range(1,rows):
        for j in range(columns):
            ax[i,j].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='grey', alpha=0.1, zorder=-2)
            ax[i,j].axvspan(t_sim_pre_, t_sim_pre_ + transition_time_ * t_sim_, facecolor='grey', alpha=0.1, zorder=-1, hatch='///',
                            edgecolor = 'grey')
    
    if shading:
        facecol = 'grey'
        al = 0.5
        for i in range(rows):
            for times in control_time_exc:
                if (times == control_time_exc[0]):
                    ax[i,0].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1, label=cntrl_time_legend[0])
                else:
                    ax[i,0].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1)
            for times in control_time_inh:
                if (times == control_time_inh[0]):
                    ax[i,1].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1, label=cntrl_time_legend[0])
                else:
                    ax[i,1].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1)
    
    
    leg_col_number = 2 + int( math.ceil( len( control_array )/2. ) )
    if len( control_array ) > 4:
        leg_col_number -= 1
    leg = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol = leg_col_number)
    leg.set_in_layout(True)

    cols = ['Excitatory', 'Inhibitory']
            
    for a, col in zip(ax[0,:], cols):
        a.annotate(col, xy=(0.5, 1.05), xytext=(0,5), xycoords='axes fraction', textcoords='offset points',
                   size=20, ha='center', va='baseline', weight='bold')
    
    fig.tight_layout()
        
    if not filename_ == '':
        plt.savefig(os.path.join(path_, filename_), bbox_inches='tight')
        
        
def plot_control_rate(model, control_array, cost_node_array, weights_array, t_sim_, t_sim_pre_, t_sim_post_, initial_params_,
                         target_, path_, filename_ = '', shading = False, transition_time_ = 0., labels_ = [],
                         precision_variables_ = [0,1], print_cost_ = False):
    
    dt = model.params.dt  
    control_ = control_array[0]
        
    model.params.duration = (control_.shape[2] - 1.) * dt
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    i3 = int(round( (t_sim_pre_ + transition_time_ * t_sim_) / dt, 1) + 1 )
    
    N = model.params.N
    T = control_.shape[2] - i1 - i2
    
    i_p = weights_array[0][0]
    for w in weights_array:
        if i_p != w[0]:
            print("WARNING: Precision cost weight differs, cannot consistently compute cost of uncontrolled system.")
    
    
    init_vars = model.init_vars
    for iv in range(len(init_vars)):
        if (type(model.params[init_vars[iv]]) == np.float64 or type(model.params[init_vars[iv]]) == float):
            model.params[init_vars[iv]] = initial_params_[iv]
        elif model.params[init_vars[iv]].ndim == 2:
            model.params[init_vars[iv]][0,:] = initial_params_[iv]
        else:
            model.params[init_vars[iv]][0] = initial_params_[iv]
            

    # no control
    state_ = fo.updateState(model, model.getZeroControl())

    output_vars = model.output_vars
    
    control_time_exc = []
    control_time_inh = []
    cntrl_limit_scaled = 10 * 1e-3
    cntrl_limit = cntrl_limit_scaled * 5. # 1e-3 nA (factor 5 because capacitance)

    for t in range(len(model.t)):
        if (np.abs(control_[0,0,t]) > cntrl_limit):
            control_time_exc.append(dt * t)
        if model.name == "aln" or model.name == "aln-control":
            if (np.abs(control_[0,1,t]) > cntrl_limit):
                control_time_inh.append(dt * t)
    
    columns = len(model.output_vars)-1
    rows = 3
            
    fig, ax = plt.subplots(rows, columns, figsize=(8, 6) )#, linewidth=8, edgecolor='grey')
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    y_labels_rates = ['Rates exc. [Hz]', 'Rates inh. [Hz]', 'Adaptation current [pA]']
    y_labels_control = ['Control rate to E [Hz]', 'Control rate to I [Hz]']
    sim_legend = ['Uncontrolled rate', 'Control', 'Control current', 'Control rate']
    target_legend = ['Target']
    cntrl_time_legend = ['Control > {} pA'.format(cntrl_limit_scaled * 1000), 'Control active', 'Transition time']
    
    if labels_ == []:
        for control_ in control_array:
            labels_.append("")
            
    n_colors = len(control_array)
    color_array = np.zeros(( 2 + len(control_array) ))
    color_array[0] = 0.0
    color_array[1] = 1.
    color_distance = 0.2
    color_array[2:] = np.linspace(color_array[0] + color_distance, color_array[1] - color_distance, n_colors)
    colors_ = cmap(color_array)
    ### 0: target, 1: uncontrolled rate, 2...: control inputs
    
    target_trans = target_.copy()
    for t_ in range(T):
        if t_ < transition_time_ * T:
            for n_ in range(N):
                for v_ in range(2):
                        target_trans[n_,v_,t_] = - 1000.
                            
    #### UNCONTROLLED ACTIVITY
    
    ax[0,0].plot(model.t, model[output_vars[0]][0,:], linewidth = 1., label=sim_legend[0], color=colors_[0])
    ax[0,1].plot(model.t, model[output_vars[1]][0,:], linewidth = 1., color=colors_[0])
    for i in range(columns):
        ax[0,i].set(xlabel='t [ms]', ylabel=y_labels_rates[i])
        ax[0,i].set_xlim([model.t[0],model.t[-1]])
        #ax[0,0].axvspan(control_time_exc[0], control_time_exc[1], facecolor='0.1', alpha=0.2, zorder=-100)
        #ax[0,0].axvspan(0, 50, facecolor='0.7', alpha=0.2, zorder=-100)
        
    ##### PLOT TARGET
    i3 = i1 # plot full target
    if (i2 == 0):
        ax[0,0].plot(model.t[i3:], target_[0,0,i3-i1:], '--', linewidth = 3, label=target_legend[0], color=colors_[1])
        ax[0,1].plot(model.t[i3:], target_[0,1,i3-i1:], '--', linewidth = 3, color=colors_[1])
    else:
        ax[0,0].plot(model.t[i3:-i2], target_[0,0,i3-i1:], '--', linewidth = 3, label=target_legend[0], color=colors_[1])
        ax[0,1].plot(model.t[i3:-i2], target_[0,1,i3-i1:], '--', linewidth = 3, color=colors_[1])
    
    ##################### go through all controls in control array 
    
    for c_ind in range(len(control_array)):
        
        control_ = control_array[c_ind]
                
        model.run(control=control_)
        
        ax[0,0].plot(model.t, model[output_vars[0]][0,:], linewidth = 1., color=colors_[c_ind+2],
                     label=labels_[c_ind] )
        ax[0,1].plot(model.t, model[output_vars[1]][0,:], linewidth = 1., color=colors_[c_ind+2])
        
                
                               
        for i in range(columns):
                                    
            ax[1,i].plot(model.t, control_[0,i+2,:]*1000., linewidth = 1., color=colors_[c_ind+2] )
            ax[1,i].set(xlabel='t [ms]', ylabel=y_labels_control[0])
            ax[1,i].set_xlim([model.t[0],model.t[-1]])
            
            ax[2,i].plot(model.t, control_[0,i+4,:]*1000., linewidth = 1., color=colors_[c_ind+2] )
            
            ax[2,i].set(xlabel='t [ms]', ylabel=y_labels_control[1])
            ax[2,i].set_xlim([model.t[0],model.t[-1]])
            
    #####################
    
    ax[0,0].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='grey', alpha=0.1, zorder=-2, label=cntrl_time_legend[1])
    ax[0,0].axvspan(t_sim_pre_, t_sim_pre_ + transition_time_ * t_sim_, facecolor='grey', alpha=0.1, zorder=-1, hatch='///',
                            edgecolor = 'g', label=cntrl_time_legend[2])
    ax[0,1].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='grey', alpha=0.1, zorder=-2)
    ax[0,1].axvspan(t_sim_pre_, t_sim_pre_ + transition_time_ * t_sim_, facecolor='grey', alpha=0.1, zorder=-1, hatch='///',
                            edgecolor = 'g')
        
    for i in range(1,rows):
        for j in range(columns):
            ax[i,j].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='grey', alpha=0.1, zorder=-2)
            ax[i,j].axvspan(t_sim_pre_, t_sim_pre_ + transition_time_ * t_sim_, facecolor='grey', alpha=0.1, zorder=-1, hatch='///',
                            edgecolor = 'grey')
    
    if shading:
        facecol = 'grey'
        al = 0.5
        for i in range(rows):
            for times in control_time_exc:
                if (times == control_time_exc[0]):
                    ax[i,0].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1, label=cntrl_time_legend[0])
                else:
                    ax[i,0].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1)
            for times in control_time_inh:
                if (times == control_time_inh[0]):
                    ax[i,1].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1, label=cntrl_time_legend[0])
                else:
                    ax[i,1].axvspan(times, times+dt, facecolor=facecol, alpha=al, zorder=-1)
    
    
    leg_col_number = 2 + int(len(control_array)/2)
    leg = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol = leg_col_number)
    leg.set_in_layout(True)

    cols = ['Excitatory', 'Inhibitory']
            
    for a, col in zip(ax[0,:], cols):
        a.annotate(col, xy=(0.5, 1.05), xytext=(0,5), xycoords='axes fraction', textcoords='offset points',
                   size=20, ha='center', va='baseline', weight='bold')
    
    fig.tight_layout()
        
    if not filename_ == '':
        plt.savefig(os.path.join(path_, filename_), bbox_inches='tight')
        
        
def plot_control_array(model, control_array, cost_total_array_, weights_array, t_sim_, t_sim_pre_, t_sim_post_, initial_params_,
                         target_, ylim_, path_, filename_ = '', transition_time_array = 0.,
                         precision_variables_ = [0,1]):
    
    dt = model.params.dt
    if model.name == "aln" or model.name == "aln-control":
        control_factor = model.params.C/1000.
    else:
        control_factor = 1.
        
    control_ = control_array[0,0]
        
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    i3 = int(round( (t_sim_pre_ + transition_time_array[0] * t_sim_) / dt, 1) + 1 )
    
    N = model.params.N
    T = control_.shape[2]
    time_ = np.arange(0,T*dt,dt)
    
    init_vars = model.init_vars
    for iv in range(len(init_vars)):
        if (type(model.params[init_vars[iv]]) == np.float64 or type(model.params[init_vars[iv]]) == float):
            model.params[init_vars[iv]] = initial_params_[iv]
        elif model.params[init_vars[iv]].ndim == 2:
            model.params[init_vars[iv]][0,:] = initial_params_[iv]
        else:
            model.params[init_vars[iv]][0] = initial_params_[iv]
    
    columns = control_array.shape[0]
    rows = control_array.shape[1]
            
    # x axis energy, y axis sparsity
    fig, ax = plt.subplots(rows, columns, figsize=(24, 24) )#, linewidth=8, edgecolor='grey')
    plt.subplots_adjust(left=0.001, bottom=0.001, right=0.999, top=0.999, wspace=0.001, hspace=0.001)
    
    ##################### go through all controls in control array 
    
    for s_ in range(control_array.shape[1]):
        for e_ in range(control_array.shape[0]):
        
            control_ = control_array[e_,s_,:,:,:]
            
            if s_ == rows-1 and e_ == 0:
                ax[rows-s_-1,e_].plot(time_, control_[0,0,:] * control_factor, linewidth = 1., label="excitatory control current")
                ax[rows-s_-1,e_].plot(time_, control_[0,1,:] * control_factor, linewidth = 1., label="inhibitory control current")
                ax[rows-s_-1,e_].plot(time_, control_[0,2,:] * control_factor, linewidth = 1., label="EE control rate")
                ax[rows-s_-1,e_].plot(time_, control_[0,3,:] * control_factor, linewidth = 1., label="IE control rate")
                ax[rows-s_-1,e_].plot(time_, control_[0,4,:] * control_factor, linewidth = 1., label="EI control rate")
                ax[rows-s_-1,e_].plot(time_, control_[0,5,:] * control_factor, linewidth = 1., label="II control rate")
            else:
                ax[rows-s_-1,e_].plot(time_, control_[0,0,:] * control_factor, linewidth = 1.)
                ax[rows-s_-1,e_].plot(time_, control_[0,1,:] * control_factor, linewidth = 1.)
                ax[rows-s_-1,e_].plot(time_, control_[0,2,:] * control_factor, linewidth = 1.)
                ax[rows-s_-1,e_].plot(time_, control_[0,3,:] * control_factor, linewidth = 1.)
                ax[rows-s_-1,e_].plot(time_, control_[0,4,:] * control_factor, linewidth = 1.)
                ax[rows-s_-1,e_].plot(time_, control_[0,5,:] * control_factor, linewidth = 1.)
                
            ax[rows-s_-1,e_].text(0.97, 0.97, str(r'${:,.0f}$'.format(cost_total_array_[e_,s_])), verticalalignment='top',
                                  horizontalalignment='right', transform=ax[rows-s_-1,e_].transAxes, fontsize=14,
                                  bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 1})
            
            #####################
    
            ax[rows-s_-1,e_].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='grey', alpha=0.1, zorder=-2)
            ax[rows-s_-1,e_].axvspan(t_sim_pre_, t_sim_pre_ + transition_time_array[s_] * t_sim_, facecolor='grey',
                                     alpha=0.1, zorder=-1, hatch='///', edgecolor = 'g')
            
            ax[rows-s_-1,e_].tick_params(axis='y', which='minor', labelsize=7)
            #ax[rows-s_-1,e_].tick_params(labelrotation=90)
            ax[rows-s_-1,e_].set_ylim(ylim_[0], ylim_[1])
            if e_ != 0:
                ax[rows-s_-1,e_].get_yaxis().set_ticklabels([])
            if s_ != 0:
                ax[rows-s_-1,e_].get_xaxis().set_ticklabels([])
    
    leg_col_number = 2
    leg = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol = leg_col_number)
    leg.set_in_layout(True)

    col_labels = [r'$w_s = $' + str(r'${:,.0f}$'.format(we_)) for we_ in weights_array[:,0,0]]
            
    for a, col in zip(ax[0,:], col_labels):
        a.annotate(col, xy=(0.5, 1.02), xytext=(0,5), xycoords='axes fraction', textcoords='offset points',
                   size=20, ha='center', va='baseline')
        
    row_labels = [r'$t_{trans} = $' + str(r'${:,.1f}$'.format(ws_)) for ws_ in weights_array[0,:,1]]
            
    for a, row in zip(ax[:,0], row_labels[::-1]):
        a.annotate(row, xy=(-0.25, 0.5), xytext=(0,5), xycoords='axes fraction', textcoords='offset points',
                   size=20, ha='left', va='center', rotation=90)
    
    fig.tight_layout()
        
    if not filename_ == '':
        plt.savefig(os.path.join(path_, filename_), bbox_inches='tight')