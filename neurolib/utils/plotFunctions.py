import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams['axes.grid'] = True
plt.rcParams['hatch.linewidth'] = 1.

cmap=cm.get_cmap('viridis')

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
    
def plot_traces(model, control_):
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
    plt.show()
    
    
# plot uncontrolled dynamics, controlled dynamics
def plot_control(model, control_array, cost_node_array, t_sim_, t_sim_pre_, t_sim_post_, initial_params_, target_, path_, filename_ = '',
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
        control_ = control_array[c_ind]
        
        model.run(control=control_)
        
        ax[0,0].plot(model.t, model[output_vars[0]][0,:], linewidth = 1., color=colors_[c_ind+2],
                     label=labels_[c_ind] )
        ax[0,1].plot(model.t, model[output_vars[1]][0,:], linewidth = 1., color=colors_[c_ind+2])
            #ax[0,i].set(xlabel='t [ms]', ylabel=y_labels_rates[i])

        for i in range(columns):
            ax[1,i].plot(model.t, control_[0,i,:] * control_factor, linewidth = 1., color=colors_[c_ind+2] ) # divide by five to take into account capacitance
            ax[1,i].set(xlabel='t [ms]', ylabel=y_labels_control[1])
            ax[1,i].set_xlim([model.t[0],model.t[-1]])
            
            # ee, ei, ie, ii
            ax[2,i].plot(model.t, control_[0,i+2,:], linewidth = 1., color=colors_[c_ind+2] )
            ax[2,i].set(xlabel='t [ms]', ylabel=y_labels_control[2])
            ax[2,i].set_xlim([model.t[0],model.t[-1]])
            
            ax[3,i].plot(model.t, control_[0,i+4,:], linewidth = 1., color=colors_[c_ind+2] )
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