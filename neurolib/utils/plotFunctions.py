import os
import numpy as np
import matplotlib.pyplot as plt

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
    
    
# plot uncontrolled dynamics, controlled dynamics
def plot_control(model, control_, t_sim_, t_sim_pre_, t_sim_post_, initial_params_, target_, path_, filename_ = 'control_aln.png'):
    
    dt = model.params.dt
    if model.name == "aln":
        control_factor = model.params.C/1000.
    else:
        control_factor = 1.
    model.params.duration = (control_.shape[2] - 1.) * dt
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    
    init_vars = model.init_vars
    for iv in range(len(init_vars)):
        if (type(model.params[init_vars[iv]]) == np.float64 or type(model.params[init_vars[iv]]) == float):
            model.params[init_vars[iv]] = initial_params_[iv]
        elif model.params[init_vars[iv]].ndim == 2:
            model.params[init_vars[iv]][0,0] = initial_params_[iv]
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
        if model.name == "aln":
            if (np.abs(control_[0,1,t]) > cntrl_limit):
                control_time_inh.append(dt * t)
        
    fig, ax = plt.subplots(3, len(model.output_vars), figsize=(16, 12), linewidth=8, edgecolor='grey')
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    y_labels_rates = ['Rates exc. [Hz]', 'Rates inh. [Hz]', 'Adaptation current [pA]']
    y_labels_control = ['Control exc. [nA]', 'Control inh. [nA]']
    sim_legend = ['Rate', 'Rate', 'Adaptation current']
    target_legend = ['Target']
    cntrl_time_legend = ['Control > {} pA'.format(cntrl_limit_scaled * 1000), 'Control active']
    
    sim_legend = ['Rate', 'mue', 'sigma_bar', 'sigma_e', 'tau_e']
    y_labels_rates = ['Rates exc. [Hz]', 'mue', 'sigma_bar', 'sigma_e', 'tau_e']
    y_labels_control = ['Control exc. [nA]', 'Control inh. [nA]']
    
    if len(model.output_vars) > 1:
        for i in range(len(model.output_vars)):
            ax[0,i].plot(model.t, model[output_vars[i]][0,:], label=sim_legend[i])
            ax[0,i].set(xlabel='t [ms]', ylabel=y_labels_rates[i])
        #ax[0,0].axvspan(control_time_exc[0], control_time_exc[1], facecolor='0.1', alpha=0.2, zorder=-100)
        #ax[0,0].axvspan(0, 50, facecolor='0.7', alpha=0.2, zorder=-100)
        
        model.run(control=control_)
        
        for i in range(len(model.output_vars)):
            ax[1,i].plot(model.t, model[output_vars[i]][0,:], label=sim_legend[i])
            ax[1,i].set(xlabel='t [ms]', ylabel=y_labels_rates[i])
            
        
        for i in range(len(control_vars)):
            ax[2,i].plot(model.t, control_[0,i,:] * control_factor) # divide by five to take into account capacitance
            ax[2,i].set(xlabel='t [ms]', ylabel=y_labels_control[i])
    else:
        ax[0].plot(model.t, model[output_vars[0]][0,:], label=sim_legend[1])
        ax[0].set(xlabel='t [ms]', ylabel=y_labels_rates[1])
        
        model.run(control=control_)
        
        ax[1].plot(model.t, model[output_vars[0]][0,:], label=sim_legend[1])
        ax[1].set(xlabel='t [ms]', ylabel=y_labels_rates[1])
        ax[2].plot(model.t, control_[0,0,:] * control_factor) # divide by five to take into account capacitance
        ax[2].set(xlabel='t [ms]', ylabel=y_labels_control[1])
        

    if len(model.output_vars) > 1:
        
        """
        for i in range(3):
            for j in range(len(model.target_output_vars)):
                ax[i,j].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='0.4', alpha=0.1, zorder=-2,
                        label=cntrl_time_legend[1])
            for times in control_time_exc:
                if (times == control_time_exc[0]):
                    ax[i,0].axvspan(times, times+dt, facecolor='0.1', alpha=0.2, zorder=-1, label=cntrl_time_legend[0])
                else:
                    ax[i,0].axvspan(times, times+dt, facecolor='0.1', alpha=0.2, zorder=-1)
            for times in control_time_inh:
                if (times == control_time_inh[0]):
                    ax[i,1].axvspan(times, times+dt, facecolor='0.1', alpha=0.2, zorder=-1, label=cntrl_time_legend[0])
                else:
                    ax[i,1].axvspan(times, times+dt, facecolor='0.1', alpha=0.2, zorder=-1)
        """
    
    
        if (i2 == 0):
            for j in range(len(model.target_output_vars)):
                ax[0,j].plot(model.t[i1:], target_[0,j,:], '--', label=target_legend[0])
                ax[1,j].plot(model.t[i1:], target_[0,j,:], '--', label=target_legend[0])
        else:
            for j in range(len(model.target_output_vars)):
                ax[0,j].plot(model.t[i1:-i2], target_[0,j,:], '--', label=target_legend[0])
                ax[1,j].plot(model.t[i1:-i2], target_[0,j,:], '--', label=target_legend[0])
    
    else:
        """
        for i in range(3):
            ax[i].axvspan(t_sim_pre_, t_sim_pre_ + t_sim_, facecolor='0.4', alpha=0.1, zorder=-2,
                        label=cntrl_time_legend[1])
        """
    
        if (i2 == 0):
            ax[0].plot(model.t[i1:], target_[0,0,:], '--', label=target_legend[0])
            ax[1].plot(model.t[i1:], target_[0,0,:], '--', label=target_legend[0])
        else:
            ax[0].plot(model.t[i1:-i2], target_[0,0,:], '--', label=target_legend[0])
            ax[1].plot(model.t[i1:-i2], target_[0,0,:], '--', label=target_legend[0])
        
        
        #state = aln.getZeroState()
        #for i in range(len(output_vars)):
        #    state[:,i,:] = aln[output_vars[i]][:,:]
        #cost = aln.cost(state, target_, control_)
        
        #ax[2,2].plot(aln.t, cost)
        #ax[2,2].set(xlabel='t [ms]', ylabel='Cost')
        
        #ax[2,2].plot([-2.75, 1.01], [1.03, 1.03], transform=ax[2,2].transAxes, clip_on=False, c='grey', linewidth = 3)
        #ax[2,2].plot([-0.2, -0.2], [3.45, -0.15], transform=ax[2,2].transAxes, clip_on=False, c='grey', linewidth = 3)
    

    """
    for i in range(2):
        for j in range(len(model.output_vars)):
            ax[i,j].legend(loc='upper right')
            
    rows = ['Uncontrolled', 'Controlled']
            
    for a, row in zip(ax[:,0], rows):
        a.annotate(row, xy=(-0.05, 0.5), xytext=(-a.yaxis.labelpad - 15, 0), rotation = 90,
                xycoords=a.yaxis.label, textcoords='offset points', size=20, ha='right', va='center', weight='bold')
        
    cols = ['Excitatory', 'Inhibitory']
            
    for a, col in zip(ax[0,:], cols):
        a.annotate(col, xy=(0.5, 1.05), xytext=(0,5), xycoords='axes fraction', textcoords='offset points',
                   size=20, ha='center', va='baseline', weight='bold')
    """
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_, filename_))