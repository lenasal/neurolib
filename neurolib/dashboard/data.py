import numpy as np
import plotly.graph_objs as go
import pickle
import os
from pathlib import Path

from . import layout as layout
from . import functions as f

background_color = layout.getcolors()[0]
cmap = layout.getcolormap()

background_dx_ = 0.025
background_dy_ = background_dx_

step_current_duration = layout.step_current_duration
max_step_current = layout.max_step_current

def set_parameters(model):
    model.params.sigma_ou = 0.
    model.params.mue_ext_mean = 0.
    model.params.mui_ext_mean = 0.
    model.params.ext_exc_current = 0.
    model.params.ext_inh_current = 0.
    
    # NO ADAPTATION
    model.params.IA_init = 0.0 * np.zeros((model.params.N, 1))  # pA
    model.params.a = 0.
    model.params.b = 0.
    
def remove_from_background(x, y, exc_1, inh_1):
    i = 0
    while i in range(len(x)):
        for j in range(len(exc_1)):
            if np.abs(x[i] - exc_1[j]) < 1e-4 and np.abs(y[i] - inh_1[j]) < 1e-4:
                x = np.delete(x, i)
                y = np.delete(y, i)
                break
        i += 1
    
    return x, y

def get_data_background(exc_1, inh_1, exc_2, inh_2, exc_3, inh_3, exc_4, inh_4):
    background_x, background_y = get_background(layout.x_plotrange[0], layout.x_plotrange[1], background_dx_,
                                                layout.y_plotrange[0], layout.y_plotrange[1], background_dy_)
    
    background_x, background_y = remove_from_background(background_x, background_y, exc_1, inh_1)
    background_x, background_y = remove_from_background(background_x, background_y, exc_2, inh_2)
    background_x, background_y = remove_from_background(background_x, background_y, exc_3, inh_3)
    background_x, background_y = remove_from_background(background_x, background_y, exc_4, inh_4)
    
    return go.Scatter(
    x=background_x,
    y=background_y,
    marker=dict(
        line=dict(
            width=2,
            color=background_color,
            ),
        color=background_color,
        size=[0] * len(background_x),
        symbol='x-thin',
    ),
    mode='markers',
    name='Background',
    hoverinfo='x+y',
    opacity=1.,
    showlegend=False,
    )

def get_background(xmin, xmax, dx, ymin, ymax, dy):
    x_range = np.arange(xmin,xmax+dx,dx)
    y_range = np.arange(ymin,ymax+dy,dy)

    n_x = len(x_range)
    n_y = len(y_range)

    background_x = np.zeros(( n_x * n_y ))
    background_y = background_x.copy()

    j_ = 0

    for x_ in x_range:
        for y_ in y_range:
            background_x[j_] = x_
            background_y[j_] = y_
            j_ += 1
            
    return background_x, background_y

def get_time(model):
    return np.arange(0., step_current_duration/model.params.dt + model.params.dt, model.params.dt)

def plot_trace(model, x_, y_, trace0, trace1):
    model.params.duration = step_current_duration

    stepcontrol_ = model.getZeroControl()
    stepcontrol_ = f.step_control(model, maxI_ = max_step_current)

    model.params.ext_exc_current = x_ * 5.
    model.params.ext_inh_current = y_ * 5.
    time_ = get_time(model)

    model.run(control=stepcontrol_)
    
    trace0.x = time_
    trace0.y = model.rates_exc[0,:]
    
    trace1.x = time_
    trace1.y = model.rates_inh[0,:]
    
def get_step_current_traces(model):
    
    model.params.duration = step_current_duration
    stepcontrol_ = model.getZeroControl()
    stepcontrol_ = f.step_control(model, maxI_ = max_step_current)
    time_ = get_time(model)
    
    trace00 = go.Scatter(
        x=time_,
        y=stepcontrol_[0,0,:],
        xaxis="x2",
        yaxis="y2",
        name="External excitatory current [nA]",
        line_color=layout.darkgrey,
        showlegend=False,
        hoverinfo='x+y',
    )
    trace01 = go.Scatter(
        x=time_,
        y=stepcontrol_[0,1,:],
        xaxis="x3",
        yaxis="y2",
        name="External inhibitory current[nA]",
        line_color='rgba' + str(cmap(0)),
        showlegend=False,
        hoverinfo='x+y',
        visible=False,
    )
    return trace00, trace01

def read_data(readpath, case):
    
    not_checked = []
    exc__ = []
    inh__ = []
    no_c__ = []
    both_c__ = []
    
    # exc only
    exc_1_ = []
    inh_1_ = []
    lenx_1_ = []
    leny_1_ = []
    
    # inh only
    exc_2_ = []
    inh_2_ = []
    lenx_2_ = []
    leny_2_ = []
                
    # no control
    exc_3_ = []
    inh_3_ = []
    lenx_3_ = []
    leny_3_ = []
    
    # control in both nodes
    exc_4_ = []
    inh_4_ = []
    lenx_4_ = []
    leny_4_ = []
    
    file_ = os.sep + 'bi' + '.pickle'
    
    
    if not Path(readpath + file_).is_file():
        print("data not found")
        return [exc__, inh__, no_c__, both_c__,
            exc_1_, inh_1_, lenx_1_, leny_1_ ,
            exc_2_, inh_2_, lenx_2_, leny_2_ ,
            exc_3_, inh_3_, lenx_3_, leny_3_ ,
            exc_4_, inh_4_, lenx_4_, leny_4_, None, None, None, None]
    

    with open(readpath + file_,'rb') as file:
        load_array= pickle.load(file)
    ext_exc = load_array[0]
    ext_inh = load_array[1]

    [bestControl_init, costnode_init, bestControl_0, costnode_0] = read_control(readpath, case)
    
    cost_node1 = []
    cost_node2 = []
    cost_node3 = []
    cost_node4 = []

    for i in range(len(ext_exc)):
        if type(bestControl_0[i]) is type(None):
            #print(i, " not checked yet")
            not_checked.append(i)
            continue
        elif np.amax(np.abs(bestControl_0[i][0,1,:])) < 1e-8 and np.amax(np.abs(bestControl_0[i][0,0,:])) > 1e-8:
            exc__.append(i)
            cost_node1.append(costnode_0[i])
            #print(i, " only excitatory current")
        elif np.amax(np.abs(bestControl_0[i][0,0,:])) < 1e-8 and np.amax(np.abs(bestControl_0[i][0,1,:])) > 1e-8:
            inh__.append(i)
            cost_node2.append(costnode_0[i])
            #print(i, " only inhibitory current")
        elif np.amax(np.abs(bestControl_0[i][0,0,:])) > 1e-8 and np.amax(np.abs(bestControl_0[i][0,1,:])) > 1e-8:
            #print(i, " control input in both nodes")
            both_c__.append(i)
            cost_node3.append(costnode_0[i])
        elif np.amax(np.abs(bestControl_0[i][0,0,:])) < 1e-8 and np.amax(np.abs(bestControl_0[i][0,1,:])) < 1e-8:
            #print(i, "no control input")
            no_c__.append(i)
            cost_node4.append(costnode_0[i])
        else:
            print(i, " no category")
        

    for i in range(len(ext_exc)):
        if i in exc__:
            exc_1_.append(ext_exc[i])
            inh_1_.append(ext_inh[i])

            lenx = np.amax(bestControl_0[i][0,0,:])
            if np.abs(np.amin(bestControl_0[i][0,0,:])) > np.abs(lenx):
                lenx = np.amin(bestControl_0[i][0,0,:])
            leny = np.amax(bestControl_0[i][0,1,:])
            if np.abs(np.amin(bestControl_0[i][0,1,:])) > np.abs(leny):
                leny = np.amin(bestControl_0[i][0,1,:])
            lenx_1_.append(lenx/5.)
            leny_1_.append(leny/5.)


    for i in range(len(ext_exc)):
        if i in inh__:
            exc_2_.append(ext_exc[i])
            inh_2_.append(ext_inh[i])

            lenx = np.amax(bestControl_0[i][0,0,:])
            if np.abs(np.amin(bestControl_0[i][0,0,:])) > np.abs(lenx):
                lenx = np.amin(bestControl_0[i][0,0,:])
            leny = np.amax(bestControl_0[i][0,1,:])
            if np.abs(np.amin(bestControl_0[i][0,1,:])) > np.abs(leny):
                leny = np.amin(bestControl_0[i][0,1,:])
            lenx_2_.append(lenx/5.)
            leny_2_.append(leny/5.)      

    for i in range(len(ext_exc)):
        if i in both_c__:
            exc_3_.append(ext_exc[i])
            inh_3_.append(ext_inh[i])

            lenx = np.amax(bestControl_0[i][0,0,:])
            if np.abs(np.amin(bestControl_0[i][0,0,:])) > np.abs(lenx):
                lenx = np.amin(bestControl_0[i][0,0,:])
            leny = np.amax(bestControl_0[i][0,1,:])
            if np.abs(np.amin(bestControl_0[i][0,1,:])) > np.abs(leny):
                leny = np.amin(bestControl_0[i][0,1,:])
            lenx_3_.append(lenx/5.)
            leny_3_.append(leny/5.)
            
    for i in range(len(ext_exc)):
        if i in no_c__:
            exc_4_.append(ext_exc[i])
            inh_4_.append(ext_inh[i])

            lenx = np.amax(bestControl_0[i][0,0,:])
            if np.abs(np.amin(bestControl_0[i][0,0,:])) > np.abs(lenx):
                lenx = np.amin(bestControl_0[i][0,0,:])
            leny = np.amax(bestControl_0[i][0,1,:])
            if np.abs(np.amin(bestControl_0[i][0,1,:])) > np.abs(leny):
                leny = np.amin(bestControl_0[i][0,1,:])
            lenx_4_.append(lenx/5.)
            leny_4_.append(leny/5.)
                        
    return [exc__, inh__, both_c__, no_c__, 
            exc_1_, inh_1_, lenx_1_, leny_1_ ,
            exc_2_, inh_2_, lenx_2_, leny_2_ ,
            exc_3_, inh_3_, lenx_3_, leny_3_ ,
            exc_4_, inh_4_, lenx_4_, leny_4_,
            cost_node1, cost_node2, cost_node3, cost_node4]

def read_control(readpath, case):
    
    with open(readpath + os.sep + 'control_init_' + str(case) + '.pickle','rb') as file:
        load_array = pickle.load(file)

    bestControl_init = load_array[0]
    bestState_init = load_array[1]
    cost_init = load_array[2]
    runtime_init = load_array[3]
    grad_init = load_array[4]
    phi_init = load_array[5]
    costnode_init = load_array[6]
    weights_init = load_array[7] 
    
    with open(readpath + os.sep + 'control_0_' + str(case) + '.pickle','rb') as file:
        load_array = pickle.load(file)

    bestControl_0 = load_array[0]
    bestState_0 = load_array[1]
    cost_0 = load_array[2]
    runtime_0 = load_array[3]
    grad_0 = load_array[4]
    phi_0 = load_array[5]
    costnode_0 = load_array[6]
    weights_0 = load_array[7]    
    
    return [bestControl_init, costnode_init, bestControl_0, costnode_0]

def get_scatter_data(exc_1, inh_1, exc_2, inh_2, exc_3, inh_3, exc_4, inh_4):

    data1 = go.Scatter(
        x=exc_1,
        y=inh_1,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(3)),
                ),
            color='rgba' + str(cmap(3)),
            size=[layout.markersize] * len(exc_1)
        ),
        mode='markers',
        name='Excitatory current only',
        hoverinfo='x+y',
        uid='123bla',
        )
    
    if len(exc_1) == 0:
        data1.x = [None]
        data1.y = [None]

    data2 = go.Scatter(
        x=exc_2,
        y=inh_2,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(0)),
                ),
            color='rgba' + str(cmap(0)),
            size=[layout.markersize] * len(exc_2),
        ),
        mode='markers',
        name='Inhibitory current only',
        hoverinfo='x+y',
        uid='2',
        )
    
    if len(exc_2) == 0:
        data2.x = [None]
        data2.y = [None]
    
    
    data3 = go.Scatter(
        x=exc_3,
        y=inh_3,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(2)),
                ),
            color='rgba' + str(cmap(2)),
            size=[layout.markersize] * len(exc_3)
        ),
        mode='markers',
        name='Control in both nodes',
        hoverinfo='x+y',
        uid='3',
        )
    
    if len(exc_3) == 0:
        data3.x = [None]
        data3.y = [None]

    data4 = go.Scatter(
        x=exc_4,
        y=inh_4,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(7)),
                ),
            color='rgba' + str(cmap(7)),
            size=[layout.markersize] * len(exc_4),
        ),
        mode='markers',
        name='No control result',
        hoverinfo='x+y',
        uid='4',
        )
    
    if len(exc_4) == 0:
        data4.x = [None]
        data4.y = [None]
    
    return data1, data2, data3, data4

def update_data(fig, e1, i1, e2, i2, e3, i3, e4, i4):
    
    data1 = fig.data[1]
    data1.x = e1
    data1.y = i1
    data1.marker.size=[layout.markersize] * len(e1)
    if len(e1) == 0:
        data1.x = [None]
        data1.y = [None]
        
    data2 = fig.data[2]
    data2.x = e2
    data2.y = i2
    data2.marker.size=[layout.markersize] * len(e2)
    if len(e2) == 0:
        data2.x = [None]
        data2.y = [None]
        
    data3 = fig.data[3]
    data3.x = e3
    data3.y = i3
    data3.marker.size=[layout.markersize] * len(e3)
    if len(e3) == 0:
        data3.x = [None]
        data3.y = [None]
        
    data4 = fig.data[4]
    data4.x = e4
    data4.y = i4
    data4.marker.size=[layout.markersize] * len(e4)
    if len(e4) == 0:
        data4.x = [None]
        data4.y = [None]