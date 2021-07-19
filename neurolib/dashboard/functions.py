import numpy as np
import plotly.graph_objs as go
from . import layout as layout

arrowhead_ = 2
arrowsize_ = 1
arrowwidth_ = 2
arrowcolor_ = layout.darkgrey

def setmarkersize(index_, final_, trace_):
    s = list(trace_.marker.size)
    for ind_s in range(len(s)):
        if ind_s == index_:
            s[ind_s] = final_
    trace_.marker.size = s 
    
def setdefaultmarkersize(default_, trace_):
    s = list(trace_.marker.size)
    for ind_s in range(len(s)):
        s[ind_s] = default_
    trace_.marker.size = s 
    
    
def get_x_arrow(x0,y0,xlen):
    
    reshape = 1.
    while np.abs(xlen) > 0.4:
        reshape *= 0.5
        xlen *= reshape
    while np.abs(xlen) < 0.02:
        reshape *= 2.
        xlen *= reshape
            
    arrow = go.Annotation(
        x=x0+xlen,
        y=y0,
        xref="x",
        yref="y",
        showarrow=True,
        arrowhead=arrowhead_,
        arrowsize=arrowsize_,
        arrowwidth=arrowwidth_,
        arrowcolor=arrowcolor_,
        axref='x',
        ayref='y',
        ax=x0,
        ay=y0,
        )
    return arrow, reshape

def get_x_rescale_annotation(reshape,x0,y0,xlen):  
    ann = go.Annotation(
        x=x0+xlen*reshape,
        y=y0+0.02,
        xref="x",
        yref="y",
        text='*' + str(1./reshape),
        showarrow=False
        )
    return ann

def get_y_arrow(x0,y0,ylen):
    
    reshape = 1.
    
    while np.abs(ylen) > 0.8:
        reshape *= 0.5
        ylen *= reshape
    while np.abs(ylen) < 0.02:
        reshape *= 2.
        ylen *= reshape
                
    arrow = go.Annotation(
        x=x0,
        y=y0+ylen,
        xref="x",
        yref="y",
        showarrow=True,
        arrowhead=arrowhead_,
        arrowsize=arrowsize_,
        arrowwidth=arrowwidth_,
        arrowcolor=arrowcolor_,
        axref='x',
        ayref='y',
        ax=x0,
        ay=y0,
        )
    return arrow, reshape

def get_y_rescale_annotation(reshape,x0,y0,ylen):  
    ann = go.Annotation(
        x=x0+0.03,
        y=y0+0.9*ylen*reshape,
        xref="x",
        yref="y",
        text='*' + str(1./reshape),
        showarrow=False
        )
    return ann

def step_control(model, maxI_ = 1.):
    control_ = model.getZeroControl()
    for i_time in range(control_.shape[2]):
        if ( float(i_time/control_.shape[2]) < 0.1):
            control_[:,:1,i_time] = - maxI_
        elif ( float(i_time/control_.shape[2]) > 0.5 and float(i_time/control_.shape[2]) < 0.6 ):
            control_[:,:1,i_time] = maxI_
    return control_