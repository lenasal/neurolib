import numpy as np
import plotly.graph_objs as go
import matplotlib.cm as cm
from pathlib import Path
import os

#### MEASURES FOR FIGURE LAYOUT

bifurcation_width = 700.
bifurcation_height = 1000.

traces_width = 600.
traces_height = 300.

x_plotrange = [0.,0.7]
y_plotrange = [0.,1.]

grid_resolution = 0.025
grid_resolution_granular = 0.005

x1_axis_end = 1.
y1_axis_end = x1_axis_end * ( bifurcation_width / bifurcation_height ) * ( y_plotrange[1] / x_plotrange[1] )

legend_x = x1_axis_end - 0.01
legend_y = 0.01

info_x = 0.
info_y = 1.
label_bistable_x = 0.78
label_bistable_y = 0.75
label_LC_x = 0.35
label_LC_y = 0.15
label_oscup_x = 0.48
label_oscup_y = 0.3
label_down_x = 0.3
label_down_y = 0.5
label_up_x = 0.75
label_up_y = 0.25

time_axis_length = 0.26
y2_axis_height = 0.24

x2_axis_start = x1_axis_end + 0.01
x2_axis_end = x2_axis_start + time_axis_length
x3_axis_end = 1.
x3_axis_start = x3_axis_end - time_axis_length

y3_axis_end = 0.97
y3_axis_start = y1_axis_end + 0.05
y2_axis_end = y3_axis_end
y2_axis_start = y2_axis_end - y2_axis_height

exc_x = (x2_axis_end + x2_axis_start)/2.
inh_x = (x3_axis_end + x3_axis_start)/2.
e_i_y = 1.

y_buttons = info_y - 0.001
dist_buttons = 0.048
x_button = 0.19

control_trace_x = x1_axis_end + 0.02
control_trace_y = y1_axis_end
control_trace_size = 0.6

markersize = 8
background_markersize = 18

step_current_duration = 2000.
max_step_current = 3.
simulation_duration = 400.


def get_rgb_string_from_rgba(rgba_val):
    rgb_array = [int(float(rgba_val[0]) * 255.),
                 int(float(rgba_val[1]) * 255.),
                 int(float(rgba_val[2]) * 255.)]
    rgb_str = 'rgb(' + str(rgb_array[0]) + ',' + str(rgb_array[1]) + ',' +str(rgb_array[2]) + ')'
    return rgb_str

cmap=cm.get_cmap('tab10')

darkgrey = 'rgb(100,100,100)'
midgrey = 'rgb(200,200,200)'
lightgrey='rgb(250,250,250)'
color_bi_updown = get_rgb_string_from_rgba(cmap(9))
color_LC = get_rgb_string_from_rgba(cmap(8))
color_bi_uposc = get_rgb_string_from_rgba(cmap(6))

def getcolors():
    return darkgrey, midgrey, lightgrey, color_bi_updown, color_LC, color_bi_uposc

def getcolormap():
    return cmap

text_fontsize = 14
buttons_fontsize = 12

font_info = dict(
        size=text_fontsize,
        color=darkgrey,
        )

font_buttons = dict(
        size=buttons_fontsize,
        color=darkgrey,
        )


##### LABELS
def get_info_text():
    return go.Annotation(
        x=info_x,
        y=info_y,
        text=("<b>Task:</b><br><br>"
              "<b>Cost measure:</b><br><br>"
              "<b>Cost weights:</b><br><br>"
              "<b>Initial guess:</b><br><br>"
              "<b>Sim. duration:"),#"</b><br><br>"
              #"<b>Precision cost:"),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=font_info,
        align="left",
        bordercolor=darkgrey,
        borderwidth=1,
        borderpad=8,
        bgcolor=lightgrey,
        opacity=0.8,
        captureevents=True,
        )

def get_label_bistable():
    return go.Annotation(
        x=label_bistable_x,
        y=label_bistable_y,
        text=("bistable<br>up/ down"),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=text_fontsize,
            color=color_bi_updown,
            ),
        align="center",
        )

def get_label_osc():
    return go.Annotation(
        x=label_LC_x,
        y=label_LC_y,
        text=("LC"),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=text_fontsize,
            color=color_LC,
            ),
        align="center",
        )

def get_label_osc_up():
    return go.Annotation(
        x=label_oscup_x,
        y=label_oscup_y,
        text=("bistable<br>up/ osc"),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=text_fontsize,
            color=color_bi_uposc,
            ),
        align="center",
        )

def get_label_down():
    return go.Annotation(
        x=label_down_x,
        y=label_down_y,
        text=("down"),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=text_fontsize,
            color=midgrey
            ),
        align="center",
        )

def get_label_up():
    return go.Annotation(
        x=label_up_x,
        y=label_up_y,
        text=("up"),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=text_fontsize,
            color=midgrey
            ),
        align="center",
        )

def get_label_exc():
    return go.Annotation(
        x=exc_x,
        xanchor='center',
        y=e_i_y,
        yanchor='top',
        text=("<b>Excitatory node</b>"),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=text_fontsize,
            color=darkgrey,
            ),
        align="center",
        )

def get_label_inh():
    return go.Annotation(
        x=inh_x,
        xanchor='center',
        y=e_i_y,
        yanchor='top',
        text=("<b>Inhibitory node</b>"),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=text_fontsize,
            color=darkgrey,
            ),
        align="center",
        )

def get_label_cost():
    return go.Annotation(
        x=control_trace_x,
        xanchor='left',
        y=-0.04,
        yanchor='bottom',
        text=("P:<br>"
              "S:<br>"
              "E:"),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=text_fontsize,
            color=darkgrey,
            ),
        align="left",
        )

def change_cost_layout(cost, case_):
    if True:
        return ("P:   " + str(round(cost[0][0][0],4)) + " (e) + " + str(round(cost[0][0][1],4)) + " (i)" + "<br>" # precision, network node, type node
                "S:   " + str(round(cost[2][0][0],4)) + " (e) + " + str(round(cost[2][0][1],4)) + " (i)" + "<br>"
                "E:   " + str(round(cost[1][0][0],4)) + " (e) + " + str(round(cost[1][0][1],4)) + " (i)" + "<br>")
    else:
        return ("P:   " + str(round(cost[0][0][0],4)) + " (e) + " + str(round(cost[0][0][1],4)) + " (i)" + "<br>" # precision, network node, type node
                "S:   " + str(round(cost[2][0][2],4)) + " (ee) + " + str(round(cost[2][0][3],4)) + " (ei) + " + str(round(cost[2][0][4],4)) + " (ie) + " + str(round(cost[2][0][5],4)) + " (ii)""<br>"
                "E:   " + str(round(cost[1][0][2],4)) + " (ee) + " + str(round(cost[1][0][3],4)) + " (ei) + " + str(round(cost[1][0][4],4)) + " (ie) + " + str(round(cost[1][0][5],4)) + " (ii)""<br>")


def boundary_path(p_e, p_i):
    polygon = "M" + str(p_e[0]) + "," + str(p_i[0])
    for i in range(1,len(p_e)):
        polygon = polygon + "L" + str(p_e[i]) + "," + str(p_i[i])
    polygon = polygon + ("Z")
    return polygon

def get_layout_bifurcation():
    return go.Layout(
    width = bifurcation_width,
    height = bifurcation_height,
    margin=dict(l=10, r=10, t=10, b=10, pad=0),
    paper_bgcolor=midgrey,
    plot_bgcolor=lightgrey,
    hovermode='closest',
    legend=dict(
        yanchor="bottom",
        y=legend_y,
        xanchor="right",
        x=legend_x,
    ),
    #title=dict(text="Case 00000",font=dict(size=1,color=midgrey),pad=dict(l=2, r=2, t=2, b=2),),
    #legend_title="Legend Title",
    xaxis=dict(
        domain=[0., 1.],
        range=x_plotrange,
        constrain="domain",
        tick0=0.,
        dtick=0.1,
        gridcolor=midgrey,
        title=dict(
            text="External excitatory current [nA]",
            standoff=0.,
                  ),
    ),
    yaxis=dict(
        domain=[0., 1.],
        range=y_plotrange,
        #scaleanchor = "x",
        scaleratio = 1,
        tick0=0.,
        dtick=0.1,
        gridcolor=midgrey,
        title=dict(
            text="External inhibitory current [nA]",
            standoff=0.,
                  ),
        
    ),
    )

def get_layout_exc():
    
    return go.Layout(
    width = traces_width,
    height = traces_height,
    margin=dict(l=10, r=10, t=50, b=10, pad=0),
    paper_bgcolor=midgrey,
    plot_bgcolor=lightgrey,
    hovermode='closest',
    legend=dict(
        yanchor="bottom",
        y=legend_y,
        xanchor="right",
        x=legend_x,
    ),
    title=dict(text="Excitatory node",font=dict(size=24,color=darkgrey),pad=dict(l=00, r=2, t=2, b=2)),
    xaxis=dict(
        domain=[0., 1.],
        range=[0.,step_current_duration],
        constrain="domain",
        tick0=0.,
        dtick=step_current_duration/5.,
        gridcolor=midgrey,
        title=dict(
            text="Simulation time [ms]",
            standoff=0.,
                  ),
        position=0.,
        zeroline=True,
        zerolinecolor=darkgrey,
        zerolinewidth=1, 
    ), 
    yaxis=dict(
        domain=[0., 1.],
        anchor="x",
        range=[-max_step_current-1., max_step_current+1.],
        constrain="domain",
        tick0=0.,
        dtick=1.,
        gridcolor=midgrey,
        title=dict(
            text="Control current [nA]",
            standoff=0.,
                  ),
        #position=x2_axis_end,
        zeroline=True,
        zerolinecolor=darkgrey,
        zerolinewidth=1,
    ),
    yaxis2=dict(
        domain=[0., 1.],
        anchor="x",
        range=[0.,200.],
        constrain="domain",
        tick0=0.,
        dtick=25.,
        gridcolor=midgrey,
        title=dict(
            text="Activity [Hz]",
            standoff=10.,
                  ),
        side='right',
        zeroline=True,
        zerolinecolor=darkgrey,
        zerolinewidth=1,
        overlaying="y",
    ),
    )

def get_layout_cntrl_exc():
    
    return go.Layout(
    width = traces_width,
    height = 2. * traces_height,
    margin=dict(l=10, r=10, t=50, b=10, pad=0),
    paper_bgcolor=midgrey,
    plot_bgcolor=lightgrey,
    hovermode='closest',
    legend=dict(
        yanchor="bottom",
        y=legend_y,
        xanchor="right",
        x=legend_x,
    ),
    title=dict(text="Excitatory node",font=dict(size=24,color=darkgrey),pad=dict(l=0, r=2, t=2, b=2)),
    xaxis=dict(
        domain=[0., 1.],
        range=[0.,simulation_duration],
        constrain="domain",
        tick0=0.,
        dtick=simulation_duration/8.,
        gridcolor=midgrey,
        title=dict(
            text="Simulation time [ms]",
            standoff=0.,
                  ),
        position=0.,
        zeroline=True,
        zerolinecolor=darkgrey,
        zerolinewidth=1, 
    ), 
    yaxis=dict(
        domain=[0.55, 1.],
        anchor="x",
        range=[0., 75.],
        constrain="domain",
        tick0=0.,
        dtick=25.,
        gridcolor=midgrey,
        title=dict(
            text="Activity [Hz]",
            standoff=0.,
                  ),
        zeroline=True,
        zerolinecolor=darkgrey,
        zerolinewidth=1,
    ),
    xaxis2=dict(
        domain=[0., 1.],
        range=[0.,simulation_duration],
        constrain="domain",
        tick0=0.,
        dtick=simulation_duration/8.,
        gridcolor=midgrey,
        title=dict(
            text="Simulation time [ms]",
            standoff=0.,
                  ),
        position=0.,
        zeroline=True,
        zerolinecolor=darkgrey,
        zerolinewidth=1, 
    ),
    yaxis2=dict(
        domain=[0., 0.45],
        anchor="x",
        range=[-0.2,2.],
        constrain="domain",
        tick0=0.,
        dtick=0.4,
        gridcolor=midgrey,
        title=dict(
            text="Control current [nA]",
            standoff=10.,
                  ),
        zeroline=True,
        zerolinecolor=darkgrey,
        zerolinewidth=1,
    ),
    )

def get_button():
    return dict(
        buttons=[],
        direction="down",
        pad={"r": 10, "t": 0},
        showactive=True,
        x=x_button,
        xanchor="left",
        yanchor="top",
        font=font_buttons,
        bordercolor=darkgrey,
        borderwidth=1,
        )

def get_updatemenus():
    
    # relayout: args properties of annotation
    # update: args properties of data
    # animate: args properties of data
    # restyle:
    
    button0 = get_button()
    button0.update(buttons=list([
                dict(
                    args=[{'updatemenus[0].active':0}],
                    label="Low to high",
                    method="relayout",
                ),
                dict(
                    args=[{'updatemenus[0].active':1}],
                    label="High to low",
                    method="relayout",
                ),
                ]))
    button0.update(y=y_buttons)
    
    button1 = get_button()
    button1.update(buttons=list([
                dict(
                    args=[{'updatemenus[1].active':0}],
                    label="Sparsity",
                    method="relayout"
                ),
                dict(
                    args=[{'updatemenus[1].active':1}],
                    label="Energy",
                    method="relayout"
                ),
            ]))
    button1.update(y=y_buttons-dist_buttons)     
    
    return [button0, button1]

def get_updatemenus_final():
    
    # relayout: args properties of annotation
    # update: args properties of data
    # animate: args properties of data
    # restyle:
    
    button0 = get_button()
    button0.update(buttons=list([
                dict(
                    args=[{'updatemenus[0].active':0}],
                    label="Low to high",
                    method="relayout",
                ),
                dict(
                    args=[{'updatemenus[0].active':1}],
                    label="High to low",
                    method="relayout",
                ),
                ]))
    button0.update(y=y_buttons)
    
    button1 = get_button()
    button1.update(buttons=list([
                dict(
                    args=[{'updatemenus[1].active':0}],
                    label="Sparsity",
                    method="relayout"
                ),
                dict(
                    args=[{'updatemenus[1].active':1}],
                    label="Energy",
                    method="relayout"
                ),
            ]))
    button1.update(y=y_buttons-dist_buttons)     
    
    return [button0, button1]
        
def get_img(img_path, control_case, index):
    img_source = img_path + str(control_case) + os.sep + str(index) + '.png'
        
    if not Path(img_source).is_file():
        print("image not found")
        return dict()
    
    #print("show image with path", img_source)
    return dict(
        source=img_source,
        xref="paper",
        yref="paper",
        #sizing="contain",
        x=control_trace_x,
        y=control_trace_y,
        sizex=control_trace_size,
        sizey=control_trace_size,
        opacity=1.,
        xanchor="left",
        yanchor="top",
        )

def get_empty_traces():
    trace10 = go.Scatter(
        x=[],
        y=[],
        xaxis="x",
        yaxis="y2",
        name="Excitatory activity",
        line_color='rgba' + str(cmap(3)),
        showlegend=False,
        hoverinfo='x+y',
    )
    trace11 = go.Scatter(
        x=[],
        y=[],
        xaxis="x",
        yaxis="y2",
        name="Inhibitory activity",
        line_color='rgba' + str(cmap(0)),
        showlegend=False,
        hoverinfo='x+y',
    )
    return trace10, trace11

def get_empty_state():
    state_e = go.Scatter(
        x=[],
        y=[],
        xaxis="x",
        yaxis="y",
        name="Excitatory activity",
        line_color='rgba' + str(cmap(3)),
        showlegend=False,
        hoverinfo='x+y',
    )
    state_i = go.Scatter(
        x=[],
        y=[],
        xaxis="x",
        yaxis="y",
        name="Inhibitory activity",
        line_color='rgba' + str(cmap(0)),
        showlegend=False,
        hoverinfo='x+y',
    )
    return state_e, state_i

def get_empty_control():
    cntrl_e = go.Scatter(
        x=[],
        y=[],
        xaxis="x2",
        yaxis="y2",
        name="Control current to excitatory node",
        line_color='rgba' + str(cmap(3)),
        showlegend=False,
        hoverinfo='x+y',
    )
    cntrl_i = go.Scatter(
        x=[],
        y=[],
        xaxis="x2",
        yaxis="y2",
        name="Control current to inhibitory node",
        line_color='rgba' + str(cmap(0)),
        showlegend=False,
        hoverinfo='x+y',
    )
    return cntrl_e, cntrl_i

def get_bistable_paths(boundary_bi_exc, boundary_bi_inh):
    return dict(
        type="path",
        path=boundary_path(boundary_bi_exc, boundary_bi_inh),
        fillcolor=color_bi_updown,
        line_color=color_bi_updown,
        opacity=0.2,
        layer="below",
    )


def get_osc_path(boundary_LC_exc, boundary_LC_inh):
    return dict(
        type="path",
        path=boundary_path(boundary_LC_exc, boundary_LC_inh),
        fillcolor=color_LC,
        line_color=color_LC,
        opacity=0.2,
        layer="below",
    )


def get_LC_up_path(boundary_LC_up_exc, boundary_LC_up_inh):
    return dict(
        type="path",
        path=boundary_path(boundary_LC_up_exc, boundary_LC_up_inh),
        fillcolor=color_bi_uposc,
        line_color=color_bi_uposc,
        line_width=1.,
        opacity=0.2,
        layer="below",
    )