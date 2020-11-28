import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def heatmap(mat,**kwargs):
    which_x = kwargs.get('show_x',np.arange(mat.shape[1]))
    which_y = kwargs.get('show_y',np.arange(mat.shape[0]))
    xlabels = kwargs.get('xlabels',np.arange(mat.shape[1]))
    ylabels = kwargs.get('ylabels',np.arange(mat.shape[0]))
    colorscale = kwargs.get('colorscale',px.colors.sequential.Viridis)
    zmid = kwargs.get('zmid',None)
    fig = go.Figure(data=go.Heatmap(
                        z=mat[which_y][:,which_x],zmid=zmid,colorscale=colorscale))
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = np.arange(len(which_x)),
            ticktext = xlabels[which_x]
        ),
        yaxis = dict(
            tickmode = 'array',
            tickvals = np.arange(len(which_y)),
            ticktext = ylabels[which_y]
        ),
        margin=dict(l=100, r=100, t=20, b=20),
    )
    return fig

def scatter3D(vecs,**kwargs):
    id_label = kwargs.get('id_label','ID')
    color_label = kwargs.get('color_label','Color')
    item_labels = kwargs.get('item_labels',None)
    show_items = kwargs.get('show_items',None)
    if len(vecs.shape)==2:
        keywords = {k:[v] for k,v in kwargs.items() if not(k in ['id_label', 'color_label', 'item_labels','show_items'])}
        vecs = np.stack([vecs])
    else:
        keywords = {k:v for k,v in kwargs.items() if not(k in ['id_label', 'color_label', 'item_labels','show_items'])}
    if show_items is None:
        show_items = np.stack([np.arange(vecs.shape[2])]*vecs.shape[0])
    if item_labels is None:
        item_labels = np.arange(vecs.shape[2])
    
    fig = go.Figure(data=[go.Scatter3d(x=vecs[j,0,show_items[j]],
                                y=vecs[j,1,show_items[j]],
                                z=vecs[j,2,show_items[j]],
                                   mode='markers',
                                   marker={k:v[j] for k,v in keywords.items()},
                                   customdata=item_labels[show_items[j]],
                                   hovertemplate ='<i>'+id_label+'</i>: %{customdata}<br> <i>'+color_label+'</i>:%{marker.color} <br>')                   
                        for j in range(vecs.shape[0])])
    return fig

def scatter2D(vecs,**kwargs):
    id_label = kwargs.get('id_label','ID')
    color_label = kwargs.get('color_label','Color')
    item_labels = kwargs.get('item_labels',None)
    show_items = kwargs.get('show_items',None)
    names = kwargs.get('names',None)
    if len(vecs.shape)==2:
        keywords = {k:[v] for k,v in kwargs.items() if not(k in ['id_label', 'color_label', 'item_labels','show_items','names'])}
        vecs = np.stack([vecs])
    else:
        keywords = {k:v for k,v in kwargs.items() if not(k in ['id_label', 'color_label', 'item_labels','show_items','names'])}
    if show_items is None:
        show_items = np.stack([np.arange(vecs.shape[2])]*vecs.shape[0])
    if item_labels is None:
        item_labels = np.arange(vecs.shape[2])
    if names is None:
        names = np.arange(vecs.shape[0]).astype(str)
    
    fig = go.Figure(data=[go.Scatter(x=vecs[j,0,show_items[j]],
                                y=vecs[j,1,show_items[j]],
                                   mode='markers',
                                   marker={k:v[j] for k,v in keywords.items()},
                                   customdata=item_labels[show_items[j]],
                                   hovertemplate ='<i>'+id_label+'</i>: %{customdata}<br> <i>'+color_label+'</i>:%{marker.color} <br>',
                                   name=str(names[j]))                   
                        for j in range(vecs.shape[0])])
    return fig