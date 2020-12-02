---
layout: docs
title: scatter3D
parent: visualization
def: 'scatter3D(x, y, z, agg=None, layout=None, show_items=None, **kwargs)'
excerpt: 'Generates a 3-dimensional scatter plot of given coordinate vectors; optionally plots them on separate traces based on an aggregation.'
permalink: /docs/visualization/scatter3D/
---
Generates a 3-dimensional scatter plot of given coordinate vectors; optionally plots them on separate traces based on an aggregation.
Returns a [Plotly `Figure`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html).

## Additional Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `x` | | `np.ndarray` | The $$x$$-coordinates of the data points. |
| `y` | | `np.ndarray` | The $$y$$-coordinates of the data points. |
| `z` | | `np.ndarray` | The $$y$$-coordinates of the data points. |
| `agg` | Keyword | `Aggregation` | An Aggregation of the indices of `x`, `y` and `z`. |
| `show_items` | Keyword | `np.ndarray` | A one-dimensional array of which indices of `x`, `y` and `z` are to be shown. |
| `layout` | Keyword | `dict` | Updates values for the [Plotly `Figure` `layout`](https://plotly.com/python/reference/layout/). |
| `**kwargs` | Further keyword arguments | | Keyword arguments for the [Plotly Scatter3d trace](https://plotly.com/python/reference/scatter3d/). If an attribute is given as a single string or float, will be applied to all data points.  If as an array of length `x.shape[0]`, will be applied separately to each data point. If an an array of length `agg.clusters.size`, will be applied separately to each cluster. |