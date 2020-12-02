---
layout: docs
title: dendrogram
parent: visualization
def: 'dendrogram(hier, line=None, layout=None, show_progress=False, **kwargs)'
excerpt: 'Generates a stacked bar plot of a given array of vectors.'
permalink: /docs/visualization/dendrogram/
---
Generates a dendrogram of a hierarchical clustering scheme. Uses [Plotly Shapes](https://plotly.com/python/shapes/) to draw the dendrogram and a scatter plot to highlight clusters at their branching points.
Returns a [Plotly `Figure`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html).

## Additional Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `hier` | | `Hierarchy` | A `Hierarchy` which is to be plotted as a dendrogram. |
| `line` | Keyword | `dict` | Formatting information for the [Plotly line](https://plotly.com/python/reference/layout/shapes/#layout-shapes-items-shape-line) shape. If an attribute is given as a single string or float, will be applied to all lines. If as an array of length hier.clusters.size, will be applied separately to the lines immediately beneath each cluster. |
| `layout` | Keyword | `dict` | Updates values for the [Plotly `Figure` `layout`](https://plotly.com/python/reference/layout/). |
| `show_progress` | Keyword | `bool` | Whether to show a tqdm progress bar as the dendrogram is generated. |
| `**kwargs` | Further keyword arguments | | Keyword arguments for the [Plotly `Scatter` trace](https://plotly.com/python/reference/scatter/). If an attribute is given as a single string or float, will be applied to all branch points. If as an array of length `hier.clusters.size`, will be applied separately to each cluster's branch point.|