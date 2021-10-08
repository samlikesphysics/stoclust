---
layout: docs
title: heatmap
parent: visualization
def: 'heatmap(mat, show_x=None, show_y=None, xlabels=None, ylabels=None,  layout=None, **kwargs)'
excerpt: 'Generates a heatmap of a given matrix: that is, displays the matrix as a table of colored blocks such that the colors correspond to matrix values.'
permalink: /docs/visualization/heatmap/
---
Generates a heatmap of a given matrix: that is, displays the matrix as a table of colored blocks such that the colors correspond to matrix values. Returns a [Plotly `Figure`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html).

## Additional Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `mat` | | `np.ndarray` | The matrix whose values are being visualized in a heatmap. |
| `show_x` | Keyword | `np.ndarray` | An array of the column indices which are to be shown, in the order they should be shown. |
| `show_y` | Keyword | `np.ndarray` | An array of the row indices which are to be shown, in the order they should be shown. |
| `xlabels` | Keyword | `np.ndarray` or `Index` | An array or group of how the columns should be labeled on the plot. |
| `ylabels` | Keyword | `np.ndarray` or `Index` | An array or group of how the rows should be labeled on the plot. |
| `layout` | Keyword | `dict` | Updates values for the [Plotly `Figure` `layout`](https://plotly.com/python/reference/layout/). |
| `**kwargs` | Further keyword arguments | | Keyword arguments for the [Plotly `Heatmap` trace](https://plotly.com/python/reference/heatmap/). |