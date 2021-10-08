---
layout: docs
title: bars
parent: visualization
def: 'bars(mat, show_x=None, show_y=None, xlabels=None, ylabels=None, layout=None, **kwargs)'
excerpt: 'Generates a stacked bar plot of a given array of vectors.'
permalink: /docs/visualization/bars/
---
Generates a stacked bar plot of a given array of vectors; the rows index the horizontally separate bars and the columns index the stack heights.
Returns a [Plotly `Figure`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html).

## Additional Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `mat` | | `np.ndarray` | The matrix whose values are being visualized in a stacked bar plot. |
| `show_x` | Keyword | `np.ndarray` | An array of the row indices (horizontally separate bars) which are to be shown, in the order they should be shown. |
| `show_y` | Keyword | `np.ndarray` | An array of the column indices (stacked bars) which are to be shown, in the order they should be shown. |
| `xlabels` | Keyword | `np.ndarray` or `Index` | An array or group of how the rows should be labeled on the plot. |
| `ylabels` | Keyword | `np.ndarray` or `Index` | An array or group of how the columns should be labeled on the plot. |
| `layout` | Keyword | `dict` | Updates values for the [Plotly `Figure` `layout`](https://plotly.com/python/reference/layout/). |
| `**kwargs` | Further keyword arguments | | Keyword arguments for the [Plotly `Bar` trace](https://plotly.com/python/reference/bar/). If an attribute is given as a single string or float, will be applied to all bars. If as an array of length `mat.shape[1]`, will be applied separately to each layer of the stack.|