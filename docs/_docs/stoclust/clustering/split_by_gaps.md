---
layout: docs
title: split_by_gaps
parent: clustering
def: 'split_by_gaps(vec, num_gaps = 1, group = None)'
excerpt: 'Aggregates the indices of a vector based on gaps between index values. The number of gaps is specified, and the largest gaps in the sorted array are used to cluster values.'
permalink: /docs/clustering/split_by_gaps/
---

Aggregates the indices of a vector based on gaps between index values.
The number of gaps is specified,
and the largest gaps in the sorted array
are used to cluster values.

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `vec` | | `np.ndarray` | A one-dimensional array of values.|
| `num_gaps` | Keyword | `int` | The number of gaps to use to break `vec` into clusters.|
| `group` | Keyword | `Group` | The group which labels the indices of `vec`, and which will be the item set of the returned `Aggregation`. |

## Example

A visual example works best for understanding
gap clustering. Below we generate a random vector
of 50 components, apply gap clustering with 3 gaps,
and plot the sorted vector components colored by cluster.

{% highlight python %}
import stoclust.clustering as clust
import stoclust.visualization as viz
import numpy as np

vec = np.random.rand(50)
agg = clust.split_by_gaps(vec,num_gaps=3)

rank = np.empty_like(vec)
rank[np.argsort(vec)] = np.arange(len(vec))

fig = viz.scatter2D(
    rank,vec,agg=agg,mode='markers',
    layout=dict(
        title = 'Sorted vector components (gap clustering)',
        xaxis = dict(
            title='Sorted rank'
        ),
        yaxis = dict(
            title='Value'
        ),
        legend = dict(
            title='Cluster'
        )
    )
)

fig.show()
{% endhighlight %}
<iframe
  src="/stoclust/assets/html/clustering/split_by_gaps.html"
  style="width:100%; height:400px;"
></iframe>