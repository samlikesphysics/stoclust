---
layout: docs
title: meyer_wessell
parent: clustering
def: 'meyer_wessell(st_mat, min_times_same = 5, vector_clustering = None, index = None)'
excerpt: 'Given a square column-stochastic matrix describing the strength of the relationship between pairs of items, determines an aggregation of the items using the dynamical approach of Meyer and Wessell..'
permalink: /docs/clustering/meyer_wessell/
---

Given a column-stochastic matrix (that is, one with `np.sum(st_mat,axis=1)=[1 ... 1]`) describing the strength
of the relationship between pairs of items,
determines an aggregation of the items using the dynamical
approach of Meyer and Wessell. The algorithm
is inherently random, though fairly stable, and so may
be used as a one-shot measure but will be more reliable
in an ensemble.

A column-stochastic matrix $$\mathbf{T}$$ will, by the Perron-Frobenius theorem,
have a uniform vector $$\mathbf{u} = (1,...,1)$$ as a fixed point, that is:

$$
\sum_{j}T_{ij} u_j = u_i
$$

This fixed point is unique as long as 
the stochastic matrix is not reducible
into disconnected components. If it is almost reducible
(that is, if there are strongly connected communities with
weak connections between them), the vector $$\mathbf{T}^t \mathbf{x}$$ for some
non-uniform $$\mathbf{x}$$ will achieve
uniformity among the connected components before achieving global
uniformity. 

The Meyer-Wessell approach relies on applying the
column-stochastic matrix $$\mathbf{T}$$ to a random initial vector $$\mathbf{x}$$ and
detecting communities by identifying clusters of components which
achieve uniformity long before global uniformity is reached.
This is done by iteratively applying $$\mathbf{T}$$ to $$\mathbf{x}$$ and, at each iteration,
performing some kind of vector clustering on $$\mathbf{T}^t \mathbf{x}$$. When
the resulting Aggregation ceases to change ove a long
enough number of iterations, it is returned as the final Aggregation.

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `st_mat` | | `np.ndarray` | A square column-stochastic matrix describing a Markov dynamic.|
| `min_times_same` | Keyword | `int` | The number of iterations after which, if the clustering has not changed, the algorithm halts. |
| `vector_clustering` | Keyword | `function` | The particular method of vector clustering which should be used in the algorithm. Should receive a vector as the sole input and return an Aggregation. |
| `index` | Keyword | `Index` | The `Index` which labels the indices of `st_mat`, and which will be the item set of the returned `Aggregation`. |

## Example

First we will generate a sample dataset
with two touching circles:

{% highlight python %}
import stoclust.visualization as viz
import stoclust as sc
import numpy as np
import pandas as pd

n = 500

samples = np.concatenate([
    sc.examples.gen_disk(num_samples=n,rad1=0,rad2=1),
    sc.examples.gen_disk(num_samples=n,rad1=0,rad2=1)+
    np.array([2,0])
])

agg = sc.Aggregation(
    pd.Index(np.arange(samples.shape[0])),
    pd.Inex(['Left','Right']),
    {
        0:np.arange(0,n),
        1:np.arange(n,2*n)
    }
)
{% endhighlight %}
<iframe
  src="/stoclust/assets/html/clustering/disks.html"
  style="width:100%; height:215px;"
></iframe><br>

Next, we apply the Meyer-Wessell algorithm
with `min_times_same = 20`. It experiences 
confusion at the boundary but otherwise
identifies the two dominant structures.

{% highlight python %}
dist = sc.distance.euclid(samples)

T = sc.utils.stoch(
    np.exp(-dist/0.05) - np.eye(dist.shape[0])
)

new_agg = clust.meyer_wessell(T,min_times_same=20)

fig = viz.scatter2D(
    samples[:,0],samples[:,1],
    agg=new_agg, mode='markers',
    text=new_agg.items.elements.astype(str),
    hoverinfo='text,name',
    layout = {
        'margin':dict(l=10, r=10, t=10, b=10)
    }
)

fig.show()
{% endhighlight %}
<iframe
  src="/stoclust/assets/html/clustering/meyer_wessell.html"
  style="width:100%; height:215px;"
></iframe>

## Reference

Meyer, Carl D. and Charles D. Wessell. “Stochastic Data Clustering.” SIAM J. Matrix Analysis and Application 33-4: 1214-1236. 2012, [doi: 10.1137/100804395](https://epubs.siam.org/doi/abs/10.1137/100804395).