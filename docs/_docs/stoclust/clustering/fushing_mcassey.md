---
layout: docs
title: fushing_mcassey
parent: clustering
def: 'fushing_mcassey(st_mat, max_visits=5, time_quantile_cutoff=0.95, index=None)'
excerpt: 'Given a stochastic matrix describing the strength of the relationship between pairs of items, determines an aggregation of the items using the regulated random walk approach of Fushing and McAssey.'
permalink: /docs/clustering/fushing_mcassey/
---

Given a square column-stochastic matrix 
(that is, one with `np.sum(st_mat,axis=1)=[1 ... 1]`) describing the strength
of the relationship between pairs of items,
determines an `Aggregation` of the items using
the regulated random walk approach of Fushing and McAssey.
The algorithm is inherently random
and highly unstable as a single-shot approach,
but may be used in an ensemble to determine a 
useful similarity matrix.

Suppose `st_mat` is given by the Markov matrix $$\mathbf{T}$$.
A regulated random walk is taken using $$\mathbf{T}$$ as the initial
transition probabilities, and modifying these probabilities
to remove from circulation any node which has been visited
at least `max_visits` times (this prevents the walk from
being stuck in a cluster for too long). The time between removals
is recorded; the highest values (determined by `time_quantile_cutoff`)
determine the number of clusters (it is interpreted that a sudden, long
removal time after many short removal times indicates 
one has left a highly-explored cluster and entered an unexplored one).

A node which was removed and for which $$>50\%$$ of its visits
prior to removal were in a particular time-interval is placed in the cluster
associated with that time interval; all other nodes remain unclustered.

This algorithm will not return useful results after a single run,
but if an ensemble of runs is collected it may be used to
derive a similarity matrix, based on how often two nodes are in
a cluster together over the many runs.

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `st_mat` | | `np.ndarray` | A square stochastic matrix describing a Markov dynamic. |
| `max_visits` | Keyword | `int` | The maximum number of visits to a node before it is removed in the regulated random walk. |
| `time_quantile_cutoff` | Keyword | `float` | The quantile of the length of time between node removals, which is used to determine the number of clusters. |
| `index` | Keyword | `Index` | The `Index` which labels the indices of `st_mat`, and which will be the item set of the returned `Aggregation`. |

## Example

First we will generate a sample dataset with two concentric rings:

{% highlight python %}
import stoclust.visualization as viz
import stoclust as sc
import numpy as np
import pandas as pd

n1 = 200
n2 = 50

samples = np.concatenate([
    sc.examples.gen_disk(
        num_samples=n1,
        rad1=1.3,rad2=1.6
    ),
    sc.examples.gen_disk.gen_disk(
        num_samples=n2,
        rad1=0,rad2=0.2
    )
])

agg = sc.Aggregation(
    pd.Index(np.arange(n1+n2)),
    pd.Index(['Outer','Inner']),
    {
        0:np.arange(0,n1),
        1:np.arange(n1,n1+n2)
    }
)

viz.scatter2D(
    samples[:,0],samples[:,1],
    agg=agg, mode='markers',
    text=agg.items.elements.astype(str),
    hoverinfo='text',
    layout = {
        'margin':dict(l=10, r=10, t=10, b=10)
    }
)
{% endhighlight %}
<iframe
  src="/stoclust/assets/html/clustering/rings.html"
  style="width:100%; height:375px;"
></iframe><br>

The Fushing-McAssey algorithm is not designed to be only
used once, like other clustering algorithms in this
package; rather, it is intended to be used with the
ensemble feature.

The following is also an example of applying
the `stoclust.ensemble.random_clustering` method,
which applies the given clustering method a given number
of times in order to generate an ensemble of
block matrices, which can then be averaged to give
an overall similarity matrix, which we visualize.

{% highlight python %}
dist = sc.distance.euclid(samples)
T = sc.utils.stoch(
    np.exp(-dist/0.05) - np.eye(dist.shape[0])
)

ens = sc.ensemble.random_clustering(
    T,
    clustering_method = sc.clustering.fushing_mcassey,
    ensemble_size = 500
)

fig = viz.heatmap(
    np.average(ens,axis=0)-np.eye(ens.shape[1])
)

fig.show()
{% endhighlight %}
<iframe
  src="/stoclust/assets/html/clustering/fm_heatmap.html"
  style="width:100%; height:350px;"
></iframe><br>

This similarity matrix shows a clear separation between
the first 200 indices (the outer ring) and the final 50 indices
(the inner ring). Normalizing this similarity matrix and using
the subleading eigenvalue for spectral clustering
gives the appropriate division:

{% highlight python %}
import scipy.linalg as la

TT = sc.utils.stoch(
    np.average(ens,axis=0)-np.eye(ens.shape[1])
)

new_agg = sc.clustering.shi_malik(
    TT,
    eig_thresh=np.sort(la.eig(TT)[0])[-2]
)

fig = viz.scatter2D(
    samples[:,0],samples[:,1],
    agg=new_agg, mode='markers',
    text=new_agg.items.elements.astype(str),
    hoverinfo='text',
    layout = {
        'margin':dict(l=10, r=10, t=10, b=10)
    }
)

fig.show()
{% endhighlight %}
<iframe
  src="/stoclust/assets/html/clustering/fushing_mcassey.html"
  style="width:100%; height:375px;"
></iframe>

## Reference

H. Fushing and M. P. McAssey, "Time, temperature, and data cloud geometry," in *Phys. Rev. E*, vol. 82, 061110, Dec. 2010, [doi: 10.1103/PhysRevE.82.061110](https://doi.org/10.1103/PhysRevE.82.061110).