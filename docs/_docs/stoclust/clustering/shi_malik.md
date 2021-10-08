---
layout: docs
title: shi_malik
parent: clustering
def: 'shi_malik(st_mat, eig_thresh=0.95, tol=0, index=None)'
excerpt: 'Given a square column-stochastic matrix describing the strength of the relationship between pairs of items, determines an aggregation of the items using the spectral approach of Shi and Malik.'
permalink: /docs/clustering/shi_malik/
---

Given a square column-stochastic matrix (that is, one with `np.sum(st_mat,axis=1)=[1 ... 1]`) describing the strength
of the relationship between pairs of items,
determines an `Aggregation` of the items using
the spectral approach of Shi and Malik.

A column-stochastic matrix $$\mathbf{T}$$ will always have a leading
eigenvalue of 1 and a leading uniform right-eigenvector, 
$$\mathbf{u}=(1,...,1)$$, which is a fixed point of the map:

$$
\sum_{j}T_{ij} u_j = u_i
$$

If $$\mathbf{T}$$ has no disconnected components then u is the
unique fixed point (up to a constant scaling) 
and the sub-leading eigenvalue
is strictly less than one; otherwise, the eigenvalue
1 is degenerate. In the first case, if the sub-leading
eigenvalue is close to 1, then the sub-leading
right-eigenvector $$\mathbf{y}$$ may be used to partition the indices into
two slowly-decaying communities.

The Shi-Malik algorithm is recursive, taking
the sub-leading eigenvector of $$\mathbf{T}$$ (as long as the
corresponding eigenvalue is above a threshold),
using it to bipartition the indices, and then
repeating these steps on the partitions with a reweighted
matrix. This implementation cuts the vector $$\mathbf{T}$$ by value,
by default into components $$y_i>0$$ and $$y_i<=0$$, because of the
orthogonality relationship

$$
\left<\mathbf{y}\right>_{\pi} = \sum_i y_i \pi_i = 0
$$

which indicates that the mean value of $$\mathbf{y}$$
under the stationary distribution $$\pi$$(left-eigenvector of $$\mathbf{T}$$)
must always be zero, making this a value of significance.

The algorithm halts when no community has a sub-leading
eigenvector above the threshold, and the final partitioning
is returned as an `Aggregation`.

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `st_mat` | | `np.ndarray` | A square column-stochastic matrix describing a Markov dynamic.|
| `eig_thresh` | Keyword | `float` | The smallest value the subleading eigenvalue may have to continue the recursion. |
| `cut` | Keyword | `float` | The value used to "cut" the subleading eigenvector into two clusters. |
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
    pd.Index(['Left','Right']),
    {
        0:np.arange(0,n),
        1:np.arange(n,2*n)
    }
)
{% endhighlight %}
<iframe
  src="/stoclust/assets/html/clustering/disks2.html"
  style="width:100%; height:215px;"
></iframe><br>

Next, we apply the Shi-Malik algorithm
with `eig_thresh = 0.98`. This results
in a crisp division into two circles.

{% highlight python %}
dist = sc.distance.euclid(samples)

T = sc.utils.stoch(
    np.exp(-dist/0.05) - np.eye(dist.shape[0])
)

new_agg = clust.shi_malik(T,eig_thresh=0.98)

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
  src="/stoclust/assets/html/clustering/shi_malik_1.html"
  style="width:100%; height:215px;"
></iframe><br>

However, note that lowering the threshold
even a little to `0.95` results in the appearance
of spurious sub-clusters:

{% highlight python %}
dist = sc.distance.euclid(samples)

T = sc.utils.stoch(
    np.exp(-dist/0.05) - np.eye(dist.shape[0])
)

# Note the new value of eig_thresh!
new_agg = clust.shi_malik(T,eig_thresh=0.95)

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
  src="/stoclust/assets/html/clustering/shi_malik_0.html"
  style="width:100%; height:215px;"
></iframe>

## Reference

Jianbo Shi and J. Malik, "Normalized cuts and image segmentation," in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no. 8, pp. 888-905, Aug. 2000, [doi: 10.1109/34.868688](https://ieeexplore.ieee.org/document/868688).