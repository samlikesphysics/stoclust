---
layout: docs
title: split_by_quantiles
parent: clustering
def: 'split_by_quantiles(vec, quantiles=0.95, group = None)'
excerpt: 'Cuts the vector at specific quantiles rather than rigid values. Assumes right-continuity of the cumulative distribution function.'
permalink: /docs/clustering/split_by_quantiles/
---

Like [`split_by_vals`](/docs/clustering/split_by_vals), but cuts the vector at specific quantiles
rather than rigid values. Assumes right-continuity of the 
cumulative distribution function.

For a vector $$v_i$$ let

$$
F(x) = \frac{\left|\{i: v_i \leq x\}\right|}{\mathrm{len}( \mathbf{v})}
$$

be the ratio of components that are less than or equal to $$x$$. (This is
effectively the cumulative distribution function of $$\mathbf{v}$$'s values.)
Given value $$p\in [0,1]$$, define the quantile function as

$$
Q(p) = \min\{x:p\leq F(x)\}
$$

The method `split_by_quantiles` receives a monotonically increasing 
list of *quantiles*, $$(p_1,\dots,p_k)$$, each between 0 and 1, 
and returns

$$
    C_\ell = \begin{cases}
        \{i : v_i \leq Q(p_1)\} & \ell=0\\
        \{i : Q(p_\ell) < v_i \leq Q(p_{\ell+1})\} & 0 < \ell < k \\
        \{i : v_i > Q(p_k) \} & \ell=k
    \end{cases}
$$

If any of these clusters are empty, they will be discarded
and the clusters relabeled to begin at $$0$$ and end at $$N-1$$
where $$N$$ is the final number of clusters.

Note the asymmetry in that clusters will always contain their upper bound
but not their lower bound (this is in line with the standard
right-continuity of the cumulative distribution function).
To change the orientation of this asymmetry, the user
can pass `-vec` as the argument instead.


## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `vec` | | `np.ndarray` | A one-dimensional array of values.|
| `quantiles` | Keyword | `float`, `list` or `np.ndarray` | Quantiles which will be used to divide the vector components.|
| `group` | Keyword | `Group` | The group which labels the indices of `vec`, and which will be the item set of the returned `Aggregation`. |

## Examples

First we provide a direct example which shows the
behavior as well as subtleties of the quantile function.
We pass the vector $$\mathbf{v} = (0,0,1,1,2,2,3,3,4,4,5,5)$$
and the quantiles $$\{0,0.5,0.9\}$$:
{% highlight python %}
import stoclust.clustering as clust
import numpy as np

vec = np.array([0,0,1,1,2,2,3,3,4,4])
clust.split_by_quantiles(vec,quantiles=[0.0,0.5,0.9])
{% endhighlight %}

<code>Aggregation(</code><br>
<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Group([2, 3, 4, 5])</code><br>
<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Group([0, 1])</code><br>
<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Group([6, 7, 8, 9])</code><br>
<code>)</code><br>

Note first that the $$0.5$$ quantile divides the vector into components
with values $$\{0,1,2\}$$ and $$\{3,4\}$$. This is because

$$ 
Q(0.5) = \min\{2,3,4\} = 2
$$

and because our clustering scheme will divide the vector
into components less than or equal to $$Q(p)$$ and greater than $$Q(p)$$.

Next, the $$0$$ quantile
separates the lowest value $$0$$ from the rest; 
this is because

$$ 
Q(0) = \min\{0,1,2,3,4\} = 0
$$

and so a cluster is created for elements less than *or equal* to zero.

Finally, the $$0.9$$ quantile
has no effect---we have

$$ 
Q(0.9) = \min\{4\} = 4
$$

and the cluster with elements greater than $$4$$
is empty.

The second example is visual. We randomly generate
a vector with 50 components, its values drawn from a uniform 
distribution and then squared. After quantile clustering
and plotting the sorted values, we see that 
the quantiles $$\{0.2,0.7,0.8\}$$ result in clusters with
10, 25, 5 and 10 elements, respectively, corresponding
proportionally to the gaps in the quantiles. 
Though the upper quintile only contains $$20\%$$ of the
components, it covers over $$50\%$$ of the range of values.

{% highlight python %}
import stoclust.visualization as viz

vec = np.random.rand(50)**2
agg = clust.split_by_quantiles(vec,quantiles=[0.2,0.7,0.8])

rank = np.empty_like(vec)
rank[np.argsort(vec)] = np.arange(len(vec))

fig = viz.scatter2D(
    rank,vec,agg=agg,mode='markers',
    layout=dict(
        title = 'Sorted vector components (quantile clustering)',
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
  src="/stoclust/assets/html/clustering/split_by_quantiles.html"
  style="width:100%; height:400px;"
></iframe>