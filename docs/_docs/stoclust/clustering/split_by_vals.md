---
layout: docs
title: split_by_vals
parent: clustering
def: 'split_by_vals(vec, cuts=0, index = None, tol=0)'
excerpt: 'Aggregates the indices of a vector based on specified values at which to cut the sorted array. Assumes the right-continuity of the cumulative distribution function.'
permalink: /docs/clustering/split_by_vals/
---

Aggregates the indices of a vector based on specified
values at which to cut the sorted array. Assumes the
right-continuity of the cumulative distribution function.

Given a vector $$v_i$$ and a
list of cuts $$(c_1,\dots,c_k)$$, the function
will attempt to return clusters defined by:

$$
    C_\ell = \begin{cases}
        \{i : v_i \leq c_1\} & \ell=0\\
        \{i : c_\ell < v_i \leq c_{\ell+1}\} & 0 < \ell < k \\
        \{i : v_i > c_k \} & \ell=k
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
| `cuts` | Keyword | `float`, `list` or `np.ndarray` | Values which will be used to divide the vector components.|
| `index` | Keyword | `Index` | The index which labels the indices of `vec`, and which will be the item set of the returned `Aggregation`. |

## Examples
First we provide a direct example which shows the
behavior as well as subtleties of the quantile function.
We pass the vector $$\mathbf{v} = (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)$$
and the cut-values $$\{-0.1,0,0.5,0.9,1.0\}$$:

{% highlight python %}
import stoclust.clustering as clust
import numpy as np
vec = np.arange(0,1,0.1)
agg = clust.split_by_vals(
    vec,
    cuts=[-0.1,0,0.5,0.9,1.0]
)

print(vec)
agg
{% endhighlight %}

<code>[0.&nbsp;&nbsp;0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]</code><br>
`Aggregation(`<br>
<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Index([1,2,3,4,5])</code><br>
<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Index([0])</code><br>
<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Index([6,7,8,9])</code><br>
`)`<br>

Note that, as $$-0.1$$ and $$1.0$$ are outside of the range of the vector's values,
they generate no nontrivial clusters. Furthermore, $$0.9$$ can only
divide the components into those with values less than or equal to $$0.9$$–––which
is all of them–––or those with values greater than $$0.9$$–––which is empty.
So, only the cuts $$0$$ and $$0.5$$ are used, dividing the components
into three parts: those with $$v_i\leq 0$$, those with
$$0 < v_i\leq 0.5$$, and those with $$v_i> 0.5$$.

The next example is visual.
We randomly generate
a vector with 50 components, its values drawn from a uniform 
distribution. After value clustering
we plot the sorted values, colored by cluster.

{% highlight python %}
vec = np.random.rand(50)
agg = sc.clustering.split_by_vals(vec,cuts=[0.2,0.5,0.9])

rank = np.empty_like(vec)
rank[np.argsort(vec)] = np.arange(len(vec))

fig = viz.scatter2D(
    rank,vec,agg=agg,mode='markers',
    layout=dict(
        title = 'Sorted vector components (value clustering)',
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
fig.write_html('split_by_vals.html')
fig.show()
{% endhighlight %}
<iframe
  src="/stoclust/assets/html/clustering/split_by_vals.html"
  style="width:100%; height:400px;"
></iframe>