---
layout: docs
title: meyer_wessell
parent: clustering
def: 'meyer_wessell(st_mat, min_times_same = 5, vector_clustering = None, group = None)'
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
| `group` | Keyword | `Group` | The group which labels the indices of `st_mat`, and which will be the item set of the returned `Aggregation`. |

## Reference

Meyer, Carl D. and Charles D. Wessell. “Stochastic Data Clustering.” SIAM J. Matrix Analysis and Application 33-4: 1214-1236. 2012, [doi: 10.1137/100804395](https://epubs.siam.org/doi/abs/10.1137/100804395).