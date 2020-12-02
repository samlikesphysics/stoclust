---
layout: docs
title: fushing_mcassey
parent: clustering
def: 'fushing_mcassey(st_mat, max_visits=5, time_quantile_cutoff=0.95, group=None)'
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
| `group` | Keyword | `Group` | The group which labels the indices of `st_mat`, and which will be the item set of the returned `Aggregation`. |

## Reference

H. Fushing and M. P. McAssey, "Time, temperature, and data cloud geometry," in *Phys. Rev. E*, vol. 82, 061110, Dec. 2010, [doi: 10.1103/PhysRevE.82.061110](https://doi.org/10.1103/PhysRevE.82.061110).