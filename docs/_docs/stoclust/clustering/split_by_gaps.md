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