---
layout: docs
title: split_by_vals
parent: clustering
def: 'split_by_vals(vec, cuts=0, group = None, tol=0)'
excerpt: 'Aggregates the indices of a vector based on specified values at which to cut the sorted array. Assumes the right-continuity of the cumulative distribution function.'
permalink: /docs/clustering/split_by_vals/
---

Aggregates the indices of a vector based on specified
values at which to cut the sorted array. Assumes the
right-continuity of the cumulative distribution function.

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `vec` | | `np.ndarray` | A one-dimensional array of values.|
| `cuts` | Keyword | `float`, `list` or `np.ndarray` | Values which will be used to divide the vector components.|
| `group` | Keyword | `Group` | The group which labels the indices of `vec`, and which will be the item set of the returned `Aggregation`. |