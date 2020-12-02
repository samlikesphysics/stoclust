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

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `vec` | | `np.ndarray` | A one-dimensional array of values.|
| `quantiles` | Keyword | `float`, `list` or `np.ndarray` | Quantiles which will be used to divide the vector components.|
| `group` | Keyword | `Group` | The group which labels the indices of `vec`, and which will be the item set of the returned `Aggregation`. |