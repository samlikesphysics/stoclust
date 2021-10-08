---
layout: docs
title: Aggregation
parent: stoclust
permalink: /docs/Aggregation/
children: 1
list_title: 'Methods'
def: 'Aggregation(item_group,cluster_group,agg_dict)'
---

A class for describing partitions of Indices into clusters.

Aggregations are defined by three primary attributes:
their Index of items, their Index of cluster labels,
and a dictionary whose keys are cluster indices and whose
values are arrays of item indices, indicating which cluster
contains which items.

Attributes that can be obtained are self.items and self.clusters.
Aggregations act like dictionaries in that the cluster labels
may be called as indices. That is, for an aggregation A, and cluster c,
A[c] results in an Index containing the items in cluster c.
When treated as an iterator, A returns tuples of the form (c,A[c]),
much like the dictionary items() iterator.
The length of an Aggregation, len(A), is the number of clusters.

## Attributes

| Attribute | Visibility | Description |
| --- | --- | --- |
| `items` | *Public* | An `Index` whose elements are divided into categories by the `Aggregation`. |
| `clusters` | *Public* | An `Index` whose elements are labels corresponding to the main clusters. |
| `_aggregations` | *Private* | A `dict` whose keys are cluster indices and whose values are arrays containing the indices of items. It is better for the user to retrieve the clustering information either through treating the `Aggregation` like a dictionary, or through the public method `as_dict`, which will also return a `dict` but with cluster labels as keys and `Index`es of items as values.|