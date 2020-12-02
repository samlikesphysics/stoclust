---
layout: docs
title: Aggregation
parent: stoclust
permalink: /docs/Aggregation/
children: 1
list_title: 'Methods'
def: 'Aggregation(item_group,cluster_group,agg_dict)'
---

A class for describing partitions of `Group`s into clusters.

`Aggregation`s are defined by three primary attributes:
their `Group` of items, their `Group` of cluster labels,
and a `dict` whose keys are cluster indices and whose
values are arrays of item indices, indicating which cluster
contains which items.

Attributes that can be obtained are `self.items` and `self.clusters`.
`Aggregation`s act like dictionaries in that the cluster labels
may be called as indices. That is, for an   `Aggregation` `A`, and cluster `c`,
`A[c]` results in a `Group` containing the items in cluster `c` with superset `A.items`.
When treated as an iterator, `A` returns tuples of the form `(c,A[c])`,
much like the dictionary `items()` iterator.
The length of an `Aggregation`, `len(A)`, is the number of clusters.

## Attributes

| Attribute | Visibility | Description |
| --- | --- | --- |
| `items` | *Public* | A `Group` whose elements are divided into categories by the `Aggregation`. |
| `clusters` | *Public* | A `Group` whose elements are labels corresponding to the main clusters. |
| `_aggregations` | *Private* | A `dict` whose keys are cluster indices and whose values are arrays containing the indices of items. It is better for the user to retrieve the clustering information either through treating the `Aggregation` like a dictionary, or through the public method `as_dict`, which will also return a `dict` but with cluster labels as keys and `Group`s of items as values.|