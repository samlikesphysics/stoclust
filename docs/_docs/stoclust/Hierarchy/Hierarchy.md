---
layout: docs
title: Hierarchy
parent: stoclust
permalink: /docs/Hierarchy/
children: 1
list_title: 'Methods'
def: 'Hierarchy(items_group,cluster_group,cluster_children)'
---

A class for describing a hierarchical clustering.

`Hierarchy`s are defined by three primary attributes:
their `Group` of items, their `Group` of cluster clusters,
and a `dict` whose keys are cluster indices and whose
values are tuples. The first element of the tuple is a *scale parameter* which must increase from subset to superset,
and the second is an array containing the indices
of each immediate child cluster of the key cluster.

Attributes that can be obtained are `self.items` and `self.clusters`.
Hierarchies act like dictionaries in that the clusters
may be called as indices. That is, for a `Hierarchy` `H`, and cluster `c`,
`H[c]` results in a `Group` containing all items under cluster `c`.
When treated as an iterator, `H` returns tuples of the form `(c,H[c])`,
much like the dictionary `items()` iterator.
The length of a Hierarchy, `len(H)`, is the number of distinct clusters.


## Attributes

| Attribute | Visibility | Description |
| --- | --- | --- |
| `items` | *Public* | A `Group` whose elements are divided into categories by the `Aggregation`. |
| `clusters` | *Public* | A `Group` whose elements are labels corresponding to the main clusters. |
| `_children` | *Private* | A `dict` whose keys are cluster indices and whose values are arrays containing the indices of child clusters. It is better for the user to retrieve the clustering information either through treating the `Hierarchy` like a dictionary, or through the public methods such as `cluster_children` and `cluster_groups`.|
| `_scales` | *Private* | An array whose indices correspond to clusters and whose entries give the scale parameter of each cluster. It is better for the user to retrieve this information through the public method `get_scales` and to modify it through the method `set_scales`.|