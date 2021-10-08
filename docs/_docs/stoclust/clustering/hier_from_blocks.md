---
layout: docs
title: hier_from_blocks
parent: clustering
def: 'hier_from_blocks(block_mats, scales=None, index=None)'
excerpt: 'Given a parameterized ensemble of block matrices, each more coarse-grained than the last, constructs a corresponding Hierarchy object.'
permalink: /docs/clustering/hier_from_blocks/
---
Given a parameterized ensemble of block matrices, each more coarse-grained than the last,
constructs a corresponding Hierarchy object.

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `block_mats` | | `np.ndarray` | A three-dimensional array. The first dimension is the ensemble dimension, and the remaining two dimensions are equal. |
| `scales` | Keyword | `np.ndarray` | A one-dimensional monotonically increasing array, giving a scale parameter for each ensemble. |
| `index` | Keyword | `Index` | The `Index` which labels the indices of block_mats.shape[1], and which will be the item set of the returned Aggregation. |