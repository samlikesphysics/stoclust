---
layout: docs
title: smooth_block
parent: ensemble
def: 'smooth_block(block_mats, window=7, cutoff=0.5)'
excerpt: 'Given an ensemble of matrices and a clustering method, returns the result of clustering each trial as an ensemble of block matrices.'
permalink: /docs/ensemble/smooth_block/
---
Given an ordered ensemble of block matrices, uses a smoothing method to ensure hierarchical consistency.
Returns a three-dimensional array the same shape as block_mats.

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `block_mats` | | `np.ndarray` | A three dimensional array, the first dimension of which is the ensemble index and the remaining two are square. |
| `window` | Keyword | `int` | The window to be used in the smoothing technique. |
| `cutoff` | Keyword | `float` | The cutoff for whether a smoothed value should be set to 0 or 1. |