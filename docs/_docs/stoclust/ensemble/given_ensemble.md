---
layout: docs
title: given_ensemble
parent: ensemble
def: 'given_ensemble(mats, clustering_method, show_progress=False)'
excerpt: 'Given an ensemble of matrices and a clustering method, returns the result of clustering each trial as an ensemble of block matrices.'
permalink: /docs/ensemble/given_ensemble/
---
Given an ensemble of matrices and a clustering method, returns the result of clustering each trial as an ensemble of block matrices.
The output is a three-dimensional array
of the same shape as `mat`.

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `mat` | | `np.ndarray` | A three dimensional array, the first dimension of which is the ensemble index and the remaining two are square. |
| `clustering_method` | | `function` | Any function which takes a square matrix and returns an `Aggregation`. |
| `show_progress` | Keyword | `bool` | Whether or not to display a tqdm progress bar. |