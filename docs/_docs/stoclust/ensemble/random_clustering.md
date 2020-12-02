---
layout: docs
title: random_clustering
parent: ensemble
def: 'random_clustering(mat, clustering_method, ensemble_size=100, show_progress=False)'
excerpt: 'Given a matrix and a function describing a random clustering method, returns an ensemble of block matrices.'
permalink: /docs/ensemble/random_clustering/
---
Given a matrix and a function describing a random clustering method, returns an ensemble of block matrices. The output is a three-dimensional array
whose first dimension indexes the ensemble trials,
and whose remaining two dimensions have the same
shape as `mat`.

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `mat` | | `np.ndarray` | A square array, of whatever format is required by the `clustering_method`. |
| `clustering_method` | | `function` | Any function which takes a square matrix and returns an `Aggregation`; ideally one which uses random methods. |
| `ensemble_size` | Keyword | `int` | The number of ensembles to run. |
| `show_progress` | Keyword | `bool` | Whether or not to display a tqdm progress bar. |