---
layout: docs
title: from_noise
parent: ensemble
def: 'from_noise(vecs, noise_map, ensemble_size=100, show_progress=False)'
excerpt: 'Given an ensemble of matrices and a clustering method, returns the result of clustering each trial as an ensemble of block matrices.'
permalink: /docs/ensemble/from_noise/
---
Given an array of vectors and a function which takes each vector to a randomly-generated ensemble, creates the ensemble of randomly-generated instances for each given vector. The output is a three-dimensional array
whose first dimension indexes the ensemble,
and whose remaining two dimensions have the same
shape as `vecs`.

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `vecs` | | `np.ndarray` | A two-dimensional array, the first dimension of which indexes the vectors, and the remaining dimension is the dimension of the vectors.|
| `noise_map` | | `function` | Any function which takes two positional arguments (a vector and an ensemble size) and generates a random ensemble of vectors (with ensemble index first). |
| `ensemble_size` | Keyword | `int` | The number of ensembles to run. |
| `show_progress` | Keyword | `bool` | Whether or not to display a tqdm progress bar. |