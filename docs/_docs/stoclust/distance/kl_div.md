---
layout: docs
title: kl_div
parent: distance
def: 'kl_div(probs)'
excerpt: 'Given an array of probability distributions, returns a matrix of Kullback-Liebler divergences between the distributions.'
permalink: /docs/distance/kl_div/
---
Given an array of probability distributions, returns a matrix of Kullback-Liebler divergences between the distributions.

The input `probs` has the form of a column-stochastic matrix: that is, 
that is, `np.sum(probs,axis=1) == 1`.
The output is a square matrix `D` with length and width
`probs.shape[0]`, so that `D[i,j]` is the KL divergence of `probs[i]` over `probs[j]`.

Mathematically, if `probs[i,j]` is given by the matrix $$T_{i,j}$$, then 
`D[i,j]` is given by $$D_{i,j}$$:

$$
D_{i,j} = \sum_{k} T_{i,k} \log\left(\frac{T_{i,k}}{T_{j,k}}\right)
$$