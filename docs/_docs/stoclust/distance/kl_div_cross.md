---
layout: docs
title: kl_div_cross
parent: distance
def: 'kl_div_cross(probs)'
excerpt: 'Given a Gram matrix, computes the corresponding distance of the basis vectors.'
permalink: /docs/distance/kl_div_cross/
---
Given a Markov matrix, returns a matrix of cross-Kullback-Liebler divergences between the distributions.

The input `probs` has the form of a square column-stochastic matrix: that is, 
that is, `np.sum(probs,axis=1) == 1`.
The output is a square matrix `D` with length and width
`probs.shape[0]`.

Mathematically, if `probs[i,j]` is given by the matrix $$T_{i,j}$$, then 
`D[i,j]` is given by $$D_{i,j}$$:

$$
D_{i,j} = T_{i,i} \log\left(\frac{T_{i,i}}{T_{j,j}}\right)
+T_{i,j} \log\left(\frac{T_{i,j}}{T_{j,i}}\right)
+
\sum_{k\neq j,i} T_{i,k} \log\left(\frac{T_{i,k}}{T_{j,k}}\right)
$$