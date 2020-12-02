---
layout: docs
title: sinkhorn_knopp
parent: utils
def: 'sinkhorn_knopp(mat, num_iter=100, rescalings=False)'
excerpt: 'Given a matrix of weights, generates a bistochastic matrix using the Sinkhorn-Knopp algorithm.'
permalink: /docs/examples/sinkhorn_knopp/
---
Given a matrix of weights, generates a bistochastic matrix using the Sinkhorn-Knopp algorithm.

A bistochastic matrix is one where the sum of rows
is always equal to one, and the sum of columns is always
equal to one. That is, $$T_{ij}$$ is bistochastic
if 
$$
\sum_{j} T_{ij} = \sum_{j} T_{ji} = 1
$$
for all $$i$$.

For any matrix $$M$$, there is a unique pair of
diagonal matrices $$D_L$$ and $$D_R$$ such that
$$D_L^{-1} M D_R^{-1}$$ is bistochastic. The
Sinkhorn-Knopp algorithm determines these matrices
iteratively. This function will return the resulting
bistochastic matrix and, optionally, the diagonal weights
of the rescaling matrices.