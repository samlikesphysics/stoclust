---
layout: docs
title: from_gram
parent: distance
def: 'from_gram(g)'
excerpt: 'Given a Gram matrix, computes the corresponding distance of the basis vectors.'
permalink: /docs/distance/from_gram/
---
Given a Gram matrix, computes the corresponding
distance of the basis vectors.

Mathematically, if the input `g[i,j]` is given by the matrix $$g_{i,j}$$, then the output
`D[i,j]` is given by $$D_{i,j}$$:

$$
D_{i,j} = g_{i,j} - g_{i,i} - g_{j,j}
$$