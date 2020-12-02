---
layout: docs
title: euclid
parent: distance
def: 'euclid(vecs)'
excerpt: 'Given a set of vectors, determines the Euclidean distance between them.'
permalink: /docs/distance/euclid/
---
Given a set of vectors, returns a matrix with the Euclidean distance between them.

Mathematically, if the input `vecs[i,j]` is given by the matrix $$v_{i,j}$$, then the output
`D[i,j]` is given by $$D_{i,j}$$:

$$
D_{i,j} = \sqrt{\sum_{k} \left(v_{i,k} - v_{j,k}\right)^2}
$$