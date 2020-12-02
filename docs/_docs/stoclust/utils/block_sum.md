---
layout: docs
title: block_sum
parent: utils
def: 'block_sum(a, blocks, axis=0)'
excerpt: 'Given an array and a list of index blocks
along a given axis, delivers a new array whose indices
correspond to blocks and whose
entries are sums over the old values.'
permalink: /docs/utils/block_sum/
---
Given an array and a list of index blocks
along a given axis, delivers a new array whose indices
correspond to blocks and whose
entries are sums over the old values.

For example, consider the following matrix with the blocks indicated by horizontal lines, and
the resulting sum over blocks:

$$
\left[
\begin{array}{c|cc|c}
  0 & 1 & 2 & 3\\\hline
  4 & 5 & 6 & 7\\
  8 & 9 & 10 & 11
\end{array}
\right] \mapsto 
\left[
\begin{array}{ccc}
  0 & 3 & 3 \\
  12 & 30 & 18
\end{array}
\right]
$$

This can be achieved by calling `block_sum` twice,
first with `blocks = [[0],[1,2]]` and `axis=0`,
second with `blocks = [[0],[1,2],[3]]` and `axis=1`.