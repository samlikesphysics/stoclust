---
layout: docs
title: stoch
parent: utils
def: 'stoch(weights, axis=-1)'
excerpt: 'Reweights each row along a given axis such that sums along that axis are one.'
permalink: /docs/examples/stoch/
---
Reweights each row along a given axis such that sums along that axis are one.

If `weights` is a $$d$$-dimensional array
of the form $$W_{k_1\dots k_d}$$ and axis $$j$$ is specified, then this returns the array $$T_{k_1\dots k_d}$$ given by:

$$
T_{k_1\dots k_d} = \frac{W_{k_1\dots k_d}}{\sum_{k_j} W_{k_1\dots k_j\dots k_d}}
$$