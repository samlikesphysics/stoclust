---
layout: docs
title: meyer_wessell
parent: clustering
def: 'meyer_wessell(bi_mat, min_times_same = 5, vector_clustering = None, group = None)'
excerpt: 'Given a bistochastic matrix describing the strength of the relationship between pairs of items, determines an aggregation of the items using the dynamical approach of Meyer and Wessell..'
permalink: /stoclust/clustering/meyer_wessell/
---

Given a bistochastic matrix describing the strength
of the relationship between pairs of items,
determines an aggregation of the items using the dynamical
approach of Meyer and Wessell. The algorithm
is inherently random, though fairly stable, and so may
be used as a one-shot measure but will be more reliable
in an ensemble.