---
layout: docs
title: fushing_mcassey
parent: clustering
def: 'fushing_mcassey(st_mat, max_visits=5, time_quantile_cutoff=0.95, group=None)'
excerpt: 'Given a stochastic matrix describing the strength of the relationship between pairs of items, determines an aggregation of the items using the regulated random walk approach of Fushing and McAssey.'
permalink: /stoclust/clustering/fushing_mcassey/
---

Given a stochastic matrix describing the strength
of the relationship between pairs of items,
determines an aggregation of the items using
the regulated random walk approach of Fushing and McAssey.
The algorithm is inherently random
and highly unstable as a single-shot approach,
but may be used in an ensemble to determine a 
useful similarity matrix.