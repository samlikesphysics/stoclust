---
layout: docs
title: split_by_quantiles
parent: clustering
def: 'split_by_quantiles(vec, quantiles=0.95, group = None)'
excerpt: 'Cuts the vector at specific quantiles rather than rigid values. Assumes right-continuity of the cumulative distribution function.'
permalink: /stoclust/clustering/split_by_quantiles/
---

Like [`split_by_vals`](/stoclust/clustering/split_by_vals), but cuts the vector at specific quantiles
rather than rigid values. Assumes right-continuity of the 
cumulative distribution function.