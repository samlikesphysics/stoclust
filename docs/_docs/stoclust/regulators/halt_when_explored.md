---
layout: docs
title: halt_when_explored
parent: regulators
def: 'halt_when_explored(time,at_node,node_data)'
excerpt: 'Modifies transition probabilities by making it impossible to transition to any node which has been visited already at least max_visits times; when a node is removed in this way, reports the ID of the removed node.'
permalink: /docs/regulators/halt_when_explored/
---
Halts when all nodes have been visited at least once. Assumes that node_data is being used to count visits; does not work otherwise. Therefore, be careful to use with an appropriate regulator.