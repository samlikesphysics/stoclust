---
layout: docs
title: node_removal_regulator
parent: regulators
def: 'node_removal_regulator(time, probs, at_node, node_data, max_visits=5)'
excerpt: 'Modifies transition probabilities by making it impossible to transition to any node which has been visited already at least max_visits times; when a node is removed in this way, reports the ID of the removed node.'
permalink: /docs/regulators/node_removal_regulator/
---
Modifies transition probabilities by making it impossible to transition
to any node which has been visited already at least max_visits times.
Uses node_data to track the number of previous visits.
When a node is removed, reports the ID of the removed node;
otherwise, no report is made.

## Additional Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `max_visits` | Keyword | `int` | The number of times a node is visted before being removed. |