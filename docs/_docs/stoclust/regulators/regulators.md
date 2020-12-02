---
layout: docs
title: regulators
parent: stoclust
permalink: /docs/regulators/
children: 1
list_title: 'Functions'
---

Contains some pre-defined regulators and halting
conditions for use in the markov_random_walk method.

A regulator is any function which receives the following arguments...

| Arguments | Type | Description |
| --- | --- | --- |
| `time` | `int` | The timestep of the random walk. |
| `probs` | `np.ndarray` | The original transition probabilities of the walk. |
| `at_node` | `int` | The index of the current node. |
| `node_data` | `np.ndarray` | An array in which "node data" may be stored, to make the random walk memoryful. |

...and returns the following outputs...

| Returns | Type | Description |
| --- | --- | --- |
| `is_report` | `bool` | Whether a report is to be logged. |
| `report` | `object` | The report, if `is_report == True`, or `NoneType` if `is_report == False`. |
| `new_probs` | `np.ndarray` | The modified transition probabilities. |

...and, optionally, which modifies the node_data array *in-place*
in order to update the memory of the walk.

A halting condition is similar but also simpler. It is any function
which receives the following arguments...

| Arguments | Type | Description |
| --- | --- | --- |
| `time` | `int` | The timestep of the random walk. |
| `at_node` | `int` | The index of the current node. |
| `node_data` | `np.ndarray` | An array in which "node data" may be stored, to make the random walk memoryful. |

...and returns a single boolean output, which if `True` will result
in the halting of the random walk.