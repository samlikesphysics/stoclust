---
layout: docs
title: markov_random_walk
parent: simulation
def: 'markov_random_walk(probs, initial=None, group=None, regulator=None, halt=None, max_time=100)'
excerpt: 'Given a set of transition probabilities, generates a random walk
through the available nodes. Further, the user may pass regulator and halt functions with complex rules for altering transition probabilities during the walk and determining when to halt the walk, as well as delivering specific reports about notable events during the walk.'
permalink: /docs/regulators/node_removal_regulator/
---
Given a set of transition probabilities, generates a random walk
through the available nodes. Further, the user may pass regulator and halt functions with complex rules for altering transition probabilities during the walk and determining when to halt the walk, as well as delivering specific reports about notable events during the walk.

The initial node, if not specified,
is randomly selected. The method returns two items.
The first is the report array `R`,
dimensions `M x 2`, where the reports are indexed by the axis `M`.
`R[i,0]` is the content of the report and `R[i,1]` is the time of the report.
The second is the path `P`, an `N`-dimensional vector, where `N` is the number of steps
in the random walk and `P[j]` is the node at time `j`.

The walk may be regulated. This involves passing a regulator,
which is a function that takes the simulation time, 
the transition probabilities, the current node, 
and an array of node data. At each time, 
the regulator returns a `bool` indicating
whether a report is to be made, the content of the report,
and a new set of transition probabilities determined by
the available information. The regulator also
updates node_data in-place.

Lastly, one can either specify a maximum length
of time for the walk (using the keyword `max_time`) 
or a more general halting condition (using the keyword halt).
A halting condition takes the time, current node, and node data. 
If neither are specified, the maximum length will
be set to 100 steps.

For more details on regulators and halting conditions, see the documentation on the submodule [`regulators`](/stoclust/stoclust/regulators/).

## Arguments

| Arguments |  | Type | Description |
| --- | --- | --- | --- |
| `probs` | | `np.ndarray` | A square Markov matrix indicating the transition probabilities for the walk. |
| `initial` | Keyword | flexible | The initial node. If `group` is not `None`, then the type of `initial` should be the type of the elements of `group`. Otherwise, `initial` should be the index of the initial node. If not specified, a random node will be chosen. |
| `group` | Keyword | `Group` | A `Group` whose elements label the indices of `probs`. If specified, inputs like `initial` and outputs like the path refer to nodes by their labels. If not specified, nodes will be referred to in inputs and outputs by their index. |
| `regulator` | Keyword | `function` | The regulator determines how the probability matrix will be modified over the course of the walk, and what events will be noted in reports. See the submodule [`regulators`](/stoclust/stoclust/regulators/) for more details. If not specified, a trivial regulator will be used which never modifies the transition matrix and returns no reports. |
| `halt` | Keyword | `function` | The halt condition determines under what conditions the walk should stop. See the submodule [`regulators`](/stoclust/stoclust/regulators/) for more details. If not specified, a trivial halt condition to stop after a specified number of steps will be used; the number of steps can be changed using the `max_time` argument. |
| `max_time` | Keyword | `int` | If a halt condition is not specified, then the walk is halted automatically after `max_time` steps. The default is set to 100. |