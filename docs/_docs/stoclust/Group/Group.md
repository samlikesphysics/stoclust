---
layout: docs
title: Group
parent: stoclust
permalink: /docs/Group/
children: 1
list_title: 'Methods'
def: 'Group(elements, superset=None)'
---

A cross between an array and a set, which contains
no duplicate elements but allows indexing and can
allows cross-referencing between a `Group` and its sub-`Group`s.

A `Group` is defined by its initializing array containing elements,
and optionally by an additional keyword parameter that
points to a superset `Group`.

Attributes that can be obtained are self.elements (the initializing array of elements),
`self.ind` (a dictionary whose keys are elements and values are indices),
and `self.in_superset` (an array whose indices correspond
to `self.elements`, but whose entries give the corresponding indices
for the same elements in `superset.elements`). A superset may be defined
after instantiation using the method `self.set_super`.

Aggregations act like arrays in that the elements
may be called by indices. That is, for a group `G`, `G[j]`
returns the same thing as `G.elements[j]`. Furthermore,
`G` may be called as in iterator, which would yield the
same result as calling G.elements as an iterator,
and the boolean statement `i in G` yields the same
result as `i in G.elements`. `len(G)` is just the number of elements.
Lastly, `Group`s may be added together, which results
in the union of their elements, but the superset must be redefined
afterwards.

## Attributes

| Attribute | Description |
| --- | --- |
| `elements` | An array containing the unique elements of the `Group`. It is recommended that elements be either strings or integers. |
| `ind` | A dictionary whose keys are in `elements` and whose values are the corresponding indices of the elements in the array `elements`. |
| `in_superset` | *(optional)* If a superset is specified on initialization or later set, this will be an array, whose values are the indices of `elements` in the superset. |