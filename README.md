# `stoclust` is a package of modularized methods for stochastic and ensemble clustering techniques. 

By modular, I mean that there are few methods in this package which act as a single pipeline for clustering a dataset–––rather, the methods each form a unit of what might be a larger clustering routine.

These modular units are designed to be compatible with general clustering methods from
other packages, like `scipy.clustering` or `sklearn.cluster`. However, we also provide
specific methods for implementing clustering algorithms whose underlying mathematics
is rooted in stochastic analysis and dynamics. Additionally, one can add a stochastic
twist to any clustering method by using ensemble clustering, which uses randomness to
probe the stability and robustness of clustering results.

The core of our package is currently:

1. The three classes `Group`, `Aggregation` and `Hierarchy`, which respectively
formalize the notion of a set of items, a single clustering or partition of a set, and a
hierarchical clustering of a set, each in a manner that is amicable to `numpy` indexing
and allows cross-referencing between subsets and supersets;

2. The `ensemble` module, which can be used to generate noisy ensembles from a base
dataset and to apply clustering methods to already-generated ensembles

3. The `clustering` module, which contains functions implementing selected
stochastic clustering techniques;

4. The `simulation` and `regulators` modules, which currently allows the generation
of regulated Markov random walks.

In addition to these are several auxiliary modules such as
`distance`, which contains methods for calculating simple distance metrics from data;
`visualization`, which contains methods for easily generating Plotly visualizations
of data and clusters; and
`utils`, which contains useful miscellaneous functions.

Check out our [**Documentation**](https://samlikesphysics.github.io/stoclust/docs/) for further info!

# Installation

# Dependencies