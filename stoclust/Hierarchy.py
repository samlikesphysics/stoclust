import numpy as np
from stoclust.Group import Group
from stoclust.Aggregation import Aggregation
from stoclust import utils
from functools import reduce

class Hierarchy:
    """
    A class for describing a hierarchical clustering.

    Hierarchies are defined by three primary attributes:
    their Group of items, their Group of cluster clusters,
    and a dictionary whose keys are cluster indices and whose
    values are tuples. The first element of the tuple is a scale
    parameter which must increase from subset to superset,
    and the second is an array containing the indices
    of each immediate child cluster of the key cluster.

    Attributes that can be obtained are self.items and self.clusters.
    Hierarchies act like dictionaries in that the cluster clusters
    may be called as indices. That is, for a Hierarchy H, and cluster c,
    H[c] results in a Group containing all items under cluster c.
    When treated as an iterator, H returns tuples of the form (c,H[c]),
    much like the dictionary items() iterator.
    The length of a Hierarchy, len(H), is the number of distinct clusters.

    Methods
    -------
    cluster_children:       Returns a dictionary where keys are cluster labels and
                            the values are the immediate child clusters.
    cluster_groups:         Returns a dictionary where keys are cluster labels and
                            the values are all items under the key cluster.
    clusters_containing:    Returns a Group containing all cluster labels for clusters
                            that contain the given items.
    at_scale:               Returns the aggregation corresponding to the coarsest
                            partition made from clusters not exceeding the given scale.
    join:                   Returns the smallest cluster that is a supercluster of all
                            given clusters.
    measure:                Given a field (array of values over items), gives the partial
                            sums of the field over each cluster in the form of an array
                            whose indices correspond to those of self.clusters.
    get_ultrametric:        Returns a Parisi matrix P of nested diagonal blocks, such that
                            P[i,j] is the smallest scale at which i and j are in the
                            same cluster.
    get_scales:             Returns the array of scales, indexed like self.clusters.
    set_scales:             Allows the user to define the scales through an array
                            indexed like self.clusters.
    """
    def __init__(self,items_group,cluster_group,cluster_children):
        self.items = items_group
        self.clusters = cluster_group
        self._children = {k:v[1] for k,v in cluster_children.items()}
        self._children.update({cluster_group.ind[j]:[] for j in items_group})

        self._scales = np.zeros([self.clusters.size])
        self._scales[self.items.size:] = np.array([cluster_children[k][0] 
                                                   for k in range(self.items.size,self.clusters.size)])

    def __iter__(self):
        return iter(self.cluster_groups().items())
    
    def __repr__(self):
        return 'Hierarchy('+str({self.clusters[k]:(self._scales[k],self.clusters[v]) 
                                 for k,v in self._children.items()})+')'

    def __getitem__(self,key):
        return self.cluster_groups()[key]

    def __len__(self):
        return self.clusters.size

    def cluster_children(self):
        """
        Returns a dictionary where keys are cluster labels and
        the values are the immediate child clusters.
        """
        return  {self.clusters[k]:Group(self.clusters.elements[self._children[k]],superset=self.clusters)
                 for k in self._children.keys()}

    def cluster_groups(self):
        """
        Returns a dictionary where keys are cluster labels and
        the values are all items under the key cluster.
        """
        C = self._children
        C_groups = {}
        for k in C.keys():
            if len(C[k])>0:
                final_items = list(C[k])
                done = False
                while not(done):
                    new_final_items = []
                    done = True
                    for l in final_items:
                        if len(C[l])>1:
                            done = False
                            new_final_items = new_final_items + list(C[l])
                        else:
                            new_final_items = new_final_items + [l]
                    final_items = new_final_items
                C_groups[self.clusters[k]] = Group(self.items[np.array(final_items)],superset=self.items)
            else:
                C_groups[self.clusters[k]] = Group(np.array([self.items[k]]),superset=self.items)
        return C_groups
    
    def at_scale(self,scale):
        """
        Returns the aggregation corresponding to the coarsest
        partition made from clusters not exceeding the given scale.
        """
        C_tree = self._children
        C_less_than = {k:C_tree[k] for k in C_tree.keys() if self._scales[k]<=scale}
        C_top = {k:C_less_than[k] for k in C_less_than.keys() if utils.is_canopy(k,C_less_than)}
        C_groups = self.cluster_groups()
        C_top_lists = {k:C_groups[self.clusters[k]].in_superset for k in C_top.keys()}
        C_top_group = Group(self.clusters.elements[np.array(list(C_top_lists.keys()))],superset=self.clusters)
        return Aggregation(self.items,C_top_group,
                           {C_top_group.ind[self.clusters.elements[k]]:C_top_lists[k] 
                            for k in C_top_lists.keys()})

    def join(self,cluster_list):
        """
        Returns the smallest cluster that is a supercluster of all given clusters.
        """
        C_groups = self.cluster_groups()
        items = list(reduce(lambda x,y:x+y,[C_groups[c] for c in cluster_list],Group(np.array([]))).elements)
        rivals = self.clusters_containing(items)
        lens = np.array([len(self.cluster_groups()[c]) for c in rivals.elements])
        return rivals[np.argmin(lens)]

    def clusters_containing(self,items_list):
        """
        Returns a Group containing all cluster labels for clusters
        that contain the given items.
        """
        C_groups = self.cluster_groups()
        containing_groups = []
        for key,grp in C_groups.items():
            if np.all(np.array([(i in grp) for i in items_list])):
                containing_groups.append(key)
        return Group(np.array(containing_groups),superset=self.clusters)

    def measure(self,field,axis=0):
        """
        Given a field (array of values over items), gives the partial
        sums of the field over each cluster in the form of an array
        whose indices correspond to those of self.clusters.
        Can also be applied to multidimensional arrays, along
        a specified axis.
        """
        if axis != 0:
            new_field = np.moveaxis(field,[0,axis],[axis,0])
        else:
            new_field = field
        C_groups = self.cluster_groups()
        measure = np.stack([np.sum(new_field[C_groups[k].in_superset],axis=0) for k in self.clusters])
        if axis != 0:
            new_measure = np.moveaxis(measure,[0,axis],[axis,0])
        else:
            new_measure = measure
        return new_measure

    def get_ultrametric(self):
        """
        Returns a Parisi matrix P of nested diagonal blocks, such that
        P[i,j] is the smallest scale at which i and j are in the same cluster.
        """
        umet = np.zeros([self.items.size,self.items.size])
        scales = np.unique(self._scales)
        for s in scales:
            bmat = self.at_scale(s).block_mat()
            umet = umet + (bmat-(umet>0))*s
        return umet - np.diag(np.diag(umet))

    def set_scales(self,scales):
        """
        Allows the user to define the scales through an array
        indexed like self.clusters.
        """
        self._scales = scales

    def get_scales(self):
        """
        Returns the array of scales, indexed like self.clusters.
        """
        return self._scales