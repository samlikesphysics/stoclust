import numpy as np
from stoclust.Group import Group

class Aggregation:
    """
    A class for describing partitions of Groups into clusters.

    Aggregations are defined by three primary attributes:
    their Group of items, their Group of cluster labels,
    and a dictionary whose keys are cluster indices and whose
    values are arrays of item indices, indicating which cluster
    contains which items.

    Attributes that can be obtained are self.items and self.clusters.
    Aggregations act like dictionaries in that the cluster labels
    may be called as indices. That is, for an aggregation A, and cluster c,
    A[c] results in a Group containing the items in cluster c.
    When treated as an iterator, A returns tuples of the form (c,A[c]),
    much like the dictionary items() iterator.
    The length of an Aggregation, len(A), is the number of clusters.

    Methods
    -------
    block_mat:      Returns a block-diagonal matrix whose indices 
                    correspond to items and which contains a block
                    for every cluster.

    by_cluster:     Returns an array B, whose indices correspond to items,
                    such that B[j] is the cluster containing self.items[j].
                    
    as_dict:        Returns a dictionary whose keys are from self.clusters
                    and whose values are Groups corresponding to said clusters.
    """
    def __init__(self,item_group,cluster_group,agg_dict):
        self.items = item_group
        self.clusters = cluster_group
        self._aggregations = agg_dict

 #   @classmethod
 #   def from_labels(cls,items,labels):
 #       a = np.unique(labels,axis=0)
 #       [tuple(list(a[j])) for j in np.arange(a.shape[0])]
 #       clusters = Group(np.unique(labels,axis=0))
 #       agg = {clusters.ind[k]: np.where(np.array([labels[j]==k for j in np.arange(labels.shape[0])]))[0]
 #              for k in clusters}
 #       return cls(items,clusters,agg)
#        for j in self.items:
#            num_containing = 0
#            for c in self.clusters:
#                occurrences = len(np.where(self.items.ind[j] == self._aggregations[self.clusters.ind[c]])[0])
#                if occurrences > 1:
#                    raise ValueError('Cluster '+str(c)+' contains duplicate items.')
#                elif occurrences == 1:
#                    num_containing += 1
#            if num_containing == 0:
#                raise ValueError('Item '+str(j)+' is not in a cluster.')
#            elif num_containing > 1:
#                raise ValueError('Item '+str(j)+' is in multiple clusters.')

    def __iter__(self):
        return iter(self.as_dict().items())
    
    def __repr__(self):
        return 'Aggregation('+str({self.clusters[k]:self.items[v] for k,v in self._aggregations.items()})+')'

    def __getitem__(self,key):
        return Group(self.items.elements[self._aggregations[self.clusters.ind[key]]],superset=self.items)

    def __len__(self):
        return self.clusters.size

    def block_mat(self):
        """
        Returns a block-diagonal matrix whose indices 
        correspond to items and which contains a block
        for every cluster.
        """
        pmat = np.zeros([self.items.size,self.items.size])
        for k,g in self._aggregations.items():
            pmat[np.ix_(g,g)] = np.ones([len(g),len(g)])
        return pmat

    def by_cluster(self):
        """
        Returns an array B, whose indices correspond to items,
        such that B[j] is the cluster containing self.items[j].
        """
        bylab = np.zeros([self.items.size])
        for j in np.arange(self.clusters.size):
            bylab[self._aggregations[j]] = (j*np.ones([len(self._aggregations[j])]))
        return bylab.astype(int)

    def as_dict(self):
        """
        Returns a dictionary whose keys are from self.clusters
        and whose values are Groups corresponding to said clusters.
        """
        return {self.clusters[k]:Group(self.items.elements[self._aggregations[k]],superset=self.items)
                for k in self._aggregations.keys()}

#    def __mul__(self,other):
#        labels1 = self.by_cluster()
#        labels2 = other.by_cluster()
#        new_labels = np.array([(self.clusters[labels1[j]],other.clusters[labels2[j]]) 
#                               for j in np.arange(labels1.shape[0])])
#        return Aggregation.from_labels(self.items,new_labels)

        