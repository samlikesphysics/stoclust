import numpy as np
import scipy.linalg as la
from stoclust import utils
from stoclust.Group import Group
from stoclust.Aggregation import Aggregation
from stoclust.Hierarchy import Hierarchy
from stoclust import simulation
from stoclust import regulators

def split_by_gaps(vec,num_gaps = 1,group = None):
    """
    Aggregates the indices of a vector based on gaps between index values.
    The number of gaps is specified by num_gaps,
    and the largest num_gaps gaps in the sorted array
    are used to cluster values.
    """
    if group is None:
        group = Group(np.arange(len(vec)))

    sort_inds = np.argsort(vec)
    gap_locs = np.flip(np.argsort(np.diff(vec[sort_inds])))[0:num_gaps]
    ordered_gaps = np.sort(gap_locs)

    agg_dict = {k+1:sort_inds[ordered_gaps[k]+1:ordered_gaps[k+1]+1] for k in range(0,num_gaps-1)}
    agg_dict.update({0:sort_inds[:ordered_gaps[0]+1],
                     num_gaps:sort_inds[ordered_gaps[-1]+1:]})
    return Aggregation(group,Group(np.arange(num_gaps+1)),agg_dict)

def split_by_quantiles(vec,quantiles=0.95,group = None):
    """
    Like split_by_vals, but cuts the vector at specific quantiles
    rather than rigid values. Assumes right-continuity of the 
    cumulative distribution function.
    """
    num = len(vec)
    if group is None:
        group = Group(np.arange(len(vec)))

    if not(isinstance(quantiles,np.ndarray)):
        if isinstance(quantiles,list):
            quantiles = np.array(quantiles)
        else:
            quantiles = np.array([quantiles])

    quantiles = quantiles[np.where(np.logical_and(quantiles>1/len(vec),quantiles<=1))]
        
    cuts = np.sort(vec)[(np.floor(quantiles*len(vec))-1).astype(int)]
    return split_by_vals(vec,cuts=cuts,group=group)

def split_by_vals(vec,cuts=0,group = None,tol=0):
    """
    Aggregates the indices of a vector based on specified
    values at which to cut the sorted array. Assumes the
    right-continuity of the cumulative distribution function.
    """
    if group is None:
        group = Group(np.arange(len(vec)))

    if not(isinstance(cuts,np.ndarray)):
        if isinstance(cuts,list):
            cuts = np.array(cuts)+tol
        else:
            cuts = np.array([cuts])+tol
    
    cuts = cuts[np.where(np.logical_and(cuts>np.amin(vec),cuts<=np.amax(vec)))[0]]

    if len(cuts)==0:
        return Aggregation(group,Group(np.array([0])),{0:np.arange(len(vec))})
    else:
        agg_dict = {k+1:np.where(np.logical_and(vec >= cuts[k],
                                                vec < cuts[k+1]))[0]
                    for k in range(0,len(cuts)-1)}
        agg_dict.update({0:np.where(vec < cuts[0])[0],
                        len(cuts):np.where(vec >= cuts[len(cuts)-1])[0]})

        return Aggregation(group,Group(np.arange(len(cuts)+1)),agg_dict)

def meyer_wessell(bi_mat,min_times_same = 5,vector_clustering = None,group = None):
    """
    Given a bistochastic matrix describing the strength
    of the relationship between pairs of items,
    determines an aggregation of the items using the dynamical
    approach of Meyer and Wessell. The algorithm
    is inherently random, though fairly stable, and so may
    be used as a one-shot measure but will be more reliable
    in an ensemble.
    """
    method = vector_clustering
    if group is None:
        group = Group(np.arange(bi_mat.shape[0]))

    if method is None:
        eigs,vecs = la.eig(bi_mat)
        eig_agg = split_by_gaps(eigs)
        k = len(eig_agg[1])
        if k>1:
            method = lambda v:split_by_gaps(v,num_gaps = k-1,group = group)
        else:
            method = lambda v:Aggregation(group,Group(np.array([0])),{0:np.arange(len(group))})

    x = np.random.rand(bi_mat.shape[0])
    times_same = 1
    new_agg = method(x)
    by_cluster = method(x).by_cluster()
    while times_same < min_times_same:
        x = bi_mat@x
        new_agg = method(x)
        new_by_cluster = new_agg.by_cluster()
        if np.all(new_by_cluster == by_cluster):
            times_same += 1
        else:
            times_same = 1
        by_cluster = new_by_cluster
    return new_agg
    
def shi_malik(st_mat,eig_thresh=0.95,tol=0,group=None):
    """
    Given a stochastic matrix describing the strength
    of the relationship between pairs of items,
    determines an aggregation of the items using
    the spectral approach of Shi and Malik.
    """
    if group is None:
        group = Group(np.arange(st_mat.shape[0]))

    num_items = group.size
    clusts = Aggregation(group,Group(np.array([0])),{0:np.arange(len(group))})
    change = True
    while change:
        new_clusts = []
        change = False
        for k,c in clusts:
            if len(c)>1:
                T = utils.stoch(st_mat[np.ix_(c.in_superset,c.in_superset)])
                eigs,evecs = la.eig(T)
                einds = np.flip(np.argsort(np.abs(eigs)))
                if eigs[einds[1]]>eig_thresh:
                    y = np.real(evecs[:,einds[1]])
                    ind_agg = split_by_vals(y/np.sum(y),group=c,tol=tol)
                    if ind_agg.clusters.size>1:
                        new_clusts.append(c.in_superset[ind_agg[0].in_superset])
                        new_clusts.append(c.in_superset[ind_agg[1].in_superset])
                    else:
                        ind_agg = split_by_gaps(y,group=c)
                        new_clusts.append(c.in_superset[ind_agg[0].in_superset])
                        new_clusts.append(c.in_superset[ind_agg[1].in_superset])
                    change = True
                else:
                    new_clusts.append(c.in_superset)
            else:
                new_clusts.append(c.in_superset)

        new_agg = {j:new_clusts[j] for j in range(len(new_clusts))}
        clusts = Aggregation(group,Group(np.arange(len(new_clusts))),new_agg)
    return clusts
        
def fushing_mcassey(st_mat,max_visits=5,time_quantile_cutoff=0.95,group=None):
    """
    Given a stochastic matrix describing the strength
    of the relationship between pairs of items,
    determines an aggregation of the items using
    the regulated random walk approach of Fushing and McAssey.
    The algorithm is inherently random
    and highly unstable as a single-shot approach,
    but may be used in an ensemble to determine a 
    useful similarity matrix.
    """
    if group is None:
        group = Group(np.arange(st_mat.shape[0]))

    reg = lambda t,ps,an,nd: regulators.node_removal_regulator(t,ps,an,nd,max_visits=max_visits)
    hlt = lambda t,an,nd: regulators.halt_when_explored(t,an,nd)
    reports, path = simulation.markov_random_walk(st_mat,regulator=reg,halt=hlt,group=group)
    times = np.concatenate([np.array([reports[0,1]]),np.diff(reports[:,1])])
    clust_times = reports[split_by_quantiles(times,quantiles=np.array([time_quantile_cutoff]))[1].in_superset,1]

    if clust_times[0]>0:
        clust_times = np.concatenate([np.array([0]),clust_times])
    clusters = {j+group.size:[] for j in range(len(clust_times))}
    
    for k in group:
        if k in np.unique(reports[:,0]):
            block_counts = np.add.reduceat((path==k),clust_times)
            props = block_counts/np.sum(block_counts)
            if np.any(props>0.5):
                clusters[np.where(props>0.5)[0][0]+group.size].append(group.ind[k])
            else:
                clusters[group.ind[k]] = [group.ind[k]]
        else:
            clusters[group.ind[k]] = [group.ind[k]]
    cluster_names = np.array([k for k in clusters.keys() if len(clusters[k])>0])
    agg_dict = {j:np.array(clusters[cluster_names[j]]) 
                for j in range(len(cluster_names))}
    return Aggregation(group,Group(cluster_names),agg_dict)

def hier_from_blocks(block_mats,scales=None,group=None):
    if group is None:
        group = Group(np.arange(block_mats.shape[1]))
    if scales is None:
        scales = np.arange(block_mats.shape[0])

    current_agg = Aggregation(group,Group(np.arange(group.size)),{j:np.array([j]) for j in range(group.size)})
    new_block_mats = block_mats.copy()
    cluster_children = {}
    proper_clusters = 0
    for j in range(block_mats.shape[0]):
        st_mat = utils.stoch(new_block_mats[j])
        agg = shi_malik(st_mat,eig_thresh=0.99,group=current_agg.clusters,tol=1e-6)
        new_agg_dict = {}
        new_agg_names = []
        num_clusters = 0
        for k,c in agg:
            if c.size>1:
                cluster_children[proper_clusters+group.size] = (scales[j],
                                                                current_agg.clusters.elements[c.in_superset])
                new_agg_dict[num_clusters] = c.in_superset
                new_agg_names.append(proper_clusters+group.size)
                proper_clusters += 1
                num_clusters += 1
            else:
                new_agg_dict[num_clusters] = c.in_superset
                new_agg_names.append(c[0])
                num_clusters += 1
        new_agg = Aggregation(current_agg.clusters,
                              Group(np.array(new_agg_names)),
                              new_agg_dict)

        blocks = [list(new_agg[new_agg.clusters[k]].in_superset) for k in range(new_agg.clusters.size)]
        new_block_mats = (utils.block_sum(utils.block_sum(new_block_mats,blocks,axis=1),
                                          blocks,axis=2)>0).astype(float)
        current_agg = new_agg

    labels = Group(np.array(list(group.elements)+list(cluster_children.keys())))

    return Hierarchy(group,labels,cluster_children)