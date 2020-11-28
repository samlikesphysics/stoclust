import numpy as np
import scipy.linalg as la
from functools import reduce

def subspace(T,eig_thresh=0.8,num_samples=10000):
    """
    Takes a Markov matrix T and an eigenvalue threshold,
    and returns the basis and dual vectors of items within that 
    high-eigenvalue subspace.
    """
    num_items = T.shape[0]
    vals, vl, vr = la.eig(T,left=True)
    sort = np.flip(np.argsort(vals))
    vecsl = vl[:,sort].astype(float)
    vecsr = (vr[:,sort].astype(float)).T
    d = np.amax(np.where(np.abs(vals[sort])>eig_thresh))
    basis = vecsl[:,0:d+1]
    dual = vecsr[0:d+1,:]
    return basis, dual

def projector(basis,dual):
    """
    Delives the projector corresponding to a given basis and dual.
    """
    inners = dual@dual.T
    pi = basis@inners@basis.T
    return pi

def stoch(weights,axis=-1):
    """
    Returns an array M normalized so that np.sum(M,axis=axis) is
    an array of all ones.
    """
    if isinstance(axis,tuple):
        axes = list(axis)
    elif axis==-1:
        axes = [len(weights.shape)-1]
    else:
        axes = [axis]
    shape = np.array(weights.shape)
    all_axes = list(range(len(weights.shape)))
    other_axes = list(set(all_axes) - set(axes))
    reorg = axes + other_axes
    new_weights = np.moveaxis(weights,reorg,all_axes)
    new_targets = list(range(len(axes)))
    sums = np.sum(new_weights,axis=tuple(new_targets))
    prop_sums = sums + (sums==0).astype(int)
    sums_full = np.outer(np.ones([np.prod(shape[(axes)])]),prop_sums).reshape(list(shape[(axes)])
                                                            +list(shape[(other_axes)]))
    return np.moveaxis(new_weights/sums_full,all_axes,reorg)

def sinkhorn_knopp(mat,num_iter=100,rescalings=True):
    """
    A bistochastic matrix is one for which 
    np.sum(M,axis=0)=1 and np.sum(M,axis=1)=1.
    For every positive-definite matrix M, there are unique
    arrays dL and dR such that B = np.diag(dL)@M@np.diag(dR)
    is bistochastic, which can be found using the Sinkhorn-Knopp
    algorithm. This method returns B and, optionally, dL and dR.
    """
    dR = np.ones([mat.shape[1]])
    dL = np.ones([mat.shape[0]])
    for j in range(num_iter):
        dL = np.sum(mat@np.diag(dR),axis=1)**(-1)
        dR = np.sum(np.diag(dL)@mat,axis=0)**(-1)
    if rescalings:
        return dL, np.diag(dL)@mat@np.diag(dR), dR
    else:
        return np.diag(dL)@mat@np.diag(dR)

def is_canopy(k,C_less_than):
    """
    A check used by the Hierarchy class to determine whether
    a given cluster index is a top-level cluster among
    a given subset of clusters.
    """
    is_cpy = True
    for j in C_less_than:
        if (len(C_less_than[j])>1) and (k in C_less_than[j]):
            is_cpy = False
    return is_cpy

def block_sum(a, blocks, axis=0):
    """
    Given an array and an aggregation of its indices
    along a given axis, delivers a new array whose indices
    correspond to clusters of the old indices and whose
    entries are sums over the old values.
    """
    if axis==0:
        move_a = a.copy()
    else:
        move_a = np.moveaxis(a,[0,axis],[axis,0])
    order = reduce(lambda x,y:x+y, blocks,[])
    order_a = move_a[order]
    lengths = np.array([len(b) for b in blocks])
    slices = np.concatenate([np.array([0]),np.cumsum(lengths)[:-1]])
    red_a = np.add.reduceat(order_a,slices,axis=0)
    if axis==0:
        final_a = red_a.copy()
    else:
        final_a = np.moveaxis(red_a,[0,axis],[axis,0])
    return final_a