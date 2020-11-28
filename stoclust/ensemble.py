import numpy as np
from tqdm import tqdm
import scipy.linalg as la
from functools import reduce
from math import floor,ceil
from stoclust import utils
from stoclust.Group import Group
from stoclust.Aggregation import Aggregation
from stoclust import clustering

def random_clustering(st_mat,clustering_method,ensemble_size=100,show_progress=False):
    if show_progress:
        ensemble_iter = tqdm(range(ensemble_size))
    else:
        ensemble_iter = range(ensemble_size)
    return np.stack([clustering_method(st_mat).block_mat() for k in ensemble_iter])

def given_ensemble(st_mats,clustering_method,show_progress=False):
    if show_progress:
        ensemble_iter = tqdm(range(st_mats.shape[0]))
    else:
        ensemble_iter = range(st_mats.shape[0])
    return np.stack([clustering_method(st_mats[j]).block_mat() for j in ensemble_iter])

def smooth_block(block_mats,window=7,cutoff=0.5):
    height = block_mats.shape[0]
    width = block_mats.shape[1]
    indices = reduce(lambda x,y:x+y,[[np.arange(max(0,j-floor(window/2)),
                                                min(height,j+ceil(window/2)))[0],
                                        np.arange(max(0,j-2),min(height,j+3))[-1]] for j in range(height)],[])
    sums = np.add.reduceat(block_mats,indices,axis=0)[0::2]
    nums = np.add.reduceat(np.ones([height]),indices,axis=0)[0::2]
    bmats_avg = sums/np.outer(nums,np.ones([width,width])).reshape([height,width,width])
    upper = (bmats_avg>cutoff).astype(float)
    new_bmats = np.stack([(la.expm(upper[j])>0).astype(float) for j in range(height)])
    result = np.zeros([height,width,width])
    result[-1] = new_bmats[-1]
    for j in range(height-1):
        result[height-j-2] = new_bmats[height-j-2]*result[height-j-1]
    return result