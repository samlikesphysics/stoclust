import numpy as np
from stoclust import ensemble

def from_gram(g):
    norms = np.diag(g)
    return g/np.sqrt((norms[None,:]*norms[:,None]))

def from_random_clustering(weights,**kwargs):
    parisi_ensemble = ensemble.random_clustering(weights,**kwargs)
    return np.average(parisi_ensemble,axis=0)