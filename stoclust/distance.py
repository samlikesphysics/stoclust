import numpy as np
import scipy.special as sp
from stoclust import gram

def kl_div(probs):
    """
    Given a column-stochastic matrix (that is,
    np.sum(probs,axis=1)=1), generates a square
    matrix of dimension probs.shape[0],
    whose i,j element is the KL divergence
    of probs[i] over probs[j].
    """
    dist = sp.kl_div(probs[:,None,:],probs[None,:,:])
    return np.sum(dist,axis=-1)

def euclid(vecs):
    """
    Given a set of vectors, determines the Euclidean distance between them.
    """
    g = gram.from_vecs(vecs)
    return from_gram(g)

def from_gram(g):
    """
    Given a Gram matrix g, computes the corresponding
    distance of the basis vectors.
    """
    norms = np.diag(g)
    return norms[None,:]+norms[:,None]-2*g

def kl_div_cross(probs):
    """
    Given a square column-stochastic matrix (that is,
    np.sum(probs,axis=1)=1), generates a square
    matrix of dimension probs.shape[0],
    whose i,j element is given by the formula

    d[i,j] = sum_(k != i,j) probs[i,k] log(probs[i,k]/probs[j,k])
             + probs[i,j] log(probs[i,j]/probs[j,i])
    """
    dist = sp.kl_div(probs[:,None,:],probs[None,:,:])
    diags = sp.kl_div(probs,probs.T)
    dist[np.arange(probs.shape[0]),:,np.arange(probs.shape[0])] = 0
    dist[:,np.arange(probs.shape[0]),np.arange(probs.shape[0])] = 0
    return np.sum(dist,axis=-1) + diags