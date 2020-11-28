import numpy as np
from tqdm import tqdm

def from_vecs(vecs):
    return vecs@vecs.T

def stocheq_iter(g,T,label_weights=None):
    """
    Given 
    """
    new_g = np.array([[stocheq_step(g,T,i,j,label_weights=label_weights) for j in range(g.shape[1])] for i in tqdm(range(g.shape[0]))])
    return new_g/np.sum(new_g)

def stocheq_step(g,T,i,j,label_weights=None):
    if label_weights is None:
        label_weights = np.eye(T.shape[0])
    if len(label_weights.shape)==4:
        Aij = np.tensordot(np.tensordot(T[:,i,:],label_weights[:,i,:,j],axes=[0,0]),T[:,j,:],axes=[1,0])
    else:
        Aij = np.tensordot(np.tensordot(T[:,i,:],label_weights,axes=[0,0]),T[:,j,:],axes=[1,0])
    new_gij = np.sum(g*Aij)
    return new_gij

def trnsdeq_iter(h,T):
    """
    Given 
    """
    new_h = np.array([[[trnsdeq_step(h,T,i,j,k) for k in range(h.shape[2])] 
                      for j in (range(h.shape[1]))]
                      for i in tqdm(range(h.shape[0]))])
    return new_h/np.sum(new_h)

def trnsdeq_step(h,T,i,j,k):
    Aij = T[i,j,:,:][:,:,None]*T[i,k,:,:][:,None,:]
    new_hijk = np.sum(h*Aij)
    return new_hijk

