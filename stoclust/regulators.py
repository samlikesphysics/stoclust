import numpy as np
from stoclust import utils

def node_removal_regulator(time,probs,at_node,node_data,max_visits=5):
    """
    Modifies transition probabilities by making it impossible to transition
    to any node which has been visited already at least max_visits times.
    Uses node_data to track the number of previous visits.
    When a node is removed, reports the ID of the removed node;
    otherwise, no report is made.
    """
    old_ind = np.where(node_data>=max_visits)[0]
    node_data[at_node] += 1
    new_ind = np.where(node_data>=max_visits)[0]
    new_probs = utils.stoch(probs*((node_data<max_visits).astype(int)[None,:]))
    updated = np.where(np.logical_not(np.isin(new_ind,old_ind)))[0]
    
    if len(updated>0):
        return True, new_ind[updated[0]], new_probs
    else:
        return False, None, new_probs

def halt_when_explored(time,at_node,node_data):
    """
    Halts when all nodes have been visited at least once.
    Assumes that node_data is being used to count visits;
    does not work otherwise. Therefore, be careful to use
    with an appropriate regulator.
    """
    return np.all(node_data>0)

def halt_after_time(time,at_node,node_data,max_time=100):
    """
    Halts after a fixed duration of time.
    """
    return time>=max_time