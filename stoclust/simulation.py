import numpy as np
from tqdm import tqdm
from stoclust import utils
from stoclust import regulators
from stoclust.Group import Group

def markov_random_walk(probs,initial=None,group=None,regulator=None,halt=None,**kwargs):
    """
    Given a set of transition probabilities, generates a random walk
    through the available nodes. The initial node, if not specified,
    is randomly selected. The method returns two items.
    The first is the report array R,
    dimensions Mx2, where the reports are indexed by the axis M.
    R[i,0] is the content of the report and R[i,1] is the time of the report.
    The second is the path P, an N-dimensional vector, where N is the number of steps
    in the random walk and P[j] is the node at time j.

    The walk may be regulated. This involves passing a regulator,
    which is a function that takes the simulation time, 
    the transition probabilities, the current node, 
    and an array of node data. At each time, 
    the regulator returns a Boolean indicating
    whether a report is to be made, the content of the report,
    and a new set of transition probabilities determined by
    the available information. The regulator also
    updates node_data in-place.

    Lastly, one can either specify a maximum length
    of time for the walk (using the keyword max_time) 
    or a more general halting condition (using the keyword halt).
    A halting condition takes the time, current node, and node data.
    If neither are specified, the maximum length will
    be set to 100 steps.
    """
    if group is None:
        group = Group(np.arange(probs.shape[0]))

    if regulator is None:
        regulator = lambda t,ps,an,nd: (False,None,ps)

    if halt is None:
        max_time = kwargs.get('max_time',100)
        halt = lambda t,an,nd: regulators.halt_after_time(t,an,nd,max_time=max_time)

    if initial is None:
        initial_ind = np.random.choice(np.arange(probs.shape[0]))
    else:
        initial_ind = group.ind[initial]
    
    tol = kwargs.get('tol',1e-6)
    
    reports = []
    locations = []
    node_data = np.zeros([probs.shape[0]])

    t = 0
    current = initial_ind
    will_report, report, new_probs = regulator(t,probs,current,node_data)
    if will_report:
            reports.append([report,t])
    locations.append(initial)
    t += 1

    while not(halt(t,current,node_data)):
        if np.sum(new_probs[current,:])<tol:
            remaining = np.where(np.sum(new_probs,axis=0)>tol)[0]
            sequel = np.random.choice(remaining)
        else:
            sequel = np.random.choice(np.arange(probs.shape[0]),p=new_probs[current,:])
        
        will_report, report, new_probs = regulator(t,probs,sequel,node_data)
        if will_report:
            reports.append([report,t])

        current = sequel
        locations.append(group.elements[current])
        t += 1
    
    return np.array(reports), np.array(locations)