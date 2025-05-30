import numpy as np
import networkx as nx
import pandas as pd
from collections import Counter

from .blockmodels import SBM



def _de_instanciate(G:SBM) -> nx.Graph:

    assert isinstance(G, SBM), "G must be of type SBM or inheret from SBM."
    #TODO: Further asserts. 
    nxG = G.to_Nx()

    return nxG


def _get_community_sizes(G:SBM) -> dict:

    G = _de_instanciate(G)
    communities:dict = nx.get_node_attributes(G, 'communities')

    counts = Counter(communities.values())
    community_sizes = [counts[i] for i in range(max(counts)+1)]

    return community_sizes


def _get_feature_matrix(G:SBM) -> np.array:
    G = _de_instanciate(G)
    feats:dict = nx.get_node_attributes(G, 'features') 

    return np.array([x for x in feats.values()])


def get_node_degrees(G:SBM) -> dict:
    """
    :param Graph: A nx Graph.
    returns: a dictionary with the number of edges per community.
    """
    G = _de_instanciate(G)
    return dict(G.degree(G.nodes()))


# ---- Connectivity between communities, targets and feature-clusters ----
    

def get_group_connectivity(G:SBM, group_by:str): 

    """
    
    """
    assert group_by in ['communities', 'feature-cluster', 'targets'],\
        "Group must be 'communities', 'feature-cluster' or 'targets'."

    edges:list = G.edges(data=False)
    group:dict = nx.get_node_attributes(G, group_by)
    g_set:int = len(set(group.values()))  # Number of targets

    counter = {i: {j: 0 for j in range(0, g_set)} for i in range(0, g_set)}
    # {0: {0:_,1:_,2:_},1: {0:_,1:_,2:_}, 2: {0:_,1:_,2:_}} example for nt = 3
    for e in edges:
        i, j = e 
        target_i = group[i]
        target_j = group[j] 

        if target_i == target_j:
            counter[target_i][target_j] += 1

        else:
            counter[target_i][target_j] += 1
            counter[target_j][target_i] += 1

    return pd.DataFrame(counter)


# TODO: Add

# Correlation between cat~num and cat~cat
#
