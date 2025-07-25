from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score
    )
from statsmodels.multivariate.manova import MANOVA

from .blockmodels import SBM
from .utils import CramersV


def _de_instanciate(G:SBM) -> nx.Graph:

    """
    Helper to convert the SBM to a NetworkX Graph.
    """

    assert isinstance(G, SBM), "G must be of type SBM or inheret from SBM."
    #TODO: Further asserts. 
    nxG = G.to_Nx()

    return nxG


def _get_community_sizes(G:SBM) -> dict:
    """
    Gets the number of nodes per community from a NetworkX Graph.
    """

    G = _de_instanciate(G)
    communities:dict = nx.get_node_attributes(G, 'communities')

    counts = Counter(communities.values())
    community_sizes = [counts[i] for i in range(max(counts)+1)]

    return community_sizes


def _get_feature_matrix(G:SBM) -> np.array:

    """
    Gets the feature matrix from a NetworkX Graph.
    """

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
    Gets the connectivity between communities, feature-clusters and targets in a Matrix.

    Args:
        G: A SBM-type Graph.
        group_by: String to specify what group connectivity to get.
    """
    assert group_by in ['communities', 'feature-cluster', 'targets'],\
        "Group must be either 'communities', 'feature-cluster' or 'targets'."

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


def get_label_correlations(G:SBM) -> pd.DataFrame:

    """
    Computes label correlations of community, feature cluster and targets.
    :return: pandas data.frame.
    """
    labels_1 = G.y
    labels_2 = G.cluster_labels
    labels_3 = G.community_labels

    correlations = pd.DataFrame({

        "Y~F": [normalized_mutual_info_score(labels_1, labels_2),
                CramersV(labels_1, labels_2),
                adjusted_rand_score(labels_1, labels_2)],

        "Y~C": [normalized_mutual_info_score(labels_1, labels_3),
                CramersV(labels_1, labels_3),
                adjusted_rand_score(labels_1, labels_3)],

    },
        index=["NMI", "CV", "ARI"]
    )

    return correlations

def feature_target_manova(G:SBM):
    """

    :return: Wilks lambda in [0, 1]
    """

    y:np.array = G.y
    X:np.array = G.X
    m:int = X.shape[1]
    

    data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(data, columns=[f"X{i + 1}" for i in range(m)] + ['Group'])
    df['Group'] = df['Group'].astype(int)

    formula = " + ".join(df.columns[:-1]) + " ~ " + df.columns[-1]
    manova = MANOVA.from_formula(formula, data=df)
    manova_result = manova.mv_test()

    # manova_result.results['Group']['stat']:
    # Wilks' lambda  0.946569  6, 992.0  9.332549  0.0
    # Pillai's trace 0.053431  6, 992.0  9.332549  0.0
    # ...

    wilks_lambda = manova_result.results['Group']['stat'].iloc[0, 0]
    p_value = manova_result.results['Group']['stat'].iloc[0, -1]

    return np.round(wilks_lambda, 3), np.round(p_value, 3)


def _simple_edge_homophily(G:SBM):
    
    y:np.array = G.y
    n:int = G.n
    G:nx.Graph = _de_instanciate(G)

    labels = np.array(y)
    total_neighbors = np.array([degree for _, degree in G.degree()])

    same_label_neighbors = np.zeros(n, dtype=int)

    neighbors_list = [list(G.neighbors(node)) for node in range(n)]

    for node, neighbors in enumerate(neighbors_list):
        same_label_neighbors[node] = np.sum(labels[neighbors] == labels[node])

    return np.sum(same_label_neighbors) / np.sum(total_neighbors)


def edge_homophily(G:SBM, adjusted:bool = False):
    """
    Computes homophilly meassure from Lim et al. (2021).
    :return:
    """

    if not adjusted:
        return _simple_edge_homophily(G)
    
    y:np.array = G.y
    n:int = G.n
    n_targets:int = G.n_targets
    G:nx.Graph = _de_instanciate(G)

    total_neighbors = np.array([tpl[1] for tpl in list(G.degree())])  # tpl: (node, ngbhr)
 
    n_y_k = np.bincount(y)

    same_label_neighbors = np.zeros(n, dtype=int)

    for node in range(n):
        neighbors = list(G.neighbors(node))
        # total_neighbors[node] = len(neighbors)
        same_label_neighbors[node] = sum(y[neighbor] == y[node] for neighbor in neighbors)

    h_k = np.zeros(n_targets)
    for l in range(n_targets):
        numerator = sum(same_label_neighbors[np.where(y == l)])
        denominator = sum(total_neighbors[np.where(y == l)])
        h_k[l] = numerator/denominator  # indexed 0, 1, ..., tau


    h_hat = (
        (1/(n_targets-1)) * sum(np.maximum(np.zeros(n_targets), h_k - (n_y_k / n)))
        )
    
    return h_hat