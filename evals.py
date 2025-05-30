import numpy as np
import networkx as nx
import pandas as pd
from collections import Counter

from BlockModels import SBM


class Evals:
    def __init__(self, G:nx.Graph | SBM):

        if isinstance(G, SBM):  # Also covers subclasses (DCSBM, ADCSBM, ...) 
            self.G = G.to_Nx()

        elif isinstance(G, nx.Graph):
            self.G = G

        else:
            raise TypeError("G must be of type nx.Graph or inheret from SBM")
        
        assert all(nx.get_node_attributes(self.G, attr) for attr in ['communities']), \
            "G must have node attributes 'communities'"
        
        self.is_directed:bool = nx.is_directed(self.G)


    def _get_community_sizes(self) -> dict:

        communities:dict = nx.get_node_attributes(self.G, 'communities')

        counts = Counter(communities.values())
        community_sizes = [counts[i] for i in range(max(counts)+1)]

        return community_sizes


    def _get_feature_matrix(self) -> np.array:
        feats:dict = nx.get_node_attributes(self.G, 'features') 

        return np.array([x for x in feats.values()])


    def get_number_of_edges_per_node(self) -> dict:
        """
        :param Graph: A nx Graph.
        returns: a dictionary with the number of edges per community.
        """
        return dict(self.G.degree(self.G.nodes()))


    def _get_empirical_B_from_undirected(self):
        """
        
        """
        community_sizes = self._get_community_sizes()
        com = nx.get_node_attributes(self.G, 'communities')

        edges:list = self.G.edges(data=False) # [(0, 3), (0, 4), ...
        k:int = len(community_sizes) # community_sizes e.g.: [10, 20, 30]
        B_emp = np.zeros((k, k))

        for (node_i, node_j) in edges:
            com_of_node_i, com_of_node_j = com[node_i], com[node_j]
            B_emp[com_of_node_i, com_of_node_j] += 1

        upper_triangular = np.triu(B_emp)

        diagonal = np.diag(np.diag(upper_triangular))
        B_emp = upper_triangular + upper_triangular.T - diagonal

        return B_emp


    def _get_empirical_B_from_directed(self):
        # TODO implement
        raise NotImplementedError


    def get_number_edges_between_communities(self):
        
        if self.is_directed:
            return self._get_empirical_B_from_directed()
        else:
            return self._get_empirical_B_from_undirected()


    def get_number_edges_between_targets(self):
        # Note:This works for undirected Graphs. 

        edges:list = self.G.edges(data=False)
        targets:dict = nx.get_node_attributes(self.G, 'targets')
        nt:int = len(set(targets.values()))  # Number of targets

        counter = {i: {j: 0 for j in range(0, nt)} for i in range(0, nt)}
        # {0: {0:_,1:_,2:_},1: {0:_,1:_,2:_}, 2: {0:_,1:_,2:_}} example for nt = 3
        for e in edges:
            i, j = e 
            target_i = targets[i]
            target_j = targets[j] 

            if target_i == target_j:
                counter[target_i][target_j] += 1

            else:
                counter[target_i][target_j] += 1
                counter[target_j][target_i] += 1

        return pd.DataFrame(counter)
        