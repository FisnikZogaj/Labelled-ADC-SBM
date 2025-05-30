import itertools
import random

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

class SBM:

    def __init__(self, community_sizes:any, B:np.array, rs:int=None):

        """
        Simple implementation of the Stochastic Block Model, which serves as a Base class 
        for further extensions of the model.  
        :params: community_sizes: E.g.: [70, 50, 100]
        :params: B: Block Matrix
        :params: rs: random state (seed)
        """
        # ---- initial inputs ----
        self.community_sizes = community_sizes
        self.B = B
        self.rs = rs        

        # ---- Attributes after computations ----
        self.n = sum(community_sizes)
        self.community_labels = self._assign_community_labels()
        self.A = None
        self.graph:nx.Graph = self._gen_graph()
    
    def _assign_community_labels(self):
        """
        Assigngs community labels based on the community size vector.
        """
        return np.concatenate([
            [i] * size
            for i, size
            in enumerate(self.community_sizes)
            ])

    def __getattr__(self, name) -> nx.Graph:
        return getattr(self.graph, name)

    def __repr__(self):
        return str(self.A)

    def _gen_graph(self):
        """
        Generates the Basic Stochastic Block Model Graph as an NetworkX graph. 
        """
        if self.rs: 
            np.random.seed(self.rs)

        prob_matrix = self.B[self.community_labels[:, None], self.community_labels[None, :]]

        upper_triangle = np.triu(np.random.rand(self.n, self.n), 1)
        edges = upper_triangle < np.triu(prob_matrix, 1)

        self.A = edges + edges.T
        G = nx.from_numpy_array(self.A)

        labels_dict = {i: int(label) for i, label in enumerate(self.community_labels)}
        nx.set_node_attributes(G=G, values=labels_dict, name='communities')

        return G

    def to_Nx(self):
        """
        Return the NetworkX Graph. 
        """
        return self.graph



class DCSBM(SBM):

    def __init__(self, community_sizes:any, B:np.array, theta:any, model:str='bernoulli', rs:int=None):

        # ---- new params, before parent class is called
        self.theta = theta
        self.model = model

        super().__init__(community_sizes, B, rs)        
        
        

    def _gen_graph(self):
        """
        Overrides the _gen_graph method from the parent class. Degree Corrected Stochastic Block Model.
        """
        if self.rs:
            np.random.seed(self.rs)

        rng = np.random.default_rng(seed=self.rs)
        θ_outer = np.outer(a=self.theta, b=self.theta)

        block_probs = self.B[self.community_labels[:, None], self.community_labels[None, :]]
        P = θ_outer * block_probs

        if self.model == 'bernoulli':
            upper = np.triu(rng.random((self.n, self.n)), 1)
            mask = upper < np.triu(P, 1)
            self.A = mask + mask.T

        elif self.model == 'poisson':
            upper = np.triu(rng.poisson(P), 1)
            self.A = upper + upper.T

        else:
            raise ValueError(f"Unknown model type: {self.model}")
        
        G = nx.from_numpy_array(self.A)

        labels_dict = {i: int(label) for i, label in enumerate(self.community_labels)}
        nx.set_node_attributes(G=G, values=labels_dict, name='communities')
        
        return G
    


class ADCSBM(DCSBM):
    def __init__(
            self,
            community_sizes:any,
            B:np.array,
            theta:any,
            X:np.array,
            cluster_labels:any, 
            model:str='bernoulli',
            rs:int=None
            ):
        """
        Further extend the DCSBM to include a feauture Matrix X.
        """
        super().__init__(community_sizes, B, theta, model, rs)

        assert X.shape[0] == self.n, (
            f'''X must have the same number of rows as the number of nodes in the graph!\n
            X.shape[0]: {X.shape[0]} != self.n: {self.n}\n'''
            )
        
        self.X:np.array = X
        self.cluster_labels = cluster_labels

        self._add_features()


    def _add_features(self):
        """
        Add features to the node attributes of the graph.
        """
        node_feature_zip = zip(
            range(self.n),
            [x for x in self.X]
            )
        
        feature_cluster_zip = zip(
            range(self.n),
            [c for c in self.cluster_labels]
            )
        
        node_feature_dict = dict(node_feature_zip)
        feature_cluster_dict = dict(feature_cluster_zip)

        nx.set_node_attributes(
            G=self.graph,
            values=node_feature_dict, 
            name='features'
            )
        
        nx.set_node_attributes(
            G=self.graph,
            values=feature_cluster_dict, 
            name='feature-cluster'
            )



class LADCSBM(ADCSBM):
    def __init__(
            self,
            community_sizes:any,
            B:np.array,
            theta:any,
            X:np.array,
            cluster_labels:any,
            model:str='bernoulli',
            seed:int=None
            ):
        super().__init__(community_sizes, B, theta, X, cluster_labels, model, seed)

        self.y = None
        self.n_targets = None 

    def set_y(self, y:np.array):
        """
        Set the labels for the nodes in the graph directly from an array.
        """
        self.y = y
        self.n_targets = None  # number of targets...

    def set_y_from_X(self, omega:np.array, eps:float=2.0):
        """
        Generate the labels for the nodes in the graph from the features.
        """
        """
        :param task: ["regression","binary","multiclass"]
        :param weights: array of numbers specifying the importance of each feature
        (order is relevant to match the feature matrix!)
        A vector if not multiclass, else a matrix with m_rows = number of classes, n_col = number of features
        E.g.: weights = np.array([0.5, 1.0, 2.0, 2.0])
        :param feature_info: if "cluster": betas for dummies are generated, else raw coefficients for numeric feature values
        :param eps: Variance of the error component, high variances will lead to heavy Y-mixing between clusters
        :return: targets
        """

        feat_mat = np.hstack((
            pd.get_dummies(self.cluster_labels).to_numpy(dtype=np.float16),
            pd.get_dummies(self.community_labels).to_numpy(dtype=np.float16)
            ))

        beta = np.ones(omega.shape) * omega

        error = np.random.normal(0, eps, (self.n, beta.shape[0]))

        logits = np.dot(feat_mat, beta.T) + error
        probabilities = softmax(logits, axis=1)
        
        self.y = np.argmax(probabilities, axis=1)
        self.n_targets = probabilities.shape[1]  # number of targets...

        node_target_zip = zip(
            range(self.n),
            self.y.astype(int)
            ) 

        nx.set_node_attributes(
            G=self.graph,
            values=dict(node_target_zip), 
            name='targets'
            )


    