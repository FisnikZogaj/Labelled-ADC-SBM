import numpy as np
import pandas as pd
import networkx as nx

from scipy.special import softmax 

def _cartprod(*arrays):
    N = len(arrays)
    return np.transpose(np.meshgrid(*arrays, indexing="ij"), np.roll(np.arange(N + 1), -1)).reshape(-1, N)

def _symmetrize(graph, method="triu"):
    if method == "triu":
        graph = np.triu(graph)
    elif method == "tril":
        graph = np.tril(graph)
    elif method == "avg":
        graph = (np.triu(graph) + np.tril(graph)) / 2
    else:
        raise ValueError("invalid method")
    graph = graph + graph.T - np.diag(np.diag(graph))
    return graph


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
    def __init__(self, community_sizes, B: np.array, theta, rs: int = None):
        self.theta = np.asarray(theta)
        super().__init__(community_sizes, B, rs)


    def _gen_graph(self):
        """
        Graspy(Version: 0.2.0)-like degree-corrected SBM (undirected, no loops).
        - Uses global np.random so np.random.seed(rs) reproduces graspy.
        - Only Bernoulli-style sampling (no 'model' choices).
        - If theta provided, it's normalized per-block and used as selection weights
          when choosing which pairs become edges (graspy procedure).
        """
        if self.rs is not None:
            np.random.seed(self.rs)

        n = self.n
        K = len(self.community_sizes)

        # community ranges
        cmties = []
        counter = 0
        for size in self.community_sizes:
            cmties.append(range(counter, counter + size))
            counter += size

        # prepare dcProbs normalized per block
        dcProbs = None
        if self.theta is not None:
            dcProbs = self.theta.astype(float).copy()
            for i, indices in enumerate(cmties):
                idx = np.array(list(indices))
                s = dcProbs[idx].sum()
                if s > 0:
                    dcProbs[idx] = dcProbs[idx] / s
                else:
                    dcProbs[idx] = 0.0

        A = np.zeros((n, n), dtype=int)
        block_probs = self.B

        for i in range(K):
            for j in range(i, K):
                cprod = _cartprod(np.arange(cmties[i].start, cmties[i].stop), np.arange(cmties[j].start, cmties[j].stop))
                v1 = cprod[:, 0].astype(int)
                v2 = cprod[:, 1].astype(int)
                triu = np.ravel_multi_index((v1, v2), (n, n))
                block_p = block_probs[i, j]

                if len(triu) == 0:
                    continue

                pchoice = np.random.uniform(size=len(triu))
                if dcProbs is not None:
                    num_edges = int((pchoice < block_p).sum())
                    if num_edges == 0:
                        continue
                    edge_dist = dcProbs[v1] * dcProbs[v2]
                    support = (edge_dist > 0).sum()
                    if support == 0:
                        continue
                    if num_edges > support:
                        num_edges = support
                    probs = edge_dist / edge_dist.sum()
                    chosen = np.random.choice(triu, size=num_edges, replace=False, p=probs)
                    rr, cc = np.unravel_index(chosen, (n, n))
                    A[rr, cc] = 1
                else:
                    chosen = triu[pchoice < block_p]
                    if chosen.size == 0:
                        continue
                    rr, cc = np.unravel_index(chosen, (n, n))
                    A[rr, cc] = 1

        # no loops and symmetrize like graspy
        A = A - np.diag(np.diag(A))
        A = _symmetrize(A, method="triu")
        self.A = A.astype(int)
        G = nx.from_numpy_array(self.A)
        labels_dict = {i: int(label) for i, label in enumerate(self.community_labels)}
        nx.set_node_attributes(G=G, values=labels_dict, name="communities")
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


    

        

    # def _gen_graph(self):
    #     """
    #     Overrides the _gen_graph method from the parent class. Degree Corrected Stochastic Block Model.
    #     """
    #     if self.rs:
    #         np.random.seed(self.rs)

    #     rng = np.random.default_rng(seed=self.rs)
    #     θ_outer = np.outer(a=self.theta, b=self.theta)

    #     block_probs = self.B[self.community_labels[:, None], self.community_labels[None, :]]
    #     P = θ_outer * block_probs

    #     if self.model == 'bernoulli':
    #         upper = np.triu(rng.random((self.n, self.n)), 1)
    #         mask = upper < np.triu(P, 1)
    #         self.A = mask + mask.T

    #     elif self.model == 'poisson':
    #         upper = np.triu(rng.poisson(P), 1)
    #         self.A = upper + upper.T

    #     else:
    #         raise ValueError(f"Unknown model type: {self.model}")
        
    #     G = nx.from_numpy_array(self.A)

    #     labels_dict = {i: int(label) for i, label in enumerate(self.community_labels)}
    #     nx.set_node_attributes(G=G, values=labels_dict, name='communities')
        
    #     return G