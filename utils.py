from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def CramersV(labels_1, labels_2):
    contingency_table = pd.crosstab(pd.Series(labels_1), pd.Series(labels_2))
    chi2, _, _, _ = chi2_contingency(contingency_table)
    phi2 = chi2 / np.sum(contingency_table.values)
    r, k = contingency_table.shape

    return np.sqrt(phi2 / min(k - 1, r - 1))

def generate_theta(
        community_sizes,
        distribution="exponential",
        seed=None,
        **kwargs) -> np.array:
    """
    Generate values for theta (degree corrections) using the specified probability distribution.
    
    :param community_sizes: List of community sizes, e.g., [70, 50, 100]
    :param distribution: String specifying the distribution to use (default: "exponential")
    :param seed: Random seed for reproducibility
    :param **kwargs: Distribution-specific parameters
    
    :return: numpy array of generated values
    """
    rng = np.random.default_rng(seed)
    
    distribution_funcs = {
        "exponential": rng.exponential,
        "normal": rng.normal,
        "uniform": rng.uniform,
        "gamma": rng.gamma,
        "beta": rng.beta,
        "poisson": rng.poisson,
        "lognormal": rng.lognormal,
    }
    
    if distribution.lower() not in distribution_funcs:
        raise ValueError(f"Unsupported distribution: {distribution}. "
                         f"Supported distributions: {', '.join(distribution_funcs.keys())}")
    
    dist_func = distribution_funcs[distribution.lower()]
    
    return np.concatenate([
        dist_func(size=size, **kwargs)
        for size in community_sizes
    ])


def generate_theta_powerlaw(
        community_sizes,
        alpha=2.0,
        dmin=1.0,
        dmax=5.0,
        seed=None) -> np.array:
    """
    Implementation of the powerlaw distribution from the paper
    "Synthetic Graph Generation to Benchmark Graph Learning" 
    (Tsitsulin et al. 2022) https://arxiv.org/pdf/2204.01376.

    :param community_sizes: E.g.: [70, 50, 100]
    :param alpha: Exponential Scale
    :param dmin: ...
    :param dmax: ...
    :param seed: random state.

    returns: theta
    """
    rng = np.random.default_rng(seed)
    thetas = []
    for size in community_sizes:
        u = rng.uniform(0, 1, size)
        t = ((dmax**(1-alpha) - dmin**(1-alpha)) * u + dmin**(1-alpha)) ** (1 / (1 - alpha))
        thetas.append(t)
    return np.concatenate(thetas)


def generate_B(
        m: int,
        b_range: tuple,
        w_range: tuple,
        rs: int = False) -> np.array:
    """
    B is the matrix that is used for the SBM as well as for generating proper covariance matrices.

    :param m: row and col dimension
    :param b_range: between range -> triangular part
    :param w_range: within range -> diagonal part
    :param rs: random state
    """
    if rs:
        np.random.seed(rs)

    B = np.random.uniform(*b_range, size=(m, m))
    B = np.tril(B) + np.tril(B, -1).T
    diag_elements = np.random.uniform(*w_range, size=m)
    np.fill_diagonal(B, diag_elements)

    return B


def generate_omega(
        n_targets: int,
        n_communities: int,
        k_clusters: int,
        w_x: float,
        w_com: float,
        distribution: str = "uniform",
        seed: int = None,
        **kwargs) -> np.array:
    """
    Generate omega matrix with cluster and community weights drawn from a flexible distribution.
    
    :param n_targets: Number of targets
    :param n_communities: Number of communities
    :param k_clusters: Number of clusters
    :param w_x: Diagonal boost for cluster weights
    :param w_com: Diagonal boost for community weights
    :param distribution: Distribution to sample from (default: "uniform")
    :param seed: Random seed
    :param kwargs: Additional distribution parameters
    
    :return: Concatenated omega matrix of shape (n_targets, k_clusters + n_communities)
    """
    rng = np.random.default_rng(seed)
    
    distribution_funcs = {
        "exponential": rng.exponential,
        "normal": rng.normal,
        "uniform": rng.uniform,
        "gamma": rng.gamma,
        "beta": rng.beta,
        "poisson": rng.poisson,
        "lognormal": rng.lognormal,
    }

    if distribution.lower() not in distribution_funcs:
        raise ValueError(f"Unsupported distribution: {distribution}. "
                         f"Supported: {', '.join(distribution_funcs)}")
    
    dist_func = distribution_funcs[distribution.lower()]

    x_betas:np.array = dist_func(size=(n_targets, k_clusters), **kwargs)
    community_betas:np.array = dist_func(size=(n_targets, n_communities), **kwargs)

    np.fill_diagonal(x_betas, x_betas.diagonal() + w_x)
    np.fill_diagonal(community_betas, community_betas.diagonal() + w_com)

    return np.hstack((x_betas, community_betas))


def generate_X(
        n_observations: int,  
        mu: list,
        sigma: list,
        w: any,
        normalize: bool = False
        ) -> Tuple[np.array, np.array]:
    """
    :param n_observations: number of observations (nodes)
    :param n_feat_clust: number of feature clusters
    :param mu: list of tuples corresponding to the cluster means
        num of features (tuple-length) and number of components (list-length)
        e.g.: [(1,1),(2,2),(2,3)] -> 3 clusters in 2-dimensional space
    :param sigma: list of Covariance matrices
    :param w: mixture weights for feature cluster sizes
        (either list of probabilities summing up to one, or the actual number of nodes assigned to the clusters)

    returns: (X, cluster_labels)

    Generate the node feature matrix:
    Add numeric features. The X values for each component are sorted (asc. order)
        so indexing is straightforward (beneficial for Adj. Matrix?)
    """
    assert len(mu) == len(sigma) == len(w), \
        f"Different dimensions chosen for mu-{len(mu)}-, sigma-{len(sigma)}-, w-{len(w)}-. "
    
    assert [len(m) for m in mu] == [s.shape[0] for s in sigma], \
        f"Different dimensions chosen for mu-{[len(m) for m in mu]}- and sigma-{[s.shape[0] for s in sigma]}-."
    
    n_feat_clust = len(mu)

    if np.isclose(sum(w), 1, atol=1.0e-8):
        # w are probs -> Sample clustersizes at random.
        cluster_labels = np.sort(
            np.random.choice(n_feat_clust, size=n_observations, p=w)
        )

    elif sum(w) == n_observations:
        num = np.arange(len(w), dtype=np.int64)
        cluster_labels = np.repeat(num, w)

    elif sum(w) != n_observations:
        print(f"\033[91mWarning: Sum of X-Cluster adjusted by {n_observations-sum(w)}! \033[0m")
        w[0] += (n_observations-sum(w))

        num = np.arange(len(w), dtype=np.int64)
        cluster_labels = np.repeat(num, w)

    X = np.array([
        np.random.multivariate_normal(mu[label], sigma[label])
        for label in cluster_labels
        ])

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, cluster_labels


if __name__=="__main__":
    B = generate_B(3, (.4, .5), (.6, .7), 24)
    X = generate_X(
        n_nodes=15,
        n_feat_clust=3,
        mu=[(-10,-10, -10),(0,0,0),(10,10,10)],
        sigma=[np.eye(2), np.eye(2), np.eye(2)],
        w=[5, 5, 5]
        ) 
    print(np.round(X, 2))