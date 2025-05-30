# from BlockModels import *
# from src.evals import *
# from src.utils import *
# from src.plotting import *
# from src.utils import generate_B, generate_theta, generate_X
from blockmodels import *

if __name__=="__main__":

    # 0) ---- Generate example ADC_SBM ----

    community_sizes=[20, 30, 20]
    N = sum(community_sizes)

    B = generate_B(
        m=3,
        b_range=(.1, .1),
        w_range=(.6, .7),
        rs=24
        )

    X, cl = generate_X(
        n_observations=N,
        mu=[(-10,-10),(0,0),(10,10)],
        sigma=[np.eye(2), np.eye(2), np.eye(2)],
        w=community_sizes  # overlapping clusters and communities
        )

    theta = generate_theta(  # degree-corrections 
        community_sizes,
        distribution="exponential",
        seed=24,
        scale=1.0
        )

    omega = generate_omega(
            n_targets=3,
            n_communities=len(community_sizes),
            k_clusters=3,
            w_x=2.0,
            w_com=3.0
        )

    ladcsbm = LADCSBM(
        community_sizes=community_sizes,
        B=B,
        theta=theta,
        X=X,
        cluster_labels=cl
        )

    ladcsbm.set_y_from_X(omega=omega, eps=2.0)

    G = ladcsbm.to_Nx()

    com:dict = nx.get_node_attributes(G, 'communities')
    feat:dict = nx.get_node_attributes(G, 'features')
    edges:list = G.edges(data=False)
    targets = nx.get_node_attributes(G, 'targets')

    # print(com, "\n")
    # print(feat, "\n")
    # print(edges, "\n")
    print(targets, "\n")

    # 1) ---- Scheck edge_distribution per group ----

