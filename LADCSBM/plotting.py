import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns


def plot_theta_distribution(
        theta: np.array,
        bins: int = 50
        ):
    """
    Plot the distribution of a of degree corrections.

    :param data: NumPy array (can be 1D or 2D)
    :param bins: Number of histogram bins
    :param title: Plot title
    """
    flat_theta = theta.flatten()  # flatten in case it's 2D
    sns.histplot(flat_theta, bins=bins, kde=True)

    if sum(theta) ==1.0:
        plt.title("Distribution of Normalized Degree Corrections")

    else:
        plt.title("Distribution of raw Degree Corrections")

    plt.xlabel("θ")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()




def plot_edge_densities(
        G:nx.Graph,
        by_community:bool=True
        ) -> None:
    """
    Given a NetworkX graph, plot the distribution of edge densities either by community or overall.

    :param G: a NetworkX graph. 
    :param by_community: specifying, whether the plot should be grouped by community. 
    """
    # Extract node degrees
    node_degrees = [G.degree[node] for node in G.nodes]

    if by_community:
        # Extract community assignments
        community_dict = nx.get_node_attributes(G, 'communities')
        if not community_dict:
            raise ValueError("Graph does not have 'community' node attributes")

        community_labels = sorted(set(community_dict.values()))

        # Group edge densities by community
        densities_by_community = {
            l: [G.degree[node] for node in G.nodes if community_dict[node] == l]
            for l in community_labels
        }

        # Plot densities per community
        plt.figure(figsize=(10, 6))
        for label, densities in densities_by_community.items():
            sns.kdeplot(densities, label=f"Community {label}", fill=True)
        plt.title("Edge Density Distribution by Community")
    else:
        # Plot overall density
        plt.figure(figsize=(10, 6))
        sns.kdeplot(node_degrees, label="All Nodes", fill=True)
        plt.title("Overall Edge Density Distribution")

    plt.xlabel("Edge Density")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

