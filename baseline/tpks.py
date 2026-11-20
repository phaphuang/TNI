import networkx as nx
import numpy as np

def compute_TP(G, sigma=1.0, mass=None):
    """
    Compute Topology Potential (TP) for each node in graph G.
    By default, if mass is not provided, each node's mass (m_j) is assumed to be 1.
    TP(v) = sum_{u != v} m(u) * exp(-[d(v,u)]^2 / sigma^2)
    """
    # If no mass is provided, assume unit mass for all nodes.
    if mass is None:
        mass = {node: 1.0 for node in G.nodes()}
    
    TP = {}
    # Precompute all-pairs shortest path lengths. Depending on the size of G,
    # you might instead compute single_source_shortest_path_length in a loop.
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    for v in G.nodes():
        potential = 0.0
        dists = path_lengths[v]
        for u, d in dists.items():
            if u == v:
                continue
            potential += mass[u] * np.exp(- (d**2) / (sigma**2))
        TP[v] = potential
    return TP

def compute_TPKS(G, sigma=1.0):
    """
    Compute "Topology Potential with mass parameter identified by k-shell" (TPKS).
    Here the mass of each node is taken as its k-shell value.
    """
    # Compute k-shell for each node. (NetworkX's core_number function returns the k-shell decomposition.)
    ks = nx.core_number(G)
    # Use the TP computation with mass = k-shell values.
    return compute_TP(G, sigma=sigma, mass=ks)