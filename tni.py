import networkx as nx
import numpy as np
from baseline.hvgc import compute_hvgc
import scipy.linalg as la

def compute_global_communicability(G):
    """
    Compute the global communicability measure for each node.
    Here we use NetworkX's communicability exponential function,
    and for each node we sum its communicability with respect to all other nodes.
    """
    communicability = nx.communicability_exp(G)  # Returns a dict-of-dicts.
    global_comm = {}
    for i in G.nodes():
        global_comm[i] = (1/2) * sum(communicability[i].values())
    return global_comm

def compute_global_communicability_weighted(G, weight='weight'):
    """
    Compute the global communicability measure for each node in a weighted graph.
    This implementation normalizes weights based on node degrees before computing
    the matrix exponential, as suggested in the literature.
    """
    # Get the adjacency matrix with weights
    A = nx.to_numpy_array(G, weight=weight)
    
    # Compute the generalized degree for each node
    degrees = np.sum(A, axis=0)
    
    # Normalize the adjacency matrix
    # Avoid division by zero by adding a small epsilon where degree is zero
    epsilon = 1e-10
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + epsilon))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    
    # Compute the matrix exponential
    exp_A = la.expm(A_norm)
    
    # Convert back to a dict-of-dicts format similar to nx.communicability_exp
    communicability = {}
    for i in range(len(G.nodes())):
        communicability[i] = {j: exp_A[i, j] for j in range(len(G.nodes()))}
    
    # Compute global communicability for each node
    global_comm = {}
    for i in G.nodes():
        global_comm[i] = (1/2) * sum(communicability[i].values())
    
    return global_comm

def compute_TNI(G, R, network_type="unweighted"):
    """
    Compute the Triplet Node Influence (TNI) for each node.
    
    Steps:
    1. Compute the local measure degrees.
    2. Compute the gravity measure over nodes within distance R.
    3. Compute the global communicability.
    4. Compute TNI as the product of the three measures.

    """
    TNI = {}

    if network_type == "unweighted":
        # Local influence component
        LI = dict(nx.degree(G))
        # Gravity component
        gravity = compute_hvgc(G, R)
        # Global communicability component
        GI = compute_global_communicability(G)
    
    elif network_type == "weighted":
        # Local influence component
        LI = dict(G.degree(weight='weight'))
        # Mesoscale gravity component
        gravity = compute_hvgc(G, R)
        # Global communicability component
        GI = compute_global_communicability_weighted(G, weight='weight')

    for node in G.nodes():
        TNI[node] = LI[node] * gravity[node] * GI[node]
    
    # Return communicability matrix
    CM = GI.copy()

    return CM, TNI