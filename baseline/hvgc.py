import networkx as nx
import numpy as np


# Compute H-index for each node
def compute_h_index(G, node, degrees):
    neighbor_degrees = sorted([degrees[neighbor] for neighbor in G.neighbors(node)], reverse=True)
    h_index = sum(1 for i, deg in enumerate(neighbor_degrees) if deg >= i + 1)
    return h_index

# Compute HV values
def compute_hv(h_indices, G, node, degrees):
    h_index = h_indices[node]
    hv_value = sum(degrees[neighbor] for neighbor in G.neighbors(node) if degrees[neighbor] >= h_index)
    return hv_value

# Compute structural hole constraint coefficient c(i)
def compute_constraint_coefficient(G, node):
    neighbors = list(G.neighbors(node))
    pij_values = {}
    total_weight = len(neighbors)
    
    # Compute pij values
    for j in neighbors:
        pij = 1 / total_weight
        pij_values[j] = pij
    
    # Compute c(i)
    constraint = 0
    for j in neighbors:
        inner_sum = sum(pij_values[w] * pij_values[j] for w in set(G.neighbors(j)) & set(neighbors))
        constraint += (pij_values[j] + inner_sum) ** 2
    return constraint

# Compute HVGC values
def cal_hvgc(G, node, R, constraints, hv_values, shortest_paths):
    
    hvgc_value = 0
    for other_node in G.nodes():
        if other_node == node:
            continue
        # Safely retrieve distance with default infinity
        distance = shortest_paths[node].get(other_node, float('inf'))
        if other_node != node and distance <= R:
            distance_squared = distance**2
            hvgc_value += (
                np.exp(-constraints[node]) * hv_values[node] * hv_values[other_node] / distance_squared
            )

    return hvgc_value

# Compute HVGC values
def compute_hvgc(G, R):

    degrees = dict(G.degree())
    h_indices = {node: compute_h_index(G, node, degrees) for node in G.nodes()}
    hv_values = {node: compute_hv(h_indices, G, node, degrees) for node in G.nodes()}
    constraints = {node: compute_constraint_coefficient(G, node) for node in G.nodes()}
    # Compute shortest path distances
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    hvgc_values = {}

    for node in G.nodes():

        hvgc_values[node] = cal_hvgc(G, node, R, constraints, hv_values, shortest_paths)
    
    return hvgc_values
