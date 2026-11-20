import networkx as nx
import numpy as np
from scipy.stats import kendalltau
import EoN

# Load real network data to generate the network
def load_graph_data(folder_name, name):
    if name == 'karate':
        G = nx.karate_club_graph()
    else:
        try:
            G = nx.read_edgelist(f"{folder_name}/{name}.edgelist", nodetype=np.int64)
        except:
            # edgelist contains simple weights, specify the data format explicitly
            G = nx.read_edgelist(f"{folder_name}/{name}.edgelist", nodetype=np.int64, data=(('weight', float),))
            # G = nx.read_edgelist(f"{folder_name}/{name}.edgelist", nodetype=np.int64, data=False)

    # Identify self-loop edges
    self_loops = list(nx.selfloop_edges(G))

    # Remove all self-loop edges from the graph
    G.remove_edges_from(self_loops)
    
    N = nx.number_of_nodes(G)
    mapping = dict(zip(G, range(N)))
    G = nx.relabel_nodes(G, mapping)
    return G

def cal_Kendall_tau_coefficient(X, Y):
    tau, p_value = kendalltau(X, Y)
    return tau

# epidemic threshold: beta_c
def get_beta_c(G, N):
    k = sum([G.degree(i) for i in G.nodes()])/N
    square_k = sum([G.degree(i)**2 for i in G.nodes()])/N

    return k/(square_k-k)

def normalize_weights(G):
    """Normalize weights to a consistent scale (e.g., [0.1, 1.0])"""
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    min_weight = min(weights)
    max_weight = max(weights)
    
    # Avoid division by zero if all weights are equal
    if max_weight == min_weight:
        return G
    
    # Create a copy of the graph to avoid modifying the original
    H = G.copy()
    
    # Scale weights to range [0.1, 1.0]
    for u, v in H.edges():
        normalized_weight = 0.1 + 0.9 * (H[u][v]['weight'] - min_weight) / (max_weight - min_weight)
        H[u][v]['normalized_weight'] = normalized_weight
    
    return H

# Monotonicity index formula
def monotonicity_index(graph, centrality):
    N = len(graph.nodes)
    centrality_values = list(centrality.values())
    _, counts = np.unique(centrality_values, return_counts=True)
    numerator = sum(count * (count - 1) for count in counts)
    denominator = N * (N - 1)
    monotonicity = (1 - numerator / denominator) ** 2
    return monotonicity

# Monotonicity index formula
def sir_monotonicity_index(graph, sir_score):
    N = len(graph.nodes)
    centrality_values = sir_score
    _, counts = np.unique(centrality_values, return_counts=True)
    numerator = sum(count * (count - 1) for count in counts)
    denominator = N * (N - 1)
    monotonicity = (1 - numerator / denominator) ** 2
    return monotonicity

# def calculate_robustness_curve(G, scores, attack_strategy='targeted'):
#     """Calculate robustness curve (R-curve) for a network.
    
#     Parameters:
#     G (NetworkX Graph): Input network
#     centrality_measure: 'betweenness', 'degree', or 'closeness'
#     attack_strategy: 'targeted' or 'random'
    
#     Returns:
#     tuple: (x_values, y_values, R_value)
#     """
#     G = G.copy()
#     n = len(G)
#     lcc_sizes = []
    
#     # Get node removal order
#     if attack_strategy == 'targeted':
#         removal_order = sorted(scores, key=scores.get, reverse=True)
#     else:  # Random attack
#         removal_order = list(G.nodes())
#         np.random.shuffle(removal_order)

#     # Initial LCC
#     lcc = max(nx.connected_components(G), key=len)
#     lcc_sizes.append(len(lcc)/n)

#     # Simulate node removal
#     for i, node in enumerate(removal_order):
#         if node in G:
#             G.remove_node(node)
#             if G.number_of_nodes() == 0:
#                 break
#             current_lcc = max(nx.connected_components(G), key=len)
#             lcc_sizes.append(len(current_lcc)/n)

#     # Calculate robustness value (area under curve)
#     # x = np.linspace(0, 1, len(lcc_sizes))
#     # R = np.trapezoid(lcc_sizes, x) / len(lcc_sizes)
#     x = [i/len(removal_order) for i in range(len(lcc_sizes))]
#     # R = np.trapz(lcc_sizes, x) / max(x)
#     R = np.trapezoid(lcc_sizes, x) / max(x)
    
#     return x, lcc_sizes, R

def calculate_robustness_curve(G, scores):
    """
    Calculate network robustness value R according to Definition 3.
    
    Parameters:
    G (NetworkX Graph): Input network
    centrality_measure: 'degree', 'betweenness', or 'closeness'
    
    Returns:
    tuple: (R_value, x_values, y_values)
    """
    G = G.copy()
    n = G.number_of_nodes()
    gcc_original = len(max(nx.connected_components(G), key=len))
        
    nodes_to_remove = sorted(scores, key=scores.get, reverse=True)
    
    # Track GCC size after each removal
    gcc_sizes = []
    G_temp = G.copy()
    
    for node in nodes_to_remove:
        G_temp.remove_node(node)
        if G_temp.number_of_nodes() == 0:
            break
        gcc_size = len(max(nx.connected_components(G_temp), key=len))
        gcc_sizes.append(gcc_size / gcc_original)  # Normalize by original GCC size
    
    # Calculate R value according to the formula
    R = sum(gcc_sizes) / n
    
    # Generate x values for plotting
    x_values = [(i+1)/n for i in range(len(gcc_sizes))]
    
    return x_values, gcc_sizes, R

# def plot_robustness_results(results, title="Network Efficiency Ratio vs. Fraction of Nodes Removed"):
#     """
#     Plot the network efficiency ratio against the fraction of nodes removed for USAir97
#     """
#     #plt.figure(figsize=(7, 5))

#     # Define colors similar to the original figure
#     colors = {
#         #'RA': '#1AAB40',    # Orange
#         #'DC': '#1f77b4',    # Blue
#         #'DC_AND': '#8c564b',  # Light Orange
#         'CM': '#4A2D75',    # Gray
#         #'TPKS': '#e377c2',  # Pink
#         #'DCGM': '#ff7f0e',  # Orange
#         #'GGC': '#2ca02c',   # Green
#         'DCGM_PLUS': '#E645AB', # Red
#         'DCGM_PLUS_PLUS': '#6B007B', # Purple
#         'HVGC': '#E66C37',
#         'GLC': '#12239E',
#         'TNI': '#118DFF',   # Black
#     }
    
#     # Plot each attack strategy
#     for i, (strategy, color) in enumerate(colors.items()):
#         if strategy in results:
#             #print(results[strategy])
#             (x, y, R) = results[strategy][0]
#             plt.plot(x, y, label=strategy+" "+ f"(R={R:.4f})", color=color)
#             plt.fill_between(x, 0, y, alpha=0.05, color='lightblue')
    
#     # Set plot properties
#     #plt.title("Comparative Robustness Analysis")
#     plt.legend(fontsize=10)
#     plt.tight_layout()
    
#     return plt