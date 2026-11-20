import networkx as nx
import numpy as np
import igraph as ig
from utils import load_graph_data

def compute_dc_plus(G):
    """Calculate DC⁺ centrality"""
    dc_plus = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        avg_deg = np.mean([G.degree(n) for n in neighbors]) if neighbors else 0
        dc_plus[node] = G.degree(node) * avg_deg
    return dc_plus

def compute_dcgm_plus(G, R):
    """Calculate DCGM⁺ centrality"""
    dcgm_plus = {}
    dc_plus = compute_dc_plus(G)
    for node in G.nodes():
        sum_influence = 0
        for neighbor in G.nodes():
            if node != neighbor and nx.shortest_path_length(G, node, neighbor) <= R:
                sum_influence += (dc_plus[node] * dc_plus[neighbor]) / (nx.shortest_path_length(G, node, neighbor) ** 2)
        dcgm_plus[node] = sum_influence
    return dcgm_plus

def compute_dcgm_plus_plus(G, R):
    """Calculate DCGM⁺⁺ centrality"""
    dcgm_plus_plus = {}
    dc_plus = compute_dc_plus(G)
    dcgm = {}
    for node in G.nodes():
        sum_influence = 0
        for neighbor in G.nodes():
            if node != neighbor and nx.shortest_path_length(G, node, neighbor) <= R:
                sum_influence += (G.degree(node) * G.degree(neighbor)) / (nx.shortest_path_length(G, node, neighbor) ** 2)
        dcgm[node] = sum_influence
    for node in G.nodes():
        dcgm_plus_plus[node] = dc_plus[node] * dcgm[node]
    return dcgm_plus_plus

def DC_plus(G):
    av_nei_deg = nx.average_neighbor_degree(G)
    DC_AND = {i : G.degree(i)*av_nei_deg[i] for i in G.nodes()}

    return DC_AND

# epidemic threshold: beta_c
def get_beta_c(G, N):
    k = sum([G.degree(i) for i in G.nodes()])/N
    square_k = sum([G.degree(i)**2 for i in G.nodes()])/N

    return k/(square_k-k)

# Calculate the shortest path distance matrix of the network
def get_distance_matrix(G, N):
    g = ig.Graph.from_networkx(G)
    DM = np.array(g.shortest_paths()).reshape(N,N)

    return DM

# references: Li H, Shang Q, Deng Y. A generalized gravity model for influential spreaders identification in complex networks[J].
# Chaos, Solitons & Fractals, 2021, 143: 110456.
def cal_SP(G):
    SP = {}
    for i in G.nodes():
        ci = nx.clustering(G, i)
        SP[i] = np.exp(-2.0*ci)*G.degree(i)
    return SP

def baseline(folder_name, name, R):
    if name == "01_Karate":
        G = nx.karate_club_graph()
    else:
        G = load_graph_data(folder_name, name)
    N, M = len(G.nodes()), len(G.edges())
    # print(N, M)

    beta_c = get_beta_c(G, N)

    DC = dict(nx.degree(G))
    SP = cal_SP(G)
    DC_AND = DC_plus(G)

    # Calculate centrality scores
    g = ig.Graph.from_networkx(G)
    DCGM = {}
    GGC = {}
    DCGM1 = {}
    DCGM2 = {}
    
    nodes = G.nodes()

    for i in nodes:
        s1 = 0
        s2 = 0
        s3 = 0
        ball_i_R = set(g.neighborhood(i, order=R))
        ball_i_R.remove(i)
        for j in ball_i_R:
            dij = g.distances(i, j)[0][0]
            if dij>0:
                s1 += (DC[i] * DC[j]) / (dij ** 2)
                s2 += (SP[i] * SP[j]) / (dij ** 2)
                s3 += (DC_AND[i] * DC_AND[j]) / (dij ** 2)
        DCGM[i] = s1
        GGC[i] = s2
        DCGM1[i] = s3
        DCGM2[i] = DC_AND[i]*s1
    
    return beta_c, DC, DC_AND, DCGM, GGC, DCGM1, DCGM2