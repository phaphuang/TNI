import networkx as nx
import numpy as np
from collections import defaultdict

def calculate_pc(G):
    """Calculate the potential of nodes forming clusters"""
    pc = {}
    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        k_i = len(neighbors)
        
        sum_k_in = 0
        for j in neighbors:
            # Count links connecting j to i and i's neighbors
            k_j_in = 1  # Link to i
            for neighbor in neighbors:
                if G.has_edge(j, neighbor):
                    k_j_in += 1
            sum_k_in += k_j_in
        
        pc[i] = k_i * sum_k_in
    
    return pc

def detect_clusters(G, lambda_param=0.5):
    """Detect clusters and identify global critical nodes"""
    N = G.number_of_nodes()
    target_nodes = int(N * lambda_param)
    
    # Initialize
    clusters = []
    nodes_in_clusters = set()
    pc_values = calculate_pc(G)
    
    while len(nodes_in_clusters) < target_nodes:
        # Find node with maximum pc value
        remaining_nodes = {n: pc for n, pc in pc_values.items() if n not in nodes_in_clusters}
        if not remaining_nodes:
            break
            
        pc_max = max(remaining_nodes.values())
        candidates = [n for n, pc in remaining_nodes.items() if pc == pc_max]
        
        # Select node with highest degree
        initial_node = max(candidates, key=lambda n: G.degree(n))
        
        # Form initial cluster
        cluster = {initial_node}
        for neighbor in G.neighbors(initial_node):
            if neighbor not in nodes_in_clusters and pc_values.get(neighbor, 0) >= pc_max / 2:
                cluster.add(neighbor)
        
        # Expand cluster (3 iterations as per three degrees of influence rule)
        for _ in range(3):
            new_additions = set()
            neighbors_to_check = set()
            
            for node in cluster:
                neighbors_to_check.update(n for n in G.neighbors(node) if n not in cluster)
            
            # Sort by increasing degree
            neighbors_to_check = sorted(neighbors_to_check, key=lambda n: G.degree(n))
            
            for i in neighbors_to_check:
                if i in nodes_in_clusters:
                    continue
                    
                # Count links to cluster and outside
                k_in = sum(1 for n in G.neighbors(i) if n in cluster)
                k_out = G.degree(i) - k_in
                
                if k_in >= k_out:
                    new_additions.add(i)
            
            cluster.update(new_additions)
        
        # Add cluster and update nodes in clusters
        clusters.append(cluster)
        nodes_in_clusters.update(cluster)
        
        # Set pc values to 0 for nodes in the cluster
        for node in cluster:
            pc_values[node] = 0
    
    # Find global critical nodes (highest degree node in each cluster)
    global_critical_nodes = []
    for cluster in clusters:
        if cluster:
            critical_node = max(cluster, key=lambda n: G.degree(n))
            global_critical_nodes.append(critical_node)
    
    return clusters, global_critical_nodes

def calculate_k_shell(G):
    """Calculate k-shell values for all nodes"""
    k_shell = {}
    remaining_graph = G.copy()
    k = 1
    
    while remaining_graph.number_of_nodes() > 0:
        # Find nodes with degree <= k
        while True:
            to_remove = [n for n in remaining_graph.nodes() if remaining_graph.degree(n) <= k]
            if not to_remove:
                break
                
            for node in to_remove:
                k_shell[node] = k
                remaining_graph.remove_node(node)
        
        k += 1
    
    return k_shell

def calculate_local_influence(G, k_shell):
    """Calculate local influence (NCC) for all nodes"""
    local_influence = {}
    
    for i in G.nodes():
        li = sum(k_shell[j] for j in G.neighbors(i))
        local_influence[i] = li
    
    return local_influence

def compute_glc(G, lambda_param=0.5):
    """Calculate GLC for all nodes"""
    # Step 1: Detect clusters and find global critical nodes
    clusters, global_critical_nodes = detect_clusters(G, lambda_param)
    
    # Step 2: Calculate k-shell values
    k_shell = calculate_k_shell(G)
    
    # Step 3: Calculate local influence
    local_influence = calculate_local_influence(G, k_shell)
    
    # Step 4: Calculate shortest paths to global critical nodes
    shortest_paths = {}
    for node in G.nodes():
        shortest_paths[node] = {}
        for gcn in global_critical_nodes:
            try:
                shortest_paths[node][gcn] = nx.shortest_path_length(G, node, gcn)
            except nx.NetworkXNoPath:
                shortest_paths[node][gcn] = float('inf')
    
    # Step 5: Calculate GLC
    glc = {}
    for i in G.nodes():
        global_influence = 0
        for u in global_critical_nodes:
            if shortest_paths[i][u] != float('inf'):
                global_influence += local_influence[u] / (2 ** shortest_paths[i][u])
        
        glc[i] = local_influence[i] * global_influence
    
    return glc

# Example usage
def main():
    # Create a sample graph
    G = nx.karate_club_graph()
    
    # Calculate GLC with lambda=0.5
    glc_values = compute_glc(G, lambda_param=0.5)
    
    # Sort nodes by GLC value
    sorted_nodes = sorted(glc_values.items(), key=lambda x: x[1], reverse=True)
    
    # Print top 10 influential nodes
    print("Top 10 influential nodes by GLC:")
    for node, value in sorted_nodes[:10]:
        print(f"Node {node}: GLC = {value:.2f}")

if __name__ == "__main__":
    main()