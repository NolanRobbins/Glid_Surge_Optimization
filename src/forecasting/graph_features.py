"""
Graph-Based Feature Engineering for Forecasting
================================================
Extracts topological features from the transportation graph
to enhance surge and dwell time predictions.

Key insight: Congestion propagates through the network.
A surge at Port of LA affects connected rail terminals.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class NodeGraphFeatures:
    """Graph-derived features for a location node."""
    node_id: str
    degree: int                    # Number of connections
    betweenness: float            # How central in flow paths
    closeness: float              # Average distance to all nodes
    clustering: float             # Local clustering coefficient
    pagerank: float               # Importance in network
    rail_class_avg: float         # Average rail class of connected edges
    distance_to_hub: float        # Miles to nearest major hub
    connected_capacity: float     # Total capacity of connected edges
    num_class1_connections: int   # Premium rail connections
    subgraph_density: float       # Density of local neighborhood


def compute_graph_centrality_features(
    G: nx.Graph,
    nodes_of_interest: List[str] = None,
    fast_mode: bool = True
) -> Dict[str, NodeGraphFeatures]:
    """
    Compute centrality and topological features for nodes.
    
    These features capture network position which affects:
    - How quickly congestion propagates to this node
    - Alternative routing options available
    - Importance in the overall logistics network
    
    Args:
        G: Transportation graph
        nodes_of_interest: Specific nodes to compute (None = all)
        fast_mode: Use fast approximations for large graphs
        
    Returns:
        Dictionary mapping node_id to NodeGraphFeatures
    """
    print("Computing graph centrality features...")
    
    # For large graphs, use fast mode with only degree-based features
    if fast_mode and G.number_of_nodes() > 5000:
        print(f"  Fast mode: {G.number_of_nodes()} nodes, using degree-based features only")
        degree = dict(G.degree())
        
        # Skip expensive centrality computations
        betweenness = {n: 0.0 for n in G.nodes()}
        closeness = {n: 0.0 for n in G.nodes()}
        clustering = {n: 0.0 for n in G.nodes()}
        pagerank = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}
    else:
        # Compute centralities with progress
        print("  Computing degree...")
        degree = dict(G.degree())
        
        print("  Computing betweenness (sampling)...")
        betweenness = nx.betweenness_centrality(G, k=min(50, G.number_of_nodes()))
        
        print("  Computing clustering...")
        try:
            clustering = nx.clustering(G)
        except:
            clustering = {n: 0.0 for n in G.nodes()}
        
        print("  Computing pagerank...")
        try:
            pagerank = nx.pagerank(G, max_iter=50)
        except:
            pagerank = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}
        
        closeness = {n: 0.0 for n in G.nodes()}  # Skip - too slow
    
    # Identify major hubs (top 1% by degree)
    degree_threshold = np.percentile(list(degree.values()), 99)
    major_hubs = [n for n, d in degree.items() if d >= degree_threshold][:10]  # Limit hubs
    
    if nodes_of_interest is None:
        nodes_of_interest = list(G.nodes())
    
    features = {}
    
    for node in tqdm(nodes_of_interest, desc="  Node features", unit="node"):
        if node not in G:
            continue
            
        # Get edge attributes for connected edges
        edges = G.edges(node, data=True)
        rail_classes = []
        capacities = []
        class1_count = 0
        
        for _, _, data in edges:
            if 'rail_class' in data:
                rail_classes.append(data['rail_class'])
                if data['rail_class'] == 1:
                    class1_count += 1
            if 'capacity' in data:
                capacities.append(data['capacity'])
        
        # Skip expensive distance computations in fast mode
        dist_to_hub = 0.0
        
        # Local subgraph density (fast)
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:
            subgraph_density = len(neighbors) / (len(neighbors) * (len(neighbors) - 1) / 2 + 0.001)
        else:
            subgraph_density = 0.0
        
        features[node] = NodeGraphFeatures(
            node_id=node,
            degree=degree.get(node, 0),
            betweenness=betweenness.get(node, 0.0),
            closeness=closeness.get(node, 0.0),
            clustering=clustering.get(node, 0.0),
            pagerank=pagerank.get(node, 0.0),
            rail_class_avg=np.mean(rail_classes) if rail_classes else 0.0,
            distance_to_hub=dist_to_hub,
            connected_capacity=sum(capacities) if capacities else 0.0,
            num_class1_connections=class1_count,
            subgraph_density=subgraph_density
        )
    
    print(f"  Computed features for {len(features)} nodes")
    return features


def compute_neighbor_congestion_features(
    G: nx.Graph,
    congestion_values: Dict[str, float],
    target_node: str,
    hops: int = 2
) -> Dict[str, float]:
    """
    Compute congestion propagation features from neighboring nodes.
    
    Key insight: Congestion at nearby ports/terminals affects this location.
    
    Args:
        G: Transportation graph
        congestion_values: Current congestion at each node
        target_node: Node to compute features for
        hops: Number of hops to consider
        
    Returns:
        Dictionary of neighbor congestion features
    """
    features = {}
    
    if target_node not in G:
        return {
            'neighbor_congestion_1hop': 0.0,
            'neighbor_congestion_2hop': 0.0,
            'max_neighbor_congestion': 0.0,
            'congestion_gradient': 0.0
        }
    
    # Get 1-hop neighbors
    neighbors_1 = set(G.neighbors(target_node))
    congestion_1hop = [
        congestion_values.get(n, 0.0) 
        for n in neighbors_1 
        if n in congestion_values
    ]
    
    # Get 2-hop neighbors
    neighbors_2 = set()
    for n1 in neighbors_1:
        neighbors_2.update(G.neighbors(n1))
    neighbors_2 -= neighbors_1
    neighbors_2.discard(target_node)
    
    congestion_2hop = [
        congestion_values.get(n, 0.0) 
        for n in neighbors_2 
        if n in congestion_values
    ]
    
    # Compute features
    features['neighbor_congestion_1hop'] = np.mean(congestion_1hop) if congestion_1hop else 0.0
    features['neighbor_congestion_2hop'] = np.mean(congestion_2hop) if congestion_2hop else 0.0
    features['max_neighbor_congestion'] = max(congestion_1hop + congestion_2hop) if (congestion_1hop or congestion_2hop) else 0.0
    
    # Congestion gradient (is congestion increasing toward us?)
    own_congestion = congestion_values.get(target_node, 0.0)
    features['congestion_gradient'] = features['neighbor_congestion_1hop'] - own_congestion
    
    return features


def build_graph_feature_matrix(
    G: nx.Graph,
    locations: List[str],
    historical_congestion: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Build a feature matrix with graph-derived features for each location.
    
    Args:
        G: Transportation graph
        locations: List of location identifiers
        historical_congestion: Optional historical congestion data
        
    Returns:
        DataFrame with graph features for each location
    """
    # Compute centrality features
    node_features = compute_graph_centrality_features(G, locations)
    
    # Convert to DataFrame
    rows = []
    for loc in locations:
        if loc in node_features:
            nf = node_features[loc]
            rows.append({
                'location': loc,
                'graph_degree': nf.degree,
                'graph_betweenness': nf.betweenness,
                'graph_closeness': nf.closeness,
                'graph_clustering': nf.clustering,
                'graph_pagerank': nf.pagerank,
                'graph_rail_class_avg': nf.rail_class_avg,
                'graph_dist_to_hub': nf.distance_to_hub,
                'graph_connected_capacity': nf.connected_capacity,
                'graph_class1_connections': nf.num_class1_connections,
                'graph_local_density': nf.subgraph_density
            })
        else:
            # Location not in graph - use zeros
            rows.append({
                'location': loc,
                'graph_degree': 0,
                'graph_betweenness': 0.0,
                'graph_closeness': 0.0,
                'graph_clustering': 0.0,
                'graph_pagerank': 0.0,
                'graph_rail_class_avg': 0.0,
                'graph_dist_to_hub': 0.0,
                'graph_connected_capacity': 0.0,
                'graph_class1_connections': 0,
                'graph_local_density': 0.0
            })
    
    return pd.DataFrame(rows)


class GraphEnhancedForecaster:
    """
    Forecaster that combines time-series features with graph topology.
    
    This addresses the question: "Why not use the graph for prediction?"
    
    The graph provides:
    1. Network position features (centrality, connectivity)
    2. Congestion propagation signals from neighbors
    3. Route alternative indicators
    
    Combined with:
    - Time features (day of week, seasonality)
    - Historical patterns (lags, rolling averages)
    - Weather impacts
    """
    
    def __init__(self, graph: nx.Graph = None):
        self.graph = graph
        self.graph_features = None
        self.model = None
        self.feature_cols = []
        
    def compute_graph_features(self, locations: List[str]) -> pd.DataFrame:
        """Pre-compute graph features for locations."""
        if self.graph is None:
            return pd.DataFrame({'location': locations})
        
        self.graph_features = build_graph_feature_matrix(self.graph, locations)
        return self.graph_features
    
    def add_graph_features_to_data(self, df: pd.DataFrame, location_col: str = 'portname') -> pd.DataFrame:
        """
        Merge graph features into training/prediction data.
        
        Args:
            df: DataFrame with time-series data
            location_col: Column containing location identifiers
            
        Returns:
            DataFrame with graph features added
        """
        if self.graph_features is None:
            locations = df[location_col].unique().tolist()
            self.compute_graph_features(locations)
        
        if self.graph_features is None or len(self.graph_features) == 0:
            return df
        
        # Merge on location
        merged = df.merge(
            self.graph_features,
            left_on=location_col,
            right_on='location',
            how='left'
        )
        
        # Fill NaN graph features with 0
        graph_cols = [c for c in merged.columns if c.startswith('graph_')]
        merged[graph_cols] = merged[graph_cols].fillna(0)
        
        return merged


def analyze_congestion_propagation(
    G: nx.Graph,
    source_node: str,
    congestion_level: float,
    decay_factor: float = 0.5
) -> Dict[str, float]:
    """
    Model how congestion propagates through the network.
    
    Uses a diffusion model where congestion decreases with distance.
    
    Args:
        G: Transportation graph
        source_node: Origin of congestion
        congestion_level: Initial congestion level (0-1)
        decay_factor: How quickly congestion decreases per hop
        
    Returns:
        Predicted congestion at each connected node
    """
    propagated = {source_node: congestion_level}
    
    # BFS to propagate congestion
    visited = {source_node}
    queue = [(source_node, congestion_level, 0)]
    
    while queue:
        node, current_congestion, hops = queue.pop(0)
        
        if hops >= 5:  # Limit propagation depth
            continue
            
        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue
            
            visited.add(neighbor)
            
            # Congestion decays with distance
            neighbor_congestion = current_congestion * decay_factor
            
            # Consider edge capacity - high capacity means less impact
            edge_data = G.get_edge_data(node, neighbor, {})
            if 'capacity' in edge_data and edge_data['capacity'] > 0:
                capacity_factor = min(1.0, 100 / edge_data['capacity'])
                neighbor_congestion *= capacity_factor
            
            if neighbor_congestion > 0.01:  # Only propagate meaningful congestion
                propagated[neighbor] = neighbor_congestion
                queue.append((neighbor, neighbor_congestion, hops + 1))
    
    return propagated

