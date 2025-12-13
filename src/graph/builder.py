"""
Graph Builder for Multi-Modal Transportation Network
=====================================================
Constructs NetworkX graphs from rail and road data.
"""

import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Point
from geopy.distance import geodesic
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    GLID_CLIENTS, GLID_VEHICLE, RAIL_CLASS_CONSTRAINTS,
    CLASS_1_RAILROADS, US_PORTS, RAIL_TERMINALS
)


def build_rail_graph(
    nodes_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame,
    add_constraints: bool = True
) -> nx.Graph:
    """
    Build a NetworkX graph from rail nodes and lines.
    
    Args:
        nodes_gdf: GeoDataFrame of rail nodes
        lines_gdf: GeoDataFrame of rail line segments
        add_constraints: Whether to add rail class constraints
        
    Returns:
        NetworkX Graph with nodes and edges
    """
    print("Building rail network graph...")
    G = nx.Graph()
    
    # Add nodes
    print("  Adding nodes...")
    for idx, row in tqdm(nodes_gdf.iterrows(), total=len(nodes_gdf), desc="  Nodes"):
        node_id = row['FRANODEID']
        G.add_node(
            node_id,
            lat=row.geometry.y if row.geometry else None,
            lon=row.geometry.x if row.geometry else None,
            state=row.get('STATE'),
            county_fips=row.get('STCYFIPS'),
            node_type='rail_node',
            passenger_station=row.get('PASSNGRSTN'),
        )
    
    # Add edges (rail segments)
    print("  Adding edges...")
    for idx, row in tqdm(lines_gdf.iterrows(), total=len(lines_gdf), desc="  Edges"):
        from_node = row['FRFRANODE']
        to_node = row['TOFRANODE']
        
        # Skip if nodes don't exist
        if from_node not in G.nodes or to_node not in G.nodes:
            continue
        
        # Calculate edge weight (distance in miles)
        distance_miles = row.get('MILES', 1.0)
        
        # Determine rail class and get constraints
        owner = row.get('RROWNER1', '')
        is_class1 = owner in CLASS_1_RAILROADS
        
        # Default to Class 3 constraints, Class 4 for Class I railroads
        rail_class = 4 if is_class1 else 3
        constraints = RAIL_CLASS_CONSTRAINTS.get(rail_class, RAIL_CLASS_CONSTRAINTS[3])
        
        # Calculate travel time based on speed limit
        travel_time_hours = distance_miles / constraints['max_speed_mph']
        
        edge_attrs = {
            'distance_miles': distance_miles,
            'travel_time_hours': travel_time_hours,
            'owner': owner,
            'is_class1': is_class1,
            'rail_class': rail_class,
            'max_speed_mph': constraints['max_speed_mph'],
            'min_headway_min': constraints['min_headway_minutes'],
            'edge_type': 'rail',
            'subdivision': row.get('SUBDIV'),
            'tracks': row.get('TRACKS', 1),
        }
        
        G.add_edge(from_node, to_node, **edge_attrs)
    
    print(f"  Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def add_location_nodes(
    G: nx.Graph,
    locations: Dict[str, Dict],
    node_type: str = 'location'
) -> nx.Graph:
    """
    Add custom location nodes (ports, clients, terminals) to graph.
    
    Args:
        G: Existing NetworkX graph
        locations: Dictionary of locations with lat/lon
        node_type: Type label for the nodes
        
    Returns:
        Updated graph with new nodes
    """
    for loc_id, loc_data in locations.items():
        node_id = f"{node_type}_{loc_id}"
        G.add_node(
            node_id,
            lat=loc_data['lat'],
            lon=loc_data['lon'],
            name=loc_data.get('name', loc_id),
            node_type=node_type,
            **{k: v for k, v in loc_data.items() if k not in ['lat', 'lon', 'name']}
        )
    
    return G


def connect_locations_to_graph(
    G: nx.Graph,
    locations: Dict[str, Dict],
    node_type: str = 'location',
    max_connection_miles: float = 10.0,
    avg_road_speed_mph: float = 30.0
) -> nx.Graph:
    """
    Connect location nodes to nearest rail nodes via road edges.
    
    Args:
        G: Graph with rail nodes and location nodes
        locations: Dictionary of locations
        node_type: Type of location nodes
        max_connection_miles: Maximum distance to connect
        avg_road_speed_mph: Average road speed for travel time
        
    Returns:
        Graph with road connections added
    """
    print(f"Connecting {node_type} nodes to rail network...")
    
    # Get all rail nodes with coordinates
    rail_nodes = [
        (n, d) for n, d in G.nodes(data=True)
        if d.get('node_type') == 'rail_node' and d.get('lat') and d.get('lon')
    ]
    
    for loc_id, loc_data in tqdm(locations.items(), desc=f"  {node_type}"):
        loc_node_id = f"{node_type}_{loc_id}"
        loc_coords = (loc_data['lat'], loc_data['lon'])
        
        # Find nearest rail nodes
        nearest_nodes = []
        for rail_node_id, rail_data in rail_nodes:
            rail_coords = (rail_data['lat'], rail_data['lon'])
            distance = geodesic(loc_coords, rail_coords).miles
            
            if distance <= max_connection_miles:
                nearest_nodes.append((rail_node_id, distance))
        
        # Connect to nearest nodes (up to 3)
        nearest_nodes.sort(key=lambda x: x[1])
        for rail_node_id, distance in nearest_nodes[:3]:
            travel_time_hours = distance / avg_road_speed_mph
            G.add_edge(
                loc_node_id,
                rail_node_id,
                distance_miles=distance,
                travel_time_hours=travel_time_hours,
                edge_type='road',
                connection_type=f'{node_type}_to_rail'
            )
    
    return G


def extract_subgraph_radius(
    G: nx.Graph,
    center_lat: float,
    center_lon: float,
    radius_miles: float = 50.0
) -> nx.Graph:
    """
    Extract subgraph within a radius of a center point.
    
    Args:
        G: Full graph
        center_lat: Center latitude
        center_lon: Center longitude
        radius_miles: Radius in miles
        
    Returns:
        Subgraph containing only nodes within radius
    """
    center_coords = (center_lat, center_lon)
    
    nodes_in_radius = []
    for node, data in G.nodes(data=True):
        if data.get('lat') and data.get('lon'):
            node_coords = (data['lat'], data['lon'])
            distance = geodesic(center_coords, node_coords).miles
            if distance <= radius_miles:
                nodes_in_radius.append(node)
    
    subgraph = G.subgraph(nodes_in_radius).copy()
    return subgraph


def extract_client_subgraphs(
    G: nx.Graph,
    clients: Dict[str, Dict] = None,
    radius_miles: float = 50.0
) -> Dict[str, nx.Graph]:
    """
    Extract subgraphs for each Glid client location.
    
    Args:
        G: Full graph
        clients: Dictionary of client locations (default: GLID_CLIENTS)
        radius_miles: Radius for each subgraph
        
    Returns:
        Dictionary mapping client IDs to subgraphs
    """
    if clients is None:
        clients = GLID_CLIENTS
    
    subgraphs = {}
    for client_id, client_data in clients.items():
        print(f"Extracting subgraph for {client_data['name']}...")
        subgraphs[client_id] = extract_subgraph_radius(
            G,
            center_lat=client_data['lat'],
            center_lon=client_data['lon'],
            radius_miles=radius_miles
        )
        print(f"  Subgraph: {subgraphs[client_id].number_of_nodes()} nodes, "
              f"{subgraphs[client_id].number_of_edges()} edges")
    
    return subgraphs


def build_multimodal_graph(
    rail_nodes: gpd.GeoDataFrame,
    rail_lines: gpd.GeoDataFrame,
    include_clients: bool = True,
    include_ports: bool = True,
    include_terminals: bool = True
) -> nx.Graph:
    """
    Build complete multi-modal transportation graph.
    
    Args:
        rail_nodes: Rail node GeoDataFrame
        rail_lines: Rail line GeoDataFrame
        include_clients: Add Glid client locations
        include_ports: Add US port locations
        include_terminals: Add rail terminal locations
        
    Returns:
        Complete multi-modal NetworkX graph
    """
    # Build base rail graph
    G = build_rail_graph(rail_nodes, rail_lines)
    
    # Add location nodes
    if include_clients:
        G = add_location_nodes(G, GLID_CLIENTS, 'client')
        G = connect_locations_to_graph(G, GLID_CLIENTS, 'client')
    
    if include_ports:
        ports_dict = {f"port_{i}": p for i, p in enumerate(US_PORTS)}
        G = add_location_nodes(G, ports_dict, 'port')
        G = connect_locations_to_graph(G, ports_dict, 'port')
    
    if include_terminals:
        terminals_dict = {f"terminal_{i}": t for i, t in enumerate(RAIL_TERMINALS)}
        G = add_location_nodes(G, terminals_dict, 'terminal')
        G = connect_locations_to_graph(G, terminals_dict, 'terminal')
    
    print(f"\nFinal multi-modal graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


if __name__ == "__main__":
    # Test graph building
    from data.loaders import load_rail_nodes, load_rail_lines
    
    nodes = load_rail_nodes()
    lines = load_rail_lines()
    
    G = build_multimodal_graph(nodes, lines)
    
    # Test subgraph extraction for Woodland
    woodland = GLID_CLIENTS['port_of_woodland']
    subgraph = extract_subgraph_radius(G, woodland['lat'], woodland['lon'], 50)
    print(f"\nWoodland 50-mile subgraph: {subgraph.number_of_nodes()} nodes")

