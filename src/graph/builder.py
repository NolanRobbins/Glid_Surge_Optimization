"""
Graph Builder for Multi-Modal Transportation Network
=====================================================
Constructs NetworkX graphs from rail and road data.
Optimized for NVIDIA RAPIDS (cuML/cuDF) with CPU fallback.
"""

import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from geopy.distance import geodesic
from tqdm import tqdm
import sys
from pathlib import Path

# GPU Acceleration Imports
try:
    import cudf
    import cuml
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    HAS_GPU = True
    print("🚀 NVIDIA RAPIDS detected: GPU acceleration enabled for graph building")
except ImportError:
    HAS_GPU = False
    from scipy.spatial import cKDTree
    print("⚠ NVIDIA RAPIDS not found: Falling back to CPU (KDTree)")

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
    """
    print("Building rail network graph...")
    G = nx.Graph()
    
    # Add nodes
    print("  Adding nodes...")
    # Convert to pandas if it's a cudf DataFrame (NetworkX expects CPU objects)
    if 'cudf' in sys.modules and isinstance(nodes_gdf, cudf.DataFrame):
        nodes_iter = nodes_gdf.to_pandas().iterrows()
        total_nodes = len(nodes_gdf)
    else:
        nodes_iter = nodes_gdf.iterrows()
        total_nodes = len(nodes_gdf)

    for idx, row in tqdm(nodes_iter, total=total_nodes, desc="  Nodes"):
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
    
    # Add edges
    print("  Adding edges...")
    if 'cudf' in sys.modules and isinstance(lines_gdf, cudf.DataFrame):
        lines_iter = lines_gdf.to_pandas().iterrows()
        total_lines = len(lines_gdf)
    else:
        lines_iter = lines_gdf.iterrows()
        total_lines = len(lines_gdf)

    for idx, row in tqdm(lines_iter, total=total_lines, desc="  Edges"):
        from_node = row['FRFRANODE']
        to_node = row['TOFRANODE']
        
        if from_node not in G.nodes or to_node not in G.nodes:
            continue
        
        distance_miles = row.get('MILES', 1.0)
        owner = row.get('RROWNER1', '')
        is_class1 = owner in CLASS_1_RAILROADS
        rail_class = 4 if is_class1 else 3
        constraints = RAIL_CLASS_CONSTRAINTS.get(rail_class, RAIL_CLASS_CONSTRAINTS[3])
        travel_time_hours = distance_miles / constraints['max_speed_mph']

        # Preserve the actual rail segment geometry so downstream routing can
        # return a polyline that follows the rail line instead of drawing
        # straight chords between rail nodes.
        geometry_coords: Optional[List[List[float]]] = None
        geom: Optional[BaseGeometry] = getattr(row, "geometry", None)
        if geom is not None and not getattr(geom, "is_empty", True):
            try:
                if geom.geom_type == "LineString":
                    geometry_coords = [[float(x), float(y)] for (x, y) in geom.coords]
                elif geom.geom_type == "MultiLineString":
                    coords: List[List[float]] = []
                    for part in geom.geoms:
                        coords.extend([[float(x), float(y)] for (x, y) in part.coords])
                    geometry_coords = coords if coords else None
            except Exception:
                geometry_coords = None
        
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

        if geometry_coords:
            edge_attrs["geometry_coords"] = geometry_coords
        
        G.add_edge(from_node, to_node, **edge_attrs)
    
    print(f"  Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def add_location_nodes(
    G: nx.Graph,
    locations: Dict[str, Dict],
    node_type: str = 'location'
) -> nx.Graph:
    """Add custom location nodes (ports, clients, terminals) to graph."""
    for loc_id, loc_data in locations.items():
        node_id = f"{node_type}_{loc_id}"
        # Store original name if available to avoid duplicates if ID changes
        name = loc_data.get('name', loc_id)
        
        # Check if node already exists by name (handling potential re-runs)
        existing_node = None
        # Optimization: Don't iterate all nodes if we don't have to
        # But here we rely on the caller to manage IDs correctly
        
        G.add_node(
            node_id,
            lat=loc_data['lat'],
            lon=loc_data['lon'],
            name=name,
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
    Automatically chooses GPU (cuML) or CPU (KDTree) backend.
    """
    
    # 1. Extract Rail Nodes
    rail_nodes = []
    rail_coords = []
    
    for n, d in G.nodes(data=True):
        if d.get('node_type') == 'rail_node' and d.get('lat') is not None and d.get('lon') is not None:
            rail_nodes.append(n)
            rail_coords.append([d['lat'], d['lon']])
            
    if not rail_nodes:
        print("  ⚠ No rail nodes found with coordinates!")
        return G
    
    rail_coords_np = np.array(rail_coords, dtype=np.float32)
    
    # 2. Extract Location Nodes
    loc_ids = []
    loc_coords = []
    loc_node_ids = []
    
    for loc_id, loc_data in locations.items():
        loc_node_id = f"{node_type}_{loc_id}"
        
        # Verify node exists in graph (handle name matching if needed)
        if loc_node_id not in G:
             # Try matching by name attribute if ID doesn't match directly
            found = False
            target_name = loc_data.get('name')
            if target_name:
                for n, d in G.nodes(data=True):
                    if d.get('node_type') == node_type and d.get('name') == target_name:
                        loc_node_id = n
                        found = True
                        break
            if not found:
                continue
                
        loc_ids.append(loc_id)
        loc_node_ids.append(loc_node_id)
        loc_coords.append([loc_data['lat'], loc_data['lon']])
        
    if not loc_coords:
        return G
        
    loc_coords_np = np.array(loc_coords, dtype=np.float32)
    
    print(f"Connecting {len(loc_coords)} {node_type}s to {len(rail_nodes)} rail nodes...")
    
    # 3. Nearest Neighbor Search
    k_neighbors = 20
    connections_made = 0
    
    if HAS_GPU:
        print("  ⚡ Using GPU (cuML) for spatial search...")
        # cuML expects float32
        import cudf
        import cuml
        
        # Create cuML NearestNeighbors model
        knn = cuNearestNeighbors(n_neighbors=k_neighbors)
        knn.fit(rail_coords_np)
        
        # Query
        distances, indices = knn.kneighbors(loc_coords_np)
        
        # Convert back to numpy for iteration (distances are Euclidean in lat/lon space, need verification)
        # Note: cuML returns squared L2 distance by default for some metrics, but standard is Euclidean
        indices = indices.get() if hasattr(indices, 'get') else indices
        
    else:
        print("  🐢 Using CPU (KDTree) for spatial search...")
        tree = cKDTree(rail_coords_np)
        # Approximate degree-to-miles conversion for search radius
        # 1 deg ≈ 69 miles. max_connection_miles / 50 is a safe upper bound in degrees
        search_radius = max_connection_miles / 50.0 
        distances, indices = tree.query(loc_coords_np, k=k_neighbors, distance_upper_bound=search_radius)
        
    # 4. Create Edges (verify with Geodesic)
    for i, loc_node_id in enumerate(loc_node_ids):
        loc_lat, loc_lon = loc_coords[i]
        
        valid_connections = []
        
        for j in range(k_neighbors):
            idx = indices[i][j]
            
            # Handle KDTree infinite index (no neighbor found within bound)
            if idx == len(rail_nodes) or idx == -1:
                continue
                
            rail_node_id = rail_nodes[idx]
            rail_lat, rail_lon = rail_coords[idx]
            
            # Precise calculation using geodesic (miles)
            dist_miles = geodesic((loc_lat, loc_lon), (rail_lat, rail_lon)).miles
            
            if dist_miles <= max_connection_miles:
                valid_connections.append((rail_node_id, dist_miles))
        
        # Sort by actual miles and take top 3
        valid_connections.sort(key=lambda x: x[1])
        
        for rail_node_id, dist_miles in valid_connections[:3]:
            travel_time_hours = dist_miles / avg_road_speed_mph
            G.add_edge(
                loc_node_id,
                rail_node_id,
                distance_miles=dist_miles,
                travel_time_hours=travel_time_hours,
                edge_type='road',
                connection_type=f'{node_type}_to_rail'
            )
            connections_made += 1
            
    print(f"  ✓ Created {connections_made} connections")
    return G


def extract_subgraph_radius(
    G: nx.Graph,
    center_lat: float,
    center_lon: float,
    radius_miles: float = 50.0
) -> nx.Graph:
    """Extract subgraph within a radius using crude lat/lon filter first."""
    center_coords = (center_lat, center_lon)
    
    # 1 deg lat approx 69 miles
    deg_radius = radius_miles / 60.0
    
    nodes_in_radius = []
    for node, data in G.nodes(data=True):
        lat = data.get('lat')
        lon = data.get('lon')
        if lat and lon:
            # Fast box filter
            if abs(lat - center_lat) < deg_radius and abs(lon - center_lon) < deg_radius:
                # Precise check
                distance = geodesic(center_coords, (lat, lon)).miles
                if distance <= radius_miles:
                    nodes_in_radius.append(node)
    
    subgraph = G.subgraph(nodes_in_radius).copy()
    return subgraph


def build_multimodal_graph(
    rail_nodes: gpd.GeoDataFrame,
    rail_lines: gpd.GeoDataFrame,
    include_clients: bool = True,
    include_ports: bool = True,
    include_terminals: bool = True
) -> nx.Graph:
    """Build complete multi-modal transportation graph."""
    
    # Build base rail graph
    G = build_rail_graph(rail_nodes, rail_lines)
    
    # Add location nodes
    if include_clients:
        G = add_location_nodes(G, GLID_CLIENTS, 'client')
        G = connect_locations_to_graph(G, GLID_CLIENTS, 'client')
    
    if include_ports:
        # Convert list of dicts to dict of dicts if needed, or assume US_PORTS is correct format
        # US_PORTS is list of dicts: [{'name':..., 'lat':..., 'lon':...}]
        # Need to create ID-based dict
        ports_dict = {p['name']: p for p in US_PORTS}
        G = add_location_nodes(G, ports_dict, 'port')
        G = connect_locations_to_graph(G, ports_dict, 'port')
    
    if include_terminals:
        terminals_dict = {t['name']: t for t in RAIL_TERMINALS}
        G = add_location_nodes(G, terminals_dict, 'terminal')
        G = connect_locations_to_graph(G, terminals_dict, 'terminal')
    
    print(f"\nFinal multi-modal graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G
