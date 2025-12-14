"""
Routing and Path Finding for Glid Vehicles
===========================================
Implements route optimization algorithms.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from geopy.distance import geodesic

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GLID_VEHICLE, RAIL_CLASS_CONSTRAINTS


@dataclass
class RouteMetrics:
    """Metrics for an evaluated route."""
    path: List[Any]
    total_distance_miles: float
    total_time_hours: float
    rail_distance_miles: float
    road_distance_miles: float
    num_segments: int
    is_valid: bool
    violations: List[str]


def find_shortest_path(
    G: nx.Graph,
    source: Any,
    target: Any,
    weight: str = 'distance_miles'
) -> Tuple[List[Any], float]:
    """
    Find shortest path between two nodes.
    
    Args:
        G: NetworkX graph
        source: Source node ID
        target: Target node ID
        weight: Edge attribute to use as weight
        
    Returns:
        Tuple of (path as node list, total weight)
    """
    try:
        path = nx.shortest_path(G, source, target, weight=weight)
        path_length = nx.shortest_path_length(G, source, target, weight=weight)
        return path, path_length
    except nx.NetworkXNoPath:
        return [], float('inf')


def find_optimal_route(
    G: nx.Graph,
    source: Any,
    target: Any,
    optimize_for: str = 'time',
    max_distance: float = None,
    respect_constraints: bool = True
) -> RouteMetrics:
    """
    Find optimal route considering Glid vehicle constraints.
    
    Args:
        G: NetworkX graph
        source: Source node
        target: Target node
        optimize_for: 'time' or 'distance'
        max_distance: Maximum allowed distance (default: GLID_VEHICLE.max_route_miles)
        respect_constraints: Whether to apply rail class constraints
        
    Returns:
        RouteMetrics object with path and metrics
    """
    if max_distance is None:
        max_distance = GLID_VEHICLE.max_route_miles
    
    weight = 'travel_time_hours' if optimize_for == 'time' else 'distance_miles'
    
    # Find path
    path, total_weight = find_shortest_path(G, source, target, weight)
    
    if not path:
        return RouteMetrics(
            path=[],
            total_distance_miles=float('inf'),
            total_time_hours=float('inf'),
            rail_distance_miles=0,
            road_distance_miles=0,
            num_segments=0,
            is_valid=False,
            violations=["No path found"]
        )
    
    # Calculate metrics
    metrics = calculate_route_metrics(G, path)
    
    # Check constraints
    violations = []
    if metrics.total_distance_miles > max_distance:
        violations.append(f"Distance {metrics.total_distance_miles:.1f} exceeds max {max_distance}")
    
    if metrics.total_distance_miles < GLID_VEHICLE.min_route_miles:
        violations.append(f"Distance {metrics.total_distance_miles:.1f} below min {GLID_VEHICLE.min_route_miles}")
    
    metrics.violations = violations
    metrics.is_valid = len(violations) == 0
    
    return metrics


def calculate_route_metrics(G: nx.Graph, path: List[Any]) -> RouteMetrics:
    """
    Calculate detailed metrics for a path.
    
    Args:
        G: NetworkX graph
        path: List of node IDs in path order
        
    Returns:
        RouteMetrics object
    """
    if len(path) < 2:
        return RouteMetrics(
            path=path,
            total_distance_miles=0,
            total_time_hours=0,
            rail_distance_miles=0,
            road_distance_miles=0,
            num_segments=0,
            is_valid=False,
            violations=["Path too short"]
        )
    
    total_distance = 0
    total_time = 0
    rail_distance = 0
    road_distance = 0
    
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i+1], {})
        
        dist = edge_data.get('distance_miles', 0)
        time = edge_data.get('travel_time_hours', 0)
        edge_type = edge_data.get('edge_type', 'unknown')
        
        total_distance += dist
        total_time += time
        
        if edge_type == 'rail':
            rail_distance += dist
        else:
            road_distance += dist
    
    return RouteMetrics(
        path=path,
        total_distance_miles=total_distance,
        total_time_hours=total_time,
        rail_distance_miles=rail_distance,
        road_distance_miles=road_distance,
        num_segments=len(path) - 1,
        is_valid=True,
        violations=[]
    )


def find_round_trip_routes(
    G: nx.Graph,
    origin: Any,
    destinations: List[Any],
    min_backhaul_load: float = 0.6
) -> List[Dict]:
    """
    Find optimal round-trip routes that maximize backhaul.
    
    Args:
        G: NetworkX graph
        origin: Origin node (Glid vehicle home base)
        destinations: List of potential destination nodes
        min_backhaul_load: Minimum backhaul load factor
        
    Returns:
        List of route dictionaries with outbound and return legs
    """
    routes = []
    
    for dest in destinations:
        # Find outbound route
        outbound = find_optimal_route(G, origin, dest)
        if not outbound.is_valid:
            continue
        
        # Find return route (may be different)
        return_route = find_optimal_route(G, dest, origin)
        if not return_route.is_valid:
            continue
        
        # Calculate round-trip efficiency
        total_distance = outbound.total_distance_miles + return_route.total_distance_miles
        total_time = outbound.total_time_hours + return_route.total_time_hours
        
        routes.append({
            'destination': dest,
            'outbound': outbound,
            'return': return_route,
            'total_distance_miles': total_distance,
            'total_time_hours': total_time,
            'rail_percentage': (outbound.rail_distance_miles + return_route.rail_distance_miles) / total_distance * 100
        })
    
    # Sort by efficiency (time per mile)
    routes.sort(key=lambda x: x['total_time_hours'] / max(x['total_distance_miles'], 1))
    
    return routes


def validate_route_for_glid(
    route: RouteMetrics,
    check_rail_class: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate a route against all Glid vehicle constraints.
    
    Args:
        route: RouteMetrics to validate
        check_rail_class: Whether to check rail class restrictions
        
    Returns:
        Tuple of (is_valid, list of violation messages)
    """
    violations = list(route.violations)  # Start with existing violations
    
    # Distance constraints
    if route.total_distance_miles > GLID_VEHICLE.max_route_miles:
        violations.append(
            f"Route distance ({route.total_distance_miles:.1f} mi) exceeds "
            f"maximum ({GLID_VEHICLE.max_route_miles} mi)"
        )
    
    if route.total_distance_miles < GLID_VEHICLE.min_route_miles:
        violations.append(
            f"Route distance ({route.total_distance_miles:.1f} mi) below "
            f"minimum ({GLID_VEHICLE.min_route_miles} mi)"
        )
    
    # Time estimates with turnaround
    estimated_time = route.total_time_hours + GLID_VEHICLE.turnaround_time_hours
    
    return len(violations) == 0, violations


if __name__ == "__main__":
    # Test with a simple graph
    G = nx.Graph()
    G.add_edge('A', 'B', distance_miles=10, travel_time_hours=0.5, edge_type='road')
    G.add_edge('B', 'C', distance_miles=25, travel_time_hours=0.75, edge_type='rail')
    G.add_edge('C', 'D', distance_miles=15, travel_time_hours=0.4, edge_type='rail')
    
    route = find_optimal_route(G, 'A', 'D')
    print(f"Route A->D: {route.path}")
    print(f"Distance: {route.total_distance_miles} miles")
    print(f"Time: {route.total_time_hours} hours")
    print(f"Valid: {route.is_valid}, Violations: {route.violations}")







