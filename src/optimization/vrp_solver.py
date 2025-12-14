"""
Vehicle Routing Problem (VRP) Solver
=====================================
Optimizes multi-vehicle routing for Glid fleet.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx

# Try to import OR-Tools
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GLID_VEHICLE, OPTIMIZATION


@dataclass 
class VRPSolution:
    """Solution to the VRP."""
    routes: List[List[int]]  # List of routes (node indices)
    total_distance: float
    total_time: float
    vehicles_used: int
    objective_value: float
    is_optimal: bool


class VRPSolver:
    """
    Solver for the Vehicle Routing Problem.
    
    Optimizes routes for multiple Glid vehicles to:
    - Visit all pickup/delivery locations
    - Minimize total distance/time
    - Respect vehicle capacity constraints
    - Maximize backhaul utilization
    """
    
    def __init__(
        self,
        num_vehicles: int = 5,
        vehicle_capacity: float = 40.0,
        max_route_distance: float = 100.0  # Round trip
    ):
        """
        Initialize VRP solver.
        
        Args:
            num_vehicles: Number of Glid vehicles available
            vehicle_capacity: Capacity per vehicle (tons)
            max_route_distance: Maximum route distance (miles)
        """
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.max_route_distance = max_route_distance
    
    def solve(
        self,
        distance_matrix: np.ndarray,
        demands: List[float],
        depot_index: int = 0,
        time_limit_seconds: int = 30
    ) -> VRPSolution:
        """
        Solve the VRP given distance matrix and demands.
        
        Args:
            distance_matrix: NxN matrix of distances between nodes
            demands: Demand at each node (tons)
            depot_index: Index of the depot node
            time_limit_seconds: Solver time limit
            
        Returns:
            VRPSolution object
        """
        if not HAS_ORTOOLS:
            return self._solve_greedy(distance_matrix, demands, depot_index)
        
        num_nodes = len(distance_matrix)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            num_nodes,
            self.num_vehicles,
            depot_index
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 100)  # Scale to int
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(demands[from_node] * 10)  # Scale to int
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [int(self.vehicle_capacity * 10)] * self.num_vehicles,
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Distance constraint per vehicle
        routing.AddDimension(
            transit_callback_index,
            0,
            int(self.max_route_distance * 100),
            True,
            'Distance'
        )
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit_seconds
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._extract_solution(manager, routing, solution)
        else:
            return self._solve_greedy(distance_matrix, demands, depot_index)
    
    def _extract_solution(self, manager, routing, solution) -> VRPSolution:
        """Extract solution from OR-Tools solver."""
        routes = []
        total_distance = 0
        vehicles_used = 0
        
        for vehicle_id in range(self.num_vehicles):
            route = []
            index = routing.Start(vehicle_id)
            route_distance = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            
            route.append(manager.IndexToNode(index))  # Add end depot
            
            if len(route) > 2:  # Non-empty route
                routes.append(route)
                total_distance += route_distance / 100  # Unscale
                vehicles_used += 1
        
        return VRPSolution(
            routes=routes,
            total_distance=total_distance,
            total_time=total_distance / GLID_VEHICLE.avg_speed_mph,
            vehicles_used=vehicles_used,
            objective_value=solution.ObjectiveValue() / 100,
            is_optimal=True
        )
    
    def _solve_greedy(
        self,
        distance_matrix: np.ndarray,
        demands: List[float],
        depot_index: int
    ) -> VRPSolution:
        """
        Greedy fallback solver when OR-Tools not available.
        """
        num_nodes = len(distance_matrix)
        unvisited = set(range(num_nodes)) - {depot_index}
        routes = []
        total_distance = 0
        
        for vehicle in range(self.num_vehicles):
            if not unvisited:
                break
            
            route = [depot_index]
            current = depot_index
            route_load = 0
            route_distance = 0
            
            while unvisited:
                # Find nearest unvisited node
                nearest = None
                nearest_dist = float('inf')
                
                for node in unvisited:
                    dist = distance_matrix[current][node]
                    if dist < nearest_dist:
                        # Check capacity
                        if route_load + demands[node] <= self.vehicle_capacity:
                            # Check distance constraint
                            potential_distance = route_distance + dist + distance_matrix[node][depot_index]
                            if potential_distance <= self.max_route_distance:
                                nearest = node
                                nearest_dist = dist
                
                if nearest is None:
                    break
                
                route.append(nearest)
                unvisited.remove(nearest)
                route_distance += nearest_dist
                route_load += demands[nearest]
                current = nearest
            
            # Return to depot
            route.append(depot_index)
            route_distance += distance_matrix[current][depot_index]
            total_distance += route_distance
            
            if len(route) > 2:
                routes.append(route)
        
        return VRPSolution(
            routes=routes,
            total_distance=total_distance,
            total_time=total_distance / GLID_VEHICLE.avg_speed_mph,
            vehicles_used=len(routes),
            objective_value=total_distance,
            is_optimal=False
        )
    
    def solve_from_graph(
        self,
        G: nx.Graph,
        pickup_nodes: List[Any],
        depot_node: Any,
        demands: Dict[Any, float] = None
    ) -> VRPSolution:
        """
        Solve VRP using NetworkX graph.
        
        Args:
            G: NetworkX graph
            pickup_nodes: List of nodes to visit
            depot_node: Depot node
            demands: Optional demand dictionary
            
        Returns:
            VRPSolution
        """
        # Build distance matrix from graph
        all_nodes = [depot_node] + list(pickup_nodes)
        n = len(all_nodes)
        
        distance_matrix = np.zeros((n, n))
        for i, node_i in enumerate(all_nodes):
            for j, node_j in enumerate(all_nodes):
                if i != j:
                    try:
                        dist = nx.shortest_path_length(G, node_i, node_j, weight='distance_miles')
                    except nx.NetworkXNoPath:
                        dist = float('inf')
                    distance_matrix[i][j] = dist
        
        # Build demands list
        demand_list = [0]  # Depot has 0 demand
        for node in pickup_nodes:
            if demands and node in demands:
                demand_list.append(demands[node])
            else:
                demand_list.append(1)  # Default demand
        
        return self.solve(distance_matrix, demand_list)


if __name__ == "__main__":
    # Test VRP solver
    solver = VRPSolver(num_vehicles=3)
    
    # Create test distance matrix (5 nodes including depot)
    distance_matrix = np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 12, 18, 22],
        [15, 12, 0, 8, 14],
        [20, 18, 8, 0, 10],
        [25, 22, 14, 10, 0]
    ])
    
    demands = [0, 10, 15, 8, 12]  # Depot has 0 demand
    
    solution = solver.solve(distance_matrix, demands)
    
    print(f"VRP Solution:")
    print(f"  Vehicles used: {solution.vehicles_used}")
    print(f"  Total distance: {solution.total_distance:.1f} miles")
    print(f"  Total time: {solution.total_time:.1f} hours")
    print(f"  Is optimal: {solution.is_optimal}")
    print(f"  Routes:")
    for i, route in enumerate(solution.routes):
        print(f"    Vehicle {i+1}: {' -> '.join(map(str, route))}")







