"""
Route adapter that bridges the existing graph builder/routing utilities to
API-friendly helpers. Builds a rail-first graph from local data once, then
computes routes between arbitrary lon/lat points and returns GeoJSON-friendly
results.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from geopy.distance import geodesic

from src.data.loaders import load_rail_lines, load_rail_nodes
from src.graph.builder import build_rail_graph
from src.graph.routing import RouteMetrics, find_optimal_route


class RouteAdapter:
    """
    Lightweight wrapper that loads local rail data, builds a NetworkX graph, and
    exposes a method to compute optimized routes between arbitrary lon/lat
    coordinates.
    """

    def __init__(
        self,
        max_connection_miles: float = 15.0,
        road_speed_mph: float = 30.0,
        filter_us_only: bool = True,
    ) -> None:
        self.max_connection_miles = max_connection_miles
        self.road_speed_mph = road_speed_mph
        self.filter_us_only = filter_us_only

        self.nodes_gdf = None
        self.lines_gdf = None
        self.graph: Optional[nx.Graph] = None

    # ------------------------------------------------------------------ loaders
    def build_graph(self) -> None:
        """Load local rail data and construct the base graph."""
        self.nodes_gdf = load_rail_nodes(filter_us_only=self.filter_us_only)
        self.lines_gdf = load_rail_lines(filter_us_only=self.filter_us_only)
        self.graph = build_rail_graph(self.nodes_gdf, self.lines_gdf)

    def is_ready(self) -> bool:
        return self.graph is not None

    # ----------------------------------------------------------------- helpers
    def _nearest_rail_nodes(
        self, lat: float, lon: float, k: int = 3
    ) -> List[Tuple[Any, float]]:
        """
        Find up to k nearest rail nodes to a coordinate.
        Falls back to the closest nodes even if outside max_connection_miles.
        """
        if not self.graph:
            return []

        candidates: List[Tuple[Any, float]] = []
        for node_id, data in self.graph.nodes(data=True):
            node_lat = data.get("lat")
            node_lon = data.get("lon")
            if node_lat is None or node_lon is None:
                continue
            distance = geodesic((lat, lon), (node_lat, node_lon)).miles
            candidates.append((node_id, distance))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[1])
        within = [c for c in candidates if c[1] <= self.max_connection_miles]
        selected = within[:k] if within else candidates[:k]
        return selected

    def _attach_point(
        self, G: nx.Graph, point_id: str, lat: float, lon: float
    ) -> List[Tuple[Any, float]]:
        """Attach an external point to nearby rail nodes using road edges."""
        G.add_node(point_id, lat=lat, lon=lon, node_type="external")
        nearest = self._nearest_rail_nodes(lat, lon)

        for node_id, distance in nearest:
            travel_time_hours = distance / max(self.road_speed_mph, 1e-3)
            G.add_edge(
                point_id,
                node_id,
                distance_miles=distance,
                travel_time_hours=travel_time_hours,
                edge_type="road",
                connection_type="external",
            )

        return nearest

    @staticmethod
    def _path_to_coordinates(G: nx.Graph, path: List[Any]) -> List[List[float]]:
        """Convert a node path to a list of [lon, lat] coordinates."""
        coords: List[List[float]] = []
        for node_id in path:
            data = G.nodes.get(node_id, {})
            lat = data.get("lat")
            lon = data.get("lon")
            if lat is None or lon is None:
                continue
            coords.append([lon, lat])
        return coords

    # ------------------------------------------------------------- public API
    def compute_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        optimize_for: str = "time",
    ) -> Dict[str, Any]:
        """
        Compute an optimized route between origin/destination lon/lat pairs.

        Args:
            origin: (lon, lat)
            destination: (lon, lat)
            optimize_for: 'time' or 'distance'
        """
        if not self.graph:
            raise ValueError("Graph not initialized. Call build_graph() first.")

        origin_lon, origin_lat = origin
        dest_lon, dest_lat = destination

        # Work on a copy to avoid mutating the cached base graph
        G = self.graph.copy()

        origin_links = self._attach_point(G, "origin", origin_lat, origin_lon)
        dest_links = self._attach_point(G, "destination", dest_lat, dest_lon)

        if not origin_links or not dest_links:
            raise ValueError("Could not attach origin/destination to the rail network.")

        metrics: RouteMetrics = find_optimal_route(
            G, source="origin", target="destination", optimize_for=optimize_for
        )

        coords = self._path_to_coordinates(G, metrics.path)
        feature = {
            "type": "Feature",
            "properties": {
                "optimize_for": optimize_for,
                "total_distance_miles": metrics.total_distance_miles,
                "total_time_hours": metrics.total_time_hours,
                "rail_distance_miles": metrics.rail_distance_miles,
                "road_distance_miles": metrics.road_distance_miles,
                "num_segments": metrics.num_segments,
                "is_valid": metrics.is_valid,
                "violations": metrics.violations,
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        }

        return {
            "route": feature,
            "metrics": asdict(metrics),
            "path_nodes": metrics.path,
            "origin_links": origin_links,
            "destination_links": dest_links,
        }

    def rail_lines_geojson(self, max_features: Optional[int] = 2000) -> Dict[str, Any]:
        """Return rail lines as GeoJSON (optionally truncated)."""
        if self.lines_gdf is None:
            return {"type": "FeatureCollection", "features": []}

        gdf = self.lines_gdf
        if max_features:
            gdf = gdf.head(max_features)
        return json.loads(gdf.to_json())

    def rail_nodes_geojson(self, max_features: Optional[int] = 2000) -> Dict[str, Any]:
        """Return rail nodes as GeoJSON (optionally truncated)."""
        if self.nodes_gdf is None:
            return {"type": "FeatureCollection", "features": []}

        gdf = self.nodes_gdf
        if max_features:
            gdf = gdf.head(max_features)
        return json.loads(gdf.to_json())

