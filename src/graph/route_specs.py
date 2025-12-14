"""
Route Spec Loader (GeoJSON Feature)
==================================
Loads small route examples (e.g. Long Beach -> Fleet Yards) used for demos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RouteSpec:
    name: str
    origin_name: str
    destination_name: str
    distance: float
    distance_unit: str
    estimated_time: float
    estimated_time_unit: str
    # (lat, lon)
    origin_latlon: Tuple[float, float]
    destination_latlon: Tuple[float, float]
    # raw line coordinates as (lon, lat)
    line: List[Tuple[float, float]]
    properties: Dict[str, Any]


def load_route_spec(path: str | Path) -> RouteSpec:
    p = Path(path)
    obj = json.loads(p.read_text())

    if obj.get("type") != "Feature":
        raise ValueError(f"Expected GeoJSON Feature, got type={obj.get('type')}")

    props = obj.get("properties") or {}
    geom = obj.get("geometry") or {}
    if geom.get("type") != "LineString":
        raise ValueError(f"Expected LineString geometry, got {geom.get('type')}")

    coords = geom.get("coordinates") or []
    if len(coords) < 2:
        raise ValueError("Route LineString must have at least 2 coordinates")

    # coords are [lon, lat]
    start_lon, start_lat = coords[0]
    end_lon, end_lat = coords[-1]

    return RouteSpec(
        name=str(props.get("name", p.stem)),
        origin_name=str(props.get("origin", "origin")),
        destination_name=str(props.get("destination", "destination")),
        distance=float(props.get("distance", 0.0)),
        distance_unit=str(props.get("distanceUnit", "")),
        estimated_time=float(props.get("estimatedTime", 0.0)),
        estimated_time_unit=str(props.get("estimatedTimeUnit", "")),
        origin_latlon=(float(start_lat), float(start_lon)),
        destination_latlon=(float(end_lat), float(end_lon)),
        line=[(float(lon), float(lat)) for lon, lat in coords],
        properties=props,
    )



