"""
Glid Surge Optimization - Configuration
========================================
Central configuration for paths, constants, and parameters.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Data subdirectories
RAIL_NODES_PATH = DATA_DIR / "rail_nodes" / "NTAD_North_American_Rail_Network_Nodes_-3731209159413770440.geojson"
RAIL_LINES_PATH = DATA_DIR / "rail_lines" / "NTAD_North_American_Rail_Network_Lines_-536904749999911422.geojson"
TRUCK_TIMES_PATH = DATA_DIR / "truck_times" / "BTS_ATRI_County_to_County_Truck_Travel_Times.csv"
AIS_VESSEL_PATH = DATA_DIR / "AIS_vessel" / "tracking_db.csv"
PORT_ACTIVITY_PATH = DATA_DIR / "global_daily_port_activity" / "Daily_Port_Activity_Data_and_Trade_Estimates.csv"
WEATHER_DAILY_PATH = DATA_DIR / "weather" / "weather_daily_20251212.csv"
WEATHER_HOURLY_PATH = DATA_DIR / "weather" / "weather_hourly_20251212.csv"
PORTWATCH_CHOKEPOINTS_PATH = DATA_DIR / "portwatch" / "Daily_Chokepoint_Transit_Calls_and_Trade_Volume_Estimates.geojson"
PORTWATCH_DISRUPTIONS_PATH = DATA_DIR / "portwatch" / "portwatch_disruptions_database_-4099867879455026010.geojson"
LOGISTICS_FLEET_DIR = DATA_DIR / "logistics_fleet"

# ============================================================================
# GLID VEHICLE CONSTRAINTS
# ============================================================================
@dataclass
class GlidVehicleConfig:
    """Configuration for Glid autonomous rail vehicles (Raden/Glider-M)"""
    min_route_miles: float = 40.0  # Minimum first-mile distance
    max_route_miles: float = 50.0  # Maximum first-mile distance
    max_load_tons: float = 40.0    # Container capacity
    avg_speed_mph: float = 25.0    # Average operating speed
    turnaround_time_hours: float = 0.5  # Time to load/unload

GLID_VEHICLE = GlidVehicleConfig()

# ============================================================================
# RAIL CLASS CONSTRAINTS (FRA Classification)
# ============================================================================
RAIL_CLASS_CONSTRAINTS = {
    1: {
        "max_speed_mph": 10,
        "min_headway_minutes": 60,
        "tonnage_limit": "excepted",
        "description": "Light density, often industrial spurs"
    },
    2: {
        "max_speed_mph": 25,
        "min_headway_minutes": 45,
        "tonnage_limit": "limited",
        "description": "Branch lines, moderate traffic"
    },
    3: {
        "max_speed_mph": 40,
        "min_headway_minutes": 30,
        "tonnage_limit": "moderate",
        "description": "Secondary mainlines"
    },
    4: {
        "max_speed_mph": 60,
        "min_headway_minutes": 20,
        "tonnage_limit": "high",
        "description": "Primary mainlines (Class I)"
    },
    5: {
        "max_speed_mph": 80,
        "min_headway_minutes": 15,
        "tonnage_limit": "very_high",
        "description": "High-speed freight corridors"
    }
}

# ============================================================================
# GLID CLIENT LOCATIONS
# ============================================================================
GLID_CLIENTS: Dict[str, Dict] = {
    "port_of_woodland": {
        "name": "Port of Woodland",
        "type": "port",
        "lat": 38.6785,
        "lon": -121.7733,
        "state": "CA",
        "region": "Northern California",
        "notes": "Sacramento River port, agricultural freight"
    },
    "sierra": {
        "name": "Sierra",
        "type": "industrial",
        "lat": 39.2396,  # Approximate - Sierra Nevada foothills
        "lon": -121.0619,
        "state": "CA",
        "region": "Northern California",
        "notes": "Mountain routing, weather-sensitive"
    },
    "newlab": {
        "name": "Newlab",
        "type": "tech_hub",
        "lat": 40.6892,  # Brooklyn Navy Yard
        "lon": -73.9707,
        "state": "NY",
        "region": "NYC Metro",
        "notes": "Urban first-mile, innovation hub"
    },
    "county_of_riverside": {
        "name": "County of Riverside",
        "type": "government",
        "lat": 33.9533,
        "lon": -117.3962,
        "state": "CA",
        "region": "Southern California",
        "notes": "Inland Empire distribution hub"
    },
    "taylor_transport": {
        "name": "Taylor Transport Inc",
        "type": "carrier",
        "lat": 33.8121,  # Approximate - LA area
        "lon": -118.1714,
        "state": "CA",
        "region": "Southern California",
        "notes": "Multi-region fleet operations"
    },
    "portland_vancouver_junction": {
        "name": "Portland Vancouver Junction Railroad",
        "type": "short_line_rail",
        "lat": 45.6387,
        "lon": -122.6615,
        "state": "OR",
        "region": "Pacific Northwest",
        "notes": "Class 2-3 short line, intermodal connection"
    },
    "great_plains_industrial": {
        "name": "Great Plains Industrial Park",
        "type": "industrial",
        "lat": 38.8816,  # Kansas area
        "lon": -99.3268,
        "state": "KS",
        "region": "Central Plains",
        "notes": "Agricultural freight surge patterns"
    },
    "kansas_proving_grounds": {
        "name": "Kansas Proving Grounds",
        "type": "testing",
        "lat": 39.0473,
        "lon": -95.6752,
        "state": "KS",
        "region": "Central Plains",
        "notes": "Low-volume, high-value cargo"
    },
    "mendocino_railway": {
        "name": "Mendocino Railway",
        "type": "short_line_rail",
        "lat": 39.4457,
        "lon": -123.8053,
        "state": "CA",
        "region": "Northern California",
        "notes": "Scenic railway with freight operations"
    }
}

# ============================================================================
# POTENTIAL EXPANSION TARGETS (for presentation)
# ============================================================================
EXPANSION_TARGETS: Dict[str, Dict] = {
    "port_of_la": {
        "name": "Port of Los Angeles",
        "lat": 33.7539,
        "lon": -118.2494,
        "priority": 1,
        "notes": "Busiest container port in Western Hemisphere"
    },
    "port_of_long_beach": {
        "name": "Port of Long Beach",
        "lat": 33.7599,
        "lon": -118.2179,
        "priority": 1,
        "notes": "Green technology leader, rail efficiency"
    },
    "pacific_harbor_line": {
        "name": "Pacific Harbor Line (PHL)",
        "lat": 33.7650,
        "lon": -118.2300,
        "priority": 1,
        "notes": "Critical gatekeeper for LA/LB port rail"
    },
    "bnsf_hobart": {
        "name": "BNSF Hobart Yard",
        "lat": 33.9816,
        "lon": -118.2126,
        "priority": 2,
        "notes": "Major intermodal facility near LA"
    }
}

# ============================================================================
# KEY US PORTS (from weather data)
# ============================================================================
US_PORTS = [
    {"name": "Port of Los Angeles", "lat": 33.75387, "lon": -118.24943},
    {"name": "Port of Long Beach", "lat": 33.75988, "lon": -118.21792},
    {"name": "Port of New York/New Jersey", "lat": 40.659203, "lon": -74.15757},
    {"name": "Port of Savannah", "lat": 32.08652, "lon": -81.08842},
    {"name": "Port of Houston", "lat": 29.738659, "lon": -95.26303},
    {"name": "Port of Seattle", "lat": 47.5628, "lon": -122.351944},
    {"name": "Port of Tacoma", "lat": 47.27764, "lon": -122.3986},
    {"name": "Port of Oakland", "lat": 37.79208, "lon": -122.28128},
    {"name": "Port of Virginia (Norfolk)", "lat": 36.878994, "lon": -76.31622},
    {"name": "Port of Charleston", "lat": 32.804253, "lon": -79.9269},
    {"name": "Port of Miami", "lat": 25.767996, "lon": -80.16958},
    {"name": "Port of New Orleans", "lat": 29.925602, "lon": -90.01483},
    {"name": "Port of Baltimore", "lat": 39.257183, "lon": -76.59836},
    {"name": "Port of Philadelphia", "lat": 39.903034, "lon": -75.13817},
    {"name": "Port of Boston", "lat": 42.35753, "lon": -71.02688},
]

# ============================================================================
# KEY RAIL TERMINALS (from weather data)
# ============================================================================
RAIL_TERMINALS = [
    {"name": "BNSF Los Angeles (Hobart)", "lat": 33.9816, "lon": -118.2126},
    {"name": "UP Los Angeles (ICTF)", "lat": 33.812305, "lon": -118.23236},
    {"name": "BNSF San Bernardino", "lat": 34.093575, "lon": -117.31148},
    {"name": "UP Oakland", "lat": 37.81809, "lon": -122.29037},
    {"name": "BNSF Chicago (Cicero)", "lat": 41.834515, "lon": -87.76529},
    {"name": "UP Chicago (Global 4)", "lat": 41.659496, "lon": -87.60886},
    {"name": "BNSF Kansas City", "lat": 39.10976, "lon": -94.62058},
    {"name": "UP Kansas City", "lat": 39.081963, "lon": -94.58693},
    {"name": "UP Portland", "lat": 45.554512, "lon": -122.706566},
    {"name": "BNSF Seattle", "lat": 47.5628, "lon": -122.351944},
]

# ============================================================================
# CLASS I RAILROAD OWNERS (for filtering)
# ============================================================================
CLASS_1_RAILROADS = ["BNSF", "UP", "NS", "CSX", "CN", "CP", "KCS", "CPKC"]

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================
@dataclass
class OptimizationConfig:
    """Parameters for the optimization engine"""
    # Cost weights (α, β, γ, δ in objective function)
    dwell_time_weight: float = 0.4
    empty_miles_weight: float = 0.3
    late_arrival_weight: float = 0.2
    energy_cost_weight: float = 0.1
    
    # Time windows
    max_dwell_hours: float = 12.0  # Target max dwell time
    dispatch_window_hours: float = 4.0  # Optimal window size
    
    # Backhaul optimization
    min_backhaul_load_pct: float = 0.6  # 60% minimum return load

OPTIMIZATION = OptimizationConfig()

# ============================================================================
# FORECASTING PARAMETERS
# ============================================================================
@dataclass
class ForecastingConfig:
    """Parameters for surge/dwell prediction models"""
    prediction_horizons: List[int] = None  # Hours: [24, 48, 72]
    lag_features: List[int] = None  # Days: [1, 2, 3, 5, 7]
    rolling_windows: List[int] = None  # Days: [3, 7, 14, 30]
    
    def __post_init__(self):
        self.prediction_horizons = [24, 48, 72]
        self.lag_features = [1, 2, 3, 5, 7]
        self.rolling_windows = [3, 7, 14, 30]

FORECASTING = ForecastingConfig()

# ============================================================================
# DASHBOARD SETTINGS
# ============================================================================
DASHBOARD_CONFIG = {
    "port": 8050,
    "debug": True,
    "map_center": {"lat": 39.8283, "lon": -98.5795},  # Center of US
    "map_zoom": 4,
    "refresh_interval_seconds": 300,  # 5 minutes
}

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
